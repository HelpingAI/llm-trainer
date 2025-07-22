"""Transformer layers and components."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

from .attention import MultiHeadAttention


class FeedForward(nn.Module):
    """Position-wise feed-forward network."""
    
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1, 
                 activation: str = "gelu", use_bias: bool = True):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        
        # Two linear layers with activation in between
        self.linear1 = nn.Linear(d_model, d_ff, bias=use_bias)
        self.linear2 = nn.Linear(d_ff, d_model, bias=use_bias)
        self.dropout = nn.Dropout(dropout)
        
        # Activation function
        if activation == "relu":
            self.activation = F.relu
        elif activation == "gelu":
            self.activation = F.gelu
        elif activation == "swish":
            self.activation = F.silu  # SiLU is the same as Swish
        else:
            raise ValueError(f"Unsupported activation: {activation}")
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights."""
        nn.init.xavier_uniform_(self.linear1.weight)
        nn.init.xavier_uniform_(self.linear2.weight)
        if self.linear1.bias is not None:
            nn.init.zeros_(self.linear1.bias)
        if self.linear2.bias is not None:
            nn.init.zeros_(self.linear2.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            
        Returns:
            output: Output tensor of shape (batch_size, seq_len, d_model)
        """
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x


class TransformerBlock(nn.Module):
    """Single Transformer decoder block."""
    
    def __init__(self, d_model: int, n_heads: int, d_ff: int, 
                 dropout: float = 0.1, attention_dropout: float = 0.1,
                 activation: str = "gelu", use_bias: bool = True,
                 layer_norm_eps: float = 1e-5, pre_norm: bool = True):
        super().__init__()
        self.d_model = d_model
        self.pre_norm = pre_norm
        
        # Multi-head self-attention
        self.self_attention = MultiHeadAttention(
            d_model=d_model,
            n_heads=n_heads,
            dropout=dropout,
            use_bias=use_bias,
            attention_dropout=attention_dropout
        )
        
        # Feed-forward network
        self.feed_forward = FeedForward(
            d_model=d_model,
            d_ff=d_ff,
            dropout=dropout,
            activation=activation,
            use_bias=use_bias
        )
        
        # Layer normalization
        self.ln1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.ln2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        
        # Dropout for residual connections
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None,
                key_padding_mask: Optional[torch.Tensor] = None,
                return_attention: bool = False) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass of Transformer block.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            attention_mask: Causal mask for self-attention
            key_padding_mask: Padding mask
            return_attention: Whether to return attention weights
            
        Returns:
            output: Output tensor of shape (batch_size, seq_len, d_model)
            attention_weights: Attention weights if return_attention=True
        """
        attention_weights = None
        
        if self.pre_norm:
            # Pre-norm: LayerNorm -> Attention -> Residual
            attn_input = self.ln1(x)
            attn_output, attention_weights = self.self_attention(
                attn_input, attention_mask, key_padding_mask, return_attention
            )
            x = x + self.dropout1(attn_output)
            
            # Pre-norm: LayerNorm -> FFN -> Residual
            ff_input = self.ln2(x)
            ff_output = self.feed_forward(ff_input)
            x = x + self.dropout2(ff_output)
        else:
            # Post-norm: Attention -> Residual -> LayerNorm
            attn_output, attention_weights = self.self_attention(
                x, attention_mask, key_padding_mask, return_attention
            )
            x = self.ln1(x + self.dropout1(attn_output))
            
            # Post-norm: FFN -> Residual -> LayerNorm
            ff_output = self.feed_forward(x)
            x = self.ln2(x + self.dropout2(ff_output))
        
        return x, attention_weights


class TransformerStack(nn.Module):
    """Stack of Transformer blocks."""
    
    def __init__(self, n_layers: int, d_model: int, n_heads: int, d_ff: int,
                 dropout: float = 0.1, attention_dropout: float = 0.1,
                 activation: str = "gelu", use_bias: bool = True,
                 layer_norm_eps: float = 1e-5, pre_norm: bool = True,
                 gradient_checkpointing: bool = False):
        super().__init__()
        self.n_layers = n_layers
        self.gradient_checkpointing = gradient_checkpointing
        
        # Stack of Transformer blocks
        self.layers = nn.ModuleList([
            TransformerBlock(
                d_model=d_model,
                n_heads=n_heads,
                d_ff=d_ff,
                dropout=dropout,
                attention_dropout=attention_dropout,
                activation=activation,
                use_bias=use_bias,
                layer_norm_eps=layer_norm_eps,
                pre_norm=pre_norm
            )
            for _ in range(n_layers)
        ])
        
        # Final layer norm (for pre-norm architecture)
        self.final_ln = nn.LayerNorm(d_model, eps=layer_norm_eps) if pre_norm else None
    
    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None,
                key_padding_mask: Optional[torch.Tensor] = None,
                return_attention: bool = False) -> Tuple[torch.Tensor, Optional[list]]:
        """
        Forward pass through all Transformer layers.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            attention_mask: Causal mask for self-attention
            key_padding_mask: Padding mask
            return_attention: Whether to return attention weights from all layers
            
        Returns:
            output: Output tensor of shape (batch_size, seq_len, d_model)
            all_attention_weights: List of attention weights from all layers if return_attention=True
        """
        all_attention_weights = [] if return_attention else None
        
        for layer in self.layers:
            if self.gradient_checkpointing and self.training:
                # Use gradient checkpointing to save memory
                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs, return_attention=False)
                    return custom_forward
                
                x = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer),
                    x, attention_mask, key_padding_mask
                )[0]
                attention_weights = None
            else:
                x, attention_weights = layer(x, attention_mask, key_padding_mask, return_attention)
            
            if return_attention and attention_weights is not None:
                all_attention_weights.append(attention_weights)
        
        # Apply final layer norm for pre-norm architecture
        if self.final_ln is not None:
            x = self.final_ln(x)
        
        return x, all_attention_weights
