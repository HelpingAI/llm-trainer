"""Multi-head attention implementation."""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention mechanism."""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1,
                 use_bias: bool = True, attention_dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.scale = 1.0 / math.sqrt(self.d_head)

        # Linear projections for Q, K, V
        self.q_proj = nn.Linear(d_model, d_model, bias=use_bias)
        self.k_proj = nn.Linear(d_model, d_model, bias=use_bias)
        self.v_proj = nn.Linear(d_model, d_model, bias=use_bias)
        self.out_proj = nn.Linear(d_model, d_model, bias=use_bias)

        # Dropout layers
        self.attention_dropout = nn.Dropout(attention_dropout)
        self.output_dropout = nn.Dropout(dropout)

        self._init_weights()

    def _init_weights(self):
        """Initialize weights using Xavier uniform."""
        for module in [self.q_proj, self.k_proj, self.v_proj, self.out_proj]:
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None,
                key_padding_mask: Optional[torch.Tensor] = None,
                return_attention: bool = False) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass of multi-head attention.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            attention_mask: Causal mask of shape (seq_len, seq_len)
            key_padding_mask: Padding mask of shape (batch_size, seq_len)
            return_attention: Whether to return attention weights
            
        Returns:
            output: Attention output of shape (batch_size, seq_len, d_model)
            attention_weights: Attention weights if return_attention=True
        """
        batch_size, seq_len, d_model = x.shape

        # Linear projections
        q = self.q_proj(x)  # (batch_size, seq_len, d_model)
        k = self.k_proj(x)  # (batch_size, seq_len, d_model)
        v = self.v_proj(x)  # (batch_size, seq_len, d_model)

        # Reshape for multi-head attention
        q = q.view(batch_size, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        # Shape: (batch_size, n_heads, seq_len, d_head)

        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        # Shape: (batch_size, n_heads, seq_len, seq_len)

        # Apply masks
        if attention_mask is not None:
            # Causal mask (lower triangular)
            scores = scores.masked_fill(attention_mask == 0, float('-inf'))

        if key_padding_mask is not None:
            # Padding mask
            key_padding_mask = key_padding_mask.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(key_padding_mask, float('-inf'))

        # Apply softmax
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.attention_dropout(attention_weights)

        # Apply attention to values
        out = torch.matmul(attention_weights, v)
        # Shape: (batch_size, n_heads, seq_len, d_head)

        # Concatenate heads
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)

        # Final linear projection
        out = self.out_proj(out)
        out = self.output_dropout(out)

        if return_attention:
            return out, attention_weights
        return out, None


def create_causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
    """Create causal (lower triangular) attention mask."""
    mask = torch.tril(torch.ones(seq_len, seq_len, device=device))
    return mask


def create_padding_mask(input_ids: torch.Tensor, pad_token_id: int) -> torch.Tensor:
    """Create padding mask from input token IDs."""
    return (input_ids != pad_token_id).bool()
