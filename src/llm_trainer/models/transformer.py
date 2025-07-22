"""Main Transformer Language Model implementation."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any
import math

from ..config import ModelConfig
from .embeddings import CombinedEmbedding
from .layers import TransformerStack
from .attention import create_causal_mask, create_padding_mask


class TransformerLM(nn.Module):
    """Transformer Language Model for autoregressive text generation."""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        # Embeddings
        self.embeddings = CombinedEmbedding(
            vocab_size=config.vocab_size,
            d_model=config.d_model,
            max_seq_len=config.max_seq_len,
            padding_idx=0,  # Assuming pad token ID is 0
            use_learned_pos=config.use_learned_pos_emb,
            dropout=config.dropout
        )
        
        # Transformer stack
        self.transformer = TransformerStack(
            n_layers=config.n_layers,
            d_model=config.d_model,
            n_heads=config.n_heads,
            d_ff=config.d_ff,
            dropout=config.dropout,
            attention_dropout=config.attention_dropout,
            activation=config.activation,
            use_bias=config.use_bias,
            layer_norm_eps=config.layer_norm_eps,
            pre_norm=config.pre_norm,
            gradient_checkpointing=config.gradient_checkpointing
        )
        
        # Language modeling head
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        
        # Initialize weights
        self.apply(self._init_weights)
        
        # Tie embeddings and output weights (optional)
        if hasattr(config, 'tie_weights') and config.tie_weights:
            self.lm_head.weight = self.embeddings.token_embedding.embedding.weight
    
    def _init_weights(self, module):
        """Initialize model weights."""
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.init_std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.init_std)
            if module.padding_idx is not None:
                nn.init.zeros_(module.weight[module.padding_idx])
        elif isinstance(module, nn.LayerNorm):
            nn.init.zeros_(module.bias)
            nn.init.ones_(module.weight)
    
    def forward(self, input_ids: torch.Tensor, 
                attention_mask: Optional[torch.Tensor] = None,
                labels: Optional[torch.Tensor] = None,
                return_attention: bool = False,
                return_hidden_states: bool = False) -> Dict[str, torch.Tensor]:
        """
        Forward pass of the Transformer LM.
        
        Args:
            input_ids: Token IDs of shape (batch_size, seq_len)
            attention_mask: Attention mask of shape (batch_size, seq_len)
            labels: Target labels for language modeling loss (batch_size, seq_len)
            return_attention: Whether to return attention weights
            return_hidden_states: Whether to return hidden states
            
        Returns:
            Dictionary containing:
                - logits: Output logits of shape (batch_size, seq_len, vocab_size)
                - loss: Language modeling loss (if labels provided)
                - hidden_states: Hidden states (if return_hidden_states=True)
                - attention_weights: Attention weights (if return_attention=True)
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        
        # Create causal mask
        causal_mask = create_causal_mask(seq_len, device)
        
        # Create padding mask if attention_mask is provided
        key_padding_mask = None
        if attention_mask is not None:
            key_padding_mask = ~attention_mask.bool()
        
        # Get embeddings
        hidden_states = self.embeddings(input_ids)
        
        # Pass through transformer
        hidden_states, all_attention_weights = self.transformer(
            hidden_states,
            attention_mask=causal_mask,
            key_padding_mask=key_padding_mask,
            return_attention=return_attention
        )
        
        # Language modeling head
        logits = self.lm_head(hidden_states)
        
        # Prepare output
        output = {"logits": logits}
        
        # Compute loss if labels are provided
        if labels is not None:
            # Shift labels for next token prediction
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            # Flatten for cross entropy
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100  # Ignore padding tokens
            )
            output["loss"] = loss
        
        # Add optional outputs
        if return_hidden_states:
            output["hidden_states"] = hidden_states
        
        if return_attention:
            output["attention_weights"] = all_attention_weights
        
        return output
    
    def generate(self, input_ids: torch.Tensor, max_length: int = 100,
                 temperature: float = 1.0, top_k: Optional[int] = None,
                 top_p: Optional[float] = None, do_sample: bool = True,
                 pad_token_id: int = 0, eos_token_id: int = 3) -> torch.Tensor:
        """
        Generate text using the model.
        
        Args:
            input_ids: Input token IDs of shape (batch_size, seq_len)
            max_length: Maximum length of generated sequence
            temperature: Sampling temperature
            top_k: Top-k sampling parameter
            top_p: Nucleus sampling parameter
            do_sample: Whether to use sampling or greedy decoding
            pad_token_id: Padding token ID
            eos_token_id: End-of-sequence token ID
            
        Returns:
            generated_ids: Generated token IDs of shape (batch_size, max_length)
        """
        self.eval()
        batch_size = input_ids.shape[0]
        device = input_ids.device
        
        # Initialize generated sequence with input
        generated = input_ids.clone()
        
        with torch.no_grad():
            for _ in range(max_length - input_ids.shape[1]):
                # Forward pass
                outputs = self.forward(generated)
                logits = outputs["logits"]
                
                # Get logits for next token
                next_token_logits = logits[:, -1, :] / temperature
                
                # Apply top-k filtering
                if top_k is not None:
                    top_k = min(top_k, logits.size(-1))
                    indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                    next_token_logits[indices_to_remove] = float('-inf')
                
                # Apply top-p (nucleus) filtering
                if top_p is not None:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    
                    # Remove tokens with cumulative probability above the threshold
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    
                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    next_token_logits[indices_to_remove] = float('-inf')
                
                # Sample next token
                if do_sample:
                    probs = F.softmax(next_token_logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                else:
                    next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                
                # Append to generated sequence
                generated = torch.cat([generated, next_token], dim=1)
                
                # Check for EOS token
                if (next_token == eos_token_id).all():
                    break
        
        return generated
    
    def get_num_params(self, non_embedding: bool = False) -> int:
        """Get number of parameters in the model."""
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.embeddings.token_embedding.embedding.weight.numel()
            if hasattr(self.embeddings.pos_embedding, 'pos_embedding'):
                n_params -= self.embeddings.pos_embedding.pos_embedding.weight.numel()
        return n_params
    
    @classmethod
    def from_config(cls, config: ModelConfig) -> 'TransformerLM':
        """Create model from configuration."""
        return cls(config)
    
    def save_pretrained(self, save_directory: str) -> None:
        """Save model and configuration."""
        import os
        os.makedirs(save_directory, exist_ok=True)
        
        # Save model state dict
        model_path = os.path.join(save_directory, "pytorch_model.bin")
        torch.save(self.state_dict(), model_path)
        
        # Save configuration
        config_path = os.path.join(save_directory, "config.json")
        self.config.save(config_path)
    
    @classmethod
    def from_pretrained(cls, load_directory: str) -> 'TransformerLM':
        """Load model from directory."""
        import os
        
        # Load configuration
        config_path = os.path.join(load_directory, "config.json")
        config = ModelConfig.load(config_path)
        
        # Create model
        model = cls(config)
        
        # Load state dict
        model_path = os.path.join(load_directory, "pytorch_model.bin")
        state_dict = torch.load(model_path, map_location='cpu')
        model.load_state_dict(state_dict)
        
        return model
