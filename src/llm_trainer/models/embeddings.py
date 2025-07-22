"""Embedding layers for Transformer models."""

import math
import torch
import torch.nn as nn
from typing import Optional


class TokenEmbedding(nn.Module):
    """Token embedding layer with optional weight tying."""
    
    def __init__(self, vocab_size: int, d_model: int, padding_idx: Optional[int] = None):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=padding_idx)
        self._init_weights()
    
    def _init_weights(self):
        """Initialize embedding weights."""
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)
        if self.embedding.padding_idx is not None:
            nn.init.zeros_(self.embedding.weight[self.embedding.padding_idx])
    
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            input_ids: Token IDs of shape (batch_size, seq_len)
            
        Returns:
            embeddings: Token embeddings of shape (batch_size, seq_len, d_model)
        """
        embeddings = self.embedding(input_ids)
        # Scale embeddings by sqrt(d_model) as in original Transformer paper
        return embeddings * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""
    
    def __init__(self, d_model: int, max_seq_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)
        
        # Create positional encoding matrix
        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        
        # Create div_term for sinusoidal pattern
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        # Apply sin to even indices
        pe[:, 0::2] = torch.sin(position * div_term)
        # Apply cos to odd indices
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Add batch dimension and register as buffer
        pe = pe.unsqueeze(0)  # Shape: (1, max_seq_len, d_model)
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to input embeddings.
        
        Args:
            x: Input embeddings of shape (batch_size, seq_len, d_model)
            
        Returns:
            x: Embeddings with positional encoding added
        """
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len, :]
        return self.dropout(x)


class LearnedPositionalEmbedding(nn.Module):
    """Learned positional embeddings (alternative to sinusoidal)."""
    
    def __init__(self, max_seq_len: int, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.d_model = d_model
        self.pos_embedding = nn.Embedding(max_seq_len, d_model)
        self.dropout = nn.Dropout(dropout)
        self._init_weights()
    
    def _init_weights(self):
        """Initialize positional embedding weights."""
        nn.init.normal_(self.pos_embedding.weight, mean=0.0, std=0.02)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add learned positional embeddings to input.
        
        Args:
            x: Input embeddings of shape (batch_size, seq_len, d_model)
            
        Returns:
            x: Embeddings with positional encoding added
        """
        batch_size, seq_len, d_model = x.shape
        
        # Create position indices
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, -1)
        
        # Get positional embeddings
        pos_emb = self.pos_embedding(positions)
        
        # Add to input embeddings
        x = x + pos_emb
        return self.dropout(x)


class CombinedEmbedding(nn.Module):
    """Combined token and positional embeddings."""
    
    def __init__(self, vocab_size: int, d_model: int, max_seq_len: int,
                 padding_idx: Optional[int] = None, use_learned_pos: bool = False,
                 dropout: float = 0.1):
        super().__init__()
        
        # Token embeddings
        self.token_embedding = TokenEmbedding(vocab_size, d_model, padding_idx)
        
        # Positional embeddings
        if use_learned_pos:
            self.pos_embedding = LearnedPositionalEmbedding(max_seq_len, d_model, dropout)
        else:
            self.pos_embedding = PositionalEncoding(d_model, max_seq_len, dropout)
    
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Forward pass combining token and positional embeddings.
        
        Args:
            input_ids: Token IDs of shape (batch_size, seq_len)
            
        Returns:
            embeddings: Combined embeddings of shape (batch_size, seq_len, d_model)
        """
        # Get token embeddings
        token_emb = self.token_embedding(input_ids)
        
        # Add positional embeddings
        embeddings = self.pos_embedding(token_emb)
        
        return embeddings
