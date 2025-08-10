"""Transformer model implementations and base interfaces.

This module exposes:
- TransformerLM: our built-in Transformer implementation
- BaseLanguageModel / HuggingFaceModelWrapper: a common interface and HF wrapper

Users can plug in their own architectures by implementing BaseLanguageModel.
"""

from .transformer import TransformerLM
from .attention import MultiHeadAttention
from .layers import TransformerBlock, FeedForward
from .embeddings import PositionalEncoding, TokenEmbedding
from .base_model import BaseLanguageModel, HuggingFaceModelWrapper

__all__ = [
    "TransformerLM",
    "MultiHeadAttention",
    "TransformerBlock",
    "FeedForward",
    "PositionalEncoding",
    "TokenEmbedding",
    "BaseLanguageModel",
    "HuggingFaceModelWrapper",
]
