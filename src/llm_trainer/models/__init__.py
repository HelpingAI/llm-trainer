"""Transformer model implementations and base interfaces.

This module exposes:
- TransformerLM: our built-in Transformer implementation
- BaseLanguageModel / HuggingFaceModelWrapper: a common interface and HF wrapper
- SafeTensors utilities for efficient model saving/loading with sharding support

Users can plug in their own architectures by implementing BaseLanguageModel.
"""

from .transformer import TransformerLM
from .attention import MultiHeadAttention
from .layers import TransformerBlock, FeedForward
from .embeddings import PositionalEncoding, TokenEmbedding
from .base_model import BaseLanguageModel, HuggingFaceModelWrapper

# SafeTensors utilities (optional import)
try:
    from .safetensors_utils import (
        save_model_safetensors,
        load_model_safetensors,
        is_safetensors_available,
        convert_pytorch_to_safetensors,
        get_safetensors_metadata,
        list_safetensors_tensors
    )
    _SAFETENSORS_AVAILABLE = True
except ImportError:
    _SAFETENSORS_AVAILABLE = False
    # Create dummy functions that warn about missing SafeTensors
    def _safetensors_not_available(*args, **kwargs):
        raise ImportError("SafeTensors not available. Install with: pip install safetensors")

    def save_model_safetensors(*args, **kwargs):  # type: ignore
        _safetensors_not_available(*args, **kwargs)
    
    def load_model_safetensors(*args, **kwargs):  # type: ignore
        _safetensors_not_available(*args, **kwargs)
    
    def is_safetensors_available() -> bool:  # type: ignore
        return False
    
    def convert_pytorch_to_safetensors(*args, **kwargs):  # type: ignore
        _safetensors_not_available(*args, **kwargs)
    
    def get_safetensors_metadata(*args, **kwargs):  # type: ignore
        return _safetensors_not_available(*args, **kwargs)
    
    def list_safetensors_tensors(*args, **kwargs):  # type: ignore
        return _safetensors_not_available(*args, **kwargs)

__all__ = [
    "TransformerLM",
    "MultiHeadAttention",
    "TransformerBlock",
    "FeedForward",
    "PositionalEncoding",
    "TokenEmbedding",
    "BaseLanguageModel",
    "HuggingFaceModelWrapper",
    # SafeTensors utilities
    "save_model_safetensors",
    "load_model_safetensors",
    "is_safetensors_available",
    "convert_pytorch_to_safetensors",
    "get_safetensors_metadata",
    "list_safetensors_tensors",
]
