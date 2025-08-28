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
    
    save_model_safetensors = _safetensors_not_available
    load_model_safetensors = _safetensors_not_available
    is_safetensors_available = lambda: False
    convert_pytorch_to_safetensors = _safetensors_not_available
    get_safetensors_metadata = _safetensors_not_available
    list_safetensors_tensors = _safetensors_not_available

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
