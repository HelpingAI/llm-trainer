"""
LLM Trainer: A complete framework for training Large Language Models from scratch.

This package provides:
- Custom Transformer architecture implementation
- BPE tokenizer from scratch
- Custom tokenizer wrapper for any tokenizer (e.g., Hugging Face)
- Data loading and preprocessing pipelines
- Training infrastructure with distributed support
- Inference and generation capabilities
"""

__version__ = "0.1.0"
__author__ = "Vortex"

from .models import TransformerLM
from .tokenizer import BPETokenizer, CustomTokenizerWrapper
from .training import Trainer
from .config import ModelConfig, TrainingConfig, TokenizerConfig

__all__ = [
    "TransformerLM",
    "BPETokenizer",
    "CustomTokenizerWrapper", 
    "Trainer",
    "ModelConfig",
    "TrainingConfig",
    "TokenizerConfig"
]
