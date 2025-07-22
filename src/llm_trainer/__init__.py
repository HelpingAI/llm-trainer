"""
LLM Trainer: A complete framework for training Large Language Models from scratch.

This package provides:
- Custom Transformer architecture implementation
- BPE tokenizer from scratch
- Data loading and preprocessing pipelines
- Training infrastructure with distributed support
- Inference and generation capabilities
"""

__version__ = "0.1.0"
__author__ = "Vortex"

from .models import TransformerLM
from .tokenizer import BPETokenizer
from .training import Trainer
from .config import ModelConfig, TrainingConfig

__all__ = [
    "TransformerLM",
    "BPETokenizer", 
    "Trainer",
    "ModelConfig",
    "TrainingConfig"
]
