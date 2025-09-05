"""
LLM Trainer: A complete framework for training Large Language Models from scratch.

This package provides:
- Custom Transformer architecture implementation
- BPE tokenizer from scratch
- wordpiece tokenizer
- Data loading and preprocessing pipelines
- Training infrastructure with distributed support
- Inference and generation capabilities
"""

__version__ = "0.2.4"
__author__ = "OEvortex"
__email__ = "helpingai5@gmail.com"
__authors__ = [{"name": "OEvortex", "email": "helpingai5@gmail.com"}]

from .models import TransformerLM
from .tokenizer import BPETokenizer
from .training import (
    Trainer,
    create_optimizer
)
from .config import ModelConfig, TrainingConfig

# Export kernel optimizations
from .kernels import *
from .kernels.fused_ops import FusedLinear, FusedRMSNorm, fused_cross_entropy
from .kernels.memory_efficient import (
    gradient_checkpointing, LowVRAMLinear, 
    offload_optimizer_states, empty_cache, efficient_attention_forward
)

# Export patching system
from .patching import patch_transformers, patch_trl

__all__ = [
    "TransformerLM",
    "BPETokenizer", 
    "Trainer",
    "create_optimizer",
    "ModelConfig",
    "TrainingConfig",
    "FusedLinear",
    "FusedRMSNorm",
    "fused_cross_entropy",
    "gradient_checkpointing",
    "LowVRAMLinear",
    "offload_optimizer_states",
    "empty_cache",
    "efficient_attention_forward",
    "patch_transformers",
    "patch_trl"
]