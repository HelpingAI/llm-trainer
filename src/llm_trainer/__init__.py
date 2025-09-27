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

from importlib import import_module
from typing import Any, Dict, Tuple

__version__ = "0.2.6"
__author__ = "Abhay"
__email__ = "abhay@helpingai.co"
__authors__ = [{"name": "Abhay", "email": "abhay@helpingai.co"}]

_SYMBOL_MAP: Dict[str, Tuple[str, str]] = {
    "TransformerLM": ("llm_trainer.models", "TransformerLM"),
    "BPETokenizer": ("llm_trainer.tokenizer", "BPETokenizer"),
    "Trainer": ("llm_trainer.training.trainer", "Trainer"),
    "create_optimizer": ("llm_trainer.training.optimizer", "create_optimizer"),
    "ModelConfig": ("llm_trainer.config.model_config", "ModelConfig"),
    "TrainingConfig": ("llm_trainer.config.training_config", "TrainingConfig"),
    "FusedLinear": ("llm_trainer.kernels.fused_ops", "FusedLinear"),
    "FusedRMSNorm": ("llm_trainer.kernels.fused_ops", "FusedRMSNorm"),
    "fused_cross_entropy": ("llm_trainer.kernels.fused_ops", "fused_cross_entropy"),
    "gradient_checkpointing": ("llm_trainer.kernels.memory_efficient", "gradient_checkpointing"),
    "LowVRAMLinear": ("llm_trainer.kernels.memory_efficient", "LowVRAMLinear"),
    "offload_optimizer_states": ("llm_trainer.kernels.memory_efficient", "offload_optimizer_states"),
    "empty_cache": ("llm_trainer.kernels.memory_efficient", "empty_cache"),
    "efficient_attention_forward": ("llm_trainer.kernels.memory_efficient", "efficient_attention_forward"),
    "patch_transformers": ("llm_trainer.patching.patch_transformers", "patch_transformers"),
    "patch_trl": ("llm_trainer.patching.patch_trl", "patch_trl"),
}

__all__ = list(_SYMBOL_MAP.keys())


def __getattr__(name: str) -> Any:
    if name not in _SYMBOL_MAP:
        raise AttributeError(f"module 'llm_trainer' has no attribute '{name}'")

    module_name, attr_name = _SYMBOL_MAP[name]
    module = import_module(module_name)
    return getattr(module, attr_name)


def __dir__() -> Any:
    return sorted(list(globals().keys()) + __all__)