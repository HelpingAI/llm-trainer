"""
Utility functions for LLM fine-tuning (TRL-style).
"""
from __future__ import annotations
import torch
import torch.nn as nn
from typing import Iterable, Tuple, Union, Optional
import logging

logger = logging.getLogger(__name__)


class RunningMoments:
    """Compute running mean and variance using Welford's algorithm."""

    def __init__(self):
        self.n = 0
        self.mean = 0.0
        self.M2 = 0.0

    def update(self, x: Iterable[float]):
        """Update running statistics with new values."""
        for value in x:
            self.n += 1
            delta = value - self.mean
            self.mean += delta / self.n
            delta2 = value - self.mean
            self.M2 += delta * delta2

    def get_mean_var(self) -> Tuple[float, float]:
        """Get current mean and variance."""
        var = self.M2 / (self.n - 1) if self.n > 1 else 0.0
        return self.mean, var

    def reset(self):
        """Reset all statistics."""
        self.n = 0
        self.mean = 0.0
        self.M2 = 0.0


def compute_accuracy(preds: torch.Tensor, labels: torch.Tensor) -> float:
    """Compute accuracy between predictions and labels."""
    if not hasattr(preds, "float") or not hasattr(labels, "float"):
        return 0.0
    return float((preds == labels).float().mean().item())


def disable_dropout_in_model(model: nn.Module) -> nn.Module:
    """Disable dropout in all modules of the model."""
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.p = 0.0
        elif hasattr(module, "dropout") and hasattr(module.dropout, "p"):
            module.dropout.p = 0.0
        elif hasattr(module, "p") and isinstance(module.p, float):
            try:
                module.p = 0.0
            except Exception:
                pass
    return model


def empty_cache():
    """Clear GPU cache if CUDA is available."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def peft_module_casting_to_bf16(model: nn.Module) -> nn.Module:
    """Cast PEFT modules to bfloat16 for memory efficiency."""
    try:
        if hasattr(model, "peft_config"):
            # Only cast PEFT parameters to bf16
            for name, module in model.named_modules():
                if "lora" in name.lower() or "adapter" in name.lower():
                    module.to(dtype=torch.bfloat16)
        else:
            model = model.to(dtype=torch.bfloat16)
        return model
    except Exception as e:
        logger.warning(f"Failed to cast model to bf16: {e}")
        return model


def get_model_param_count(model: nn.Module, trainable_only: bool = False) -> int:
    """Get the number of parameters in a model."""
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def freeze_model_layers(model: nn.Module, num_layers_to_freeze: int = 0) -> nn.Module:
    """Freeze the first N layers of a transformer model."""
    if num_layers_to_freeze <= 0:
        return model

    layer_count = 0
    for name, param in model.named_parameters():
        if "layer" in name.lower() or "block" in name.lower():
            if layer_count < num_layers_to_freeze:
                param.requires_grad = False
                layer_count += 1

    logger.info(f"Frozen {layer_count} layers in the model")
    return model

