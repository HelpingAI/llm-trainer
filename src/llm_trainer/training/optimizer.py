"""Optimizer utilities."""

import torch
import torch.nn as nn
from typing import Dict, Any, List, Optional, Union
import logging


def create_optimizer(model: nn.Module,
                    optimizer_name: str = "adamw",
                    learning_rate: float = 1e-4,
                    weight_decay: float = 0.01,
                    beta1: float = 0.9,
                    beta2: float = 0.999,
                    eps: float = 1e-8,
                    momentum: float = 0.9,
                    **kwargs) -> torch.optim.Optimizer:
    """Create optimizer for model training."""

    # Get model parameters with weight decay configuration
    param_groups = get_parameter_groups(model, weight_decay)

    if optimizer_name.lower() == "adamw":
        optimizer = torch.optim.AdamW(
            param_groups,
            lr=learning_rate,
            betas=(beta1, beta2),
            eps=eps,
            weight_decay=weight_decay
        )
    elif optimizer_name.lower() == "adam":
        optimizer = torch.optim.Adam(
            param_groups,
            lr=learning_rate,
            betas=(beta1, beta2),
            eps=eps,
            weight_decay=weight_decay
        )
    elif optimizer_name.lower() == "sgd":
        optimizer = torch.optim.SGD(
            param_groups,
            lr=learning_rate,
            momentum=momentum,
            weight_decay=weight_decay
        )
    elif optimizer_name.lower() == "rmsprop":
        optimizer = torch.optim.RMSprop(
            param_groups,
            lr=learning_rate,
            eps=eps,
            weight_decay=weight_decay,
            momentum=momentum
        )
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")

    logging.info(f"Created {optimizer_name} optimizer with lr={learning_rate}")
    return optimizer


def get_parameter_groups(model: nn.Module, weight_decay: float) -> List[Dict[str, Any]]:
    """
    Create parameter groups with different weight decay settings.
    
    Typically, we don't apply weight decay to:
    - Bias terms
    - Layer normalization parameters
    - Embedding parameters
    """
    no_decay = ["bias", "LayerNorm.weight", "layernorm.weight", "layer_norm.weight"]

    # Separate parameters into groups
    decay_params = []
    no_decay_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        # Check if parameter should have weight decay
        if any(nd in name for nd in no_decay):
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    # Create parameter groups
    param_groups = [
        {
            "params": decay_params,
            "weight_decay": weight_decay,
        },
        {
            "params": no_decay_params,
            "weight_decay": 0.0,
        }
    ]

    logging.info(f"Parameter groups: {len(decay_params)} with decay, {len(no_decay_params)} without decay")
    return param_groups


class GradientClipping:
    """Gradient clipping utilities."""

    def __init__(self, max_norm: float = 1.0, norm_type: float = 2.0):
        self.max_norm = max_norm
        self.norm_type = norm_type

    def clip_gradients(self, model: nn.Module) -> float:
        """Clip gradients and return the gradient norm."""
        if self.max_norm <= 0:
            return 0.0

        # Get all parameters with gradients
        parameters = [p for p in model.parameters() if p.grad is not None]

        if not parameters:
            return 0.0

        # Clip gradients
        grad_norm = torch.nn.utils.clip_grad_norm_(
            parameters,
            max_norm=self.max_norm,
            norm_type=self.norm_type
        )

        return grad_norm.item()


class LearningRateWarmup:
    """Learning rate warmup scheduler."""

    def __init__(self,
                 optimizer: torch.optim.Optimizer,
                 warmup_steps: int,
                 initial_lr: Optional[float] = None):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.current_step = 0

        # Store initial learning rates
        if initial_lr is not None:
            self.base_lrs = [initial_lr] * len(optimizer.param_groups)
        else:
            self.base_lrs = [group['lr'] for group in optimizer.param_groups]

        # Set initial learning rate to 0
        for group in optimizer.param_groups:
            group['lr'] = 0.0

    def step(self) -> None:
        """Update learning rate for warmup."""
        if self.current_step < self.warmup_steps:
            # Linear warmup
            lr_scale = (self.current_step + 1) / self.warmup_steps

            for group, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
                group['lr'] = base_lr * lr_scale

        self.current_step += 1

    def get_lr(self) -> List[float]:
        """Get current learning rates."""
        return [group['lr'] for group in self.optimizer.param_groups]


class ExponentialMovingAverage:
    """Exponential Moving Average for model parameters."""

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}

        # Initialize shadow parameters
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self) -> None:
        """Update EMA parameters."""
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.shadow[name] = (
                    self.decay * self.shadow[name] +
                    (1.0 - self.decay) * param.data
                )

    def apply_shadow(self) -> None:
        """Apply EMA parameters to model."""
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]

    def restore(self) -> None:
        """Restore original parameters."""
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.backup:
                param.data = self.backup[name]
        self.backup.clear()


class OptimizerState:
    """Utilities for managing optimizer state."""

    @staticmethod
    def get_lr(optimizer: torch.optim.Optimizer) -> List[float]:
        """Get current learning rates from optimizer."""
        return [group['lr'] for group in optimizer.param_groups]

    @staticmethod
    def set_lr(optimizer: torch.optim.Optimizer, lr: Union[float, List[float]]) -> None:
        """Set learning rates for optimizer."""
        if isinstance(lr, (int, float)):
            lr = [lr] * len(optimizer.param_groups)

        for group, new_lr in zip(optimizer.param_groups, lr):
            group['lr'] = new_lr

    @staticmethod
    def zero_grad(optimizer: torch.optim.Optimizer, set_to_none: bool = True) -> None:
        """Zero gradients with option to set to None for memory efficiency."""
        if set_to_none:
            for group in optimizer.param_groups:
                for param in group['params']:
                    param.grad = None
        else:
            optimizer.zero_grad()

    @staticmethod
    def get_optimizer_info(optimizer: torch.optim.Optimizer) -> Dict[str, Any]:
        """Get optimizer information."""
        info = {
            "type": type(optimizer).__name__,
            "param_groups": len(optimizer.param_groups),
            "learning_rates": OptimizerState.get_lr(optimizer),
            "state_dict_keys": list(optimizer.state_dict().keys())
        }

        # Get parameter counts per group
        param_counts = []
        for group in optimizer.param_groups:
            count = sum(p.numel() for p in group['params'])
            param_counts.append(count)
        info["param_counts"] = param_counts

        return info
