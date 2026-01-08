"""Learning rate scheduler utilities."""

import math
import torch
from torch.optim.lr_scheduler import _LRScheduler
from typing import List, Optional, Union, cast
import logging


def create_scheduler(optimizer: torch.optim.Optimizer,
                    scheduler_name: str = "cosine",
                    num_training_steps: int = 1000,
                    warmup_steps: int = 100,
                    min_lr_ratio: float = 0.1,
                    **kwargs) -> Optional[_LRScheduler]:
    """Create learning rate scheduler."""

    if scheduler_name.lower() == "cosine":
        scheduler = CosineAnnealingWithWarmup(
            optimizer=optimizer,
            warmup_steps=warmup_steps,
            total_steps=num_training_steps,
            min_lr_ratio=min_lr_ratio
        )
        return cast(_LRScheduler, scheduler)
    elif scheduler_name.lower() == "linear":
        scheduler = LinearDecayWithWarmup(
            optimizer=optimizer,
            warmup_steps=warmup_steps,
            total_steps=num_training_steps,
            min_lr_ratio=min_lr_ratio
        )
        return cast(_LRScheduler, scheduler)
    elif scheduler_name.lower() == "polynomial":
        power = kwargs.get("power", 1.0)
        scheduler = PolynomialDecayWithWarmup(
            optimizer=optimizer,
            warmup_steps=warmup_steps,
            total_steps=num_training_steps,
            min_lr_ratio=min_lr_ratio,
            power=power
        )
        return cast(_LRScheduler, scheduler)
    elif scheduler_name.lower() == "constant":
        scheduler = ConstantWithWarmup(
            optimizer=optimizer,
            warmup_steps=warmup_steps
        )
        return cast(_LRScheduler, scheduler)
    elif scheduler_name.lower() == "step":
        step_size = kwargs.get("step_size", num_training_steps // 3)
        gamma = kwargs.get("gamma", 0.1)
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer=optimizer,
            step_size=step_size,
            gamma=gamma
        )
        return cast(_LRScheduler, scheduler)
    elif scheduler_name.lower() == "exponential":
        gamma = kwargs.get("gamma", 0.95)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer=optimizer,
            gamma=gamma
        )
        return cast(_LRScheduler, scheduler)
    elif scheduler_name.lower() == "none" or scheduler_name.lower() == "constant_no_warmup":
        return None
    else:
        raise ValueError(f"Unsupported scheduler: {scheduler_name}")

    logging.info(f"Created {scheduler_name} scheduler")
    return scheduler


class CosineAnnealingWithWarmup(_LRScheduler):
    """Cosine annealing learning rate scheduler with warmup."""

    def __init__(self,
                 optimizer: torch.optim.Optimizer,
                 warmup_steps: int,
                 total_steps: int,
                 min_lr_ratio: float = 0.1,
                 last_epoch: int = -1):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr_ratio = min_lr_ratio
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        """Calculate learning rate for current step."""
        if self.last_epoch < self.warmup_steps:
            # Warmup phase: linear increase
            lr_scale = (self.last_epoch + 1) / self.warmup_steps
            return [base_lr * lr_scale for base_lr in self.base_lrs]
        else:
            # Cosine annealing phase
            progress = (self.last_epoch - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            progress = min(progress, 1.0)  # Clamp to [0, 1]

            lr_scale = self.min_lr_ratio + (1 - self.min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * progress))
            return [base_lr * lr_scale for base_lr in self.base_lrs]


class LinearDecayWithWarmup(_LRScheduler):
    """Linear decay learning rate scheduler with warmup."""

    def __init__(self,
                 optimizer: torch.optim.Optimizer,
                 warmup_steps: int,
                 total_steps: int,
                 min_lr_ratio: float = 0.1,
                 last_epoch: int = -1):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr_ratio = min_lr_ratio
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        """Calculate learning rate for current step."""
        if self.last_epoch < self.warmup_steps:
            # Warmup phase: linear increase
            lr_scale = (self.last_epoch + 1) / self.warmup_steps
            return [base_lr * lr_scale for base_lr in self.base_lrs]
        else:
            # Linear decay phase
            progress = (self.last_epoch - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            progress = min(progress, 1.0)  # Clamp to [0, 1]

            lr_scale = self.min_lr_ratio + (1 - self.min_lr_ratio) * (1 - progress)
            return [base_lr * lr_scale for base_lr in self.base_lrs]


class PolynomialDecayWithWarmup(_LRScheduler):
    """Polynomial decay learning rate scheduler with warmup."""

    def __init__(self,
                 optimizer: torch.optim.Optimizer,
                 warmup_steps: int,
                 total_steps: int,
                 min_lr_ratio: float = 0.1,
                 power: float = 1.0,
                 last_epoch: int = -1):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr_ratio = min_lr_ratio
        self.power = power
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        """Calculate learning rate for current step."""
        if self.last_epoch < self.warmup_steps:
            # Warmup phase: linear increase
            lr_scale = (self.last_epoch + 1) / self.warmup_steps
            return [base_lr * lr_scale for base_lr in self.base_lrs]
        else:
            # Polynomial decay phase
            progress = (self.last_epoch - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            progress = min(progress, 1.0)  # Clamp to [0, 1]

            lr_scale = self.min_lr_ratio + (1 - self.min_lr_ratio) * ((1 - progress) ** self.power)
            return [base_lr * lr_scale for base_lr in self.base_lrs]


class ConstantWithWarmup(_LRScheduler):
    """Constant learning rate scheduler with warmup."""

    def __init__(self,
                 optimizer: torch.optim.Optimizer,
                 warmup_steps: int,
                 last_epoch: int = -1):
        self.warmup_steps = warmup_steps
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        """Calculate learning rate for current step."""
        if self.last_epoch < self.warmup_steps:
            # Warmup phase: linear increase
            lr_scale = (self.last_epoch + 1) / self.warmup_steps
            return [base_lr * lr_scale for base_lr in self.base_lrs]
        else:
            # Constant phase
            return self.base_lrs


class OneCycleLR(_LRScheduler):
    """One cycle learning rate scheduler."""

    def __init__(self,
                 optimizer: torch.optim.Optimizer,
                 max_lr: Union[float, List[float]],
                 total_steps: int,
                 pct_start: float = 0.3,
                 anneal_strategy: str = "cos",
                 div_factor: float = 25.0,
                 final_div_factor: float = 1e4,
                 last_epoch: int = -1):

        self.max_lrs = self._format_param(max_lr, optimizer, "max_lr")
        self.total_steps = total_steps
        self.pct_start = pct_start
        self.anneal_strategy = anneal_strategy
        self.div_factor = div_factor
        self.final_div_factor = final_div_factor

        # Calculate phase lengths
        self.step_up_size = int(self.total_steps * self.pct_start)
        self.step_down_size = self.total_steps - self.step_up_size

        super().__init__(optimizer, last_epoch)

    def _format_param(self, param: Union[float, List[float]],
                     optimizer: torch.optim.Optimizer, param_name: str) -> List[float]:
        """Format parameter to list matching optimizer param groups."""
        if isinstance(param, (int, float)):
            return [param] * len(optimizer.param_groups)
        elif isinstance(param, list):
            if len(param) != len(optimizer.param_groups):
                raise ValueError(f"{param_name} length {len(param)} does not match "
                               f"number of param groups {len(optimizer.param_groups)}")
            return param
        else:
            raise TypeError(f"{param_name} must be float or list of floats")

    def get_lr(self) -> List[float]:
        """Calculate learning rate for current step."""
        lrs = []

        for max_lr in self.max_lrs:
            if self.last_epoch <= self.step_up_size:
                # Increasing phase
                pct = self.last_epoch / self.step_up_size
                if self.anneal_strategy == "cos":
                    lr = max_lr / self.div_factor + (max_lr - max_lr / self.div_factor) * \
                         (1 - math.cos(math.pi * pct)) / 2
                else:  # linear
                    lr = max_lr / self.div_factor + (max_lr - max_lr / self.div_factor) * pct
            else:
                # Decreasing phase
                pct = (self.last_epoch - self.step_up_size) / self.step_down_size
                if self.anneal_strategy == "cos":
                    lr = max_lr / self.final_div_factor + (max_lr - max_lr / self.final_div_factor) * \
                         (1 + math.cos(math.pi * pct)) / 2
                else:  # linear
                    lr = max_lr - (max_lr - max_lr / self.final_div_factor) * pct

            lrs.append(lr)

        return lrs


class ReduceLROnPlateau:
    """Reduce learning rate when metric has stopped improving."""

    def __init__(self,
                 optimizer: torch.optim.Optimizer,
                 mode: str = "min",
                 factor: float = 0.1,
                 patience: int = 10,
                 threshold: float = 1e-4,
                 threshold_mode: str = "rel",
                 cooldown: int = 0,
                 min_lr: Union[float, List[float]] = 0,
                 eps: float = 1e-8,
                 verbose: bool = False):

        self.optimizer = optimizer
        self.mode = mode
        self.factor = factor
        self.patience = patience
        self.threshold = threshold
        self.threshold_mode = threshold_mode
        self.cooldown = cooldown
        self.min_lrs = self._format_param(min_lr, optimizer)
        self.eps = eps
        self.verbose = verbose

        self.best = None
        self.num_bad_epochs = 0
        self.mode_worse = None
        self.cooldown_counter = 0
        self.last_epoch = 0

        self._init_is_better()

    def _format_param(self, param: Union[float, List[float]],
                     optimizer: torch.optim.Optimizer) -> List[float]:
        """Format parameter to list matching optimizer param groups."""
        if isinstance(param, (int, float)):
            return [param] * len(optimizer.param_groups)
        return param

    def _init_is_better(self) -> None:
        """Initialize comparison function."""
        if self.mode == "min":
            self.mode_worse = float('inf')
        else:
            self.mode_worse = -float('inf')

    def _is_better(self, current: float, best: float) -> bool:
        """Check if current metric is better than best."""
        if self.mode == "min" and self.threshold_mode == "rel":
            rel_epsilon = 1.0 - self.threshold
            return current < best * rel_epsilon
        elif self.mode == "min" and self.threshold_mode == "abs":
            return current < best - self.threshold
        elif self.mode == "max" and self.threshold_mode == "rel":
            rel_epsilon = self.threshold + 1.0
            return current > best * rel_epsilon
        else:  # mode == 'max' and threshold_mode == 'abs'
            return current > best + self.threshold

    def step(self, metrics: float) -> None:
        """Update learning rate based on metric."""
        current = float(metrics)
        self.last_epoch += 1

        if self.best is None or self._is_better(current, self.best):
            self.best = current
            self.num_bad_epochs = 0
        else:
            self.num_bad_epochs += 1

        if self.cooldown_counter > 0:
            self.cooldown_counter -= 1
            self.num_bad_epochs = 0

        if self.num_bad_epochs > self.patience:
            self._reduce_lr()
            self.cooldown_counter = self.cooldown
            self.num_bad_epochs = 0

    def _reduce_lr(self) -> None:
        """Reduce learning rate."""
        for i, param_group in enumerate(self.optimizer.param_groups):
            old_lr = param_group['lr']
            new_lr = max(old_lr * self.factor, self.min_lrs[i])

            if old_lr - new_lr > self.eps:
                param_group['lr'] = new_lr
                if self.verbose:
                    logging.info(f"Reducing learning rate of group {i} to {new_lr:.4e}")

    def get_lr(self) -> List[float]:
        """Get current learning rates."""
        return [group['lr'] for group in self.optimizer.param_groups]
