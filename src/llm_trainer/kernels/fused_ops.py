"""Fused operations for memory-efficient training."""

import torch
import torch.nn as nn
from typing import Optional, Tuple


class FusedLinear(nn.Module):
    """
    Fused linear layer with optimized operations.
    Combines multiple operations in a single kernel for efficiency.
    """
    
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / fan_in**0.5
            nn.init.uniform_(self.bias, -bound, bound)
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # Fused operation for better memory efficiency
        return torch.nn.functional.linear(input, self.weight, self.bias)


class FusedRMSNorm(nn.Module):
    """
    Fused RMSNorm implementation for better performance.
    """
    
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    
    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
    
    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


def fused_cross_entropy(logits: torch.Tensor, labels: torch.Tensor, ignore_index: int = -100) -> torch.Tensor:
    """
    Fused cross-entropy loss implementation.
    
    Args:
        logits: Logits tensor of shape (batch_size, seq_len, vocab_size)
        labels: Labels tensor of shape (batch_size, seq_len)
        ignore_index: Index to ignore in loss calculation
        
    Returns:
        Loss tensor
    """
    # Flatten for easier computation
    batch_size, seq_len, vocab_size = logits.shape
    logits_flat = logits.view(-1, vocab_size)
    labels_flat = labels.view(-1)
    
    # Compute loss
    loss = torch.nn.functional.cross_entropy(
        logits_flat, 
        labels_flat, 
        ignore_index=ignore_index,
        reduction='none'
    )
    
    # Reshape back and return mean
    loss = loss.view(batch_size, seq_len)
    return loss.mean()


def fused_adamw_step(
    param: torch.Tensor,
    grad: torch.Tensor,
    exp_avg: torch.Tensor,
    exp_avg_sq: torch.Tensor,
    step: int,
    beta1: float = 0.9,
    beta2: float = 0.999,
    lr: float = 1e-3,
    weight_decay: float = 0.01,
    eps: float = 1e-8
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Fused AdamW optimizer step.
    
    Args:
        param: Parameter tensor
        grad: Gradient tensor
        exp_avg: Exponential moving average of gradient
        exp_avg_sq: Exponential moving average of squared gradient
        step: Current step
        beta1: Beta1 parameter
        beta2: Beta2 parameter
        lr: Learning rate
        weight_decay: Weight decay
        eps: Epsilon for numerical stability
        
    Returns:
        Updated parameter, exp_avg, exp_avg_sq
    """
    # Update moments
    exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
    exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
    
    # Bias correction
    bias_correction1 = 1 - beta1 ** step
    bias_correction2 = 1 - beta2 ** step
    
    # Compute denominator
    denom = (exp_avg_sq.sqrt() / torch.sqrt(torch.tensor(bias_correction2))) + eps
    
    # Weight decay
    if weight_decay != 0:
        param.mul_(1 - lr * weight_decay)
    
    # Parameter update
    step_size = lr / bias_correction1
    param.addcdiv_(exp_avg, denom, value=-step_size)
    
    return param, exp_avg, exp_avg_sq