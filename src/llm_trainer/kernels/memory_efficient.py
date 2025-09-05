"""Memory-efficient operations for low VRAM training."""

import torch
import torch.nn as nn
from typing import Optional, Tuple, List
import gc


def gradient_checkpointing(func, *args, use_reentrant: bool = True, **kwargs):
    """
    Gradient checkpointing to reduce memory usage during training.
    
    Args:
        func: Function to apply gradient checkpointing to
        *args: Arguments to pass to the function
        use_reentrant: Whether to use reentrant checkpointing
        **kwargs: Keyword arguments to pass to the function
        
    Returns:
        Output of the function
    """
    try:
        from torch.utils.checkpoint import checkpoint
        if use_reentrant:
            return checkpoint(func, *args, **kwargs)
        else:
            # Use non-reentrant checkpointing if available (PyTorch 1.13+)
            try:
                from torch.utils.checkpoint import checkpoint_sequential
                return checkpoint(func, *args, use_reentrant=False, **kwargs)
            except ImportError:
                # Fallback to reentrant
                return checkpoint(func, *args, **kwargs)
    except ImportError:
        # If checkpointing is not available, just run the function normally
        return func(*args, **kwargs)


class LowVRAMLinear(nn.Module):
    """
    Memory-efficient linear layer that offloads weights to CPU when not in use.
    """
    
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Store weights on CPU initially
        self.weight_cpu = nn.Parameter(
            torch.empty(out_features, in_features), 
            requires_grad=True
        )
        if bias:
            self.bias_cpu = nn.Parameter(
                torch.empty(out_features), 
                requires_grad=True
            )
        else:
            self.register_parameter('bias_cpu', None)
            
        self.reset_parameters()
        
        # Device tracking
        self._weight_gpu = None
        self._bias_gpu = None
        self._current_device = None
    
    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight_cpu, a=5**0.5)
        if self.bias_cpu is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight_cpu)
            bound = 1 / fan_in**0.5
            nn.init.uniform_(self.bias_cpu, -bound, bound)
    
    def _move_to_device(self, device):
        """Move parameters to the specified device."""
        if self._current_device != device:
            # Move current GPU tensors back to CPU
            if self._weight_gpu is not None:
                self.weight_cpu.data.copy_(self._weight_gpu.data)
                del self._weight_gpu
                self._weight_gpu = None
                
            if self._bias_gpu is not None and self.bias_cpu is not None:
                self.bias_cpu.data.copy_(self._bias_gpu.data)
                del self._bias_gpu
                self._bias_gpu = None
            
            # Move to new device
            self._weight_gpu = self.weight_cpu.data.to(device)
            if self.bias_cpu is not None:
                self._bias_gpu = self.bias_cpu.data.to(device)
                
            self._current_device = device
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        device = input.device
        self._move_to_device(device)
        
        bias = self._bias_gpu if self._bias_gpu is not None else None
        return torch.nn.functional.linear(input, self._weight_gpu, bias)


def offload_optimizer_states(optimizer, device: str = 'cpu'):
    """
    Offload optimizer states to reduce GPU memory usage.
    
    Args:
        optimizer: PyTorch optimizer
        device: Device to offload to (default: 'cpu')
    """
    for group in optimizer.param_groups:
        for p in group['params']:
            state = optimizer.state[p]
            if 'exp_avg' in state:
                state['exp_avg'] = state['exp_avg'].to(device)
            if 'exp_avg_sq' in state:
                state['exp_avg_sq'] = state['exp_avg_sq'].to(device)
            if 'max_exp_avg_sq' in state:
                state['max_exp_avg_sq'] = state['max_exp_avg_sq'].to(device)


def empty_cache():
    """Empty PyTorch cache to free up memory."""
    torch.cuda.empty_cache()
    gc.collect()


def efficient_attention_forward(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: Optional[torch.Tensor] = None,
    dropout_p: float = 0.0,
    is_causal: bool = False
) -> torch.Tensor:
    """
    Memory-efficient attention forward pass.
    
    Args:
        query: Query tensor of shape (batch_size, num_heads, seq_len, head_dim)
        key: Key tensor of shape (batch_size, num_heads, seq_len, head_dim)
        value: Value tensor of shape (batch_size, num_heads, seq_len, head_dim)
        attn_mask: Attention mask tensor
        dropout_p: Dropout probability
        is_causal: Whether to apply causal mask
        
    Returns:
        Output tensor of shape (batch_size, num_heads, seq_len, head_dim)
    """
    # Use PyTorch's built-in scaled dot-product attention if available
    try:
        # PyTorch 2.0+ has efficient attention
        return torch.nn.functional.scaled_dot_product_attention(
            query, key, value,
            attn_mask=attn_mask,
            dropout_p=dropout_p,
            is_causal=is_causal
        )
    except AttributeError:
        # Fallback implementation for older PyTorch versions
        # This is less memory efficient but works on older versions
        batch_size, num_heads, seq_len, head_dim = query.shape
        
        # Compute attention scores
        scores = torch.matmul(query, key.transpose(-2, -1)) / (head_dim ** 0.5)
        
        # Apply causal mask if needed
        if is_causal:
            causal_mask = torch.triu(
                torch.ones(seq_len, seq_len, device=scores.device), 
                diagonal=1
            ).bool()
            scores = scores.masked_fill(causal_mask, float('-inf'))
        
        # Apply attention mask
        if attn_mask is not None:
            scores = scores + attn_mask
        
        # Apply softmax
        attn_weights = torch.nn.functional.softmax(scores, dim=-1)
        
        # Apply dropout
        if dropout_p > 0.0:
            attn_weights = torch.nn.functional.dropout(attn_weights, p=dropout_p)
        
        # Apply attention
        return torch.matmul(attn_weights, value)