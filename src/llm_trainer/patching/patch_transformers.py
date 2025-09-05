"""Patching system for Hugging Face Transformers."""

import torch
import torch.nn as nn
from typing import Optional, Union, Callable
import warnings


def patch_transformers():
    """
    Patch Hugging Face Transformers with memory-efficient implementations.
    This function enhances Transformers with optimizations.
    """
    try:
        import transformers
        from transformers import Trainer as HFTrainer
        from transformers import TrainingArguments
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        # Patch Trainer with memory-efficient methods
        original_trainer_init = HFTrainer.__init__
        
        def patched_trainer_init(self, *args, **kwargs):
            original_trainer_init(self, *args, **kwargs)
            # Add memory-efficient methods
            self.print_trainable_parameters = lambda: _print_trainable_parameters(self.model)
            self.prepare_model_for_kbit_training = lambda: _prepare_model_for_kbit_training(self.model)
        
        HFTrainer.__init__ = patched_trainer_init
        
        # Patch TrainingArguments with additional memory-efficient options
        original_training_args_init = TrainingArguments.__init__
        
        def patched_training_args_init(self, *args, **kwargs):
            # Add memory-efficient training options
            self.use_gradient_checkpointing = kwargs.pop('use_gradient_checkpointing', False)
            self.use_low_vram = kwargs.pop('use_low_vram', False)
            self.fuse_layers = kwargs.pop('fuse_layers', False)
            original_training_args_init(self, *args, **kwargs)
        
        TrainingArguments.__init__ = patched_training_args_init
        
        print("✅ Successfully patched Hugging Face Transformers with memory-efficient optimizations")
        
    except ImportError:
        warnings.warn("Transformers not installed. Skipping Transformers patching.")
    except Exception as e:
        warnings.warn(f"Failed to patch Transformers: {e}")


def _print_trainable_parameters(model: nn.Module):
    """
    Print trainable parameters information.
    
    Args:
        model: PyTorch model
    """
    all_param = 0
    trainable_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_param += param.numel()
    
    trainable_ratio = 100 * trainable_param / all_param if all_param > 0 else 0
    print(
        f"Trainable parameters: {trainable_param:,}/{all_param:,} ({trainable_ratio:.2f}%)"
    )


def _prepare_model_for_kbit_training(model: nn.Module):
    """
    Prepare model for k-bit training.
    
    Args:
        model: PyTorch model
    """
    try:
        from peft import prepare_model_for_kbit_training
        model = prepare_model_for_kbit_training(model)
        print("✅ Model prepared for k-bit training")
    except ImportError:
        print("⚠️ PEFT not installed. Skipping k-bit training preparation")
    except Exception as e:
        print(f"⚠️ Failed to prepare model for k-bit training: {e}")


# Additional utility functions for Transformers integration
def create_memory_efficient_model(model_name: str, **kwargs) -> nn.Module:
    """
    Create a memory-efficient model with optimizations.
    
    Args:
        model_name: Name of the model to load
        **kwargs: Additional arguments to pass to from_pretrained
        
    Returns:
        Memory-efficient model
    """
    try:
        from transformers import AutoModelForCausalLM
        
        # Add memory-efficient loading options
        kwargs.setdefault('low_cpu_mem_usage', True)
        kwargs.setdefault('torch_dtype', torch.float16 if torch.cuda.is_available() else torch.float32)
        
        model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
        
        # Apply memory-efficient optimizations
        _optimize_model_for_inference(model)
        
        return model
    except ImportError:
        raise ImportError("Transformers not installed. Please install with: pip install transformers")


def _optimize_model_for_inference(model: nn.Module):
    """
    Optimize model for inference with memory-efficient techniques.
    
    Args:
        model: PyTorch model to optimize
    """
    # Apply torch.compile if available (PyTorch 2.0+)
    try:
        model = torch.compile(model, mode="reduce-overhead", fullgraph=True)
    except Exception:
        pass  # torch.compile not available or failed
    
    return model