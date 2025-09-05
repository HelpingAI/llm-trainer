"""Patching system for TRL (Transformer Reinforcement Learning)."""

import torch
import torch.nn as nn
from typing import Optional, Union, Callable, Dict, Any
import warnings


def patch_trl():
    """
    Patch TRL with memory-efficient implementations.
    This function enhances TRL with optimizations.
    """
    try:
        import trl
        from trl import SFTTrainer, DPOTrainer, PPOTrainer, RewardTrainer
        from trl import SFTConfig, DPOConfig, PPOConfig, RewardConfig
        
        # Patch SFTTrainer with memory-efficient methods
        original_sft_trainer_init = SFTTrainer.__init__
        
        def patched_sft_trainer_init(self, *args, **kwargs):
            original_sft_trainer_init(self, *args, **kwargs)
            # Add memory-efficient methods
            self.print_trainable_parameters = lambda: _print_trainable_parameters(self.model)
            self.prepare_model_for_kbit_training = lambda: _prepare_model_for_kbit_training(self.model)
            self.get_nb_trainable_parameters = lambda: _get_nb_trainable_parameters(self.model)
        
        SFTTrainer.__init__ = patched_sft_trainer_init
        
        # Patch other trainers similarly
        for trainer_class in [DPOTrainer, PPOTrainer, RewardTrainer]:
            original_init = trainer_class.__init__
            
            def patched_init(self, *args, **kwargs):
                original_init(self, *args, **kwargs)
                self.print_trainable_parameters = lambda: _print_trainable_parameters(self.model)
                self.prepare_model_for_kbit_training = lambda: _prepare_model_for_kbit_training(self.model)
                self.get_nb_trainable_parameters = lambda: _get_nb_trainable_parameters(self.model)
            
            trainer_class.__init__ = patched_init
        
        # Patch config classes with memory-efficient options
        for config_class in [SFTConfig, DPOConfig, PPOConfig, RewardConfig]:
            original_config_init = config_class.__init__
            
            def patched_config_init(self, *args, **kwargs):
                # Add memory-efficient training options
                self.use_gradient_checkpointing = kwargs.pop('use_gradient_checkpointing', False)
                self.use_low_vram = kwargs.pop('use_low_vram', False)
                self.fuse_layers = kwargs.pop('fuse_layers', False)
                original_config_init(self, *args, **kwargs)
            
            config_class.__init__ = patched_config_init
        
        print("✅ Successfully patched TRL with memory-efficient optimizations")
        
    except ImportError:
        warnings.warn("TRL not installed. Skipping TRL patching.")
    except Exception as e:
        warnings.warn(f"Failed to patch TRL: {e}")


def _print_trainable_parameters(model: nn.Module):
    """
    Print trainable parameters information.
    
    Args:
        model: PyTorch model
    """
    all_param, trainable_param = _get_nb_trainable_parameters(model)
    trainable_ratio = 100 * trainable_param / all_param if all_param > 0 else 0
    print(
        f"Trainable parameters: {trainable_param:,}/{all_param:,} ({trainable_ratio:.2f}%)"
    )


def _get_nb_trainable_parameters(model: nn.Module) -> tuple:
    """
    Get number of trainable parameters.
    
    Args:
        model: PyTorch model
        
    Returns:
        Tuple of (all_params, trainable_params)
    """
    all_param = 0
    trainable_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_param += param.numel()
    return all_param, trainable_param


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


# Factory functions for creating patched trainers
def create_memory_efficient_sft_trainer(
    model: Union[nn.Module, str],
    tokenizer: Any,
    dataset: Any,
    **kwargs
) -> Any:
    """
    Create a memory-efficient SFT trainer.
    
    Args:
        model: Model or model name
        tokenizer: Tokenizer
        dataset: Training dataset
        **kwargs: Additional arguments
        
    Returns:
        SFTTrainer instance
    """
    try:
        from trl import SFTTrainer
        
        # Add memory-efficient options
        kwargs.setdefault('use_gradient_checkpointing', True)
        kwargs.setdefault('use_low_vram', True)
        
        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=dataset,
            **kwargs
        )
        
        return trainer
    except ImportError:
        raise ImportError("TRL not installed. Please install with: pip install trl")


def create_memory_efficient_dpo_trainer(
    model: Union[nn.Module, str],
    tokenizer: Any,
    dataset: Any,
    **kwargs
) -> Any:
    """
    Create a memory-efficient DPO trainer.
    
    Args:
        model: Model or model name
        tokenizer: Tokenizer
        dataset: Training dataset
        **kwargs: Additional arguments
        
    Returns:
        DPOTrainer instance
    """
    try:
        from trl import DPOTrainer
        
        # Add memory-efficient options
        kwargs.setdefault('use_gradient_checkpointing', True)
        kwargs.setdefault('use_low_vram', True)
        
        trainer = DPOTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=dataset,
            **kwargs
        )
        
        return trainer
    except ImportError:
        raise ImportError("TRL not installed. Please install with: pip install trl")