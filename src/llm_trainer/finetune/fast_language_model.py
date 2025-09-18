"""
FastLanguageModel - Unsloth-inspired model loading and optimization

This module provides a FastLanguageModel class that mimics Unsloth's API for fast
and memory-efficient model loading, training, and saving.
"""

import torch
import torch.nn as nn
from typing import Optional, Union, Tuple, Dict, Any, List
import warnings
from pathlib import Path

try:
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        PreTrainedModel,
        PreTrainedTokenizer
    )
    _TRANSFORMERS_AVAILABLE = True
except ImportError:
    _TRANSFORMERS_AVAILABLE = False
    # Create dummy classes for type hints
    class PreTrainedModel: pass
    class PreTrainedTokenizer: pass
    class AutoModelForCausalLM: pass
    class AutoTokenizer: pass
    class BitsAndBytesConfig: pass
    warnings.warn("Transformers not available. FastLanguageModel functionality will be limited.")

try:
    from peft import (
        LoraConfig,
        get_peft_model,
        prepare_model_for_kbit_training,
        PeftModel
    )
    _PEFT_AVAILABLE = True
except ImportError:
    _PEFT_AVAILABLE = False
    # Create dummy classes for type hints
    class LoraConfig: pass
    class PeftModel: pass
    def get_peft_model(model, config): return model
    def prepare_model_for_kbit_training(model): return model
    warnings.warn("PEFT not available. LoRA functionality will be disabled.")

from ..kernels.memory_efficient import gradient_checkpointing, empty_cache
from .utils import print_trainable_parameters, get_model_max_length


class FastLanguageModel:
    """
    Unsloth-inspired FastLanguageModel for efficient model loading and training.
    
    This class provides a simple API for loading models with automatic optimizations,
    quantization support, and memory-efficient training configurations.
    """
    
    @staticmethod
    def from_pretrained(
        model_name: str,
        max_seq_length: Optional[int] = None,
        dtype: Optional[torch.dtype] = None,
        load_in_4bit: bool = False,
        load_in_8bit: bool = False,
        device_map: Union[str, Dict] = "auto",
        trust_remote_code: bool = False,
        use_gradient_checkpointing: bool = True,
        use_cache: bool = False,
        attn_implementation: Optional[str] = None,
        **kwargs
    ) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
        """
        Load a pretrained model and tokenizer with optimizations.
        
        Args:
            model_name: Model name or path
            max_seq_length: Maximum sequence length
            dtype: Model dtype (auto-detected if None)
            load_in_4bit: Enable 4-bit quantization
            load_in_8bit: Enable 8-bit quantization  
            device_map: Device mapping strategy
            trust_remote_code: Trust remote code
            use_gradient_checkpointing: Enable gradient checkpointing
            use_cache: Enable KV cache (disable for training)
            attn_implementation: Attention implementation ("flash_attention_2", "sdpa", etc.)
            **kwargs: Additional arguments
            
        Returns:
            Tuple of (model, tokenizer)
        """
        if not _TRANSFORMERS_AVAILABLE:
            raise ImportError("Transformers is required for FastLanguageModel")
            
        # Auto-detect dtype if not specified
        if dtype is None:
            if torch.cuda.is_available():
                # Use bfloat16 if supported, otherwise float16
                dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            else:
                dtype = torch.float32
                
        # Setup quantization config
        quantization_config = None
        if load_in_4bit or load_in_8bit:
            try:
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=load_in_4bit,
                    load_in_8bit=load_in_8bit,
                    bnb_4bit_compute_dtype=dtype,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4" if load_in_4bit else None,
                )
            except Exception as e:
                warnings.warn(f"Failed to setup quantization: {e}")
                quantization_config = None
                
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=trust_remote_code,
            **kwargs
        )
        
        # Set padding token if not present
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        # Load model
        model_kwargs = {
            "torch_dtype": dtype,
            "device_map": device_map,
            "trust_remote_code": trust_remote_code,
            "use_cache": use_cache,
        }
        
        if quantization_config is not None:
            model_kwargs["quantization_config"] = quantization_config
            
        if attn_implementation is not None:
            model_kwargs["attn_implementation"] = attn_implementation
            
        # Filter out None values and add remaining kwargs
        model_kwargs = {k: v for k, v in model_kwargs.items() if v is not None}
        model_kwargs.update({k: v for k, v in kwargs.items() if k not in model_kwargs})
        
        model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
        
        # Apply optimizations
        if use_gradient_checkpointing:
            model.gradient_checkpointing_enable()
            
        # Set max sequence length
        if max_seq_length is not None:
            # Try to set max_position_embeddings if available
            if hasattr(model.config, 'max_position_embeddings'):
                model.config.max_position_embeddings = max_seq_length
            if hasattr(model.config, 'max_seq_len'):
                model.config.max_seq_len = max_seq_length
                
        # Prepare for training if quantized
        if quantization_config is not None and _PEFT_AVAILABLE:
            model = prepare_model_for_kbit_training(model)
            
        return model, tokenizer
    
    @staticmethod
    def get_peft_model(
        model: PreTrainedModel,
        r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        bias: str = "none",
        target_modules: Optional[List[str]] = None,
        task_type: str = "CAUSAL_LM",
        **kwargs
    ) -> PeftModel:
        """
        Apply LoRA to a model.
        
        Args:
            model: Base model
            r: LoRA rank
            lora_alpha: LoRA alpha parameter
            lora_dropout: LoRA dropout
            bias: Bias configuration
            target_modules: Target modules for LoRA
            task_type: Task type
            **kwargs: Additional PEFT arguments
            
        Returns:
            PEFT model with LoRA applied
        """
        if not _PEFT_AVAILABLE:
            raise ImportError("PEFT is required for LoRA functionality")
            
        # Auto-detect target modules if not specified
        if target_modules is None:
            target_modules = FastLanguageModel._get_target_modules(model)
            
        lora_config = LoraConfig(
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            bias=bias,
            target_modules=target_modules,
            task_type=task_type,
            **kwargs
        )
        
        model = get_peft_model(model, lora_config)
        return model
    
    @staticmethod
    def _get_target_modules(model: PreTrainedModel) -> List[str]:
        """Auto-detect target modules for LoRA based on model architecture."""
        model_type = model.config.model_type.lower()
        
        # Common target modules for different architectures
        target_modules_map = {
            "llama": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            "mistral": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            "qwen": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            "gemma": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            "phi": ["q_proj", "k_proj", "v_proj", "dense", "fc1", "fc2"],
            "gpt2": ["c_attn", "c_proj", "c_fc"],
            "gpt_neox": ["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"],
        }
        
        return target_modules_map.get(model_type, ["q_proj", "v_proj"])
    
    @staticmethod
    def save_pretrained(
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        save_directory: Union[str, Path],
        save_method: str = "merged",
        max_shard_size: str = "5GB",
        **kwargs
    ) -> None:
        """
        Save model and tokenizer.
        
        Args:
            model: Model to save
            tokenizer: Tokenizer to save
            save_directory: Directory to save to
            save_method: "merged" or "lora_only"
            max_shard_size: Maximum shard size
            **kwargs: Additional save arguments
        """
        save_directory = Path(save_directory)
        save_directory.mkdir(parents=True, exist_ok=True)
        
        # Save tokenizer
        tokenizer.save_pretrained(save_directory, **kwargs)
        
        # Save model
        if hasattr(model, 'merge_and_unload') and save_method == "merged":
            # Merge LoRA weights and save
            merged_model = model.merge_and_unload()
            merged_model.save_pretrained(
                save_directory, 
                max_shard_size=max_shard_size,
                **kwargs
            )
        else:
            # Save as-is (LoRA adapters only if PEFT model)
            model.save_pretrained(
                save_directory,
                max_shard_size=max_shard_size, 
                **kwargs
            )
    
    @staticmethod
    def print_trainable_parameters(model: PreTrainedModel) -> None:
        """Print the number of trainable parameters."""
        print_trainable_parameters(model)
        
    @staticmethod
    def get_max_length(model: PreTrainedModel) -> int:
        """Get the maximum sequence length for the model."""
        return get_model_max_length(model)
