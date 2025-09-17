"""
Factory functions for creating fine-tuning trainers and models.

This module provides high-level factory functions that integrate all the components
(models, tokenizers, quantization, PEFT, datasets) into ready-to-use trainers.
"""

import torch
from typing import Optional, Dict, Any, Union, Tuple
import warnings
from pathlib import Path

try:
    from datasets import load_dataset, Dataset
    _DATASETS_AVAILABLE = True
except ImportError:
    _DATASETS_AVAILABLE = False
    warnings.warn("Datasets not available. Dataset loading will be limited.")

from .fast_language_model import FastLanguageModel
from .trainers import SFTTrainer, DPOTrainer, PPOTrainer, RewardTrainer
from .configs import (
    SFTConfig, DPOConfig, PPOConfig, RewardConfig,
    QuantizationConfig, LoRAConfig, load_config_from_yaml
)
from .utils import get_target_modules_for_model, print_trainable_parameters


def create_finetune_trainer(
    model_config: Dict[str, Any],
    data_config: Dict[str, Any],
    training_config: Dict[str, Any],
    trainer_type: str = "sft"
) -> Union[SFTTrainer, DPOTrainer, PPOTrainer, RewardTrainer]:
    """
    Create a fine-tuning trainer from configuration dictionaries.
    
    This is the main factory function used by the CLI and scripts.
    
    Args:
        model_config: Model configuration dictionary
        data_config: Data configuration dictionary  
        training_config: Training configuration dictionary
        trainer_type: Type of trainer ("sft", "dpo", "ppo", "reward")
        
    Returns:
        Configured trainer instance
    """
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(model_config)
    
    # Apply PEFT if configured
    if model_config.get("lora") and training_config.get("use_peft", False):
        model = apply_lora(model, model_config["lora"])
    
    # Load datasets
    train_dataset, eval_dataset = load_datasets(data_config, tokenizer)
    
    # Create training configuration
    config = create_training_config(training_config, trainer_type, model_config)
    
    # Create trainer
    trainer_classes = {
        "sft": SFTTrainer,
        "dpo": DPOTrainer,
        "ppo": PPOTrainer,
        "reward": RewardTrainer
    }
    
    trainer_class = trainer_classes.get(trainer_type, SFTTrainer)
    
    trainer = trainer_class(
        model=model,
        tokenizer=tokenizer,
        config=config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset
    )
    
    # Print model info
    print(f"✅ Created {trainer_type.upper()} trainer")
    trainer.print_trainable_parameters()
    
    return trainer


def load_model_and_tokenizer(
    model_config: Dict[str, Any]
) -> Tuple[torch.nn.Module, Any]:
    """
    Load model and tokenizer with quantization and optimizations.
    
    Args:
        model_config: Model configuration dictionary
        
    Returns:
        Tuple of (model, tokenizer)
    """
    model_name = model_config["model_name_or_path"]
    
    # Determine quantization settings
    quantization_bits = model_config.get("quantization_bits", 0)
    load_in_4bit = quantization_bits == 4
    load_in_8bit = quantization_bits == 8
    
    # Determine dtype
    torch_dtype = model_config.get("torch_dtype", "auto")
    if torch_dtype == "auto":
        if torch.cuda.is_available():
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        else:
            dtype = torch.float32
    else:
        dtype_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16
        }
        dtype = dtype_map.get(torch_dtype, torch.float16)
    
    # Load model and tokenizer
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=model_config.get("max_seq_length", 2048),
        dtype=dtype,
        load_in_4bit=load_in_4bit,
        load_in_8bit=load_in_8bit,
        device_map=model_config.get("device_map", "auto"),
        trust_remote_code=model_config.get("trust_remote_code", False),
        use_gradient_checkpointing=model_config.get("use_gradient_checkpointing", True),
        attn_implementation=model_config.get("attn_implementation")
    )
    
    print(f"✅ Loaded model: {model_name}")
    print(f"   - Dtype: {dtype}")
    print(f"   - Quantization: {'4-bit' if load_in_4bit else '8-bit' if load_in_8bit else 'None'}")
    print(f"   - Max sequence length: {FastLanguageModel.get_max_length(model)}")
    
    return model, tokenizer


def apply_lora(model: torch.nn.Module, lora_config: Dict[str, Any]) -> torch.nn.Module:
    """
    Apply LoRA to a model.
    
    Args:
        model: Base model
        lora_config: LoRA configuration dictionary
        
    Returns:
        Model with LoRA applied
    """
    # Get target modules
    target_modules = lora_config.get("target_modules")
    if target_modules == "auto" or target_modules is None:
        # Auto-detect based on model architecture
        model_name = getattr(model.config, '_name_or_path', '')
        target_modules = get_target_modules_for_model(model_name)
    
    # Apply LoRA
    model = FastLanguageModel.get_peft_model(
        model=model,
        r=lora_config.get("r", 16),
        lora_alpha=lora_config.get("alpha", 32),
        lora_dropout=lora_config.get("dropout", 0.05),
        bias=lora_config.get("bias", "none"),
        target_modules=target_modules
    )
    
    print(f"✅ Applied LoRA:")
    print(f"   - Rank: {lora_config.get('r', 16)}")
    print(f"   - Alpha: {lora_config.get('alpha', 32)}")
    print(f"   - Target modules: {target_modules}")
    
    return model


def load_datasets(
    data_config: Dict[str, Any],
    tokenizer: Any
) -> Tuple[Optional[Dataset], Optional[Dataset]]:
    """
    Load training and evaluation datasets.
    
    Args:
        data_config: Data configuration dictionary
        tokenizer: Tokenizer for processing
        
    Returns:
        Tuple of (train_dataset, eval_dataset)
    """
    if not _DATASETS_AVAILABLE:
        warnings.warn("Datasets library not available. Skipping dataset loading.")
        return None, None
    
    dataset_name = data_config.get("dataset_name")
    if not dataset_name:
        print("⚠️  No dataset specified")
        return None, None
    
    # Load dataset
    dataset_config = data_config.get("dataset_config")
    split = data_config.get("split", "train")
    
    try:
        dataset = load_dataset(dataset_name, dataset_config, split=split)
        print(f"✅ Loaded dataset: {dataset_name}")
        print(f"   - Split: {split}")
        print(f"   - Size: {len(dataset):,} examples")
        
        # Split into train/eval if needed
        validation_split = data_config.get("validation_split")
        if validation_split:
            eval_dataset = load_dataset(dataset_name, dataset_config, split=validation_split)
            print(f"   - Eval split: {validation_split} ({len(eval_dataset):,} examples)")
        else:
            # Create validation split from training data
            eval_size = data_config.get("eval_size", 0.1)
            if eval_size > 0:
                split_dataset = dataset.train_test_split(test_size=eval_size, seed=42)
                dataset = split_dataset["train"]
                eval_dataset = split_dataset["test"]
                print(f"   - Created eval split: {len(eval_dataset):,} examples ({eval_size*100:.1f}%)")
            else:
                eval_dataset = None
        
        return dataset, eval_dataset
        
    except Exception as e:
        print(f"❌ Failed to load dataset {dataset_name}: {e}")
        return None, None


def create_training_config(
    training_config: Dict[str, Any],
    trainer_type: str,
    model_config: Dict[str, Any]
) -> Union[SFTConfig, DPOConfig, PPOConfig, RewardConfig]:
    """
    Create training configuration object.
    
    Args:
        training_config: Training configuration dictionary
        trainer_type: Type of trainer
        model_config: Model configuration for additional context
        
    Returns:
        Training configuration object
    """
    # Merge model and training configs
    merged_config = {**model_config, **training_config}
    
    # Handle quantization config
    if "bnb" in model_config:
        bnb_config = model_config["bnb"]
        merged_config["quantization"] = QuantizationConfig(
            load_in_4bit=model_config.get("quantization_bits") == 4,
            load_in_8bit=model_config.get("quantization_bits") == 8,
            quant_type=bnb_config.get("quant_type", "nf4"),
            compute_dtype=bnb_config.get("compute_dtype", "bfloat16"),
            double_quant=bnb_config.get("double_quant", True)
        )
    
    # Handle LoRA config
    if "lora" in model_config:
        lora_config = model_config["lora"]
        merged_config["lora"] = LoRAConfig(
            r=lora_config.get("r", 16),
            alpha=lora_config.get("alpha", 32),
            dropout=lora_config.get("dropout", 0.05),
            bias=lora_config.get("bias", "none"),
            target_modules=lora_config.get("target_modules")
        )
    
    # Create appropriate config
    config_classes = {
        "sft": SFTConfig,
        "dpo": DPOConfig,
        "ppo": PPOConfig,
        "reward": RewardConfig
    }
    
    config_class = config_classes.get(trainer_type, SFTConfig)
    return config_class(**merged_config)


def create_model_from_config(config_path: str) -> Tuple[torch.nn.Module, Any]:
    """
    Create model and tokenizer from YAML config file.
    
    Args:
        config_path: Path to YAML configuration file
        
    Returns:
        Tuple of (model, tokenizer)
    """
    import yaml
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    model_config = config.get("model", {})
    return load_model_and_tokenizer(model_config)


def create_trainer_from_config(
    config_path: str,
    trainer_type: str = "sft"
) -> Union[SFTTrainer, DPOTrainer, PPOTrainer, RewardTrainer]:
    """
    Create trainer from YAML config file.
    
    Args:
        config_path: Path to YAML configuration file
        trainer_type: Type of trainer to create
        
    Returns:
        Configured trainer instance
    """
    import yaml
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    model_config = config.get("model", {})
    data_config = config.get("data", {})
    training_config = config.get("training", {})
    
    return create_finetune_trainer(
        model_config=model_config,
        data_config=data_config,
        training_config=training_config,
        trainer_type=trainer_type
    )


# Convenience functions for specific trainer types
def create_sft_trainer(**kwargs) -> SFTTrainer:
    """Create SFT trainer with defaults."""
    return create_finetune_trainer(trainer_type="sft", **kwargs)


def create_dpo_trainer(**kwargs) -> DPOTrainer:
    """Create DPO trainer with defaults."""
    return create_finetune_trainer(trainer_type="dpo", **kwargs)


def create_ppo_trainer(**kwargs) -> PPOTrainer:
    """Create PPO trainer with defaults."""
    return create_finetune_trainer(trainer_type="ppo", **kwargs)


def create_reward_trainer(**kwargs) -> RewardTrainer:
    """Create Reward trainer with defaults."""
    return create_finetune_trainer(trainer_type="reward", **kwargs)
