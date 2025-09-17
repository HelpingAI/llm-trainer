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

__version__ = "0.2.5"
__author__ = "Abhay"
__email__ = "abhay@helpingai.co"
__authors__ = [{"name": "Abhay", "email": "abhay@helpingai.co"}]

from .models import TransformerLM
from .tokenizer import BPETokenizer
from .training import (
    Trainer,
    create_optimizer
)
from .config import ModelConfig, TrainingConfig

# Export kernel optimizations
from .kernels import *
from .kernels.fused_ops import FusedLinear, FusedRMSNorm, fused_cross_entropy
from .kernels.memory_efficient import (
    gradient_checkpointing, LowVRAMLinear,
    offload_optimizer_states, empty_cache, efficient_attention_forward
)

# Export patching system
from .patching import patch_transformers, patch_trl

# Export fine-tuning system (Unsloth/TRL/Axolotl-inspired)
try:
    from .finetune import (
        # Core classes
        FastLanguageModel,

        # Trainers
        SFTTrainer,
        DPOTrainer,
        PPOTrainer,
        RewardTrainer,

        # Configurations
        SFTConfig,
        DPOConfig,
        PPOConfig,
        RewardConfig,
        QuantizationConfig,
        LoRAConfig,
        load_config_from_yaml,

        # Factory functions
        create_finetune_trainer,
        create_sft_trainer,
        create_dpo_trainer,
        create_ppo_trainer,
        create_reward_trainer,
        create_trainer_from_config,
        load_model_and_tokenizer,

        # Dataset utilities
        DatasetFormatter,
        DatasetProcessor,
        load_and_process_dataset,
        create_chat_template,
        apply_chat_template_to_dataset,
        filter_dataset_by_length,
        pack_dataset,

        # Export utilities
        ModelExporter,
        export_model,
        create_deployment_package,

        # Utilities
        get_model_max_length,
        print_trainable_parameters,
        prepare_model_for_kbit_training,
        apply_chat_template,
        format_instruction_dataset,
        format_chat_dataset,
        get_target_modules_for_model,
        estimate_memory_usage
    )
    _FINETUNE_AVAILABLE = True
except ImportError:
    _FINETUNE_AVAILABLE = False

__all__ = [
    "TransformerLM",
    "BPETokenizer",
    "Trainer",
    "create_optimizer",
    "ModelConfig",
    "TrainingConfig",
    "FusedLinear",
    "FusedRMSNorm",
    "fused_cross_entropy",
    "gradient_checkpointing",
    "LowVRAMLinear",
    "offload_optimizer_states",
    "empty_cache",
    "efficient_attention_forward",
    "patch_transformers",
    "patch_trl"
]

# Add fine-tuning exports if available
if _FINETUNE_AVAILABLE:
    __all__.extend([
        # Core classes
        "FastLanguageModel",

        # Trainers
        "SFTTrainer",
        "DPOTrainer",
        "PPOTrainer",
        "RewardTrainer",

        # Configurations
        "SFTConfig",
        "DPOConfig",
        "PPOConfig",
        "RewardConfig",
        "QuantizationConfig",
        "LoRAConfig",
        "load_config_from_yaml",

        # Factory functions
        "create_finetune_trainer",
        "create_sft_trainer",
        "create_dpo_trainer",
        "create_ppo_trainer",
        "create_reward_trainer",
        "create_trainer_from_config",
        "load_model_and_tokenizer",

        # Dataset utilities
        "DatasetFormatter",
        "DatasetProcessor",
        "load_and_process_dataset",
        "create_chat_template",
        "apply_chat_template_to_dataset",
        "filter_dataset_by_length",
        "pack_dataset",

        # Export utilities
        "ModelExporter",
        "export_model",
        "create_deployment_package",

        # Utilities
        "get_model_max_length",
        "print_trainable_parameters",
        "prepare_model_for_kbit_training",
        "apply_chat_template",
        "format_instruction_dataset",
        "format_chat_dataset",
        "get_target_modules_for_model",
        "estimate_memory_usage"
    ])