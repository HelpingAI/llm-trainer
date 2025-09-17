"""
LLM Trainer Fine-tuning Module

This module provides a comprehensive fine-tuning system inspired by Unsloth, TRL, and Axolotl.
It includes:
- FastLanguageModel: Unsloth-inspired model loading and optimization
- Enhanced Trainers: TRL-style SFT, DPO, PPO, and Reward trainers
- Configuration Classes: Axolotl-inspired YAML-based configuration
- Quantization Support: BitsAndBytes integration with LoRA/QLoRA
- Memory Optimizations: Advanced kernel optimizations and memory management
"""

from .fast_language_model import FastLanguageModel
from .trainers import (
    BaseFineTuneTrainer,
    SFTTrainer,
    DPOTrainer,
    PPOTrainer,
    RewardTrainer,
    ChatTrainer,

    # Trainer utilities
    SFTDataCollator,
    DPODataCollator,
    RewardDataCollator,
    compute_dpo_loss,
    compute_reward_loss,
    get_batch_logps,
    estimate_trainer_memory,

    # Registry system
    TrainerRegistry,
    TrainerInfo,

    # Utility functions
    get_trainer_class,
    list_available_trainers,
    create_trainer,
    create_sft_trainer,
    create_dpo_trainer,
    create_ppo_trainer,
    create_reward_trainer,
    create_chat_trainer,
    compare_trainers,
    get_trainer_requirements,
    validate_trainer_setup,
    print_trainer_info,
    get_recommended_trainer
)
from .configs import (
    SFTConfig,
    DPOConfig,
    PPOConfig,
    RewardConfig,
    QuantizationConfig,
    LoRAConfig,
    load_config_from_yaml
)
from .factory import (
    create_finetune_trainer,
    create_sft_trainer,
    create_dpo_trainer,
    create_ppo_trainer,
    create_reward_trainer,
    create_trainer_from_config,
    load_model_and_tokenizer
)
from .utils import (
    get_model_max_length,
    print_trainable_parameters,
    prepare_model_for_kbit_training,
    apply_chat_template,
    format_instruction_dataset,
    format_chat_dataset,
    get_target_modules_for_model,
    estimate_memory_usage
)
from .dataset_utils import (
    DatasetFormatter,
    DatasetProcessor,
    load_and_process_dataset,
    create_chat_template,
    apply_chat_template_to_dataset,
    filter_dataset_by_length,
    pack_dataset
)
from .export_utils import (
    ModelExporter,
    export_model,
    create_deployment_package
)

__all__ = [
    # Core classes
    "FastLanguageModel",

    # Trainers
    "BaseFineTuneTrainer",
    "SFTTrainer",
    "DPOTrainer",
    "PPOTrainer",
    "RewardTrainer",
    "ChatTrainer",

    # Trainer utilities
    "SFTDataCollator",
    "DPODataCollator",
    "RewardDataCollator",
    "compute_dpo_loss",
    "compute_reward_loss",
    "get_batch_logps",
    "estimate_trainer_memory",

    # Registry system
    "TrainerRegistry",
    "TrainerInfo",
    "get_trainer_class",
    "list_available_trainers",
    "create_trainer",
    "compare_trainers",
    "get_trainer_requirements",
    "validate_trainer_setup",
    "print_trainer_info",
    "get_recommended_trainer",

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
    "create_chat_trainer",
    "create_trainer_from_config",
    "load_model_and_tokenizer",

    # Utilities
    "get_model_max_length",
    "print_trainable_parameters",
    "prepare_model_for_kbit_training",
    "apply_chat_template",
    "format_instruction_dataset",
    "format_chat_dataset",
    "get_target_modules_for_model",
    "estimate_memory_usage",

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
    "create_deployment_package"
]
