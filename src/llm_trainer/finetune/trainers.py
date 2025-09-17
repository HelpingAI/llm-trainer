"""
Enhanced trainer classes inspired by TRL (Transformers Reinforcement Learning).

This module provides trainer classes that follow TRL's API patterns while integrating
with our memory-efficient optimizations and kernel implementations.

Note: This module now imports from the trainers/ subdirectory for better organization.
The trainers have been split into separate files for maintainability.
"""

# Import all trainers from the trainers subdirectory
from .trainers import (
    # Core trainer classes
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

# Export all trainer classes and utilities
__all__ = [
    # Core trainer classes
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

    # Utility functions
    "get_trainer_class",
    "list_available_trainers",
    "create_trainer",
    "create_sft_trainer",
    "create_dpo_trainer",
    "create_ppo_trainer",
    "create_reward_trainer",
    "create_chat_trainer",
    "compare_trainers",
    "get_trainer_requirements",
    "validate_trainer_setup",
    "print_trainer_info",
    "get_recommended_trainer"
]

# Legacy compatibility - keep the old class definitions as aliases
# This ensures existing code continues to work
BaseFineTuneTrainer = BaseFineTuneTrainer
SFTTrainer = SFTTrainer
DPOTrainer = DPOTrainer
PPOTrainer = PPOTrainer
RewardTrainer = RewardTrainer
ChatTrainer = ChatTrainer

# Additional convenience exports
TrainerRegistry = TrainerRegistry
TrainerInfo = TrainerInfo
