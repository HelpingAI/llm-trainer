"""
Trainer classes for fine-tuning language models.

This module provides TRL-style trainers for different fine-tuning scenarios,
focusing on chat templates and proper dataset formatting like TRL and Unsloth.

Available Trainers:
- SFTTrainer: Supervised Fine-Tuning for instruction following
- DPOTrainer: Direct Preference Optimization for alignment

Each trainer follows TRL patterns and integrates with the LLM Trainer's
memory-efficient optimizations.
"""

from .sft_trainer import SFTTrainer
from .sft_config import SFTConfig
from .dpo_trainer import DPOTrainer
from .dpo_config import DPOConfig

__all__ = [
    # Core trainers
    "SFTTrainer",
    "DPOTrainer",

    # Configuration classes
    "SFTConfig",
    "DPOConfig",
]

# Trainer mapping for convenience
TRAINER_MAPPING = {
    "sft": SFTTrainer,
    "supervised": SFTTrainer,
    "instruction": SFTTrainer,
    "dpo": DPOTrainer,
    "preference": DPOTrainer,
    "alignment": DPOTrainer,
}


def get_trainer_by_name(name: str):
    """
    Get a trainer class by name.

    Args:
        name: Name of the trainer (case-insensitive)

    Returns:
        Trainer class

    Raises:
        ValueError: If trainer name is not recognized
    """
    name = name.lower().strip()

    if name in TRAINER_MAPPING:
        return TRAINER_MAPPING[name]

    available_names = list(TRAINER_MAPPING.keys())
    raise ValueError(
        f"Unknown trainer name: '{name}'. Available trainers: {available_names}"
    )
