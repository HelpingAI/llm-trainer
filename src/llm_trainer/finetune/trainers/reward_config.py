"""
RewardConfig - placeholder config matching TRL-style API.
"""
from dataclasses import dataclass
from transformers import TrainingArguments


@dataclass
class RewardConfig(TrainingArguments):
    """Minimal Reward model config."""
    pass

