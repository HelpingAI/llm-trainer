"""
PPOConfig - placeholder config matching TRL-style API.
"""
from dataclasses import dataclass
from typing import Optional
from transformers import TrainingArguments


@dataclass
class PPOConfig(TrainingArguments):
    """Minimal PPO config inheriting from TrainingArguments (TRL-style)."""
    kl_coef: float = 0.1
    target_kl: Optional[float] = None
    whiten_rewards: bool = False

