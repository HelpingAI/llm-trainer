"""
RewardTrainer - Reward model training for RLHF (TRL-style).
"""
from typing import Optional, Union, Callable, Dict, List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from datasets import Dataset, IterableDataset
from transformers import TrainingArguments, DataCollator, PreTrainedModel, PreTrainedTokenizerBase, ProcessorMixin
from transformers.trainer_callback import TrainerCallback

try:
    from peft import PeftConfig  # type: ignore
except Exception:  # pragma: no cover
    PeftConfig = None  # type: ignore

from .base_trainer import BaseFineTuneTrainer
from .reward_config import RewardConfig
from .utils import compute_accuracy

logger = logging.getLogger(__name__)


class RewardTrainer(BaseFineTuneTrainer):
    """
    Reward model trainer for RLHF.

    This trainer is used to train reward models on preference data,
    which can then be used for PPO training or other RLHF methods.
    """

    _tag_names = ["llm-trainer", "reward"]

    def __init__(
        self,
        model: Union[str, nn.Module, PreTrainedModel],
        args: Optional[Union[RewardConfig, TrainingArguments]] = None,
        data_collator: Optional[DataCollator] = None,
        train_dataset: Optional[Union[Dataset, IterableDataset]] = None,
        eval_dataset: Optional[Union[Dataset, IterableDataset, Dict[str, Union[Dataset, IterableDataset]]]] = None,
        processing_class: Optional[Union[PreTrainedTokenizerBase, ProcessorMixin]] = None,
        compute_metrics: Optional[Callable] = None,
        callbacks: Optional[list[TrainerCallback]] = None,
        optimizers: tuple[Optional[torch.optim.Optimizer], Optional[torch.optim.lr_scheduler.LambdaLR]] = (None, None),
        preprocess_logits_for_metrics: Optional[Callable] = None,
        peft_config: Optional["PeftConfig"] = None,
    ):
        # Initialize config
        if args is None:
            args = RewardConfig(output_dir="reward-output")

        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
            optimizers=optimizers,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
            peft_config=peft_config,
        )

        # Add reward head if not present
        self._add_reward_head()

        # Reward training specific attributes
        self.margin = getattr(args, "margin", 0.0)
        self.loss_type = getattr(args, "loss_type", "ranking")

        logger.info(f"Initialized RewardTrainer with loss_type={self.loss_type}, margin={self.margin}")

    def _add_reward_head(self):
        """Add a reward head to the model if it doesn't exist."""
        if not hasattr(self.model, "reward_head"):
            # Add a simple linear layer as reward head
            hidden_size = getattr(self.model.config, "hidden_size", 768)
            self.model.reward_head = nn.Linear(hidden_size, 1)
            logger.info(f"Added reward head with hidden_size={hidden_size}")

    def compute_reward_loss(self,
                           chosen_rewards: torch.Tensor,
                           rejected_rewards: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Compute reward model loss from chosen/rejected pairs."""

        if self.loss_type == "ranking":
            # Ranking loss: chosen should have higher reward than rejected
            loss = -F.logsigmoid(chosen_rewards - rejected_rewards - self.margin).mean()
        elif self.loss_type == "mse":
            # MSE loss: chosen=1, rejected=0
            chosen_targets = torch.ones_like(chosen_rewards)
            rejected_targets = torch.zeros_like(rejected_rewards)
            loss = F.mse_loss(chosen_rewards, chosen_targets) + F.mse_loss(rejected_rewards, rejected_targets)
        else:
            raise ValueError(f"Unknown loss_type: {self.loss_type}")

        # Compute accuracy (how often chosen > rejected)
        accuracy = (chosen_rewards > rejected_rewards).float().mean()

        return {
            "loss": loss,
            "accuracy": accuracy,
            "chosen_reward_mean": chosen_rewards.mean(),
            "rejected_reward_mean": rejected_rewards.mean(),
            "reward_margin": (chosen_rewards - rejected_rewards).mean(),
        }

    def get_reward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Get reward score for input sequences."""
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)

        # Get last hidden state
        if hasattr(outputs, "last_hidden_state"):
            hidden_states = outputs.last_hidden_state
        else:
            hidden_states = outputs[0]

        # Use the last token's hidden state (before padding)
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = hidden_states.size(0)
        last_hidden = hidden_states[range(batch_size), sequence_lengths]

        # Get reward score
        reward = self.model.reward_head(last_hidden).squeeze(-1)
        return reward

    def compute_loss(self, model, inputs, return_outputs=False):
        """Compute reward model loss."""
        # Expect inputs to have 'chosen_input_ids', 'rejected_input_ids', etc.
        chosen_input_ids = inputs.get("chosen_input_ids")
        rejected_input_ids = inputs.get("rejected_input_ids")
        chosen_attention_mask = inputs.get("chosen_attention_mask")
        rejected_attention_mask = inputs.get("rejected_attention_mask")

        if chosen_input_ids is None or rejected_input_ids is None:
            # Fallback to standard loss computation
            return super().compute_loss(model, inputs, return_outputs)

        # Get rewards for chosen and rejected responses
        chosen_rewards = self.get_reward(chosen_input_ids, chosen_attention_mask)
        rejected_rewards = self.get_reward(rejected_input_ids, rejected_attention_mask)

        # Compute loss
        loss_dict = self.compute_reward_loss(chosen_rewards, rejected_rewards)
        loss = loss_dict["loss"]

        # Log metrics
        self.log(loss_dict)

        return (loss, loss_dict) if return_outputs else loss

