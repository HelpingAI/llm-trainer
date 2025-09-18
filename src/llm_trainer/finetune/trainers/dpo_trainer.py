# Copyright 2024 LLM Trainer Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Direct Preference Optimization (DPO) trainer implementation.

This module provides the DPOTrainer class for training language models
to align with human preferences without requiring a reward model,
following TRL patterns and best practices.
"""

import contextlib
import copy
import os
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from accelerate import PartialState, logging
from datasets import Dataset, IterableDataset
from transformers import (
    AutoModelForCausalLM,
    DataCollator,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    ProcessorMixin,
    TrainingArguments,
    is_wandb_available,
)
from transformers.trainer_callback import TrainerCallback
from transformers.trainer_utils import EvalPrediction
from transformers.utils import is_peft_available

from .base_trainer import BaseFineTuneTrainer
from .dpo_config import DPOConfig

if is_peft_available():
    from peft import PeftConfig, PeftModel

logger = logging.get_logger(__name__)


class DPOTrainer(BaseFineTuneTrainer):
    """
    Direct Preference Optimization Trainer inspired by TRL's DPOTrainer.
    
    This trainer implements DPO for training models to align with human preferences
    without requiring a separate reward model.
    """
    
    def __init__(
        self,
        model: Union[str, nn.Module, PreTrainedModel],
        ref_model: Optional[Union[str, nn.Module, PreTrainedModel]] = None,
        args: Optional[Union[DPOConfig, TrainingArguments]] = None,
        data_collator: Optional[DataCollator] = None,
        train_dataset: Optional[Union[Dataset, IterableDataset]] = None,
        eval_dataset: Optional[Union[Dataset, IterableDataset, dict[str, Union[Dataset, IterableDataset]]]] = None,
        processing_class: Optional[Union[PreTrainedTokenizerBase, ProcessorMixin]] = None,
        compute_metrics: Optional[Callable[[EvalPrediction], dict]] = None,
        callbacks: Optional[list[TrainerCallback]] = None,
        optimizers: tuple[Optional[torch.optim.Optimizer], Optional[torch.optim.lr_scheduler.LambdaLR]] = (None, None),
        preprocess_logits_for_metrics: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
        peft_config: Optional["PeftConfig"] = None,
    ):
        # Args
        if args is None:
            model_name = model if isinstance(model, str) else model.config._name_or_path.split("/")[-1]
            args = DPOConfig(output_dir=f"{model_name}-dpo")

        # Handle model loading
        if isinstance(model, str):
            model = self._create_model_from_path(model, args)

        # Handle reference model
        if ref_model is None:
            ref_model = self._create_reference_model(model, args)
        elif isinstance(ref_model, str):
            ref_model = self._create_model_from_path(ref_model, args)

        # Store reference model and DPO parameters
        self.ref_model = ref_model
        self.beta = getattr(args, "beta", 0.1)
        self.label_smoothing = getattr(args, "label_smoothing", 0.0)
        self.loss_type = getattr(args, "loss_type", "sigmoid")

        # Disable dropout in both models if specified
        if getattr(args, "disable_dropout", True):
            self._disable_dropout_in_model(model)
            if self.ref_model is not None:
                self._disable_dropout_in_model(self.ref_model)

        # Process datasets if provided
        if train_dataset is not None:
            train_dataset = self._prepare_dataset(train_dataset, processing_class, args, "train")
        if eval_dataset is not None:
            if isinstance(eval_dataset, dict):
                eval_dataset = {
                    key: self._prepare_dataset(dataset, processing_class, args, key)
                    for key, dataset in eval_dataset.items()
                }
            else:
                eval_dataset = self._prepare_dataset(eval_dataset, processing_class, args, "eval")

        # Initialize the parent trainer
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
        
        # Setup reference model
        self._setup_reference_model()
    
    def _setup_reference_model(self):
        """Setup the reference model for DPO (silent)."""
        if self.ref_model is None and self.model is not None:
            try:
                import copy
                self.ref_model = copy.deepcopy(self.model)
                for param in self.ref_model.parameters():
                    param.requires_grad = False
                if hasattr(self.model, "device"):
                    self.ref_model = self.ref_model.to(self.model.device)
            except Exception:
                # Fallback: use the main model as reference
                self.ref_model = self.model
    
    def _process_dpo_dataset(self, dataset: Dataset) -> Dataset:
        """Process dataset for DPO training."""
        if not _DATASETS_AVAILABLE:
            raise ImportError("datasets library required for DPO dataset processing")
        
        def tokenize_dpo(examples):
            if not self.tokenizer:
                raise ValueError("Tokenizer is required for DPO dataset processing")
            
            # Get field names from config
            prompt_field = self.config.prompt_field
            chosen_field = self.config.chosen_field
            rejected_field = self.config.rejected_field
            
            # Extract data
            prompts = examples[prompt_field]
            chosen = examples[chosen_field]
            rejected = examples[rejected_field]
            
            # Combine prompt with responses
            chosen_texts = [p + c for p, c in zip(prompts, chosen)]
            rejected_texts = [p + r for p, r in zip(prompts, rejected)]
            
            # Tokenize
            chosen_tokens = self.tokenizer(
                chosen_texts,
                truncation=True,
                max_length=self.config.max_seq_length,
                padding=False
            )
            rejected_tokens = self.tokenizer(
                rejected_texts,
                truncation=True,
                max_length=self.config.max_seq_length,
                padding=False
            )
            
            return {
                "chosen_input_ids": chosen_tokens["input_ids"],
                "chosen_attention_mask": chosen_tokens["attention_mask"],
                "rejected_input_ids": rejected_tokens["input_ids"],
                "rejected_attention_mask": rejected_tokens["attention_mask"],
            }
        
        return dataset.map(
            tokenize_dpo,
            batched=True,
            remove_columns=dataset.column_names
        )
    
    def compute_dpo_loss(
        self,
        policy_chosen_logps: torch.Tensor,
        policy_rejected_logps: torch.Tensor,
        reference_chosen_logps: torch.Tensor,
        reference_rejected_logps: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute DPO loss.
        
        Args:
            policy_chosen_logps: Log probabilities of chosen responses from policy model
            policy_rejected_logps: Log probabilities of rejected responses from policy model
            reference_chosen_logps: Log probabilities of chosen responses from reference model
            reference_rejected_logps: Log probabilities of rejected responses from reference model
            
        Returns:
            Dictionary containing loss and metrics
        """
        # Calculate log ratios
        policy_logratios = policy_chosen_logps - policy_rejected_logps
        reference_logratios = reference_chosen_logps - reference_rejected_logps
        
        # DPO loss
        if self.config.loss_type == "sigmoid":
            losses = -F.logsigmoid(self.config.beta * (policy_logratios - reference_logratios))
        elif self.config.loss_type == "hinge":
            losses = torch.relu(1 - self.config.beta * (policy_logratios - reference_logratios))
        elif self.config.loss_type == "ipo":
            # IPO loss variant
            losses = (policy_logratios - reference_logratios - 1 / (2 * self.config.beta)) ** 2
        else:
            raise ValueError(f"Unknown loss type: {self.config.loss_type}")
        
        # Apply label smoothing if configured
        if self.config.label_smoothing > 0:
            losses = losses * (1 - self.config.label_smoothing) + self.config.label_smoothing * 0.5
        
        # Calculate metrics
        chosen_rewards = self.config.beta * (policy_chosen_logps - reference_chosen_logps)
        rejected_rewards = self.config.beta * (policy_rejected_logps - reference_rejected_logps)
        reward_accuracies = (chosen_rewards > rejected_rewards).float()
        
        return {
            "loss": losses.mean(),
            "chosen_rewards": chosen_rewards.mean(),
            "rejected_rewards": rejected_rewards.mean(),
            "reward_accuracy": reward_accuracies.mean(),
            "reward_margin": (chosen_rewards - rejected_rewards).mean(),
        }
    
    def get_batch_logps(
        self,
        model: PreTrainedModel,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Calculate log probabilities for a batch.
        
        Args:
            model: Model to use
            input_ids: Input token IDs
            attention_mask: Attention mask
            labels: Labels (if None, uses input_ids)
            
        Returns:
            Log probabilities
        """
        if labels is None:
            labels = input_ids.clone()
        
        # Forward pass
        with torch.no_grad() if model == self.ref_model else torch.enable_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
        
        # Calculate log probabilities
        log_probs = F.log_softmax(logits, dim=-1)
        
        # Gather log probabilities for the labels
        labels = labels[:, 1:].contiguous()  # Shift labels
        log_probs = log_probs[:, :-1].contiguous()  # Shift logits
        
        # Get log probabilities for each token
        per_token_logps = torch.gather(log_probs, dim=2, index=labels.unsqueeze(2)).squeeze(2)
        
        # Mask out padding tokens
        attention_mask = attention_mask[:, 1:].contiguous()
        per_token_logps = per_token_logps * attention_mask
        
        # Sum over sequence length
        return per_token_logps.sum(dim=1)
    
    def compute_loss(self, model, inputs):
        """
        Compute DPO loss for a batch.
        
        Args:
            model: The model
            inputs: Batch inputs
            
        Returns:
            Loss tensor
        """
        # Extract chosen and rejected data
        chosen_input_ids = inputs["chosen_input_ids"]
        chosen_attention_mask = inputs["chosen_attention_mask"]
        rejected_input_ids = inputs["rejected_input_ids"]
        rejected_attention_mask = inputs["rejected_attention_mask"]
        
        # Get log probabilities from policy model
        policy_chosen_logps = self.get_batch_logps(
            model, chosen_input_ids, chosen_attention_mask
        )
        policy_rejected_logps = self.get_batch_logps(
            model, rejected_input_ids, rejected_attention_mask
        )
        
        # Get log probabilities from reference model
        reference_chosen_logps = self.get_batch_logps(
            self.ref_model, chosen_input_ids, chosen_attention_mask
        )
        reference_rejected_logps = self.get_batch_logps(
            self.ref_model, rejected_input_ids, rejected_attention_mask
        )
        
        # Compute DPO loss
        loss_dict = self.compute_dpo_loss(
            policy_chosen_logps,
            policy_rejected_logps,
            reference_chosen_logps,
            reference_rejected_logps,
        )
        
        # Log metrics
        if hasattr(self, '_trainer') and self._trainer:
            self.log_metrics({
                "dpo/chosen_rewards": loss_dict["chosen_rewards"].item(),
                "dpo/rejected_rewards": loss_dict["rejected_rewards"].item(),
                "dpo/reward_accuracy": loss_dict["reward_accuracy"].item(),
                "dpo/reward_margin": loss_dict["reward_margin"].item(),
            })
        
        return loss_dict["loss"]
    
    def get_preference_stats(self) -> Dict[str, float]:
        """Get statistics about preference learning."""
        if not self.train_dataset:
            return {}
        
        # This would require running inference on the dataset
        # For now, return placeholder stats
        return {
            "dataset_size": len(self.train_dataset),
            "beta": self.config.beta,
            "loss_type": self.config.loss_type,
        }
    
    def preview_preferences(self, num_examples: int = 3) -> None:
        """No-op preview to keep API compatibility (prints removed)."""
        return None
    
    def __repr__(self) -> str:
        """String representation of the DPO trainer."""
        base_repr = super().__repr__()
        dpo_info = (
            f"  beta={self.config.beta},\n"
            f"  loss_type='{self.config.loss_type}',\n"
            f"  ref_model={'Available' if self.ref_model else 'None'},\n"
        )
        return base_repr.replace(")", f"  {dpo_info})")
