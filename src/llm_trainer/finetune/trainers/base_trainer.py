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
Base trainer class for fine-tuning language models.

This module provides the base functionality for all fine-tuning trainers,
following TRL patterns and best practices for modular trainer architecture.
"""

import os
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import torch
import torch.nn as nn
from accelerate import PartialState, logging
from datasets import Dataset, IterableDataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollator,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    ProcessorMixin,
    Trainer,
    TrainingArguments,
)
from transformers.trainer_callback import TrainerCallback
from transformers.trainer_utils import EvalPrediction
from transformers.utils import is_peft_available

from ..fast_language_model import FastLanguageModel
from ..utils import print_trainable_parameters

if is_peft_available():
    from peft import PeftConfig, PeftModel

logger = logging.get_logger(__name__)


class BaseFineTuneTrainer(Trainer):
    """
    Base trainer class for fine-tuning language models.

    This class inherits from transformers.Trainer and provides common functionality
    for all specialized fine-tuning trainers, following TRL patterns.

    Args:
        model (`Union[str, PreTrainedModel]`):
            Model to be trained. Can be either:
            - A string, being the *model id* of a pretrained model hosted inside a model repo on huggingface.co
            - A [`~transformers.PreTrainedModel`] object.
        args (`TrainingArguments`, *optional*):
            Configuration for this trainer. If `None`, a default configuration is used.
        data_collator (`DataCollator`, *optional*):
            Function to use to form a batch from a list of elements of the processed dataset.
        train_dataset (`Dataset` or `IterableDataset`, *optional*):
            Dataset to use for training.
        eval_dataset (`Dataset`, `IterableDataset` or `dict[str, Union[Dataset, IterableDataset]]`, *optional*):
            Dataset to use for evaluation.
        processing_class (`PreTrainedTokenizerBase` or `ProcessorMixin`, *optional*):
            Processing class used to process the data.
        compute_metrics (`Callable[[EvalPrediction], dict]`, *optional*):
            The function that will be used to compute metrics at evaluation.
        callbacks (`list[TrainerCallback]`, *optional*):
            List of callbacks to customize the training loop.
        optimizers (`tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR]`, *optional*):
            A tuple containing the optimizer and the scheduler to use.
        preprocess_logits_for_metrics (`Callable[[torch.Tensor, torch.Tensor], torch.Tensor]`, *optional*):
            A function that preprocess the logits right before caching them at each evaluation step.
        peft_config (`PeftConfig`, *optional*):
            PEFT configuration used to wrap the model.
    """

    _tag_names = ["llm-trainer", "base"]

    def __init__(
        self,
        model: Union[str, nn.Module, PreTrainedModel],
        args: Optional[TrainingArguments] = None,
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
        # Handle model loading
        if isinstance(model, str):
            model = self._create_model_from_path(model, args)

        # Handle processing class (tokenizer/processor)
        if processing_class is None:
            model_id = model.config._name_or_path if hasattr(model.config, '_name_or_path') else "gpt2"
            processing_class = AutoTokenizer.from_pretrained(model_id)

        # PEFT configuration and model wrapping
        if peft_config is not None:
            model = self._prepare_peft_model(model, peft_config, args)

        # Initialize metrics tracking
        self._metrics = {"train": defaultdict(list), "eval": defaultdict(list)}

        # Initialize the parent Trainer
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
        )

        # Add tags for models that have been loaded with the correct transformers version
        if hasattr(self.model, "add_model_tags"):
            self.model.add_model_tags(self._tag_names)

    def _create_model_from_path(self, model_path: str, args: Optional[TrainingArguments]) -> PreTrainedModel:
        """Creates a model from a path or model identifier."""
        model_init_kwargs = {}

        # Handle torch dtype if specified in args
        if args and hasattr(args, 'torch_dtype'):
            dtype = getattr(args, 'torch_dtype', None)
            if isinstance(dtype, torch.dtype) or dtype == "auto" or dtype is None:
                pass  # dtype is already a torch.dtype or "auto" or None
            elif isinstance(dtype, str):  # it's a str, but not "auto"
                dtype = getattr(torch, dtype)
                model_init_kwargs["torch_dtype"] = dtype

        # Create model
        model = AutoModelForCausalLM.from_pretrained(model_path, **model_init_kwargs)
        return model

    def _prepare_peft_model(
        self,
        model: PreTrainedModel,
        peft_config: "PeftConfig",
        args: Optional[TrainingArguments]
    ) -> PreTrainedModel:
        """Prepares a model for PEFT training."""
        if not is_peft_available():
            raise ValueError(
                "PEFT is not installed and you passed a `peft_config`. Please install it to use PEFT models"
            )

        from peft import get_peft_model, prepare_model_for_kbit_training

        # if model is a peft model and we have a peft_config, we merge and unload it first
        if isinstance(model, PeftModel):
            model = model.merge_and_unload()

        if getattr(model, "is_loaded_in_8bit", False) or getattr(model, "is_loaded_in_4bit", False):
            prepare_model_kwargs = {"use_gradient_checkpointing": args.gradient_checkpointing if args else False}
            model = prepare_model_for_kbit_training(model, **prepare_model_kwargs)
        else:
            model = self._prepare_gradient_checkpointing(model, args)

        # get peft model with the given config
        model = get_peft_model(model, peft_config)

        return model

    def _prepare_gradient_checkpointing(self, model: PreTrainedModel, args: Optional[TrainingArguments]):
        """Prepare the gradient checkpointing for the model."""
        if args and args.gradient_checkpointing:
            # For backward compatibility with older versions of transformers
            if hasattr(model, "enable_input_require_grads"):
                model.enable_input_require_grads()
            else:
                def make_inputs_require_grad(module, input, output):
                    output.requires_grad_(True)

                model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

        return model

    def save_pretrained(self, save_directory: Union[str, Path], **kwargs):
        """
        Save model using FastLanguageModel for optimized saving.

        Args:
            save_directory: Directory to save the model
            **kwargs: Additional arguments for saving
        """
        FastLanguageModel.save_pretrained(
            self.model,
            self.processing_class,
            save_directory,
            **kwargs
        )

    def print_trainable_parameters(self):
        """Print trainable parameters information."""
        print_trainable_parameters(self.model)

    def log(self, logs: dict[str, float], start_time: Optional[float] = None) -> None:
        """
        Log metrics with additional processing.

        Args:
            logs: Dictionary of metrics to log
            start_time: Optional start time for timing calculations
        """
        mode = "train" if self.model.training else "eval"

        # Add any stored metrics
        if self._metrics[mode]:
            metrics = {key: sum(val) / len(val) for key, val in self._metrics[mode].items()}

            # Add prefix for eval metrics
            if mode == "eval":
                metrics = {f"eval_{key}": val for key, val in metrics.items()}

            logs.update(metrics)

        super().log(logs, start_time)

        # Clear stored metrics after logging
        self._metrics[mode].clear()

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        Compute the training/evaluation loss.

        This method can be overridden by subclasses to implement custom loss computation.

        Args:
            model: The model
            inputs: The inputs and targets of the model
            return_outputs: Whether to return the model outputs along with the loss
            num_items_in_batch: Number of items in the batch

        Returns:
            The loss tensor, and optionally the outputs
        """
        return super().compute_loss(model, inputs, return_outputs, num_items_in_batch)

    def create_model_card(
        self,
        model_name: Optional[str] = None,
        dataset_name: Optional[str] = None,
        tags: Union[str, list[str], None] = None,
    ):
        """
        Creates a draft of a model card using the information available to the trainer.

        Args:
            model_name: Name of the model
            dataset_name: Name of the dataset used for training
            tags: Tags to be associated with the model card
        """
        if not self.is_world_process_zero():
            return

        if hasattr(self.model.config, "_name_or_path") and not os.path.isdir(self.model.config._name_or_path):
            base_model = self.model.config._name_or_path
        else:
            base_model = None

        # normalize `tags` to a mutable set
        if tags is None:
            tags = set()
        elif isinstance(tags, str):
            tags = {tags}
        else:
            tags = set(tags)

        if hasattr(self.model.config, "unsloth_version"):
            tags.add("unsloth")

        if "JOB_ID" in os.environ:
            tags.add("hf_jobs")

        tags.update(self._tag_names)

        # Create basic model card content
        model_card_content = f"""---
base_model: {base_model or "unknown"}
tags:
{chr(10).join(f"- {tag}" for tag in sorted(tags))}
model-index:
- name: {model_name or "fine-tuned-model"}
  results: []
---

# {model_name or "Fine-tuned Model"}

This model has been fine-tuned using the LLM Trainer framework.

## Model Details

- **Base Model**: {base_model or "unknown"}
- **Fine-tuning Method**: {self.__class__.__name__}
- **Dataset**: {dataset_name or "unknown"}

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("{model_name or "path/to/model"}")
tokenizer = AutoTokenizer.from_pretrained("{model_name or "path/to/model"}")
```

## Training Details

This model was fine-tuned using the LLM Trainer framework with the {self.__class__.__name__} trainer.
"""

        # Save model card
        model_card_path = os.path.join(self.args.output_dir, "README.md")
        with open(model_card_path, "w", encoding="utf-8") as f:
            f.write(model_card_content)

    def _save_checkpoint(self, model, trial):
        """Save checkpoint with model card."""
        if self.args.hub_model_id is None:
            model_name = Path(self.args.output_dir).name
        else:
            model_name = self.args.hub_model_id.split("/")[-1]

        self.create_model_card(model_name=model_name)
        super()._save_checkpoint(model, trial)

    def __repr__(self) -> str:
        """String representation of the trainer."""
        model_name = self.model.__class__.__name__ if self.model else "None"
        train_size = len(self.train_dataset) if self.train_dataset else 0
        eval_size = len(self.eval_dataset) if self.eval_dataset else 0

        return (
            f"{self.__class__.__name__}(\n"
            f"  model={model_name},\n"
            f"  train_dataset_size={train_size},\n"
            f"  eval_dataset_size={eval_size}\n"
            f")"
        )
