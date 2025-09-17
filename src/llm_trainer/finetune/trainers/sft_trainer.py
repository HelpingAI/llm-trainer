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
Supervised Fine-Tuning (SFT) trainer implementation.

This module provides the SFTTrainer class for standard instruction-following fine-tuning,
following TRL patterns and supporting various data formats and optimization techniques.
"""

import contextlib
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Optional, Union

import torch
import torch.nn as nn
from accelerate import PartialState, logging
from datasets import Dataset, IterableDataset
from transformers import (
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
from .sft_config import SFTConfig

if is_peft_available():
    from peft import PeftConfig

logger = logging.get_logger(__name__)


class SFTTrainer(BaseFineTuneTrainer):
    """
    Trainer for Supervised Fine-Tuning (SFT) method.

    This class is a wrapper around the [`~transformers.Trainer`] class and inherits all of its attributes and methods.

    Example:

    ```python
    from datasets import load_dataset
    from llm_trainer.finetune.trainers import SFTTrainer

    dataset = load_dataset("roneneldan/TinyStories", split="train[:1%]")

    trainer = SFTTrainer(model="Qwen/Qwen2-0.5B-Instruct", train_dataset=dataset)
    trainer.train()
    ```

    Args:
        model (`Union[str, PreTrainedModel]`):
            Model to be trained. Can be either:
            - A string, being the *model id* of a pretrained model hosted inside a model repo on huggingface.co
            - A [`~transformers.PreTrainedModel`] object.
        args (`SFTConfig`, *optional*):
            Configuration for this trainer. If `None`, a default configuration is used.
        data_collator (`DataCollator`, *optional*):
            Function to use to form a batch from a list of elements of the processed dataset.
        train_dataset (`Dataset` or `IterableDataset`):
            Dataset to use for training. SFT supports both language modeling and prompt-completion types.
        eval_dataset (`Dataset`, `IterableDataset` or `dict[str, Union[Dataset, IterableDataset]]`):
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
        formatting_func (`Callable`, *optional*):
            Formatting function applied to the dataset before tokenization.
    """

    _tag_names = ["llm-trainer", "sft"]

    def __init__(
        self,
        model: Union[str, nn.Module, PreTrainedModel],
        args: Optional[Union[SFTConfig, TrainingArguments]] = None,
        data_collator: Optional[DataCollator] = None,
        train_dataset: Optional[Union[Dataset, IterableDataset]] = None,
        eval_dataset: Optional[Union[Dataset, IterableDataset, dict[str, Union[Dataset, IterableDataset]]]] = None,
        processing_class: Optional[Union[PreTrainedTokenizerBase, ProcessorMixin]] = None,
        compute_metrics: Optional[Callable[[EvalPrediction], dict]] = None,
        callbacks: Optional[list[TrainerCallback]] = None,
        optimizers: tuple[Optional[torch.optim.Optimizer], Optional[torch.optim.lr_scheduler.LambdaLR]] = (None, None),
        preprocess_logits_for_metrics: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
        peft_config: Optional["PeftConfig"] = None,
        formatting_func: Optional[Callable[[dict], str]] = None,
    ):
        # Args
        if args is None:
            model_name = model if isinstance(model, str) else model.config._name_or_path.split("/")[-1]
            args = SFTConfig(output_dir=f"{model_name}-sft")

        # Store formatting function
        self.formatting_func = formatting_func

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
    
    def _prepare_dataset(
        self,
        dataset: Union[Dataset, IterableDataset],
        processing_class: Union[PreTrainedTokenizerBase, ProcessorMixin],
        args: Union[SFTConfig, TrainingArguments],
        dataset_name: str,
    ) -> Union[Dataset, IterableDataset]:
        """
        Prepare dataset for SFT training following TRL patterns.

        Args:
            dataset: Dataset to prepare
            processing_class: Tokenizer or processor
            args: Training arguments
            dataset_name: Name of the dataset (for logging)

        Returns:
            Prepared dataset
        """
        # Build the kwargs for the `map` function
        map_kwargs = {}
        if isinstance(dataset, Dataset):  # IterableDataset does not support num_proc
            map_kwargs["num_proc"] = getattr(args, "dataset_num_proc", 1)

        with PartialState().main_process_first():
            # Apply the formatting function if any
            if self.formatting_func is not None:
                if isinstance(dataset, Dataset):  # `IterableDataset.map` does not support `desc`
                    map_kwargs["desc"] = f"Applying formatting function to {dataset_name} dataset"

                def _func(example):
                    return {"text": self.formatting_func(example)}

                dataset = dataset.map(_func, batched=False, **map_kwargs)

            # Handle different dataset formats
            first_example = next(iter(dataset))

            if "messages" in first_example:
                # Conversational format - use chat template (TRL/Unsloth style)
                if isinstance(dataset, Dataset):
                    map_kwargs["desc"] = f"Applying chat template to {dataset_name} dataset"

                def apply_chat_template_func(example):
                    if hasattr(processing_class, "apply_chat_template") and processing_class.chat_template is not None:
                        # Use the tokenizer's chat template (preferred method)
                        text = processing_class.apply_chat_template(
                            example["messages"],
                            tokenize=False,
                            add_generation_prompt=False
                        )
                    else:
                        # Fallback: simple format for models without chat template
                        text = ""
                        for message in example["messages"]:
                            role = message.get("role", "user")
                            content = message.get("content", "")
                            if role == "system":
                                text += f"System: {content}\n"
                            elif role == "user":
                                text += f"Human: {content}\n"
                            elif role == "assistant":
                                text += f"Assistant: {content}\n"
                        text = text.strip()
                    return {"text": text}

                dataset = dataset.map(
                    apply_chat_template_func,
                    remove_columns=["messages"],
                    **map_kwargs
                )

            elif "text" not in first_example:
                raise ValueError(
                    f"Dataset must contain either 'text' or 'messages' columns. "
                    f"Found columns: {list(first_example.keys())}"
                )

            # Add EOS token if needed
            if isinstance(dataset, Dataset):
                map_kwargs["desc"] = f"Adding EOS token to {dataset_name} dataset"

            def add_eos_token(example):
                if not example["text"].endswith(processing_class.eos_token):
                    example["text"] = example["text"] + processing_class.eos_token
                return example

            dataset = dataset.map(add_eos_token, **map_kwargs)

            # Tokenize the dataset
            if isinstance(dataset, Dataset):
                map_kwargs["desc"] = f"Tokenizing {dataset_name} dataset"

            def tokenize_function(examples):
                return processing_class(
                    examples["text"],
                    truncation=True,
                    padding=False,
                    max_length=getattr(args, "max_length", 2048),
                    return_overflowing_tokens=False,
                )

            dataset = dataset.map(
                tokenize_function,
                batched=True,
                remove_columns=["text"],
                **map_kwargs
            )

        return dataset
    

