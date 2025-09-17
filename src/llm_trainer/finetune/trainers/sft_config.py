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
Configuration class for Supervised Fine-Tuning (SFT) trainer.

This module provides the SFTConfig class that inherits from TrainingArguments
and adds SFT-specific configuration options.
"""

from dataclasses import dataclass, field
from typing import Literal, Optional

from transformers import TrainingArguments


@dataclass
class SFTConfig(TrainingArguments):
    """
    Configuration class for the SFTTrainer.
    
    Using [`~transformers.TrainingArguments`] as the base class, this class adds
    SFT-specific configuration options.
    
    Args:
        max_seq_length (`int`, *optional*, defaults to `None`):
            The maximum sequence length to use for the `ConstantLengthDataset` and for
            automatically creating the Dataset. If `None`, it will try to get it from the model config.
        dataset_text_field (`str`, *optional*, defaults to `"text"`):
            The name of the text field of the dataset, in case this is passed by a user, the trainer will
            automatically create a `ConstantLengthDataset` based on the `dataset_text_field`.
        formatting_func (`Callable`, *optional*, defaults to `None`):
            The formatting function to be used for creating the `ConstantLengthDataset`.
        infinite (`bool`, *optional*, defaults to `False`):
            Whether to use an infinite dataset or not. Mostly useful for training on streaming datasets.
        num_of_sequences (`int`, *optional*, defaults to `1024`):
            The number of sequences to use for the `ConstantLengthDataset`.
        chars_per_token (`float`, *optional*, defaults to `3.6`):
            The number of characters per token to use for the `ConstantLengthDataset`.
        packing (`bool`, *optional*, defaults to `False`):
            Whether to use packing or not for training. This will use a `ConstantLengthDataset`
            to pack the sequences together.
        dataset_num_proc (`int`, *optional*, defaults to `None`):
            The number of workers to use to tokenize the data. Only used when `packing=False`.
        dataset_batch_size (`int`, *optional*, defaults to `1000`):
            The number of examples to tokenize per batch. If batch_size <= 0 or batch_size == None,
            tokenize the full dataset as a single batch. Only used when `packing=False`.
        neftune_noise_alpha (`float`, *optional*, defaults to `None`):
            If not `None`, this will activate NEFTune noise embeddings. This can drastically improve model performance for instruction fine-tuning.
            Check out the original paper here: https://arxiv.org/abs/2310.05914 and the original code here: https://github.com/neelsjain/NEFTune.
        model_init_kwargs (`dict`, *optional*, defaults to `None`):
            Dict of keyword arguments to pass to the model during initialization.
        dataset_kwargs (`dict`, *optional*, defaults to `None`):
            Dict of keyword arguments to pass to the dataset during initialization.
        eval_packing (`bool`, *optional*, defaults to `None`):
            Whether to use packing or not for evaluation dataset. If `None`, will use the same value as `packing`.
    """
    
    max_seq_length: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "The maximum sequence length to use for the ConstantLengthDataset and for automatically creating the Dataset. "
                "If None, it will try to get it from the model config."
            )
        },
    )
    dataset_text_field: Optional[str] = field(
        default="text", metadata={"help": "The name of the text field of the dataset."}
    )
    formatting_func: Optional[str] = field(
        default=None,
        metadata={
            "help": "The name of the formatting function to be used for creating the ConstantLengthDataset."
        },
    )
    infinite: Optional[bool] = field(
        default=False,
        metadata={
            "help": "Whether to use an infinite dataset or not. Mostly useful for training on streaming datasets."
        },
    )
    num_of_sequences: Optional[int] = field(
        default=1024,
        metadata={"help": "The number of sequences to use for the ConstantLengthDataset."},
    )
    chars_per_token: Optional[float] = field(
        default=3.6,
        metadata={"help": "The number of characters per token to use for the ConstantLengthDataset."},
    )
    packing: Optional[bool] = field(
        default=False,
        metadata={
            "help": (
                "Whether to use packing or not for training. This will use a ConstantLengthDataset to pack the sequences together."
            )
        },
    )
    dataset_num_proc: Optional[int] = field(
        default=None,
        metadata={"help": "The number of workers to use to tokenize the data. Only used when `packing=False`."},
    )
    dataset_batch_size: int = field(
        default=1000,
        metadata={
            "help": (
                "The number of examples to tokenize per batch. If batch_size <= 0 or batch_size == None, "
                "tokenize the full dataset as a single batch. Only used when `packing=False`."
            )
        },
    )
    neftune_noise_alpha: Optional[float] = field(
        default=None,
        metadata={
            "help": (
                "If not `None`, this will activate NEFTune noise embeddings. This can drastically improve model performance for instruction fine-tuning. "
                "Check out the original paper here: https://arxiv.org/abs/2310.05914 and the original code here: https://github.com/neelsjain/NEFTune."
            )
        },
    )
    model_init_kwargs: Optional[dict] = field(
        default=None, metadata={"help": "Dict of keyword arguments to pass to the model during initialization."}
    )
    dataset_kwargs: Optional[dict] = field(
        default=None, metadata={"help": "Dict of keyword arguments to pass to the dataset during initialization."}
    )
    eval_packing: Optional[bool] = field(
        default=None,
        metadata={
            "help": "Whether to use packing or not for evaluation dataset. If `None`, will use the same value as `packing`."
        },
    )
