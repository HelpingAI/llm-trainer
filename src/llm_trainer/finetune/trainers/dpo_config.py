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
Configuration class for Direct Preference Optimization (DPO) trainer.

This module provides the DPOConfig class that inherits from TrainingArguments
and adds DPO-specific configuration options.
"""

from dataclasses import dataclass, field
from typing import Literal, Optional

from transformers import TrainingArguments


@dataclass
class DPOConfig(TrainingArguments):
    """
    Configuration class for the DPOTrainer.
    
    Using [`~transformers.TrainingArguments`] as the base class, this class adds
    DPO-specific configuration options.
    
    Args:
        beta (`float`, *optional*, defaults to `0.1`):
            The beta factor in DPO loss. Higher beta means less divergence from the original model.
        label_smoothing (`float`, *optional*, defaults to `0`):
            The robust DPO label smoothing parameter from the cDPO report and RLHF paper that should be between 0 and 0.5.
        loss_type (`str`, *optional*, defaults to `"sigmoid"`):
            The type of DPO loss to use. Either "sigmoid" the default DPO loss, "hinge" loss from SLiC paper, "ipo" from IPO paper, or "kto_pair" from the KTO paper.
        label_pad_token_id (`int`, *optional*, defaults to `-100`):
            The label pad token id. This is used to ignore the pad tokens in the loss calculation.
        padding_value (`int`, *optional*, defaults to `None`):
            The padding value if it is different to the tokenizer's pad_token_id.
        truncation_mode (`str`, *optional*, defaults to `"keep_end"`):
            The truncation mode to use, either `keep_end` or `keep_start`. This is used when the prompt + chosen/rejected responses are too long for the model.
        max_length (`int`, *optional*, defaults to `None`):
            The maximum length of the sequences in the batch. This argument is required if you want to use the default data collator.
        max_prompt_length (`int`, *optional*, defaults to `None`):
            The maximum length of the prompt. This argument is required if you want to use the default data collator.
        max_target_length (`int`, *optional*, defaults to `None`):
            The maximum length of the target. This argument is required if you want to use the default data collator and your model is an encoder-decoder.
        is_encoder_decoder (`bool`, *optional*, defaults to `None`):
            If no model is provided, we need to know if the model_init returns an encoder-decoder.
        disable_dropout (`bool`, *optional*, defaults to `True`):
            Whether or not to disable dropouts in `model` and `ref_model`.
        generate_during_eval (`bool`, *optional*, defaults to `False`):
            Whether to sample and log generations during evaluation step.
        precompute_ref_log_probs (`bool`, *optional*, defaults to `False`):
            Flag to precompute reference model log probabilities for training and evaluation datasets. This is useful when you want to train
            without the reference model and reduce the total GPU memory needed.
        dataset_num_proc (`int`, *optional*, defaults to `None`):
            The number of workers to use to tokenize the data. Defaults to None.
        model_init_kwargs (`dict`, *optional*, defaults to `None`):
            Dict of keyword arguments to pass to the model during initialization.
        ref_model_init_kwargs (`dict`, *optional*, defaults to `None`):
            Dict of keyword arguments to pass to the ref model during initialization.
        model_adapter_name (`str`, *optional*, defaults to `None`):
            Name of the train target PEFT adapter, when using LoRA with multiple adapters.
        ref_adapter_name (`str`, *optional*, defaults to `None`):
            Name of the reference PEFT adapter, when using LoRA with multiple adapters.
        reference_free (`bool`, *optional*, defaults to `False`):
            If True, we ignore the _provided_ reference model and implicitly use a reference model that assigns equal probability to all possible completions.
        force_use_ref_model (`bool`, *optional*, defaults to `False`):
            In case one passes a PEFT model for the active model and you want to use a different model for the ref_model, set this flag to True.
        f_divergence_type (`FDivergenceType`, *optional*, defaults to `FDivergenceType.REVERSE_KL`):
            The type of f-divergence to use for the DPO loss.
        f_alpha_divergence_coef (`float`, *optional*, defaults to `1.0`):
            The alpha coefficient for the f-divergence loss.
        sync_ref_model (`bool`, *optional*, defaults to `False`):
            The flag for syncing reference model weights with the active model. If set to `True`, the reference model weights are synced with the active model every `ref_model_sync_steps` steps.
        ref_model_sync_steps (`int`, *optional*, defaults to `64`):
            The number of steps after which the reference model is synced with the active model. This is used only when `sync_ref_model` is `True`.
        rpo_alpha (`float`, *optional*, defaults to `None`):
            The alpha parameter from the RPO paper. If `None`, no weighting is applied and the loss is the same as the DPO loss.
    """
    
    beta: float = field(default=0.1, metadata={"help": "The beta factor in DPO loss. Higher beta means less divergence from the original model."})
    label_smoothing: float = field(default=0, metadata={"help": "The robust DPO label smoothing parameter that should be between 0 and 0.5."})
    loss_type: Literal["sigmoid", "hinge", "ipo", "kto_pair", "bco_pair", "sppo_hard", "nca_pair", "robust"] = field(
        default="sigmoid", metadata={"help": "The type of DPO loss to use."}
    )
    label_pad_token_id: int = field(default=-100, metadata={"help": "The label pad token id."})
    padding_value: Optional[int] = field(default=None, metadata={"help": "The padding value if it is different to the tokenizer's pad_token_id."})
    truncation_mode: Literal["keep_end", "keep_start"] = field(default="keep_end", metadata={"help": "The truncation mode to use."})
    max_length: Optional[int] = field(default=None, metadata={"help": "The maximum length of the sequences in the batch."})
    max_prompt_length: Optional[int] = field(default=None, metadata={"help": "The maximum length of the prompt."})
    max_target_length: Optional[int] = field(default=None, metadata={"help": "The maximum length of the target."})
    is_encoder_decoder: Optional[bool] = field(default=None, metadata={"help": "If no model is provided, we need to know if the model_init returns an encoder-decoder."})
    disable_dropout: bool = field(default=True, metadata={"help": "Whether or not to disable dropouts in `model` and `ref_model`."})
    generate_during_eval: bool = field(default=False, metadata={"help": "Whether to sample and log generations during evaluation step."})
    precompute_ref_log_probs: bool = field(default=False, metadata={"help": "Flag to precompute reference model log probabilities."})
    dataset_num_proc: Optional[int] = field(default=None, metadata={"help": "The number of workers to use to tokenize the data."})
    model_init_kwargs: Optional[dict] = field(default=None, metadata={"help": "Dict of keyword arguments to pass to the model during initialization."})
    ref_model_init_kwargs: Optional[dict] = field(default=None, metadata={"help": "Dict of keyword arguments to pass to the ref model during initialization."})
    model_adapter_name: Optional[str] = field(default=None, metadata={"help": "Name of the train target PEFT adapter."})
    ref_adapter_name: Optional[str] = field(default=None, metadata={"help": "Name of the reference PEFT adapter."})
    reference_free: bool = field(default=False, metadata={"help": "If True, we ignore the provided reference model."})
    force_use_ref_model: bool = field(default=False, metadata={"help": "Force use of reference model for PEFT models."})
    sync_ref_model: bool = field(default=False, metadata={"help": "Sync reference model weights with the active model."})
    ref_model_sync_steps: int = field(default=64, metadata={"help": "Number of steps after which the reference model is synced."})
    rpo_alpha: Optional[float] = field(default=None, metadata={"help": "The alpha parameter from the RPO paper."})
