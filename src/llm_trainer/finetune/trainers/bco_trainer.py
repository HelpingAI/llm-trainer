from typing import Optional, Union, Callable, Dict
import torch
import torch.nn as nn
from datasets import Dataset, IterableDataset
from transformers import TrainingArguments, DataCollator, PreTrainedModel, PreTrainedTokenizerBase, ProcessorMixin
from transformers.trainer_callback import TrainerCallback

try:
    from peft import PeftConfig  # type: ignore
except Exception:
    PeftConfig = None  # type: ignore

from .base_trainer import BaseFineTuneTrainer
from .bco_config import BCOConfig


class BCOTrainer(BaseFineTuneTrainer):
    _tag_names = ["llm-trainer", "bco"]

    def __init__(
        self,
        model: Union[str, nn.Module, PreTrainedModel],
        args: Optional[Union[BCOConfig, TrainingArguments]] = None,
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
        super().__init__(
            model=model,
            args=args or BCOConfig(output_dir="bco-output"),
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

