from dataclasses import dataclass
from transformers import TrainingArguments


@dataclass
class OnlineDPOConfig(TrainingArguments):
    pass

