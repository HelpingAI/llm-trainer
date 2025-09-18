from dataclasses import dataclass
from transformers import TrainingArguments


@dataclass
class ORPOConfig(TrainingArguments):
    pass

