from dataclasses import dataclass
from transformers import TrainingArguments


@dataclass
class CPOConfig(TrainingArguments):
    pass

