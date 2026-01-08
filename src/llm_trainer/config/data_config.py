"""Data configuration for LLM training."""

from dataclasses import dataclass
from typing import Optional, List
import yaml
import json


@dataclass
class DataConfig:
    """Configuration for data loading and preprocessing."""

    # Dataset configuration
    dataset_name: str = "wikitext"
    dataset_config: Optional[str] = None
    dataset_split: str = "train"
    validation_split: Optional[str] = "validation"
    test_split: Optional[str] = "test"

    # Data preprocessing
    text_column: str = "text"
    max_length: int = 1024
    min_length: int = 10
    stride: int = 512  # For sliding window approach

    # Tokenization
    tokenizer_type: str = "bpe"  # bpe, sentencepiece, huggingface
    vocab_size: int = 50000
    special_tokens: Optional[List[str]] = None

    # Data filtering
    filter_empty_lines: bool = True
    filter_short_sequences: bool = True
    remove_duplicates: bool = False

    # Data augmentation
    use_data_augmentation: bool = False
    augmentation_prob: float = 0.1

    # Caching and streaming
    use_streaming: bool = False
    cache_dir: Optional[str] = None
    preprocessing_num_workers: int = 4

    # Custom dataset paths
    train_file: Optional[str] = None
    validation_file: Optional[str] = None
    test_file: Optional[str] = None

    # Data format
    data_format: str = "text"  # text, jsonl, csv

    # Sequence packing for efficiency
    pack_sequences: bool = True
    packing_strategy: str = "greedy"  # greedy, first_fit

    def __post_init__(self):
        """Validate and set default values."""
        if self.special_tokens is None:
            self.special_tokens = ["<pad>", "<unk>", "<bos>", "<eos>"]

        # Validation
        assert self.max_length > 0, "max_length must be positive"
        assert self.min_length > 0, "min_length must be positive"
        assert self.min_length <= self.max_length, "min_length must be <= max_length"
        assert self.vocab_size > 0, "vocab_size must be positive"
        assert 0 <= self.stride <= self.max_length, "stride must be between 0 and max_length"
        assert 0 <= self.augmentation_prob <= 1, "augmentation_prob must be between 0 and 1"
        assert self.preprocessing_num_workers >= 0, "preprocessing_num_workers must be non-negative"

        # Format validation
        valid_formats = ["text", "jsonl", "csv"]
        assert self.data_format in valid_formats, f"data_format must be one of {valid_formats}"

        valid_tokenizers = ["bpe", "sentencepiece", "huggingface"]
        assert self.tokenizer_type in valid_tokenizers, f"tokenizer_type must be one of {valid_tokenizers}"

        valid_packing = ["greedy", "first_fit"]
        assert self.packing_strategy in valid_packing, f"packing_strategy must be one of {valid_packing}"

    def save(self, path: str) -> None:
        """Save configuration to file."""
        config_dict = self.__dict__.copy()

        if path.endswith('.yaml') or path.endswith('.yml'):
            with open(path, 'w') as f:
                yaml.dump(config_dict, f, default_flow_style=False)
        elif path.endswith('.json'):
            with open(path, 'w') as f:
                json.dump(config_dict, f, indent=2)
        else:
            raise ValueError("Unsupported file format. Use .yaml, .yml, or .json")

    @classmethod
    def load(cls, path: str) -> 'DataConfig':
        """Load configuration from file."""
        if path.endswith('.yaml') or path.endswith('.yml'):
            with open(path, 'r') as f:
                config_dict = yaml.safe_load(f)
        elif path.endswith('.json'):
            with open(path, 'r') as f:
                config_dict = json.load(f)
        else:
            raise ValueError("Unsupported file format. Use .yaml, .yml, or .json")

        return cls(**config_dict)

    def to_dict(self) -> dict:
        """Convert configuration to dictionary."""
        return self.__dict__.copy()

    @classmethod
    def from_dict(cls, config_dict: dict) -> 'DataConfig':
        """Create configuration from dictionary."""
        return cls(**config_dict)
