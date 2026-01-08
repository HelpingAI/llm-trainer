"""Base tokenizer interface."""

from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Union, Any
import json
import os


class BaseTokenizer(ABC):
    """Abstract base class for tokenizers."""

    def __init__(self):
        self.vocab: Dict[str, int] = {}
        self.inverse_vocab: Dict[int, str] = {}
        self.special_tokens: Dict[str, int] = {}

        # Default special tokens
        self.pad_token = "<pad>"
        self.unk_token = "<unk>"
        self.bos_token = "<bos>"
        self.eos_token = "<eos>"

        self.pad_token_id = 0
        self.unk_token_id = 1
        self.bos_token_id = 2
        self.eos_token_id = 3

    def train(self, dataset: Union[str, Any, List[str]], vocab_size: Optional[int] = None, **kwargs) -> None:
        """Train tokenizer from various data sources.

        This method can be called in two ways:
        1. With a dataset source: train("wikitext", vocab_size=50000, ...)
        2. With a list of texts (legacy): train(texts, vocab_size=50000) where texts is List[str]

        Args:
            dataset: Data source for training. Can be:
                - str: HuggingFace dataset name (e.g., "wikitext") or local file/directory path
                - Dataset object: Already loaded HuggingFace dataset
                - List[str]: Direct list of text strings (legacy mode - if first arg is List[str])
            vocab_size: Target vocabulary size
            **kwargs: Additional arguments (text_column, max_samples, verbose, etc.)
        """
        # Check if dataset is a List[str] - legacy mode
        if isinstance(dataset, list) and len(dataset) > 0 and isinstance(dataset[0], str):
            # Legacy mode: train(texts: List[str], vocab_size: int)
            if vocab_size is None:
                raise ValueError("vocab_size is required when training from List[str]")
            return self._train_on_texts(dataset, vocab_size, **kwargs)
        
        # New flexible mode
        texts = self._load_texts_from_dataset(dataset, **kwargs)
        vocab_size = vocab_size or kwargs.get('vocab_size', 50000)
        self._train_on_texts(texts, vocab_size, **kwargs)

    @abstractmethod
    def _train_on_texts(self, texts: List[str], vocab_size: int) -> None:
        """Train the tokenizer on a corpus of texts (internal method).
        
        This is the method that concrete tokenizer classes must implement.
        """
        pass

    def _load_texts_from_dataset(self, dataset: Union[str, Any, List[str]], **kwargs) -> List[str]:
        """Load texts from various data sources."""
        text_column = kwargs.get('text_column', 'text')
        max_samples = kwargs.get('max_samples', None)
        verbose = kwargs.get('verbose', True)

        # Handle List[str] - direct text input
        if isinstance(dataset, list):
            if verbose:
                print(f"Using {len(dataset)} provided texts")
            texts = [text for text in dataset if text and text.strip()]
            if max_samples:
                texts = texts[:max_samples]
            return texts

        # Handle string - could be HuggingFace dataset name or file path
        if isinstance(dataset, str):
            # Check if it's a file path
            if os.path.exists(dataset):
                return self._load_texts_from_file(dataset, max_samples, verbose)
            else:
                # Assume it's a HuggingFace dataset name
                return self._load_texts_from_hf_dataset(dataset, **kwargs)

        # Handle Dataset object (assume it's a HuggingFace dataset)
        try:
            from datasets import Dataset, DatasetDict
            if hasattr(dataset, '__iter__'):
                if hasattr(dataset, 'column_names'):
                    # Single dataset
                    return self._load_texts_from_hf_dataset_object(dataset, text_column, max_samples, verbose)
                elif hasattr(dataset, 'keys') and hasattr(dataset, 'values'):
                    # DatasetDict - use train split by default, or first available split
                    split = kwargs.get('split', 'train')
                    if split in dataset:
                        if verbose:
                            print(f"Using '{split}' split from DatasetDict")
                        return self._load_texts_from_hf_dataset_object(dataset[split], text_column, max_samples, verbose)
                    else:
                        # Use first available split
                        first_split = list(dataset.keys())[0]
                        if verbose:
                            print(f"Split '{split}' not found, using '{first_split}' split")
                        return self._load_texts_from_hf_dataset_object(dataset[first_split], text_column, max_samples, verbose)
        except ImportError:
            pass

        # Fallback: try to iterate over the dataset
        try:
            texts = []
            for i, item in enumerate(dataset):
                if max_samples and i >= max_samples:
                    break
                if isinstance(item, str):
                    text = item
                elif isinstance(item, dict) and text_column in item:
                    text = item[text_column]
                else:
                    continue
                if text and text.strip():
                    texts.append(text.strip())
            return texts
        except:
            raise ValueError(f"Unsupported dataset type: {type(dataset)}")

    def _load_texts_from_file(self, file_path: str, max_samples: Optional[int] = None,
                             verbose: bool = True) -> List[str]:
        """Load texts from local files."""
        texts = []

        if os.path.isfile(file_path):
            # Single file
            files = [file_path]
        elif os.path.isdir(file_path):
            # Directory - find all text files
            files = []
            for root, _, filenames in os.walk(file_path):
                for filename in filenames:
                    if filename.endswith(('.txt', '.md', '.json', '.jsonl')):
                        files.append(os.path.join(root, filename))
        else:
            raise ValueError(f"Path does not exist: {file_path}")

        if verbose:
            print(f"Loading texts from {len(files)} files")

        for file_path in files:
            try:
                if file_path.endswith('.json'):
                    # JSON file - assume list of objects with text field
                    import json
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        if isinstance(data, list):
                            for item in data:
                                if isinstance(item, dict) and 'text' in item:
                                    text = item['text']
                                    if text and text.strip():
                                        texts.append(text.strip())
                                        if max_samples and len(texts) >= max_samples:
                                            break
                elif file_path.endswith('.jsonl'):
                    # JSONL file
                    import json
                    with open(file_path, 'r', encoding='utf-8') as f:
                        for line in f:
                            if line.strip():
                                item = json.loads(line)
                                if isinstance(item, dict) and 'text' in item:
                                    text = item['text']
                                    if text and text.strip():
                                        texts.append(text.strip())
                                        if max_samples and len(texts) >= max_samples:
                                            break
                else:
                    # Plain text file
                    with open(file_path, 'r', encoding='utf-8') as f:
                        for line in f:
                            line = line.strip()
                            if line:
                                texts.append(line)
                                if max_samples and len(texts) >= max_samples:
                                    break
            except Exception as e:
                if verbose:
                    print(f"Warning: Failed to read {file_path}: {e}")
                continue

            if max_samples and len(texts) >= max_samples:
                break

        if verbose:
            print(f"Loaded {len(texts)} texts")

        return texts

    def _load_texts_from_hf_dataset(self, dataset_name: str, **kwargs) -> List[str]:
        """Load texts from HuggingFace dataset."""
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError("Please install datasets: uv pip install datasets")

        dataset_config = kwargs.get('dataset_config')
        split = kwargs.get('split', 'train')
        text_column = kwargs.get('text_column', 'text')
        max_samples = kwargs.get('max_samples')
        verbose = kwargs.get('verbose', True)

        if verbose:
            print(f"Loading dataset: {dataset_name}")
            if dataset_config:
                print(f"Using config: {dataset_config}")

        # Load dataset
        try:
            if dataset_config:
                dataset = load_dataset(dataset_name, dataset_config, split=split)
            else:
                dataset = load_dataset(dataset_name, split=split)
        except ValueError as e:
            if "BuilderConfig" in str(e) and "not found" in str(e):
                if verbose:
                    print(f"Config '{dataset_config}' not found, trying with default config...")
                dataset = load_dataset(dataset_name, split=split)
            else:
                raise e

        return self._load_texts_from_hf_dataset_object(dataset, text_column, max_samples, verbose)

    def _load_texts_from_hf_dataset_object(self, dataset: Any, text_column: str = 'text',
                                          max_samples: Optional[int] = None,
                                          verbose: bool = True) -> List[str]:
        """Load texts from HuggingFace dataset object."""
        from tqdm import tqdm

        texts = []
        dataset_size = len(dataset)

        # Determine how many samples to use
        max_samples_to_use = max_samples or dataset_size

        if verbose:
            print(f"Dataset size: {dataset_size}, using {max_samples_to_use} samples")

        for i, example in enumerate(tqdm(dataset, desc="Loading texts", disable=not verbose)):
            if i >= max_samples_to_use:
                break

            text = example[text_column]
            if text and text.strip():
                texts.append(text.strip())

        if verbose:
            print(f"Loaded {len(texts)} texts")

        return texts

    @abstractmethod
    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """Encode text to token IDs."""
        pass

    @abstractmethod
    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        """Decode token IDs to text."""
        pass

    @property
    def vocab_size(self) -> int:
        """Get vocabulary size."""
        return len(self.vocab)

    def get_vocab(self) -> Dict[str, int]:
        """Get the vocabulary dictionary."""
        return self.vocab.copy()

    def token_to_id(self, token: str) -> int:
        """Convert token to ID."""
        return self.vocab.get(token, self.unk_token_id)

    def id_to_token(self, token_id: int) -> str:
        """Convert ID to token."""
        return self.inverse_vocab.get(token_id, self.unk_token)

    def add_special_tokens(self, special_tokens: Dict[str, str]) -> None:
        """Add special tokens to vocabulary."""
        for token_name, token in special_tokens.items():
            if token not in self.vocab:
                token_id = len(self.vocab)
                self.vocab[token] = token_id
                self.inverse_vocab[token_id] = token
                self.special_tokens[token] = token_id
                setattr(self, f"{token_name}_token", token)
                setattr(self, f"{token_name}_token_id", token_id)

    def save_pretrained(self, save_directory: str) -> None:
        """Save tokenizer to directory."""
        import os
        os.makedirs(save_directory, exist_ok=True)

        # Save vocabulary
        vocab_file = os.path.join(save_directory, "vocab.json")
        with open(vocab_file, 'w', encoding='utf-8') as f:
            json.dump(self.vocab, f, ensure_ascii=False, indent=2)

        # Save tokenizer config
        config = {
            "tokenizer_class": self.__class__.__name__,
            "special_tokens": self.special_tokens,
            "pad_token": self.pad_token,
            "unk_token": self.unk_token,
            "bos_token": self.bos_token,
            "eos_token": self.eos_token,
            "vocab_size": self.vocab_size
        }

        config_file = os.path.join(save_directory, "tokenizer_config.json")
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2)

    @classmethod
    def from_pretrained(cls, load_directory: str) -> 'BaseTokenizer':
        """Load tokenizer from directory."""
        import os

        # Load vocabulary
        vocab_file = os.path.join(load_directory, "vocab.json")
        with open(vocab_file, 'r', encoding='utf-8') as f:
            vocab = json.load(f)

        # Load config
        config_file = os.path.join(load_directory, "tokenizer_config.json")
        with open(config_file, 'r', encoding='utf-8') as f:
            config = json.load(f)

        # Create tokenizer instance
        tokenizer = cls()
        tokenizer.vocab = vocab
        tokenizer.inverse_vocab = {v: k for k, v in vocab.items()}
        tokenizer.special_tokens = config["special_tokens"]

        # Set special token attributes
        for token_name in ["pad", "unk", "bos", "eos"]:
            token = config[f"{token_name}_token"]
            setattr(tokenizer, f"{token_name}_token", token)
            setattr(tokenizer, f"{token_name}_token_id", vocab[token])

        return tokenizer

    def batch_encode(self, texts: List[str], add_special_tokens: bool = True,
                    padding: bool = False, max_length: Optional[int] = None) -> List[List[int]]:
        """Encode a batch of texts."""
        encoded = [self.encode(text, add_special_tokens) for text in texts]

        if padding or max_length:
            max_len = max_length or max(len(seq) for seq in encoded)
            encoded = [seq[:max_len] + [self.pad_token_id] * (max_len - len(seq))
                      for seq in encoded]

        return encoded

    def batch_decode(self, token_ids_list: List[List[int]],
                    skip_special_tokens: bool = True) -> List[str]:
        """Decode a batch of token ID sequences."""
        return [self.decode(token_ids, skip_special_tokens) for token_ids in token_ids_list]
