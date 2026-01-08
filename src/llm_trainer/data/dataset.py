"""Dataset classes for language modeling."""

import torch
from torch.utils.data import Dataset
from typing import List, Dict, Any, Optional, Union
from datasets import load_dataset, Dataset as HFDataset

from .preprocessing import TextPreprocessor, TextTokenizer, SequencePacker
from ..tokenizer import BaseTokenizer


class TextDataset(Dataset[Dict[str, torch.Tensor]]):
    """Basic text dataset for language modeling."""

    def __init__(self, texts: List[str], tokenizer: BaseTokenizer, max_length: int = 1024,
                 add_special_tokens: bool = True):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.add_special_tokens = add_special_tokens

        # Pre-tokenize all texts
        self.tokenized_texts = []
        for text in texts:
            token_ids = tokenizer.encode(text, add_special_tokens=add_special_tokens)
            # Truncate if necessary
            if len(token_ids) > max_length:
                token_ids = token_ids[:max_length]
            self.tokenized_texts.append(token_ids)

    def __len__(self) -> int:
        return len(self.tokenized_texts)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        token_ids = self.tokenized_texts[index]

        return {
            "input_ids": torch.tensor(token_ids, dtype=torch.long),
            "labels": torch.tensor(token_ids, dtype=torch.long),  # For language modeling
            "attention_mask": torch.ones(len(token_ids), dtype=torch.bool)
        }


class LanguageModelingDataset(Dataset[Dict[str, torch.Tensor]]):
    """Dataset for causal language modeling with Hugging Face datasets."""

    def __init__(self,
                 dataset_name: str,
                 dataset_config: Optional[str] = None,
                 split: str = "train",
                 tokenizer: Optional[BaseTokenizer] = None,
                 text_column: str = "text",
                 max_length: int = 1024,
                 stride: int = 512,
                 preprocessing_config: Optional[Dict[str, Any]] = None,
                 use_streaming: bool = False,
                 cache_dir: Optional[str] = None,
                 num_proc: Optional[int] = None,
                 pack_sequences: bool = False,
                 packing_strategy: str = "greedy",
                 formatting_func: Optional[Any] = None):
        """
        Args:
            dataset_name: Name of the dataset.
            dataset_config: Optional dataset config.
            split: Dataset split.
            tokenizer: Tokenizer to use.
            text_column: Name of the text column.
            max_length: Max sequence length.
            stride: Stride for sliding window.
            preprocessing_config: Preprocessing config dict.
            use_streaming: Whether to use streaming mode.
            cache_dir: Optional cache directory.
            num_proc: Number of processes for preprocessing.
            pack_sequences: Whether to pack sequences.
            packing_strategy: Packing strategy.
            formatting_func: Optional formatting function for custom sample formatting (like SFTTrainer).
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.text_column = text_column
        self.pack_sequences = pack_sequences
        self.formatting_func = formatting_func

        if use_streaming:
            raise ValueError("This class does not support streaming. Use StreamingLanguageModelingDataset instead.")

        # Load dataset
        if dataset_config:
            self.dataset = load_dataset(
                dataset_name,
                dataset_config,
                split=split,
                streaming=use_streaming,
                cache_dir=cache_dir
            )
        else:
            self.dataset = load_dataset(
                dataset_name,
                split=split,
                streaming=use_streaming,
                cache_dir=cache_dir
            )

        # Initialize preprocessor
        preprocessing_config = preprocessing_config or {}
        self.preprocessor = TextPreprocessor(**preprocessing_config)

        # Initialize tokenizer wrapper
        self.text_tokenizer = TextTokenizer(
            tokenizer=tokenizer,
            max_length=max_length,
            stride=stride,
            add_special_tokens=True
        )

        # Process dataset
        self._process_dataset(num_proc)

        # Pack sequences if requested
        if pack_sequences and tokenizer:
            self._pack_sequences(packing_strategy)

    def _process_dataset(self, num_proc: Optional[int] = None):
        """Process the dataset with preprocessing and tokenization."""
        # Apply text preprocessing
        self.dataset = self.preprocessor.process_dataset(
            self.dataset,
            text_column=self.text_column,
            num_proc=num_proc
        )

        # Apply tokenization if tokenizer is provided
        if self.tokenizer:
            self.dataset = self.text_tokenizer.tokenize_dataset(
                self.dataset,
                text_column=self.text_column,
                use_sliding_window=True,
                num_proc=num_proc
            )

    def _pack_sequences(self, strategy: str = "greedy"):
        """Pack sequences for efficiency."""
        if not self.tokenizer:
            return

        # Get all sequences
        sequences = self.dataset["input_ids"]

        # Initialize packer
        packer = SequencePacker(
            max_length=self.max_length,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            strategy=strategy
        )

        # Pack sequences
        packed_sequences = packer.pack_sequences(sequences)

        # Create new dataset with packed sequences
        self.dataset = HFDataset.from_dict({
            "input_ids": packed_sequences,
            "length": [len(seq) for seq in packed_sequences]
        })

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        item = self.dataset[index]

        # If formatting_func is provided, use it to format the sample before tokenization
        if self.formatting_func is not None:
            if self.tokenizer is None:
                raise ValueError("tokenizer must be provided when using formatting_func")
            # SFTTrainer expects formatting_func to return a list of formatted strings
            formatted = self.formatting_func(item)
            # Use the first formatted string if a list is returned
            if isinstance(formatted, list):
                formatted_text = formatted[0]
            else:
                formatted_text = formatted
            # Tokenize the formatted text
            token_ids = self.tokenizer.encode(formatted_text, add_special_tokens=True)
            if len(token_ids) > self.max_length:
                token_ids = token_ids[:self.max_length]
            input_ids = torch.tensor(token_ids, dtype=torch.long)
            attention_mask = (input_ids != self.tokenizer.pad_token_id).long() if self.tokenizer else torch.ones_like(input_ids)
            return {
                "input_ids": input_ids,
                "labels": input_ids.clone(),
                "attention_mask": attention_mask
            }

        if isinstance(item["input_ids"], list):
            input_ids = torch.tensor(item["input_ids"], dtype=torch.long)
        else:
            input_ids = item["input_ids"]  # type: ignore

        # Create attention mask (1 for real tokens, 0 for padding)
        if self.tokenizer:
            attention_mask = (input_ids != self.tokenizer.pad_token_id).long()
        else:
            attention_mask = torch.ones_like(input_ids)

        return {
            "input_ids": input_ids,
            "labels": input_ids.clone(),  # For causal LM, labels are the same as input_ids
            "attention_mask": attention_mask
        }


class StreamingLanguageModelingDataset:
    """Streaming dataset for very large datasets."""

    def __init__(self,
                 dataset_name: str,
                 dataset_config: Optional[str] = None,
                 split: str = "train",
                 tokenizer: Optional[BaseTokenizer] = None,
                 text_column: str = "text",
                 max_length: int = 1024,
                 buffer_size: int = 10000,
                 preprocessing_config: Optional[Dict[str, Any]] = None):

        self.tokenizer = tokenizer
        self.max_length = max_length
        self.text_column = text_column
        self.buffer_size = buffer_size

        # Load streaming dataset
        if dataset_config:
            self.dataset = load_dataset(
                dataset_name,
                dataset_config,
                split=split,
                streaming=True
            )
        else:
            self.dataset = load_dataset(
                dataset_name,
                split=split,
                streaming=True
            )

        # Initialize preprocessor
        preprocessing_config = preprocessing_config or {}
        self.preprocessor = TextPreprocessor(**preprocessing_config)

        # Initialize tokenizer wrapper
        self.text_tokenizer = TextTokenizer(
            tokenizer=tokenizer,
            max_length=max_length,
            add_special_tokens=True
        )

        # Buffer for processed examples
        self.buffer = []
        self.dataset_iter = iter(self.dataset)

    def _fill_buffer(self):
        """Fill buffer with processed examples."""
        while len(self.buffer) < self.buffer_size:
            try:
                example = next(self.dataset_iter)

                # Preprocess text
                text = example[self.text_column]
                processed_text = self.preprocessor.process_text(text)

                if processed_text is None:
                    continue

                # Tokenize
                if self.tokenizer:
                    token_ids = self.text_tokenizer.tokenize_text(processed_text)
                    if len(token_ids) > self.max_length:
                        token_ids = token_ids[:self.max_length]

                    # Create example
                    processed_example = {
                        "input_ids": torch.tensor(token_ids, dtype=torch.long),
                        "labels": torch.tensor(token_ids, dtype=torch.long),
                        "attention_mask": torch.ones(len(token_ids), dtype=torch.bool)
                    }

                    self.buffer.append(processed_example)

            except StopIteration:
                break

    def get_batch(self, batch_size: int) -> List[Dict[str, torch.Tensor]]:
        """Get a batch of examples."""
        # Fill buffer if needed
        if len(self.buffer) < batch_size:
            self._fill_buffer()

        # Get batch
        batch = self.buffer[:batch_size]
        self.buffer = self.buffer[batch_size:]

        return batch

    def __iter__(self):
        """Make dataset iterable."""
        return self

    def __next__(self):
        """Get next example."""
        if len(self.buffer) == 0:
            self._fill_buffer()

        if len(self.buffer) == 0:
            raise StopIteration

        return self.buffer.pop(0)
