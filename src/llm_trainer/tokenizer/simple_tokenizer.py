"""Simple whitespace-based tokenizer for beginners."""

from typing import List, Optional, Union, Any, cast
from collections import Counter
from tqdm import tqdm

from .base_tokenizer import BaseTokenizer


class SimpleTokenizer(BaseTokenizer):
    """
    Simple whitespace-based tokenizer.
    
    This is the most basic tokenizer - it splits text on whitespace.
    Perfect for beginners to understand tokenization without complexity.
    
    Key Features:
    - Extremely simple: splits on whitespace
    - No training needed (vocabulary is just all unique words)
    - Perfect for educational purposes
    - Fast and easy to understand
    """

    def __init__(self):
        super().__init__()
        self.lowercase: bool = False  # Whether to lowercase tokens

    def _tokenize(self, text: str) -> List[str]:
        """Split text into tokens (words)."""
        if self.lowercase:
            text = text.lower()

        # Simple whitespace splitting
        tokens = text.split()

        # Remove empty tokens
        tokens = [token for token in tokens if token]

        return tokens

    def _train_on_texts(self, texts: List[str], vocab_size: int, **kwargs) -> None:
        """
        Build vocabulary from texts.
        
        Args:
            texts: List of training texts
            vocab_size: Maximum vocabulary size
        """
        min_frequency = kwargs.get('min_frequency', 1)
        verbose = kwargs.get('verbose', True)
        vocab_size = vocab_size if vocab_size is not None else None
        if verbose:
            print("Building simple word-level vocabulary...")

        # Initialize special tokens
        self.vocab = {
            self.pad_token: self.pad_token_id,
            self.unk_token: self.unk_token_id,
            self.bos_token: self.bos_token_id,
            self.eos_token: self.eos_token_id
        }

        self.special_tokens = {
            self.pad_token: self.pad_token_id,
            self.unk_token: self.unk_token_id,
            self.bos_token: self.bos_token_id,
            self.eos_token: self.eos_token_id
        }

        # Count all words
        word_counts = Counter()
        for text in tqdm(texts, desc="Counting words", disable=not verbose):
            tokens = self._tokenize(text)
            word_counts.update(tokens)

        # Filter by minimum frequency
        word_counts = {word: count for word, count in word_counts.items()
                      if count >= min_frequency}

        # Sort by frequency (most frequent first)
        sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)

        # Limit vocabulary size if specified
        if vocab_size:
            sorted_words = sorted_words[:vocab_size - len(self.special_tokens)]

        # Add words to vocabulary
        for word, _ in sorted_words:
            if word not in self.vocab:
                self.vocab[word] = len(self.vocab)

        # Create inverse vocabulary
        self.inverse_vocab = {v: k for k, v in self.vocab.items()}

        if verbose:
            print(f"Vocabulary size: {len(self.vocab)} words")
            print(f"Special tokens: {len(self.special_tokens)}")
            print(f"Regular words: {len(self.vocab) - len(self.special_tokens)}")

    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """Encode text to token IDs."""
        if not text:
            return []

        # Tokenize
        tokens = self._tokenize(text)

        # Convert to token IDs
        token_ids = []
        for token in tokens:
            token_id = self.vocab.get(token, self.unk_token_id)
            token_ids.append(token_id)

        # Add special tokens
        if add_special_tokens:
            token_ids = [self.bos_token_id] + token_ids + [self.eos_token_id]

        return token_ids

    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        """Decode token IDs back to text."""
        if not token_ids:
            return ""

        # Convert token IDs to words
        tokens = []
        for token_id in token_ids:
            token = self.inverse_vocab.get(token_id, self.unk_token)

            # Skip special tokens if requested
            if skip_special_tokens and token in self.special_tokens:
                continue

            tokens.append(token)

        # Join with spaces
        return ' '.join(tokens)


    def set_lowercase(self, lowercase: bool = True) -> None:
        """Set whether to lowercase tokens."""
        self.lowercase = lowercase
        # Note: Changing this requires retraining if vocabulary was already built

    def save_pretrained(self, save_directory: str) -> None:
        """Save tokenizer to directory."""
        import os
        import json
        os.makedirs(save_directory, exist_ok=True)

        # Save base tokenizer data
        super().save_pretrained(save_directory)

        # Update config
        config_file = os.path.join(save_directory, "tokenizer_config.json")
        with open(config_file, 'r', encoding='utf-8') as f:
            config = json.load(f)

        config["tokenizer_type"] = "simple"
        config["lowercase"] = self.lowercase

        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2)

    @classmethod
    def from_pretrained(cls, load_directory: str) -> 'SimpleTokenizer':
        """Load tokenizer from directory."""
        import os
        import json

        # Load base tokenizer
        tokenizer = cast('SimpleTokenizer', super().from_pretrained(load_directory))

        # Load simple tokenizer specific config
        config_file = os.path.join(load_directory, "tokenizer_config.json")
        if os.path.exists(config_file):
            with open(config_file, 'r', encoding='utf-8') as f:
                config = json.load(f)

            tokenizer.lowercase = config.get("lowercase", False)

        return tokenizer
