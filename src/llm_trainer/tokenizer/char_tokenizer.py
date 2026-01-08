"""Character-level tokenizer implementation."""

import json
from collections import Counter
from typing import List, Optional, Union, Any, cast
from tqdm import tqdm

from .base_tokenizer import BaseTokenizer


class CharTokenizer(BaseTokenizer):
    """
    Simple character-level tokenizer.
    
    This is the simplest tokenizer - it treats each character as a token.
    Great for beginners to understand tokenization basics.
    
    Key Features:
    - Extremely simple: one character = one token
    - No training needed (vocabulary is just all unique characters)
    - Perfect for small models and educational purposes
    - Handles any Unicode character
    """

    def __init__(self):
        super().__init__()

    def _train_on_texts(self, texts: List[str], vocab_size: int, **kwargs) -> None:
        """
        Build character vocabulary from texts.
        
        Args:
            texts: List of training texts
            vocab_size: Ignored for character tokenizer (uses all characters)
        """
        min_frequency = kwargs.get('min_frequency', 1)
        verbose = kwargs.get('verbose', True)
        if verbose:
            print("Building character-level vocabulary...")

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

        # Count all characters
        char_counts = Counter()
        for text in tqdm(texts, desc="Counting characters", disable=not verbose):
            for char in text:
                char_counts[char] += 1

        # Filter by minimum frequency
        char_counts = {char: count for char, count in char_counts.items()
                      if count >= min_frequency}

        # Add all characters to vocabulary
        for char in sorted(char_counts.keys()):
            if char not in self.vocab:
                self.vocab[char] = len(self.vocab)

        # Create inverse vocabulary
        self.inverse_vocab = {v: k for k, v in self.vocab.items()}

        if verbose:
            print(f"Vocabulary size: {len(self.vocab)} characters")
            print(f"Special tokens: {len(self.special_tokens)}")
            print(f"Regular characters: {len(self.vocab) - len(self.special_tokens)}")

    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """Encode text to character token IDs."""
        if not text:
            return []

        # Convert each character to its token ID
        token_ids = []
        for char in text:
            token_id = self.vocab.get(char, self.unk_token_id)
            token_ids.append(token_id)

        # Add special tokens
        if add_special_tokens:
            token_ids = [self.bos_token_id] + token_ids + [self.eos_token_id]

        return token_ids

    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        """Decode token IDs back to text."""
        if not token_ids:
            return ""

        # Convert token IDs to characters
        chars = []
        for token_id in token_ids:
            char = self.inverse_vocab.get(token_id, self.unk_token)

            # Skip special tokens if requested
            if skip_special_tokens and char in self.special_tokens:
                continue

            chars.append(char)

        # Join characters
        return ''.join(chars)


    def save_pretrained(self, save_directory: str) -> None:
        """Save tokenizer to directory."""
        import os
        os.makedirs(save_directory, exist_ok=True)

        # Save base tokenizer data
        super().save_pretrained(save_directory)

        # Update config to indicate character tokenizer
        config_file = os.path.join(save_directory, "tokenizer_config.json")
        with open(config_file, 'r', encoding='utf-8') as f:
            config = json.load(f)

        config["tokenizer_type"] = "character"

        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2)

    @classmethod
    def from_pretrained(cls, load_directory: str) -> 'CharTokenizer':
        """Load tokenizer from directory."""
        return cast('CharTokenizer', super().from_pretrained(load_directory))
