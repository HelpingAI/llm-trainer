"""Byte-level BPE tokenizer implementation (GPT-2 style)."""

import json
from collections import defaultdict
from typing import List, Dict, Tuple, Optional, Union, Any
from tqdm import tqdm

from .base_tokenizer import BaseTokenizer


class ByteBPETokenizer(BaseTokenizer):
    """
    Byte-level BPE tokenizer (GPT-2 style).
    
    This tokenizer first converts text to bytes, then applies BPE on bytes.
    This ensures that any text can be tokenized without unknown tokens.
    
    Key Features:
    - Byte-level encoding ensures no unknown tokens
    - GPT-2 compatible tokenization
    - Handles any Unicode text
    - Efficient encoding/decoding
    """

    def __init__(self):
        super().__init__()
        self.merges: List[Tuple[int, int]] = []  # Byte pair merges
        self.byte_to_token: Dict[int, int] = {}  # Byte -> token ID mapping
        self.token_to_bytes: Dict[int, bytes] = {}  # Token ID -> bytes mapping
        self.cache: Dict[str, List[int]] = {}

        # Initialize byte vocabulary (256 bytes)
        self._init_byte_vocab()

    def _init_byte_vocab(self):
        """Initialize vocabulary with all 256 possible bytes."""
        # Add special tokens first
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

        # Add all 256 bytes to vocabulary
        for byte_val in range(256):
            byte_str = f"<0x{byte_val:02x}>"
            self.vocab[byte_str] = len(self.vocab)
            self.byte_to_token[byte_val] = self.vocab[byte_str]
            self.token_to_bytes[self.vocab[byte_str]] = bytes([byte_val])

    def _text_to_bytes(self, text: str) -> bytes:
        """Convert text to UTF-8 bytes."""
        return text.encode('utf-8')

    def _bytes_to_text(self, byte_sequence: bytes) -> str:
        """Convert bytes back to text."""
        try:
            return byte_sequence.decode('utf-8')
        except UnicodeDecodeError:
            # Fallback: replace invalid bytes
            return byte_sequence.decode('utf-8', errors='replace')

    def _get_byte_pairs(self, byte_list: List[int]) -> Dict[Tuple[int, int], int]:
        """Get frequency of adjacent byte pairs."""
        pairs = defaultdict(int)
        for i in range(len(byte_list) - 1):
            pairs[(byte_list[i], byte_list[i + 1])] += 1
        return pairs

    def _merge_bytes(self, byte_list: List[int], pair: Tuple[int, int]) -> List[int]:
        """Merge a byte pair in the byte list."""
        if len(byte_list) < 2:
            return byte_list

        new_list = []
        i = 0
        while i < len(byte_list):
            if (i < len(byte_list) - 1 and
                byte_list[i] == pair[0] and byte_list[i + 1] == pair[1]):
                # Merge the pair
                merged_byte = (pair[0] << 8) | pair[1]  # Create merged token ID
                new_list.append(merged_byte)
                i += 2
            else:
                new_list.append(byte_list[i])
                i += 1

        return new_list

    def _train_on_texts(self, texts: List[str], vocab_size: int, **kwargs) -> None:
        """
        Train byte-level BPE tokenizer.

        Args:
            texts: List of training texts
            vocab_size: Target vocabulary size (default 50257 matches GPT-2)
            min_frequency: Minimum frequency for byte pairs
            verbose: Whether to print progress
        """
        min_frequency = kwargs.get('min_frequency', 2)
        verbose = kwargs.get('verbose', True)
        if verbose:
            print("Training byte-level BPE tokenizer...")

        # Convert texts to byte sequences
        if verbose:
            print("Converting texts to bytes...")

        byte_sequences = []
        for text in tqdm(texts, desc="Converting to bytes", disable=not verbose):
            byte_seq = self._text_to_bytes(text)
            byte_sequences.append(list(byte_seq))

        # Start with byte-level vocabulary
        # We already have 256 bytes + 4 special tokens = 260 tokens
        current_vocab_size = len(self.vocab)

        # Learn merges
        num_merges = vocab_size - current_vocab_size
        self.merges = []

        if verbose:
            print(f"Learning {num_merges} byte-level BPE merges...")

        # Work with byte sequences
        working_sequences = byte_sequences.copy()

        for i in tqdm(range(num_merges), desc="Learning merges", disable=not verbose):
            # Count byte pairs
            pair_counts = defaultdict(int)
            for seq in working_sequences:
                pairs = self._get_byte_pairs(seq)
                for pair, count in pairs.items():
                    pair_counts[pair] += count

            # Filter by minimum frequency
            pair_counts = {pair: count for pair, count in pair_counts.items()
                          if count >= min_frequency}

            if not pair_counts:
                if verbose:
                    print(f"No more valid pairs at iteration {i}")
                break

            # Find most frequent pair
            best_pair = max(pair_counts, key=lambda k: pair_counts[k])

            # Add merged token to vocabulary
            merged_token = f"<merge_{i}>"
            self.vocab[merged_token] = len(self.vocab)

            # Store merge
            self.merges.append(best_pair)

            # Create mapping for merged token
            merged_bytes = bytes([best_pair[0], best_pair[1]])
            self.token_to_bytes[self.vocab[merged_token]] = merged_bytes

            # Apply merge to all sequences
            new_sequences = []
            for seq in working_sequences:
                merged_seq = self._merge_bytes(seq, best_pair)
                new_sequences.append(merged_seq)
            working_sequences = new_sequences

        # Create inverse vocabulary
        self.inverse_vocab = {v: k for k, v in self.vocab.items()}

        if verbose:
            print(f"Final vocabulary size: {len(self.vocab)}")
            print(f"Learned {len(self.merges)} merges")

    def _apply_bpe_to_bytes(self, byte_list: List[int]) -> List[int]:
        """Apply learned BPE merges to a byte sequence."""
        # Start with byte sequence
        tokens = byte_list.copy()

        # Apply each merge in order
        for merge_idx, (byte1, byte2) in enumerate(self.merges):
            new_tokens = []
            i = 0
            while i < len(tokens):
                if (i < len(tokens) - 1 and
                    tokens[i] == byte1 and tokens[i + 1] == byte2):
                    # Use merged token
                    merged_token_id = 256 + 4 + merge_idx  # Base bytes + special + merge index
                    new_tokens.append(merged_token_id)
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            tokens = new_tokens

        return tokens

    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """Encode text to token IDs using byte-level BPE."""
        if not text:
            return []

        # Check cache
        cache_key = text
        if cache_key in self.cache:
            token_ids = self.cache[cache_key].copy()
        else:
            # Convert to bytes
            byte_seq = self._text_to_bytes(text)
            byte_list = list(byte_seq)

            # Apply BPE
            token_ids = self._apply_bpe_to_bytes(byte_list)

            # Cache result
            self.cache[cache_key] = token_ids.copy()

        # Add special tokens
        if add_special_tokens:
            token_ids = [self.bos_token_id] + token_ids + [self.eos_token_id]

        return token_ids

    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        """Decode token IDs back to text."""
        if not token_ids:
            return ""

        # Remove special tokens if requested
        if skip_special_tokens:
            token_ids = [tid for tid in token_ids
                        if tid not in [self.pad_token_id, self.bos_token_id,
                                      self.eos_token_id, self.unk_token_id]]

        # Convert token IDs to bytes
        byte_sequence = bytearray()
        for token_id in token_ids:
            if token_id in self.token_to_bytes:
                byte_sequence.extend(self.token_to_bytes[token_id])
            elif token_id < 256:
                # Direct byte
                byte_sequence.append(token_id)
            else:
                # Unknown token, skip or use replacement
                continue

        # Convert bytes to text
        try:
            return bytes(byte_sequence).decode('utf-8')
        except UnicodeDecodeError:
            return bytes(byte_sequence).decode('utf-8', errors='replace')


    def save_pretrained(self, save_directory: str) -> None:
        """Save tokenizer to directory."""
        import os
        os.makedirs(save_directory, exist_ok=True)

        # Save base tokenizer data
        super().save_pretrained(save_directory)

        # Save byte-level BPE specific data
        merges_file = os.path.join(save_directory, "merges.txt")
        with open(merges_file, 'w', encoding='utf-8') as f:
            f.write("#version: 0.2 - ByteBPE\n")
            for byte1, byte2 in self.merges:
                f.write(f"{byte1} {byte2}\n")

        # Save token mappings
        mappings = {
            "token_to_bytes": {str(k): list(v) for k, v in self.token_to_bytes.items()},
            "byte_to_token": self.byte_to_token
        }

        mappings_file = os.path.join(save_directory, "byte_mappings.json")
        with open(mappings_file, 'w', encoding='utf-8') as f:
            json.dump(mappings, f, indent=2)

    @classmethod
    def from_pretrained(cls, load_directory: str) -> 'ByteBPETokenizer':
        """Load tokenizer from directory."""
        import os

        # Create instance and initialize byte vocab
        tokenizer = cls()

        # Load base tokenizer
        vocab_file = os.path.join(load_directory, "vocab.json")
        with open(vocab_file, 'r', encoding='utf-8') as f:
            vocab = json.load(f)

        tokenizer.vocab = vocab
        tokenizer.inverse_vocab = {v: k for k, v in vocab.items()}

        # Load merges
        merges_file = os.path.join(load_directory, "merges.txt")
        merges = []
        if os.path.exists(merges_file):
            with open(merges_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                for line in lines[1:]:  # Skip version line
                    line = line.strip()
                    if line:
                        parts = line.split()
                        if len(parts) == 2:
                            merges.append((int(parts[0]), int(parts[1])))

        tokenizer.merges = merges

        # Load mappings
        mappings_file = os.path.join(load_directory, "byte_mappings.json")
        if os.path.exists(mappings_file):
            with open(mappings_file, 'r', encoding='utf-8') as f:
                mappings = json.load(f)

            tokenizer.token_to_bytes = {
                int(k): bytes(v) for k, v in mappings["token_to_bytes"].items()
            }
            tokenizer.byte_to_token = {
                int(k): v for k, v in mappings["byte_to_token"].items()
            }

        tokenizer.cache = {}

        return tokenizer
