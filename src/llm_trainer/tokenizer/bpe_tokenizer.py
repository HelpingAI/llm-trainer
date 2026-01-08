"""Byte Pair Encoding (BPE) tokenizer implementation from scratch."""

import re
from collections import defaultdict, Counter
from typing import List, Dict, Tuple, Optional, cast
from tqdm import tqdm
import unicodedata

from .base_tokenizer import BaseTokenizer


class BPETokenizer(BaseTokenizer):
    """Byte Pair Encoding tokenizer implemented from scratch."""

    def __init__(self):
        super().__init__()
        self.merges: List[Tuple[str, str]] = []
        self.cache: Dict[str, List[str]] = {}

        # Regex pattern for pre-tokenization (similar to GPT-2)
        # Enhanced to handle emojis, symbols, and international characters
        # Patterns:
        # - Contractions: 's, 't, 'll, 've, 're, 'd, 'm
        # - Letters: Basic Latin, Latin Extended, Cyrillic, CJK, Arabic, etc.
        # - Numbers: 0-9 and related number symbols
        # - Emojis: Full emoji range including skin tones, flags, etc.
        # - Symbols: Mathematical, currency, technical symbols
        # - Punctuation and special characters
        pattern = (
            r"'(?:[sdmt]|ll|ve|re|d|m)|"  # Contractions
            r" ?[\u0041-\u005A\u0061-\u007A\u00C0-\u00FF\u0100-\u017F\u0180-\u024F"  # Latin letters
            r"\u1E00-\u1EFF\u0400-\u04FF\u0370-\u03FF\u0590-\u05FF\u0600-\u06FF"  # Extended Latin, Cyrillic, Greek, Hebrew, Arabic
            r"\u4E00-\u9FFF\u3040-\u309F\u30A0-\u30FF\u3400-\u4DBF]+|"  # CJK, Hiragana, Katakana
            r" ?[0-9\u0660-\u0669\u06F0-\u06F9\u07C0-\u07C9\u0966-\u096F]+|"  # Numbers (Western, Arabic, etc.)
            r" ?[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF"  # Emoticons, symbols, transport
            r"\U0001F1E0-\U0001F1FF\U00002600-\U000026FF\U00002700-\U000027BF"  # Flags, misc symbols
            r"\U0001F900-\U0001F9FF\U0001FA00-\U0001FA6F\U0001FA70-\U0001FAFF"  # Supplemental symbols
            r"\U00010000-\U0010FFFF]+|"  # Other high Unicode planes
            r" ?[^\s\w\u0041-\u005A\u0061-\u007A\u00C0-\u00FF\u0100-\u017F"  # Non-word chars excluding letters
            r"\u0180-\u024F\u1E00-\u1EFF\u0400-\u04FF\u0370-\u03FF\u0590-\u05FF"
            r"\u0600-\u06FF\u4E00-\u9FFF\u3040-\u309F\u30A0-\u30FF\u3400-\u4DBF"
            r"0-9\u0660-\u0669\u06F0-\u06F9\u07C0-\u07C9\u0966-\u096F]+|"
            r"\s+(?!\S)|\s+"  # Whitespace
        )
        self.pat = re.compile(pattern, re.IGNORECASE)

    def _get_stats(self, vocab: Dict[str, int]) -> Dict[Tuple[str, str], int]:
        """Get frequency of adjacent symbol pairs."""
        pairs = defaultdict(int)
        for word, freq in vocab.items():
            symbols = word.split()
            for i in range(len(symbols) - 1):
                pairs[(symbols[i], symbols[i + 1])] += freq
        return pairs

    def _merge_vocab(self, pair: Tuple[str, str], vocab: Dict[str, int]) -> Dict[str, int]:
        """Merge the most frequent pair in vocabulary."""
        bigram = re.escape(' '.join(pair))
        p = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')

        new_vocab = {}
        for word in vocab:
            new_word = p.sub(''.join(pair), word)
            new_vocab[new_word] = vocab[word]
        return new_vocab

    def _get_word_tokens(self, text: str) -> List[str]:
        """Pre-tokenize text into words using regex pattern."""
        # Normalize unicode
        text = unicodedata.normalize('NFC', text)

        # Find all matches
        tokens = []
        for match in self.pat.finditer(text):
            token = match.group()
            # Convert to bytes and back to handle any character
            token_bytes = token.encode('utf-8')
            token = ''.join(chr(b) for b in token_bytes)
            tokens.append(token)

        return tokens

    def _train_on_texts(self, texts: List[str], vocab_size: int, **kwargs) -> None:
        """Train BPE tokenizer on corpus."""
        min_frequency = kwargs.get('min_frequency', 2)
        verbose = kwargs.get('verbose', True)
        if verbose:
            print("Training BPE tokenizer...")

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

        # Collect word frequencies
        if verbose:
            print("Collecting word frequencies...")

        word_freqs = Counter()
        for text in tqdm(texts, desc="Processing texts", disable=not verbose):
            words = self._get_word_tokens(text)
            for word in words:
                word_freqs[word] += 1

        # Filter by minimum frequency
        word_freqs = {word: freq for word, freq in word_freqs.items()
                     if freq >= min_frequency}

        if verbose:
            print(f"Found {len(word_freqs)} unique words")

        # Initialize vocabulary with character-level tokens
        vocab = {}
        for word, freq in word_freqs.items():
            # Split word into characters and add end-of-word marker
            word_tokens = list(word) + ['</w>']
            vocab[' '.join(word_tokens)] = freq

        # Get all unique characters
        all_chars = set()
        for word in vocab:
            all_chars.update(word.split())

        # Add character tokens to vocabulary
        for char in sorted(all_chars):
            if char not in self.vocab:
                self.vocab[char] = len(self.vocab)

        if verbose:
            print(f"Initial vocabulary size: {len(self.vocab)}")

        # Learn BPE merges
        num_merges = vocab_size - len(self.vocab)
        self.merges = []

        if verbose:
            print(f"Learning {num_merges} BPE merges...")

        for i in tqdm(range(num_merges), desc="Learning merges", disable=not verbose):
            pairs = self._get_stats(vocab)
            if not pairs:
                break

            # Find most frequent pair
            best_pair = max(pairs, key=lambda k: pairs[k])

            # Merge the pair
            vocab = self._merge_vocab(best_pair, vocab)
            self.merges.append(best_pair)

            # Add merged token to vocabulary
            merged_token = ''.join(best_pair)
            if merged_token not in self.vocab:
                self.vocab[merged_token] = len(self.vocab)

        # Create inverse vocabulary
        self.inverse_vocab = {v: k for k, v in self.vocab.items()}

        if verbose:
            print(f"Final vocabulary size: {len(self.vocab)}")
            print(f"Learned {len(self.merges)} merges")


    def _apply_bpe(self, word: str) -> List[str]:
        """Apply BPE merges to a word."""
        if word in self.cache:
            return self.cache[word]

        # Split word into characters and add end-of-word marker
        word_tokens = list(word) + ['</w>']

        if len(word_tokens) == 1:
            self.cache[word] = word_tokens
            return word_tokens

        # Apply merges
        for merge in self.merges:
            i = 0
            while i < len(word_tokens) - 1:
                if (word_tokens[i], word_tokens[i + 1]) == merge:
                    # Merge the pair
                    merged = word_tokens[i] + word_tokens[i + 1]
                    word_tokens = word_tokens[:i] + [merged] + word_tokens[i + 2:]
                else:
                    i += 1

        self.cache[word] = word_tokens
        return word_tokens

    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """Encode text to token IDs."""
        if not text:
            return []

        # Pre-tokenize
        words = self._get_word_tokens(text)

        # Apply BPE to each word
        bpe_tokens = []
        for word in words:
            word_tokens = self._apply_bpe(word)
            bpe_tokens.extend(word_tokens)

        # Convert to IDs
        token_ids = []
        for token in bpe_tokens:
            token_id = self.vocab.get(token, self.unk_token_id)
            token_ids.append(token_id)

        # Add special tokens
        if add_special_tokens:
            token_ids = [self.bos_token_id] + token_ids + [self.eos_token_id]

        return token_ids

    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        """Decode token IDs to text."""
        if not token_ids:
            return ""

        # Convert IDs to tokens
        tokens = []
        for token_id in token_ids:
            token = self.inverse_vocab.get(token_id, self.unk_token)

            # Skip special tokens if requested
            if skip_special_tokens and token in self.special_tokens:
                continue

            tokens.append(token)

        # Join tokens and handle end-of-word markers
        text = ''.join(tokens)
        text = text.replace('</w>', ' ')

        # Clean up extra spaces
        text = re.sub(r'\s+', ' ', text).strip()

        return text

    def save_pretrained(self, save_directory: str) -> None:
        """Save tokenizer to directory."""
        import os
        os.makedirs(save_directory, exist_ok=True)

        # Save base tokenizer data
        super().save_pretrained(save_directory)

        # Save BPE-specific data
        merges_file = os.path.join(save_directory, "merges.txt")
        with open(merges_file, 'w', encoding='utf-8') as f:
            f.write("#version: 0.2\n")
            for merge in self.merges:
                f.write(f"{merge[0]} {merge[1]}\n")

    @classmethod
    def from_pretrained(cls, load_directory: str) -> 'BPETokenizer':
        """Load tokenizer from directory."""
        import os

        # Load base tokenizer
        tokenizer = cast('BPETokenizer', super().from_pretrained(load_directory))

        # Load BPE merges
        merges_file = os.path.join(load_directory, "merges.txt")
        merges = []

        with open(merges_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for line in lines[1:]:  # Skip version line
                line = line.strip()
                if line:
                    parts = line.split()
                    if len(parts) == 2:
                        merges.append((parts[0], parts[1]))

        tokenizer.merges = merges
        tokenizer.cache = {}

        return tokenizer

    def get_vocab_stats(self) -> Dict[str, int]:
        """Get vocabulary statistics."""
        stats = {
            "total_tokens": len(self.vocab),
            "special_tokens": len(self.special_tokens),
            "regular_tokens": len(self.vocab) - len(self.special_tokens),
            "merges": len(self.merges)
        }
        return stats
