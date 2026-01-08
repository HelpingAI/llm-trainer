"""WordPiece tokenizer implementation following BERT-style approach."""

import re
import json
import math
import unicodedata
from collections import Counter
from typing import List, Dict, Tuple, Set, Optional, Any
from tqdm import tqdm
from functools import lru_cache

from .base_tokenizer import BaseTokenizer


class WordPieceTokenizer(BaseTokenizer):
    """
    WordPiece tokenizer implementation following BERT-style approach.
    
    Key Features:
    - Likelihood-based subword merging using Score(A,B) = log(P(AB)) - log(P(A)) - log(P(B))
    - Longest-match-first encoding with continuation prefixes (##)
    - BERT-style special tokens ([PAD], [UNK], [CLS], [SEP], [MASK])
    - Full compliance with BaseTokenizer interface
    - Performance optimizations including LRU caching
    - Unicode and emoji support matching BPE capabilities
    """

    def __init__(self):
        super().__init__()

        # Override default special tokens to use BERT-style format
        self.pad_token = "[PAD]"
        self.unk_token = "[UNK]"
        self.cls_token = "[CLS]"
        self.sep_token = "[SEP]"
        self.mask_token = "[MASK]"

        # Special token IDs
        self.pad_token_id = 0
        self.unk_token_id = 1
        self.cls_token_id = 2
        self.sep_token_id = 3
        self.mask_token_id = 4

        # WordPiece-specific attributes
        self.continuation_prefix = "##"
        self.max_subword_length = 100
        self.cache: Dict[str, List[str]] = {}
        self.token_scores: Dict[str, float] = {}

        # Training data
        self.merge_history: List[Tuple[str, str, float]] = []
        self.training_stats: Dict[str, Any] = {}

        # Regex pattern for pre-tokenization (enhanced from BPE for WordPiece)
        # Similar to BERT's BasicTokenizer but with enhanced Unicode support
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

    def _initialize_training(self, texts: List[str], min_frequency: int) -> Tuple[Dict[str, int], Set[str]]:
        """Initialize character vocabulary and word frequencies."""

        # Collect word frequencies
        word_frequencies = Counter()
        for text in texts:
            words = self._get_word_tokens(text)
            word_frequencies.update(words)

        # Filter by minimum frequency
        word_frequencies = {word: freq for word, freq in word_frequencies.items()
                           if freq >= min_frequency}

        # Extract all unique characters
        characters = set()
        for word in word_frequencies:
            characters.update(word)

        # Initialize vocabulary with special tokens + characters
        vocab = set()
        vocab.update([self.pad_token, self.unk_token, self.cls_token,
                     self.sep_token, self.mask_token])
        vocab.update(characters)

        return word_frequencies, vocab

    def _calculate_likelihood_score(self, token_a: str, token_b: str,
                                   token_counts: Dict[str, int]) -> float:
        """
        Calculate likelihood improvement for merging token_a and token_b.
        
        Score(A, B) = log(P(AB)) - log(P(A)) - log(P(B))
        """
        # Count occurrences of the merged token
        merged_token = token_a + token_b
        count_merged = 0

        # Count adjacent pairs in the current tokenization
        for word_tokens, freq in token_counts.items():
            tokens = word_tokens.split()
            for i in range(len(tokens) - 1):
                if tokens[i] == token_a and tokens[i + 1] == token_b:
                    count_merged += freq

        # Get individual token counts
        count_a = sum(freq for word_tokens, freq in token_counts.items()
                     if token_a in word_tokens.split())
        count_b = sum(freq for word_tokens, freq in token_counts.items()
                     if token_b in word_tokens.split())

        # Calculate total tokens
        total_tokens = sum(len(word_tokens.split()) * freq
                          for word_tokens, freq in token_counts.items())

        # Calculate probabilities
        if count_a == 0 or count_b == 0 or count_merged == 0 or total_tokens == 0:
            return float('-inf')

        prob_a = count_a / total_tokens
        prob_b = count_b / total_tokens
        prob_merged = count_merged / total_tokens

        # Calculate likelihood score
        try:
            score = math.log(prob_merged) - math.log(prob_a) - math.log(prob_b)
            return score
        except (ValueError, ZeroDivisionError):
            return float('-inf')

    def _find_merge_candidates(self, token_counts: Dict[str, int]) -> Set[Tuple[str, str]]:
        """Find all possible merge candidates from current tokenization."""
        candidates = set()

        for word_tokens in token_counts:
            tokens = word_tokens.split()
            for i in range(len(tokens) - 1):
                candidates.add((tokens[i], tokens[i + 1]))

        return candidates

    def _apply_merge(self, merge_pair: Tuple[str, str],
                    token_counts: Dict[str, int]) -> Dict[str, int]:
        """Apply a merge to the current tokenization."""
        token_a, token_b = merge_pair
        merged_token = token_a + token_b
        new_token_counts = {}

        for word_tokens, freq in token_counts.items():
            tokens = word_tokens.split()
            new_tokens = []
            i = 0

            while i < len(tokens):
                if (i < len(tokens) - 1 and
                    tokens[i] == token_a and tokens[i + 1] == token_b):
                    # Apply the merge
                    new_tokens.append(merged_token)
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1

            new_word_tokens = ' '.join(new_tokens)
            new_token_counts[new_word_tokens] = freq

        return new_token_counts

    def _train_on_texts(self, texts: List[str], vocab_size: int, **kwargs) -> None:
        """Train WordPiece tokenizer using likelihood-based merging."""
        min_frequency = kwargs.get('min_frequency', 2)
        verbose = kwargs.get('verbose', True)
        if verbose:
            print("Training WordPiece tokenizer...")

        # Initialize special tokens
        self.vocab = {
            self.pad_token: self.pad_token_id,
            self.unk_token: self.unk_token_id,
            self.cls_token: self.cls_token_id,
            self.sep_token: self.sep_token_id,
            self.mask_token: self.mask_token_id
        }

        self.special_tokens = {
            self.pad_token: self.pad_token_id,
            self.unk_token: self.unk_token_id,
            self.cls_token: self.cls_token_id,
            self.sep_token: self.sep_token_id,
            self.mask_token: self.mask_token_id
        }

        # Phase 1: Initialize character vocabulary and word frequencies
        if verbose:
            print("Initializing character vocabulary...")

        word_frequencies, vocab_set = self._initialize_training(texts, min_frequency)

        if verbose:
            print(f"Found {len(word_frequencies)} unique words")
            print(f"Initial character vocabulary size: {len(vocab_set)}")

        # Add character tokens to vocabulary
        for char in sorted(vocab_set):
            if char not in self.vocab:
                self.vocab[char] = len(self.vocab)

        # Initialize word tokenizations (character-level)
        token_counts = {}
        for word, freq in word_frequencies.items():
            # Split word into characters
            word_tokens = ' '.join(list(word))
            token_counts[word_tokens] = freq

        # Phase 2: Learn WordPiece merges using likelihood scoring
        num_merges = vocab_size - len(self.vocab)
        self.merge_history = []

        if verbose:
            print(f"Learning {num_merges} WordPiece merges...")

        for i in tqdm(range(num_merges), desc="Learning merges", disable=not verbose):
            # Find all possible merges
            merge_candidates = self._find_merge_candidates(token_counts)

            if not merge_candidates:
                if verbose:
                    print(f"No more merge candidates found at iteration {i}")
                break

            # Score all merges using likelihood
            merge_scores = {}
            for candidate in merge_candidates:
                score = self._calculate_likelihood_score(candidate[0], candidate[1], token_counts)
                if score > float('-inf'):
                    merge_scores[candidate] = score

            if not merge_scores:
                if verbose:
                    print(f"No valid merges found at iteration {i}")
                break

            # Select best merge (highest likelihood improvement)
            best_merge = max(merge_scores, key=lambda k: merge_scores[k])
            best_score = merge_scores[best_merge]

            if best_score <= 0:
                if verbose:
                    print(f"No positive likelihood improvement at iteration {i}")
                break

            # Apply the merge
            merged_token = best_merge[0] + best_merge[1]
            self.vocab[merged_token] = len(self.vocab)
            self.merge_history.append((best_merge[0], best_merge[1], best_score))
            self.token_scores[merged_token] = best_score

            # Update word tokenizations
            token_counts = self._apply_merge(best_merge, token_counts)

        # Build continuation vocabulary (add ## prefixes)
        self._build_continuation_vocab()

        # Create inverse vocabulary
        self.inverse_vocab = {v: k for k, v in self.vocab.items()}

        # Store training statistics
        self.training_stats = {
            "final_vocab_size": len(self.vocab),
            "num_merges": len(self.merge_history),
            "num_words": len(word_frequencies),
            "min_frequency": min_frequency
        }

        if verbose:
            print(f"Final vocabulary size: {len(self.vocab)}")
            print(f"Learned {len(self.merge_history)} merges")

    def _build_continuation_vocab(self) -> None:
        """Add continuation tokens (## prefixed) to vocabulary."""
        # Get all non-special tokens
        regular_tokens = [token for token in self.vocab.keys()
                         if token not in self.special_tokens]

        # Add ## prefixed versions
        for token in regular_tokens:
            continuation_token = self.continuation_prefix + token
            if continuation_token not in self.vocab:
                self.vocab[continuation_token] = len(self.vocab)

    @lru_cache(maxsize=10000)
    def _wordpiece_encode_word(self, word: str) -> Tuple[str, ...]:
        """
        Encode a single word using WordPiece longest-match-first algorithm.
        Returns tuple for hashability in LRU cache.
        """
        if len(word) > self.max_subword_length:
            return (self.unk_token,)

        tokens = []
        start = 0

        while start < len(word):
            end = len(word)
            longest_match = None

            # Find longest matching subword
            while start < end:
                substr = word[start:end]

                # Add continuation prefix if not at word start
                if start > 0:
                    substr = self.continuation_prefix + substr

                if substr in self.vocab:
                    longest_match = substr
                    break
                end -= 1

            if longest_match is None:
                # Unable to tokenize - return UNK
                return (self.unk_token,)

            tokens.append(longest_match)
            start = end

        return tuple(tokens)

    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """Encode text to token IDs using WordPiece algorithm."""
        if not text:
            return []

        # Pre-tokenize into words
        words = self._get_word_tokens(text)

        # Apply WordPiece encoding to each word
        wordpiece_tokens = []
        for word in words:
            word_tokens = list(self._wordpiece_encode_word(word))
            wordpiece_tokens.extend(word_tokens)

        # Convert to IDs
        token_ids = []
        for token in wordpiece_tokens:
            token_id = self.vocab.get(token, self.unk_token_id)
            token_ids.append(token_id)

        # Add special tokens (BERT-style: [CLS] text [SEP])
        if add_special_tokens:
            token_ids = [self.cls_token_id] + token_ids + [self.sep_token_id]

        return token_ids

    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        """Decode token IDs to text, handling continuation tokens."""
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

        # Join tokens and handle continuation prefixes
        text_parts = []
        current_word = ""

        for token in tokens:
            if token.startswith(self.continuation_prefix):
                # Continuation token - append to current word
                current_word += token[len(self.continuation_prefix):]
            else:
                # Start of new word
                if current_word:
                    text_parts.append(current_word)
                current_word = token

        # Add the last word
        if current_word:
            text_parts.append(current_word)

        # Join with spaces and clean up
        text = ' '.join(text_parts)
        text = re.sub(r'\s+', ' ', text).strip()

        return text


    def save_pretrained(self, save_directory: str) -> None:
        """Save tokenizer to directory."""
        import os
        os.makedirs(save_directory, exist_ok=True)

        # Save base tokenizer data (override to include WordPiece-specific tokens)
        vocab_file = os.path.join(save_directory, "vocab.json")
        with open(vocab_file, 'w', encoding='utf-8') as f:
            json.dump(self.vocab, f, ensure_ascii=False, indent=2)

        # Save tokenizer config
        config = {
            "tokenizer_class": self.__class__.__name__,
            "special_tokens": self.special_tokens,
            "pad_token": self.pad_token,
            "unk_token": self.unk_token,
            "cls_token": self.cls_token,
            "sep_token": self.sep_token,
            "mask_token": self.mask_token,
            "vocab_size": self.vocab_size,
            "continuation_prefix": self.continuation_prefix,
            "max_subword_length": self.max_subword_length
        }

        config_file = os.path.join(save_directory, "tokenizer_config.json")
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2)

        # Save WordPiece-specific data
        merges_file = os.path.join(save_directory, "merges.txt")
        with open(merges_file, 'w', encoding='utf-8') as f:
            f.write("#version: 0.2 - WordPiece\n")
            for token_a, token_b, score in self.merge_history:
                f.write(f"{token_a} {token_b} {score}\n")

        # Save token scores
        scores_file = os.path.join(save_directory, "token_scores.json")
        with open(scores_file, 'w', encoding='utf-8') as f:
            json.dump(self.token_scores, f, ensure_ascii=False, indent=2)

        # Save training stats
        stats_file = os.path.join(save_directory, "training_stats.json")
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(self.training_stats, f, indent=2)

    @classmethod
    def from_pretrained(cls, load_directory: str) -> 'WordPieceTokenizer':
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
        for token_name in ["pad", "unk", "cls", "sep", "mask"]:
            if f"{token_name}_token" in config:
                token = config[f"{token_name}_token"]
                setattr(tokenizer, f"{token_name}_token", token)
                setattr(tokenizer, f"{token_name}_token_id", vocab[token])

        # Set WordPiece-specific attributes
        tokenizer.continuation_prefix = config.get("continuation_prefix", "##")
        tokenizer.max_subword_length = config.get("max_subword_length", 100)

        # Load merge history
        merges_file = os.path.join(load_directory, "merges.txt")
        merge_history = []
        if os.path.exists(merges_file):
            with open(merges_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                for line in lines[1:]:  # Skip version line
                    line = line.strip()
                    if line:
                        parts = line.split()
                        if len(parts) == 3:
                            merge_history.append((parts[0], parts[1], float(parts[2])))

        tokenizer.merge_history = merge_history

        # Load token scores
        scores_file = os.path.join(load_directory, "token_scores.json")
        if os.path.exists(scores_file):
            with open(scores_file, 'r', encoding='utf-8') as f:
                tokenizer.token_scores = json.load(f)

        # Load training stats
        stats_file = os.path.join(load_directory, "training_stats.json")
        if os.path.exists(stats_file):
            with open(stats_file, 'r', encoding='utf-8') as f:
                tokenizer.training_stats = json.load(f)

        # Clear cache
        tokenizer.cache = {}

        return tokenizer

    # Analysis and utility methods
    def get_tokenization_stats(self) -> Dict[str, Any]:
        """Get comprehensive tokenization statistics."""
        stats = {
            "vocab_size": len(self.vocab),
            "special_tokens": len(self.special_tokens),
            "regular_tokens": len(self.vocab) - len(self.special_tokens),
            "continuation_tokens": len([t for t in self.vocab if t.startswith(self.continuation_prefix)]),
            "merges": len(self.merge_history),
            "continuation_prefix": self.continuation_prefix,
            "max_subword_length": self.max_subword_length,
            "cache_size": len(self.cache)
        }

        if self.training_stats:
            stats.update(self.training_stats)

        return stats

    def analyze_word_segmentation(self, words: List[str]) -> Dict[str, List[str]]:
        """Analyze how words are segmented by the tokenizer."""
        segmentations = {}
        for word in words:
            tokens = list(self._wordpiece_encode_word(word))
            segmentations[word] = tokens
        return segmentations

    def get_merge_history(self) -> List[Tuple[str, str, float]]:
        """Get the history of merges with their likelihood scores."""
        return self.merge_history.copy()

    def calculate_compression_ratio(self, texts: List[str]) -> float:
        """Calculate compression ratio (characters per token)."""
        total_chars = sum(len(text) for text in texts)
        total_tokens = sum(len(self.encode(text, add_special_tokens=False)) for text in texts)

        if total_tokens == 0:
            return 0.0

        return total_chars / total_tokens

    def set_max_subword_length(self, length: int) -> None:
        """Set maximum subword length."""
        self.max_subword_length = length
        # Clear cache since encoding behavior changes
        self.cache.clear()
        self._wordpiece_encode_word.cache_clear()

    def set_continuation_prefix(self, prefix: str) -> None:
        """Set continuation prefix (default: ##)."""
        old_prefix = self.continuation_prefix
        self.continuation_prefix = prefix

        # Update vocabulary if already trained
        if self.vocab:
            new_vocab = {}
            for token, token_id in self.vocab.items():
                if token.startswith(old_prefix):
                    new_token = prefix + token[len(old_prefix):]
                    new_vocab[new_token] = token_id
                else:
                    new_vocab[token] = token_id

            self.vocab = new_vocab
            self.inverse_vocab = {v: k for k, v in self.vocab.items()}

        # Clear cache
        self.cache.clear()
        self._wordpiece_encode_word.cache_clear()

    def enable_caching(self, cache_size: int = 10000) -> None:
        """Enable/resize LRU cache for encoding."""
        # Update the cache size for the LRU cache decorator
        self._wordpiece_encode_word = lru_cache(maxsize=cache_size)(self._wordpiece_encode_word.__wrapped__)
