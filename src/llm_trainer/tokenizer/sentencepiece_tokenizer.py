"""SentencePiece/Unigram tokenizer implementation."""

import re
import json
import math
import unicodedata
from collections import Counter
from typing import List, Dict, Tuple, Set, Optional, Union, Any, cast
from tqdm import tqdm

from .base_tokenizer import BaseTokenizer


class SentencePieceTokenizer(BaseTokenizer):
    """
    SentencePiece Unigram tokenizer implementation.
    
    This tokenizer uses the Unigram language model approach, similar to SentencePiece.
    It's particularly good for multilingual text and handles unknown words gracefully.
    
    Key Features:
    - Unigram language model for subword selection
    - Probabilistic tokenization with multiple segmentations
    - Full Unicode and emoji support
    - Efficient encoding/decoding
    """

    def __init__(self):
        super().__init__()
        self.token_scores: Dict[str, float] = {}
        self.alpha: float = 0.0  # Smoothing parameter for unigram model
        self.max_subword_length: int = 100
        self.cache: Dict[str, List[str]] = {}

    def _get_word_tokens(self, text: str) -> List[str]:
        """Pre-tokenize text into words."""
        # Normalize unicode
        text = unicodedata.normalize('NFC', text)

        # Simple whitespace-based tokenization
        # Can be enhanced with more sophisticated patterns
        words = re.findall(r'\S+', text)
        return words

    def _initialize_vocab(self, texts: List[str], min_frequency: int = 2) -> Tuple[Dict[str, int], Set[str]]:
        """Initialize vocabulary with characters and common substrings."""
        # Collect word frequencies
        word_freqs = Counter()
        for text in texts:
            words = self._get_word_tokens(text)
            word_freqs.update(words)

        # Filter by minimum frequency
        word_freqs = {word: freq for word, freq in word_freqs.items()
                     if freq >= min_frequency}

        # Extract all unique characters
        chars = set()
        for word in word_freqs:
            chars.update(word)

        return word_freqs, chars

    def _get_subword_candidates(self, word: str) -> Set[str]:
        """Get all possible subword candidates for a word."""
        candidates = set()

        # Add all characters
        candidates.update(word)

        # Add all substrings up to max length
        for i in range(len(word)):
            for j in range(i + 1, min(i + self.max_subword_length + 1, len(word) + 1)):
                candidates.add(word[i:j])

        return candidates

    def _compute_unigram_scores(self, texts: List[str], vocab: Set[str]) -> Dict[str, float]:
        """Compute unigram scores for vocabulary items."""
        # Count occurrences of each subword
        subword_counts = Counter()
        total_count = 0

        for text in texts:
            words = self._get_word_tokens(text)
            for word in words:
                # Find all subwords in this word
                for i in range(len(word)):
                    for j in range(i + 1, min(i + self.max_subword_length + 1, len(word) + 1)):
                        subword = word[i:j]
                        if subword in vocab:
                            subword_counts[subword] += 1
                            total_count += 1

        # Compute log probabilities (scores)
        scores = {}
        for subword, count in subword_counts.items():
            if total_count > 0:
                prob = count / total_count
                scores[subword] = math.log(prob) if prob > 0 else float('-inf')
            else:
                scores[subword] = float('-inf')

        return scores

    def _viterbi_segment(self, word: str) -> Tuple[List[str], float]:
        """
        Use Viterbi algorithm to find best segmentation.
        Returns (best_segmentation, best_score).
        """
        n = len(word)
        if n == 0:
            return [], 0.0

        # DP: best_score[i] = best score for word[0:i]
        best_score = [float('-inf')] * (n + 1)
        best_score[0] = 0.0
        best_path = [None] * (n + 1)

        for i in range(1, n + 1):
            for j in range(max(0, i - self.max_subword_length), i):
                subword = word[j:i]
                if subword in self.token_scores:
                    score = best_score[j] + self.token_scores[subword]
                    if score > best_score[i]:
                        best_score[i] = score
                        best_path[i] = j

        # Reconstruct path
        if best_score[n] == float('-inf'):
            # No valid segmentation found, return as single token
            return [word], float('-inf')

        segmentation = []
        i = n
        while i > 0:
            j = best_path[i]
            segmentation.insert(0, word[j:i])
            i = j

        return segmentation, best_score[n]

    def _train_on_texts(self, texts: List[str], vocab_size: int, **kwargs) -> None:
        """Train SentencePiece Unigram tokenizer."""
        min_frequency = kwargs.get('min_frequency', 2)
        verbose = kwargs.get('verbose', True)
        if verbose:
            print("Training SentencePiece Unigram tokenizer...")

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

        # Phase 1: Initialize vocabulary
        if verbose:
            print("Initializing vocabulary...")

        word_freqs, chars = self._initialize_vocab(texts, min_frequency)

        # Add all characters to vocabulary
        for char in sorted(chars):
            if char not in self.vocab:
                self.vocab[char] = len(self.vocab)

        if verbose:
            print(f"Initial vocabulary size: {len(self.vocab)}")
            print(f"Found {len(word_freqs)} unique words")

        # Phase 2: Iterative vocabulary building
        # Start with character vocabulary
        current_vocab = set(chars)
        current_vocab.update(self.special_tokens.keys())

        # Collect all subword candidates from frequent words
        if verbose:
            print("Collecting subword candidates...")

        all_candidates = set()
        for word, freq in tqdm(word_freqs.items(), desc="Collecting candidates", disable=not verbose):
            if freq >= min_frequency:
                candidates = self._get_subword_candidates(word)
                all_candidates.update(candidates)

        # Phase 3: Build vocabulary iteratively
        num_iterations = vocab_size - len(self.vocab)
        if verbose:
            print(f"Building vocabulary with {num_iterations} iterations...")

        for iteration in tqdm(range(num_iterations), desc="Building vocab", disable=not verbose):
            # Compute scores for current vocabulary
            self.token_scores = self._compute_unigram_scores(texts, current_vocab)

            # Find best candidate to add
            best_candidate = None
            best_improvement = float('-inf')

            for candidate in all_candidates:
                if candidate in current_vocab:
                    continue

                # Test adding this candidate
                test_vocab = current_vocab | {candidate}
                test_scores = self._compute_unigram_scores(texts, test_vocab)

                # Compute improvement
                improvement = 0.0
                for word, freq in word_freqs.items():
                    if freq < min_frequency:
                        continue

                    # Compute segmentation scores
                    old_seg, old_score = self._viterbi_segment_with_vocab(word, current_vocab, self.token_scores)
                    new_seg, new_score = self._viterbi_segment_with_vocab(word, test_vocab, test_scores)

                    improvement += freq * (new_score - old_score)

                if improvement > best_improvement:
                    best_improvement = improvement
                    best_candidate = candidate

            if best_candidate is None or best_improvement <= 0:
                if verbose:
                    print(f"No improvement found at iteration {iteration}")
                break

            # Add best candidate
            current_vocab.add(best_candidate)
            self.vocab[best_candidate] = len(self.vocab)

        # Final score computation
        self.token_scores = self._compute_unigram_scores(texts, current_vocab)

        # Create inverse vocabulary
        self.inverse_vocab = {v: k for k, v in self.vocab.items()}

        if verbose:
            print(f"Final vocabulary size: {len(self.vocab)}")

    def _viterbi_segment_with_vocab(self, word: str, vocab: Set[str],
                                   scores: Dict[str, float]) -> Tuple[List[str], float]:
        """Viterbi segmentation with given vocabulary and scores."""
        n = len(word)
        if n == 0:
            return [], 0.0

        best_score = [float('-inf')] * (n + 1)
        best_score[0] = 0.0
        best_path = [None] * (n + 1)

        for i in range(1, n + 1):
            for j in range(max(0, i - self.max_subword_length), i):
                subword = word[j:i]
                if subword in vocab and subword in scores:
                    score = best_score[j] + scores[subword]
                    if score > best_score[i]:
                        best_score[i] = score
                        best_path[i] = j

        if best_score[n] == float('-inf'):
            return [word], float('-inf')

        segmentation = []
        i = n
        while i > 0:
            j = best_path[i]
            segmentation.insert(0, word[j:i])
            i = j

        return segmentation, best_score[n]

    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """Encode text to token IDs using Unigram segmentation."""
        if not text:
            return []

        # Pre-tokenize into words
        words = self._get_word_tokens(text)

        # Segment each word
        token_ids = []
        for word in words:
            if word in self.cache:
                tokens = self.cache[word]
            else:
                tokens, _ = self._viterbi_segment(word)
                self.cache[word] = tokens

            # Convert tokens to IDs
            for token in tokens:
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

            if skip_special_tokens and token in self.special_tokens:
                continue

            tokens.append(token)

        # Join tokens
        text = ''.join(tokens)

        # Clean up whitespace
        text = re.sub(r'\s+', ' ', text).strip()

        return text


    def save_pretrained(self, save_directory: str) -> None:
        """Save tokenizer to directory."""
        import os
        os.makedirs(save_directory, exist_ok=True)

        # Save base tokenizer data
        super().save_pretrained(save_directory)

        # Save SentencePiece-specific data
        scores_file = os.path.join(save_directory, "token_scores.json")
        with open(scores_file, 'w', encoding='utf-8') as f:
            json.dump(self.token_scores, f, ensure_ascii=False, indent=2)

        config_file = os.path.join(save_directory, "tokenizer_config.json")
        with open(config_file, 'r', encoding='utf-8') as f:
            config = json.load(f)

        config["tokenizer_type"] = "sentencepiece"
        config["alpha"] = self.alpha
        config["max_subword_length"] = self.max_subword_length

        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2)

    @classmethod
    def from_pretrained(cls, load_directory: str) -> 'SentencePieceTokenizer':
        """Load tokenizer from directory."""
        import os

        # Load base tokenizer
        tokenizer = cast('SentencePieceTokenizer', super().from_pretrained(load_directory))

        # Load SentencePiece-specific data
        config_file = os.path.join(load_directory, "tokenizer_config.json")
        with open(config_file, 'r', encoding='utf-8') as f:
            config = json.load(f)

        tokenizer.alpha = config.get("alpha", 0.0)
        tokenizer.max_subword_length = config.get("max_subword_length", 100)

        scores_file = os.path.join(load_directory, "token_scores.json")
        if os.path.exists(scores_file):
            with open(scores_file, 'r', encoding='utf-8') as f:
                tokenizer.token_scores = json.load(f)

        tokenizer.cache = {}

        return tokenizer
