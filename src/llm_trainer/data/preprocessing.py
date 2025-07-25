"""Text preprocessing utilities."""

import re
import unicodedata
from typing import List, Optional, Dict, Any
from datasets import Dataset


class TextPreprocessor:
    """Text preprocessing for language modeling."""
    
    def __init__(self, 
                 min_length: int = 10,
                 max_length: int = 1024,
                 filter_empty: bool = True,
                 normalize_unicode: bool = True,
                 remove_duplicates: bool = False):
        self.min_length = min_length
        self.max_length = max_length
        self.filter_empty = filter_empty
        self.normalize_unicode = normalize_unicode
        self.remove_duplicates = remove_duplicates
        
        # Compile regex patterns for efficiency
        self.whitespace_pattern = re.compile(r'\s+')
        self.control_chars_pattern = re.compile(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]')
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize text."""
        if not text:
            return ""
        
        # Normalize unicode
        if self.normalize_unicode:
            text = unicodedata.normalize('NFC', text)
        
        # Remove control characters
        text = self.control_chars_pattern.sub('', text)
        
        # Normalize whitespace
        text = self.whitespace_pattern.sub(' ', text)
        
        # Strip leading/trailing whitespace
        text = text.strip()
        
        return text
    
    def filter_text(self, text: str) -> bool:
        """Filter text based on criteria."""
        if self.filter_empty and not text.strip():
            return False
        
        text_length = len(text)
        if text_length < self.min_length or text_length > self.max_length:
            return False
        
        return True
    
    def process_text(self, text: str) -> Optional[str]:
        """Process a single text."""
        # Clean text
        cleaned_text = self.clean_text(text)
        
        # Filter text
        if not self.filter_text(cleaned_text):
            return None
        
        return cleaned_text
    
    def process_dataset(self, dataset: Dataset, text_column: str = "text",
                       num_proc: Optional[int] = None) -> Dataset:
        """Process an entire dataset."""
        def process_example(example):
            processed_text = self.process_text(example[text_column])
            example[text_column] = processed_text
            return example
        
        # Apply preprocessing
        processed_dataset = dataset.map(
            process_example,
            num_proc=num_proc,
            desc="Preprocessing texts"
        )
        
        # Filter out None values
        processed_dataset = processed_dataset.filter(
            lambda x: x[text_column] is not None,
            num_proc=num_proc,
            desc="Filtering texts"
        )
        
        # Remove duplicates if requested
        if self.remove_duplicates:
            # Get unique texts
            unique_texts = set()
            
            def is_unique(example):
                text = example[text_column]
                if text in unique_texts:
                    return False
                unique_texts.add(text)
                return True
            
            processed_dataset = processed_dataset.filter(
                is_unique,
                desc="Removing duplicates"
            )
        
        return processed_dataset


class TextTokenizer:
    """Text tokenization utilities."""
    
    def __init__(self, tokenizer, max_length: int = 1024, stride: int = 512,
                 add_special_tokens: bool = True):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.stride = stride
        self.add_special_tokens = add_special_tokens
    
    def tokenize_text(self, text: str) -> List[int]:
        """Tokenize a single text."""
        return self.tokenizer.encode(text, add_special_tokens=self.add_special_tokens)
    
    def tokenize_with_sliding_window(self, text: str) -> List[List[int]]:
        """Tokenize text with sliding window for long sequences."""
        # Tokenize full text
        token_ids = self.tokenize_text(text)
        
        if len(token_ids) <= self.max_length:
            return [token_ids]
        
        # Create sliding windows
        windows = []
        start = 0
        
        while start < len(token_ids):
            end = start + self.max_length
            window = token_ids[start:end]
            windows.append(window)
            
            # Move by stride
            start += self.stride
            
            # Stop if we've covered the entire sequence
            if end >= len(token_ids):
                break
        
        return windows
    
    def tokenize_dataset(self, dataset: Dataset, text_column: str = "text",
                        use_sliding_window: bool = True,
                        num_proc: Optional[int] = None) -> Dataset:
        """Tokenize an entire dataset."""
        def tokenize_example(example):
            text = example[text_column]
            
            if use_sliding_window:
                token_windows = self.tokenize_with_sliding_window(text)
                # Return multiple examples for sliding windows
                return {
                    "input_ids": token_windows,
                    "length": [len(window) for window in token_windows]
                }
            else:
                token_ids = self.tokenize_text(text)
                # Truncate if too long
                if len(token_ids) > self.max_length:
                    token_ids = token_ids[:self.max_length]
                
                return {
                    "input_ids": token_ids,
                    "length": len(token_ids)
                }
        
        # Apply tokenization
        tokenized_dataset = dataset.map(
            tokenize_example,
            num_proc=num_proc,
            remove_columns=dataset.column_names,
            desc="Tokenizing texts"
        )
        
        if use_sliding_window:
            # Flatten the dataset (each sliding window becomes a separate example)
            def flatten_examples(examples):
                flattened = {"input_ids": [], "length": []}
                for input_ids_list, length_list in zip(examples["input_ids"], examples["length"]):
                    flattened["input_ids"].extend(input_ids_list)
                    flattened["length"].extend(length_list)
                return flattened
            
            tokenized_dataset = tokenized_dataset.map(
                flatten_examples,
                batched=True,
                desc="Flattening sliding windows"
            )
        
        return tokenized_dataset


class SequencePacker:
    """Pack multiple sequences into fixed-length chunks for efficiency."""
    
    def __init__(self, max_length: int, pad_token_id: int = 0, 
                 eos_token_id: int = 3, strategy: str = "greedy"):
        self.max_length = max_length
        self.pad_token_id = pad_token_id
        self.eos_token_id = eos_token_id
        self.strategy = strategy
    
    def pack_sequences(self, sequences: List[List[int]]) -> List[List[int]]:
        """Pack sequences into fixed-length chunks."""
        if self.strategy == "greedy":
            return self._pack_greedy(sequences)
        elif self.strategy == "first_fit":
            return self._pack_first_fit(sequences)
        else:
            raise ValueError(f"Unknown packing strategy: {self.strategy}")
    
    def _pack_greedy(self, sequences: List[List[int]]) -> List[List[int]]:
        """Greedy packing: fill each chunk as much as possible."""
        packed_sequences = []
        current_chunk = []
        current_length = 0
        
        for seq in sequences:
            seq_with_eos = seq + [self.eos_token_id]
            seq_length = len(seq_with_eos)
            
            # If sequence is too long, truncate it
            if seq_length > self.max_length:
                seq_with_eos = seq_with_eos[:self.max_length]
                seq_length = self.max_length
            
            # If adding this sequence would exceed max_length, start new chunk
            if current_length + seq_length > self.max_length:
                if current_chunk:
                    # Pad current chunk and add to packed sequences
                    padded_chunk = current_chunk + [self.pad_token_id] * (self.max_length - current_length)
                    packed_sequences.append(padded_chunk)
                
                # Start new chunk
                current_chunk = seq_with_eos.copy()
                current_length = seq_length
            else:
                # Add to current chunk
                current_chunk.extend(seq_with_eos)
                current_length += seq_length
        
        # Add final chunk if not empty
        if current_chunk:
            padded_chunk = current_chunk + [self.pad_token_id] * (self.max_length - current_length)
            packed_sequences.append(padded_chunk)
        
        return packed_sequences
    
    def _pack_first_fit(self, sequences: List[List[int]]) -> List[List[int]]:
        """First-fit packing: find first chunk that can fit the sequence."""
        chunks = []
        chunk_lengths = []
        
        for seq in sequences:
            seq_with_eos = seq + [self.eos_token_id]
            seq_length = len(seq_with_eos)
            
            # If sequence is too long, truncate it
            if seq_length > self.max_length:
                seq_with_eos = seq_with_eos[:self.max_length]
                seq_length = self.max_length
            
            # Find first chunk that can fit this sequence
            placed = False
            for i, (chunk, chunk_length) in enumerate(zip(chunks, chunk_lengths)):
                if chunk_length + seq_length <= self.max_length:
                    chunk.extend(seq_with_eos)
                    chunk_lengths[i] += seq_length
                    placed = True
                    break
            
            # If no chunk can fit it, create new chunk
            if not placed:
                chunks.append(seq_with_eos.copy())
                chunk_lengths.append(seq_length)
        
        # Pad all chunks to max_length
        packed_sequences = []
        for chunk, chunk_length in zip(chunks, chunk_lengths):
            padded_chunk = chunk + [self.pad_token_id] * (self.max_length - chunk_length)
            packed_sequences.append(padded_chunk)
        
        return packed_sequences
