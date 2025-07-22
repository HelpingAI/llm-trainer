"""Custom tokenizer wrapper for using any tokenizer with the training pipeline."""

from typing import List, Dict, Union, Optional, Any
import os
import json
from .base_tokenizer import BaseTokenizer


class CustomTokenizerWrapper(BaseTokenizer):
    """Wrapper to make any tokenizer compatible with the training pipeline.
    
    This wrapper allows using any tokenizer (like Hugging Face AutoTokenizer)
    with the existing training infrastructure while maintaining the expected interface.
    """
    
    def __init__(self, tokenizer=None, tokenizer_name_or_path: Optional[str] = None):
        """Initialize the custom tokenizer wrapper.
        
        Args:
            tokenizer: Pre-initialized tokenizer object
            tokenizer_name_or_path: Path or name to load tokenizer from
        """
        super().__init__()
        
        if tokenizer is not None:
            self.tokenizer = tokenizer
        elif tokenizer_name_or_path is not None:
            self.tokenizer = self._load_tokenizer(tokenizer_name_or_path)
        else:
            raise ValueError("Either tokenizer or tokenizer_name_or_path must be provided")
        
        # Set up token mappings
        self._setup_special_tokens()
        self._setup_vocab()
    
    def _load_tokenizer(self, name_or_path: str):
        """Load tokenizer from Hugging Face or local path."""
        try:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(name_or_path, local_files_only=False)
            
            # Set pad_token if not available
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
                
            return tokenizer
        except ImportError:
            raise ImportError("transformers library is required for custom tokenizers. Install with: pip install transformers")
        except Exception as e:
            raise ValueError(f"Failed to load tokenizer from {name_or_path}: {e}")
    
    def _setup_special_tokens(self):
        """Setup special token mappings from the underlying tokenizer."""
        # Get special tokens from the underlying tokenizer
        if hasattr(self.tokenizer, 'pad_token') and self.tokenizer.pad_token:
            self.pad_token = self.tokenizer.pad_token
            self.pad_token_id = self.tokenizer.pad_token_id
        else:
            # Fallback if no pad token
            self.pad_token = self.tokenizer.eos_token if hasattr(self.tokenizer, 'eos_token') else "<pad>"
            self.pad_token_id = self.tokenizer.eos_token_id if hasattr(self.tokenizer, 'eos_token_id') else 0
            
        if hasattr(self.tokenizer, 'unk_token') and self.tokenizer.unk_token:
            self.unk_token = self.tokenizer.unk_token
            self.unk_token_id = self.tokenizer.unk_token_id
        else:
            self.unk_token = "<unk>"
            self.unk_token_id = getattr(self.tokenizer, 'unk_token_id', 1)
            
        if hasattr(self.tokenizer, 'bos_token') and self.tokenizer.bos_token:
            self.bos_token = self.tokenizer.bos_token
            self.bos_token_id = self.tokenizer.bos_token_id
        else:
            self.bos_token = "<bos>"
            self.bos_token_id = getattr(self.tokenizer, 'bos_token_id', 2)
            
        if hasattr(self.tokenizer, 'eos_token') and self.tokenizer.eos_token:
            self.eos_token = self.tokenizer.eos_token
            self.eos_token_id = self.tokenizer.eos_token_id
        else:
            self.eos_token = "<eos>"
            self.eos_token_id = getattr(self.tokenizer, 'eos_token_id', 3)
        
        # Update special tokens dict
        self.special_tokens = {
            self.pad_token: self.pad_token_id,
            self.unk_token: self.unk_token_id,
            self.bos_token: self.bos_token_id,
            self.eos_token: self.eos_token_id
        }
    
    def _setup_vocab(self):
        """Setup vocabulary mappings."""
        if hasattr(self.tokenizer, 'get_vocab'):
            self.vocab = self.tokenizer.get_vocab()
        else:
            # Fallback for tokenizers without get_vocab method
            self.vocab = {}
            
        # Create inverse vocab mapping
        self.inverse_vocab = {v: k for k, v in self.vocab.items()}
    
    def train(self, texts: List[str], vocab_size: int) -> None:
        """Training not supported for pre-trained tokenizers."""
        raise NotImplementedError("Training is not supported for custom pre-trained tokenizers")
    
    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """Encode text to token IDs."""
        return self.tokenizer.encode(text, add_special_tokens=add_special_tokens)
    
    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        """Decode token IDs to text."""
        return self.tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)
    
    @property
    def vocab_size(self) -> int:
        """Get vocabulary size."""
        if hasattr(self.tokenizer, 'vocab_size'):
            return self.tokenizer.vocab_size
        elif self.vocab:
            return len(self.vocab)
        else:
            # Fallback estimation
            return 50000
    
    def get_vocab(self) -> Dict[str, int]:
        """Get the vocabulary dictionary."""
        return self.vocab.copy() if self.vocab else {}
    
    def save_pretrained(self, save_directory: str) -> None:
        """Save the tokenizer to a directory."""
        if hasattr(self.tokenizer, 'save_pretrained'):
            self.tokenizer.save_pretrained(save_directory)
        else:
            # Fallback: save basic info
            os.makedirs(save_directory, exist_ok=True)
            config = {
                "tokenizer_class": "CustomTokenizerWrapper",
                "special_tokens": self.special_tokens,
                "vocab_size": self.vocab_size
            }
            with open(os.path.join(save_directory, "tokenizer_config.json"), "w") as f:
                json.dump(config, f, indent=2)
    
    @classmethod
    def from_pretrained(cls, save_directory: str):
        """Load tokenizer from a directory."""
        try:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(save_directory)
            return cls(tokenizer=tokenizer)
        except Exception:
            # Fallback: try to load from config
            config_path = os.path.join(save_directory, "tokenizer_config.json")
            if os.path.exists(config_path):
                with open(config_path, "r") as f:
                    config = json.load(f)
                raise ValueError(f"Cannot load tokenizer from {save_directory}. Original tokenizer required.")
            else:
                raise ValueError(f"No tokenizer found in {save_directory}")
    
    def train_from_dataset(self, 
                          dataset_name: str,
                          dataset_config: str = None,
                          vocab_size: int = 32000,
                          max_samples: Optional[int] = None,
                          verbose: bool = True) -> None:
        """Training from dataset not supported for pre-trained tokenizers."""
        raise NotImplementedError("Training from dataset is not supported for custom pre-trained tokenizers")
    
    def __repr__(self) -> str:
        return f"CustomTokenizerWrapper(vocab_size={self.vocab_size}, tokenizer={type(self.tokenizer).__name__})"