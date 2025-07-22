"""Base tokenizer interface."""

from abc import ABC, abstractmethod
from typing import List, Dict, Union, Optional, Tuple
import json


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
    
    @abstractmethod
    def train(self, texts: List[str], vocab_size: int) -> None:
        """Train the tokenizer on a corpus of texts."""
        pass
    
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
