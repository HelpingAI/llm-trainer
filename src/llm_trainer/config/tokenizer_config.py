"""Tokenizer configuration for LLM training."""

from dataclasses import dataclass
from typing import Optional, Dict, Any
import yaml
import json


@dataclass
class TokenizerConfig:
    """Configuration for tokenizer setup."""
    
    # Tokenizer type and source
    type: str = "bpe"  # "bpe" for built-in BPE, "custom" for external tokenizers
    name_or_path: Optional[str] = None  # HuggingFace model name or local path for custom tokenizers
    
    # Tokenizer parameters
    vocab_size: int = 32000
    max_length: int = 1024
    
    # Special token configuration  
    pad_token: Optional[str] = None
    unk_token: Optional[str] = None
    bos_token: Optional[str] = None
    eos_token: Optional[str] = None
    
    # Custom tokenizer settings
    use_fast: bool = True  # Use fast tokenizer implementation when available
    trust_remote_code: bool = False  # Whether to trust remote code for tokenizers
    
    # BPE-specific settings (only used when type="bpe")
    min_frequency: int = 2
    
    def __post_init__(self):
        """Validate configuration parameters."""
        if self.type not in ["bpe", "custom"]:
            raise ValueError(f"tokenizer.type must be 'bpe' or 'custom', got '{self.type}'")
        
        if self.type == "custom" and self.name_or_path is None:
            raise ValueError("tokenizer.name_or_path is required when type='custom'")
        
        if self.vocab_size <= 0:
            raise ValueError("vocab_size must be positive")
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'TokenizerConfig':
        """Create config from dictionary."""
        return cls(**config_dict)
    
    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'TokenizerConfig':
        """Load configuration from YAML file."""
        with open(yaml_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return cls.from_dict(config_dict.get('tokenizer', {}))
    
    @classmethod  
    def from_json(cls, json_path: str) -> 'TokenizerConfig':
        """Load configuration from JSON file."""
        with open(json_path, 'r') as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict.get('tokenizer', {}))
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            'type': self.type,
            'name_or_path': self.name_or_path,
            'vocab_size': self.vocab_size,
            'max_length': self.max_length,
            'pad_token': self.pad_token,
            'unk_token': self.unk_token,
            'bos_token': self.bos_token,
            'eos_token': self.eos_token,
            'use_fast': self.use_fast,
            'trust_remote_code': self.trust_remote_code,
            'min_frequency': self.min_frequency
        }
    
    def save(self, save_path: str) -> None:
        """Save configuration to JSON file."""
        if save_path.endswith('.yaml') or save_path.endswith('.yml'):
            with open(save_path, 'w') as f:
                yaml.dump({'tokenizer': self.to_dict()}, f, indent=2)
        else:
            with open(save_path, 'w') as f:
                json.dump(self.to_dict(), f, indent=2)