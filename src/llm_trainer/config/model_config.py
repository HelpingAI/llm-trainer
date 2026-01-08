"""Model configuration for Transformer architecture."""

from dataclasses import dataclass
import yaml
import json


@dataclass
class ModelConfig:
    """Configuration for Transformer model architecture."""

    # Model architecture
    vocab_size: int = 50000
    d_model: int = 512
    n_heads: int = 8
    n_layers: int = 6
    d_ff: int = 2048
    max_seq_len: int = 1024

    # Regularization
    dropout: float = 0.1
    attention_dropout: float = 0.1

    # Initialization
    init_std: float = 0.02

    # Positional encoding
    use_learned_pos_emb: bool = False

    # Layer normalization
    layer_norm_eps: float = 1e-5
    pre_norm: bool = True  # Pre-norm vs post-norm

    # Activation function
    activation: str = "gelu"  # gelu, relu, swish

    # Bias terms
    use_bias: bool = True

    # Gradient checkpointing for memory efficiency
    gradient_checkpointing: bool = False

    def __post_init__(self):
        """Validate configuration parameters."""
        assert self.d_model % self.n_heads == 0, "d_model must be divisible by n_heads"
        assert self.vocab_size > 0, "vocab_size must be positive"
        assert self.n_layers > 0, "n_layers must be positive"
        assert self.max_seq_len > 0, "max_seq_len must be positive"
        assert 0 <= self.dropout <= 1, "dropout must be between 0 and 1"
        assert 0 <= self.attention_dropout <= 1, "attention_dropout must be between 0 and 1"
        assert self.activation in ["gelu", "relu", "swish"], "Invalid activation function"

    @property
    def d_head(self) -> int:
        """Dimension of each attention head."""
        return self.d_model // self.n_heads

    def save(self, path: str) -> None:
        """Save configuration to file."""
        config_dict = self.__dict__

        if path.endswith('.yaml') or path.endswith('.yml'):
            with open(path, 'w') as f:
                yaml.dump(config_dict, f, default_flow_style=False)
        elif path.endswith('.json'):
            with open(path, 'w') as f:
                json.dump(config_dict, f, indent=2)
        else:
            raise ValueError("Unsupported file format. Use .yaml, .yml, or .json")

    @classmethod
    def load(cls, path: str) -> 'ModelConfig':
        """Load configuration from file."""
        if path.endswith('.yaml') or path.endswith('.yml'):
            with open(path, 'r') as f:
                config_dict = yaml.safe_load(f)
        elif path.endswith('.json'):
            with open(path, 'r') as f:
                config_dict = json.load(f)
        else:
            raise ValueError("Unsupported file format. Use .yaml, .yml, or .json")

        return cls(**config_dict)

    def to_dict(self) -> dict:
        """Convert configuration to dictionary."""
        return self.__dict__.copy()

    @classmethod
    def from_dict(cls, config_dict: dict) -> 'ModelConfig':
        """Create configuration from dictionary."""
        return cls(**config_dict)
