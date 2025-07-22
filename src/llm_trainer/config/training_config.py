"""Training configuration for LLM training."""

from dataclasses import dataclass
from typing import Optional, List, Dict, Any
import yaml
import json


@dataclass
class TrainingConfig:
    """Configuration for training process."""
    
    # Training hyperparameters
    batch_size: int = 32
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    num_epochs: int = 10
    max_steps: Optional[int] = None
    
    # Learning rate scheduling
    lr_scheduler: str = "cosine"  # cosine, linear, constant, polynomial
    warmup_steps: int = 1000
    warmup_ratio: float = 0.1
    min_lr_ratio: float = 0.1
    
    # Optimization
    optimizer: str = "adamw"  # adamw, adam, sgd
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8
    max_grad_norm: float = 1.0
    
    # Gradient accumulation
    gradient_accumulation_steps: int = 1
    
    # Mixed precision training
    use_amp: bool = True
    amp_dtype: str = "float16"  # float16, bfloat16
    
    # Checkpointing
    save_steps: int = 1000
    save_total_limit: int = 3
    checkpoint_dir: str = "./checkpoints"
    resume_from_checkpoint: Optional[str] = None
    
    # Evaluation
    eval_steps: int = 500
    eval_strategy: str = "steps"  # steps, epoch, no
    eval_accumulation_steps: Optional[int] = None
    
    # Logging
    logging_steps: int = 100
    log_level: str = "info"
    report_to: List[str] = None  # ["tensorboard", "wandb"]
    
    # Data loading
    dataloader_num_workers: int = 4
    dataloader_pin_memory: bool = True
    dataloader_drop_last: bool = True
    
    # Distributed training
    local_rank: int = -1
    world_size: int = 1
    distributed_backend: str = "nccl"
    
    # DeepSpeed configuration
    use_deepspeed: bool = False
    deepspeed_config: Optional[str] = None
    
    # Seed for reproducibility
    seed: int = 42
    
    # Early stopping
    early_stopping_patience: Optional[int] = None
    early_stopping_threshold: float = 0.0
    
    # Model compilation (PyTorch 2.0+)
    compile_model: bool = False
    compile_mode: str = "default"  # default, reduce-overhead, max-autotune
    
    def __post_init__(self):
        """Validate and set default values."""
        if self.report_to is None:
            self.report_to = ["tensorboard"]
        
        # Validation
        assert self.batch_size > 0, "batch_size must be positive"
        assert self.learning_rate > 0, "learning_rate must be positive"
        assert 0 <= self.weight_decay <= 1, "weight_decay must be between 0 and 1"
        assert self.gradient_accumulation_steps > 0, "gradient_accumulation_steps must be positive"
        assert self.warmup_steps >= 0, "warmup_steps must be non-negative"
        assert 0 <= self.warmup_ratio <= 1, "warmup_ratio must be between 0 and 1"
        assert 0 <= self.min_lr_ratio <= 1, "min_lr_ratio must be between 0 and 1"
        assert self.max_grad_norm > 0, "max_grad_norm must be positive"
        
        # Scheduler validation
        valid_schedulers = ["cosine", "linear", "constant", "polynomial"]
        assert self.lr_scheduler in valid_schedulers, f"lr_scheduler must be one of {valid_schedulers}"
        
        # Optimizer validation
        valid_optimizers = ["adamw", "adam", "sgd"]
        assert self.optimizer in valid_optimizers, f"optimizer must be one of {valid_optimizers}"
        
        # AMP validation
        valid_amp_dtypes = ["float16", "bfloat16"]
        assert self.amp_dtype in valid_amp_dtypes, f"amp_dtype must be one of {valid_amp_dtypes}"
    
    @property
    def effective_batch_size(self) -> int:
        """Calculate effective batch size considering gradient accumulation."""
        return self.batch_size * self.gradient_accumulation_steps * self.world_size
    
    def save(self, path: str) -> None:
        """Save configuration to file."""
        config_dict = self.__dict__.copy()
        
        if path.endswith('.yaml') or path.endswith('.yml'):
            with open(path, 'w') as f:
                yaml.dump(config_dict, f, default_flow_style=False)
        elif path.endswith('.json'):
            with open(path, 'w') as f:
                json.dump(config_dict, f, indent=2)
        else:
            raise ValueError("Unsupported file format. Use .yaml, .yml, or .json")
    
    @classmethod
    def load(cls, path: str) -> 'TrainingConfig':
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
    def from_dict(cls, config_dict: dict) -> 'TrainingConfig':
        """Create configuration from dictionary."""
        return cls(**config_dict)
