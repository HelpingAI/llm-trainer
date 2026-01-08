"""Training configuration for LLM training with TRL-style parameters."""

from dataclasses import dataclass
from typing import Optional, List
import yaml
import json
import torch


@dataclass
class TrainingConfig:
    """Enhanced configuration for training process with TRL-style parameters."""

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
    fp16: bool = False
    bf16: bool = False

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
    report_to: Optional[List[str]] = None  # None or list like ["tensorboard", "wandb"]

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

    # Accelerate configuration
    use_accelerate: bool = False
    accelerate_mixed_precision: str = "no"  # one of: "no", "fp16", "bf16"

    # PEFT / LoRA configuration
    use_peft: bool = False
    peft_type: str = "lora"  # lora, ada_lora, or other peft methods
    peft_r: int = 8
    peft_alpha: int = 16
    peft_dropout: float = 0.05
    peft_target_modules: Optional[List[str]] = None
    peft_bias: str = "none"  # none, all, lora_only
    peft_task_type: str = "CAUSAL_LM"

    # Device configuration
    device: str = "auto"  # auto, cpu, cuda, mps
    force_cpu: bool = False

    # Seed for reproducibility
    seed: int = 42

    # Early stopping
    early_stopping_patience: Optional[int] = None
    early_stopping_threshold: float = 0.0

    # Memory optimization
    use_gradient_checkpointing: bool = False
    use_low_vram: bool = False
    fuse_layers: bool = False
    empty_cache_steps: int = 0

    # TRL-style parameters (for compatibility)
    per_device_train_batch_size: Optional[int] = None
    per_device_eval_batch_size: Optional[int] = None
    num_train_epochs: Optional[int] = None
    learning_rate_type: Optional[str] = None  # lr_scheduler_type in TRL
    evaluation_strategy: Optional[str] = None  # eval_strategy in our implementation
    save_strategy: Optional[str] = None
    logging_strategy: Optional[str] = None
    optim: Optional[str] = None  # optimizer in our implementation

    def __post_init__(self):
        """Validate and set default values."""
        # Map TRL-style parameters to our parameters
        if self.per_device_train_batch_size is not None:
            self.batch_size = self.per_device_train_batch_size
        if self.num_train_epochs is not None:
            self.num_epochs = self.num_train_epochs
        if self.learning_rate_type is not None:
            self.lr_scheduler = self.learning_rate_type
        if self.evaluation_strategy is not None:
            self.eval_strategy = self.evaluation_strategy
        if self.optim is not None:
            self.optimizer = self.optim

        # Validation
        assert self.batch_size > 0, "batch_size must be positive"
        assert self.learning_rate > 0, "learning_rate must be positive"
        assert 0 <= self.weight_decay <= 1, "weight_decay must be between 0 and 1"
        if self.gradient_accumulation_steps <= 0:
            raise ValueError("gradient_accumulation_steps must be > 0")
        if self.warmup_steps < 0:
            raise ValueError("warmup_steps must be non-negative")
        if not (0 <= self.warmup_ratio <= 1):
            raise ValueError("warmup_ratio must be between 0 and 1")
        if not (0 <= self.min_lr_ratio <= 1):
            raise ValueError("min_lr_ratio must be between 0 and 1")
        if self.max_grad_norm <= 0:
            raise ValueError("max_grad_norm must be positive")

        # Scheduler validation
        valid_schedulers = ["cosine", "linear", "constant", "polynomial"]
        if self.lr_scheduler not in valid_schedulers:
            raise ValueError("lr_scheduler must be one of %s" % valid_schedulers)

        # Optimizer validation
        valid_optimizers = ["adamw", "adam", "sgd"]
        if self.optimizer not in valid_optimizers:
            raise ValueError("optimizer must be one of %s" % valid_optimizers)

        # Device validation
        valid_devices = ["auto", "cpu", "cuda", "mps"]
        assert self.device in valid_devices, f"device must be one of {valid_devices}"

        # Auto-adjust settings for CPU compatibility
        if self.force_cpu or self.device == "cpu":
            # Disable AMP for CPU training
            self.fp16 = False
            self.bf16 = False
            # Use gloo backend for CPU distributed training
            if self.distributed_backend == "nccl":
                self.distributed_backend = "gloo"

    @property
    def effective_batch_size(self) -> int:
        """Calculate effective batch size considering gradient accumulation."""
        return self.batch_size * self.gradient_accumulation_steps * self.world_size

    def get_effective_device(self) -> torch.device:
        """Get the effective device based on configuration and hardware availability."""
        # Force CPU if explicitly requested
        if self.force_cpu:
            return torch.device("cpu")

        # Handle explicit device selection
        if self.device == "cpu":
            return torch.device("cpu")
        elif self.device == "cuda":
            if torch.cuda.is_available():
                return torch.device("cuda")
            else:
                return torch.device("cpu")
        elif self.device == "mps":
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return torch.device("mps")
            else:
                return torch.device("cpu")
        elif self.device == "auto":
            # Use existing auto-detection logic from utils.py
            if torch.cuda.is_available():
                return torch.device("cuda")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return torch.device("mps")
            else:
                return torch.device("cpu")
        else:
            # Fallback to CPU for unknown device types
            return torch.device("cpu")

    def should_use_amp(self) -> bool:
        """Determine if AMP should be used based on device and configuration."""
        effective_device = self.get_effective_device()

        # AMP is not supported on CPU
        if effective_device.type == "cpu":
            return False

        # Return the configured AMP setting for GPU/MPS devices
        return self.fp16 or self.bf16

    def get_amp_dtype(self) -> torch.dtype:
        """Get the torch dtype for mixed precision training."""
        if self.bf16:
            return torch.bfloat16
        return torch.float16

    def get_effective_distributed_backend(self) -> str:
        """Get the effective distributed backend based on device."""
        effective_device = self.get_effective_device()

        # Use gloo for CPU, nccl for GPU
        if effective_device.type == "cpu":
            return "gloo"
        else:
            return self.distributed_backend

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
