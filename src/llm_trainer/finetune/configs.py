"""
Configuration classes for fine-tuning, inspired by Axolotl's YAML-based configuration system.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Union
import torch


@dataclass
class QuantizationConfig:
    """Configuration for model quantization."""
    
    # Quantization settings
    load_in_4bit: bool = False
    load_in_8bit: bool = False
    quant_type: str = "nf4"  # nf4, fp4
    compute_dtype: str = "bfloat16"  # bfloat16, float16, float32
    double_quant: bool = True
    
    def to_bnb_config(self):
        """Convert to BitsAndBytesConfig."""
        try:
            from transformers import BitsAndBytesConfig
            
            dtype_map = {
                "bfloat16": torch.bfloat16,
                "float16": torch.float16,
                "float32": torch.float32
            }
            
            return BitsAndBytesConfig(
                load_in_4bit=self.load_in_4bit,
                load_in_8bit=self.load_in_8bit,
                bnb_4bit_compute_dtype=dtype_map.get(self.compute_dtype, torch.bfloat16),
                bnb_4bit_use_double_quant=self.double_quant,
                bnb_4bit_quant_type=self.quant_type if self.load_in_4bit else None,
            )
        except ImportError:
            raise ImportError("transformers and bitsandbytes required for quantization")


@dataclass
class LoRAConfig:
    """Configuration for LoRA (Low-Rank Adaptation)."""
    
    # LoRA hyperparameters
    r: int = 16
    alpha: int = 32
    dropout: float = 0.05
    bias: str = "none"  # none, all, lora_only
    target_modules: Optional[List[str]] = None
    task_type: str = "CAUSAL_LM"
    
    # Advanced LoRA settings
    use_rslora: bool = False
    use_dora: bool = False
    loftq_config: Optional[Dict[str, Any]] = None
    
    def to_peft_config(self):
        """Convert to PEFT LoraConfig."""
        try:
            from peft import LoraConfig as PeftLoraConfig
            
            return PeftLoraConfig(
                r=self.r,
                lora_alpha=self.alpha,
                lora_dropout=self.dropout,
                bias=self.bias,
                target_modules=self.target_modules,
                task_type=self.task_type,
                use_rslora=self.use_rslora,
                use_dora=self.use_dora,
                loftq_config=self.loftq_config
            )
        except ImportError:
            raise ImportError("peft required for LoRA configuration")


@dataclass
class BaseTrainingConfig:
    """Base configuration for all training methods."""
    
    # Model settings
    model_name_or_path: str = ""
    torch_dtype: str = "auto"  # auto, float32, float16, bfloat16
    device_map: Union[str, Dict] = "auto"
    trust_remote_code: bool = False
    attn_implementation: Optional[str] = None  # flash_attention_2, sdpa
    
    # Training hyperparameters
    learning_rate: float = 2e-4
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 4
    gradient_accumulation_steps: int = 1
    warmup_steps: int = 100
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    
    # Optimization
    optim: str = "adamw_torch"  # adamw_torch, adamw_hf, paged_adamw_8bit
    lr_scheduler_type: str = "cosine"  # linear, cosine, constant
    
    # Sequence settings
    max_seq_length: int = 2048
    packing: bool = False
    
    # Evaluation and logging
    evaluation_strategy: str = "no"  # no, steps, epoch
    eval_steps: Optional[int] = None
    logging_steps: int = 10
    save_steps: int = 500
    save_total_limit: int = 3
    
    # Output settings
    output_dir: str = "./output"
    run_name: Optional[str] = None
    report_to: List[str] = field(default_factory=lambda: ["tensorboard"])
    
    # Memory optimizations
    gradient_checkpointing: bool = True
    dataloader_pin_memory: bool = True
    dataloader_num_workers: int = 0
    remove_unused_columns: bool = False
    
    # Mixed precision
    fp16: bool = False
    bf16: bool = True
    
    # Quantization and PEFT
    quantization: Optional[QuantizationConfig] = None
    lora: Optional[LoRAConfig] = None
    
    # Seed
    seed: int = 42


@dataclass
class SFTConfig(BaseTrainingConfig):
    """Configuration for Supervised Fine-Tuning."""
    
    # SFT-specific settings
    dataset_text_field: str = "text"
    max_seq_length: int = 2048
    packing: bool = False
    
    # Chat template settings
    chat_template: Optional[str] = None
    
    # Dataset formatting
    formatting_func: Optional[str] = None  # Function name for custom formatting
    
    def __post_init__(self):
        """Post-initialization validation."""
        if self.packing and self.max_seq_length > 4096:
            print("Warning: Packing with long sequences may cause memory issues")


@dataclass 
class DPOConfig(BaseTrainingConfig):
    """Configuration for Direct Preference Optimization."""
    
    # DPO-specific hyperparameters
    beta: float = 0.1
    loss_type: str = "sigmoid"  # sigmoid, hinge, ipo
    label_smoothing: float = 0.0
    
    # Reference model settings
    ref_model: Optional[str] = None  # If None, uses the same model
    
    # Dataset settings
    prompt_field: str = "prompt"
    chosen_field: str = "chosen" 
    rejected_field: str = "rejected"
    
    def __post_init__(self):
        """Post-initialization validation."""
        if self.beta <= 0:
            raise ValueError("DPO beta must be positive")


@dataclass
class PPOConfig(BaseTrainingConfig):
    """Configuration for Proximal Policy Optimization."""
    
    # PPO-specific hyperparameters
    ppo_epochs: int = 4
    init_kl_coef: float = 0.2
    target_kl: float = 6.0
    adap_kl_ctrl: bool = True
    
    # Value function settings
    vf_coef: float = 0.1
    cliprange: float = 0.2
    cliprange_value: float = 0.2
    
    # Generation settings
    max_new_tokens: int = 128
    temperature: float = 1.0
    top_k: int = 50
    top_p: float = 0.95
    
    # Reward model
    reward_model: Optional[str] = None
    
    def __post_init__(self):
        """Post-initialization validation."""
        if self.init_kl_coef < 0:
            raise ValueError("PPO init_kl_coef must be non-negative")


@dataclass
class RewardConfig(BaseTrainingConfig):
    """Configuration for Reward Model training."""
    
    # Reward model specific settings
    num_labels: int = 1
    
    # Dataset settings
    prompt_field: str = "prompt"
    chosen_field: str = "chosen"
    rejected_field: str = "rejected"
    
    # Loss settings
    loss_type: str = "ranking"  # ranking, regression
    margin: float = 0.0
    
    def __post_init__(self):
        """Post-initialization validation."""
        if self.num_labels < 1:
            raise ValueError("num_labels must be at least 1")


# Configuration factory functions
def create_sft_config(**kwargs) -> SFTConfig:
    """Create SFT configuration with defaults."""
    return SFTConfig(**kwargs)


def create_dpo_config(**kwargs) -> DPOConfig:
    """Create DPO configuration with defaults.""" 
    return DPOConfig(**kwargs)


def create_ppo_config(**kwargs) -> PPOConfig:
    """Create PPO configuration with defaults."""
    return PPOConfig(**kwargs)


def create_reward_config(**kwargs) -> RewardConfig:
    """Create Reward configuration with defaults."""
    return RewardConfig(**kwargs)


def load_config_from_yaml(yaml_path: str, config_type: str = "sft") -> BaseTrainingConfig:
    """
    Load configuration from YAML file.
    
    Args:
        yaml_path: Path to YAML file
        config_type: Type of config (sft, dpo, ppo, reward)
        
    Returns:
        Configuration object
    """
    import yaml
    
    with open(yaml_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    # Extract nested configurations
    model_config = config_dict.get('model', {})
    training_config = config_dict.get('training', {})
    
    # Merge configurations
    merged_config = {**model_config, **training_config}
    
    # Handle quantization config
    if 'bnb' in config_dict.get('model', {}):
        bnb_config = config_dict['model']['bnb']
        merged_config['quantization'] = QuantizationConfig(
            load_in_4bit=config_dict.get('model', {}).get('quantization_bits') == 4,
            load_in_8bit=config_dict.get('model', {}).get('quantization_bits') == 8,
            quant_type=bnb_config.get('quant_type', 'nf4'),
            compute_dtype=bnb_config.get('compute_dtype', 'bfloat16'),
            double_quant=bnb_config.get('double_quant', True)
        )
    
    # Handle LoRA config
    if 'lora' in config_dict.get('model', {}):
        lora_config = config_dict['model']['lora']
        merged_config['lora'] = LoRAConfig(
            r=lora_config.get('r', 16),
            alpha=lora_config.get('alpha', 32),
            dropout=lora_config.get('dropout', 0.05),
            bias=lora_config.get('bias', 'none'),
            target_modules=lora_config.get('target_modules')
        )
    
    # Create appropriate config type
    config_classes = {
        'sft': SFTConfig,
        'dpo': DPOConfig,
        'ppo': PPOConfig,
        'reward': RewardConfig
    }
    
    config_class = config_classes.get(config_type, SFTConfig)
    return config_class(**merged_config)
