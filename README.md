# LLM Trainer

<div align="center">

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)](https://opensource.org/licenses/Apache-2.0)
[![GitHub](https://img.shields.io/badge/GitHub-HelpingAI/llm--trainer-black.svg)](https://github.com/HelpingAI/llm-trainer)
[![SafeTensors](https://img.shields.io/badge/SafeTensors-Supported-brightgreen.svg)](https://github.com/huggingface/safetensors)
[![Version](https://img.shields.io/badge/version-0.2.4-blue.svg)](https://github.com/HelpingAI/llm-trainer/releases)

*A production-ready framework for training Large Language Models from scratch with modern PyTorch*

</div>

## What's New in v0.2.6

- **Memory Optimizations**: Efficient training with kernel optimizations
- **SafeTensors Support**: Secure model serialization with automatic sharding for large models
- **HuggingFace Integration**: Use any pretrained tokenizer via `HFTokenizerWrapper`
- **Accelerate Support**: Distributed training with `use_accelerate=true`
- **LoRA/PEFT**: Parameter-efficient fine-tuning with `use_peft=true`
- **Backward Compatible**: Existing PyTorch models continue to work
- **Patching System**: Kernel optimizations and memory-efficient training

## Features

### Core Architecture
- **Custom Transformer Implementation**: Multi-head attention, feed-forward networks, positional encodings
- **SafeTensors Integration**: Secure model serialization with automatic sharding
- **Modular Design**: Easy to extend and customize for research and production

### Tokenization
- **BPE Tokenizer**: From-scratch BPE with Unicode and emoji support
- **HuggingFace Integration**: Use any pretrained tokenizer (Mistral, Llama, GPT-2, etc.)
- **WordPiece Support**: Alternative tokenization strategies

### Data Pipeline
- **HuggingFace Datasets**: Efficient loading with preprocessing and batching
- **Memory Optimization**: Smart sequence packing and data streaming
- **Multi-Processing**: Parallel data preprocessing for faster training

### Training & Inference
- **CPU/GPU Support**: Optimized configurations for both CPU and GPU training
- **Distributed Training**: Multi-GPU support via Accelerate and DeepSpeed
- **Parameter-Efficient**: LoRA/PEFT adapters for memory-efficient fine-tuning
- **Mixed Precision**: FP16/BF16 automatic mixed precision
- **Multiple Decoding Strategies**: Greedy, beam search, nucleus (top-p), top-k sampling
- **Enhanced Trainer**: TRL-style training methods with familiar APIs
- **Memory-Efficient Optimizers**: Optimized implementations for better performance
- **Kernel Optimizations**: Fused operations for better performance
- **Low VRAM Training**: Gradient checkpointing and memory-efficient techniques

### Monitoring & Evaluation
- **TensorBoard Integration**: Real-time training metrics and visualizations
- **Weights & Biases**: Experiment tracking and hyperparameter optimization
- **Comprehensive Metrics**: Perplexity, cross-entropy loss, generation quality

## Requirements

- Python 3.8 or higher
- PyTorch 2.0 or higher
- GPU: CUDA-compatible GPU (recommended) or CPU-only mode
- Memory: 8GB RAM minimum (16GB+ recommended)

## Installation

### Basic Installation
```bash
git clone https://github.com/HelpingAI/llm-trainer.git
cd llm-trainer
pip install -e .
```

### Optional Dependencies

```bash
# Development tools
pip install -e ".[dev]"

# SafeTensors support (recommended)
pip install -e ".[safetensors]"

# Distributed training
pip install -e ".[distributed]"

# All features
pip install -e ".[full]"
```

## Quick Start

### Python API - Enhanced Training

```python
from llm_trainer import Trainer, TrainingConfig
from llm_trainer.models import TransformerLM
from llm_trainer.config import ModelConfig
from llm_trainer.tokenizer import BPETokenizer

# Create model and tokenizer
model_config = ModelConfig(
    vocab_size=32000,
    d_model=512,
    n_heads=8,
    n_layers=6,
    max_seq_len=1024
)
model = TransformerLM(model_config)
tokenizer = BPETokenizer()

# Configure training with TRL-style parameters
training_config = TrainingConfig(
    per_device_train_batch_size=4,
    gradient_accumulation_steps=2,
    learning_rate=2e-5,
    num_train_epochs=3,
    logging_steps=10,
    save_steps=100,
    optim="adamw"  # TRL-style parameter
)

# Create trainer and train
trainer = Trainer(
    model=model,
    tokenizer=tokenizer,
    config=training_config
)

# TRL-style training methods
trainer.train()  # Standard training
trainer.sft_train()  # Supervised fine-tuning
trainer.dpo_train()  # Direct preference optimization
```

### HuggingFace Integration with PEFT

```python
from llm_trainer import Trainer, TrainingConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, TaskType

# Load pretrained model and tokenizer
model = AutoModelForCausalLM.from_pretrained("gpt2")
tokenizer = AutoTokenizer.from_pretrained("gpt2")

# Configure LoRA (PEFT)
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    task_type=TaskType.CAUSAL_LM
)

# Create trainer with PEFT
trainer = Trainer(
    model=model,
    tokenizer=tokenizer,
    config=TrainingConfig(),
    peft_config=lora_config  # Pass PEFT config directly
)

# Show parameter efficiency
trainer.print_trainable_parameters()

# Train with familiar API
trainer.train()
```

### Memory-Efficient Optimizers

```python
from llm_trainer.training import create_optimizer

# Create memory-efficient optimizer
optimizer = create_optimizer(
    model,
    optimizer_name="adamw",
    learning_rate=5e-5,
    weight_decay=0.01
)
```

### Patching for Transformers/TRL

```python
from llm_trainer import patch_transformers, patch_trl

# Patch Hugging Face Transformers with memory-efficient optimizations
patch_transformers()

# Patch TRL with memory-efficient optimizations
patch_trl()

# Now you can use enhanced Transformers/TRL classes with memory-efficient methods
from transformers import Trainer, TrainingArguments
from trl import SFTTrainer

# These trainers now have enhanced methods
trainer = SFTTrainer(...)
trainer.print_trainable_parameters()  # Added by patching
trainer.prepare_model_for_kbit_training()  # Added by patching
```

### Kernel Optimizations for Fast Training

```python
from llm_trainer.kernels import (
    FusedLinear, FusedRMSNorm, fused_cross_entropy,
    gradient_checkpointing, LowVRAMLinear, empty_cache
)

# Use fused operations for better performance
fused_linear = FusedLinear(in_features=512, out_features=512)
fused_norm = FusedRMSNorm(dim=512)

# Use gradient checkpointing to reduce memory usage
def forward_pass_with_checkpointing(model, inputs):
    return gradient_checkpointing(model, inputs)

# Use low VRAM linear layers for memory-efficient training
low_vram_linear = LowVRAMLinear(in_features=512, out_features=512)

# Clear cache to free up memory
empty_cache()
```

### Command Line

```bash
# GPU Training
python scripts/train.py --config configs/small_model.yaml --output_dir ./output

# CPU Training (no GPU required)
python scripts/train.py --config configs/cpu_small_model.yaml --output_dir ./output

# Text Generation
python scripts/generate.py --model_path ./output --prompts "The quick brown fox" --interactive

# Model Evaluation
python scripts/evaluate.py --model_path ./output --dataset_config configs/eval_config.json
```

## Configuration

The framework uses YAML/JSON configuration files for reproducible experiments:

### Small Model (Quick Start)
```yaml
model:
  d_model: 512
  n_heads: 8
  n_layers: 6
  vocab_size: 32000
  max_seq_len: 1024

training:
  batch_size: 16
  learning_rate: 1e-4
  num_epochs: 3
  use_amp: true
  gradient_accumulation_steps: 4
```

### CPU-Optimized Training
```yaml
device: "cpu"
model:
  d_model: 256
  n_heads: 4
  n_layers: 4
  max_seq_len: 512

training:
  batch_size: 2
  use_amp: false
  gradient_accumulation_steps: 8
  dataloader_num_workers: 2
```

### Advanced Configuration
```yaml
model:
  d_model: 768
  n_heads: 12
  n_layers: 12

training:
  use_accelerate: true
  accelerate_mixed_precision: "fp16"
  use_peft: true
  peft_type: "lora"
  peft_r: 8
  peft_alpha: 16

# SafeTensors settings
save_format: "safetensors"
max_shard_size: "2GB"
```

## Project Structure

```
llm-trainer/
├── src/llm_trainer/              # Main package
│   ├── models/                   # Model architectures
│   │   ├── base_model.py         # Base model interface
│   │   ├── transformer.py        # Custom Transformer implementation
│   │   ├── safetensors_utils.py  # SafeTensors utilities
│   │   └── attention.py          # Attention mechanisms
│   ├── tokenizer/                # Tokenization
│   │   ├── bpe_tokenizer.py      # BPE implementation
│   │   ├── hf_tokenizer.py       # HuggingFace wrapper
│   │   └── wordpiece_tokenizer.py # WordPiece implementation
│   ├── data/                     # Data pipeline
│   │   ├── dataset.py            # Dataset classes
│   │   ├── dataloader.py         # Data loading
│   │   └── preprocessing.py      # Data preprocessing
│   ├── training/                 # Training infrastructure
│   │   ├── trainer.py            # Enhanced trainer with TRL-style APIs
│   │   ├── optimizer.py          # Standard optimizers
│   │   └── scheduler.py          # Learning rate schedulers
│   ├── kernels/                  # Kernel optimizations
│   │   ├── fused_ops.py          # Fused operations
│   │   └── memory_efficient.py   # Memory-efficient operations
│   ├── patching/                 # Patching system
│   │   ├── patch_transformers.py # Transformers patching
│   │   └── patch_trl.py          # TRL patching
│   ├── utils/                    # Utilities
│   │   ├── generation.py         # Text generation
│   │   ├── inference.py          # Inference utilities
│   │   └── metrics.py            # Evaluation metrics
│   └── config/                   # Configuration
│       ├── model_config.py       # Model configuration
│       └── training_config.py    # Training configuration
├── scripts/                      # CLI tools
│   ├── train.py                  # Training script
│   ├── generate.py               # Text generation
│   └── evaluate.py               # Model evaluation
├── configs/                      # Pre-configured setups
│   ├── small_model.yaml          # Small GPU model
│   ├── medium_model.yaml         # Medium GPU model
│   ├── cpu_small_model.yaml      # CPU-optimized small
│   └── cpu_medium_model.yaml     # CPU-optimized medium
├── examples/                     # Usage examples
│   ├── complete_pipeline.py      # End-to-end example
│   ├── safetensors_example.py    # SafeTensors demo
│   └── train_small_model.py      # Quick start example
└── docs/                         # Documentation
```

## Documentation

- [Getting Started Guide](docs/getting_started.md) — Complete setup and first steps
- [Model Architecture](docs/architecture.md) — Transformer implementation details
- [Training Guide](docs/training.md) — Comprehensive training tutorial
- [CPU Training Guide](docs/cpu_training.md) — Dedicated CPU training documentation
- [Tokenizer Details](docs/tokenizer.md) — BPE tokenizer documentation
- [API Reference](docs/api.md) — Complete API documentation

## Development

### Running Tests
```bash
pip install -e ".[dev]"
pytest tests/
```

### Code Quality
```bash
black src/ scripts/ examples/
flake8 src/ scripts/ examples/
mypy src/
```

## Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## Support

- **Bug Reports**: [GitHub Issues](https://github.com/HelpingAI/llm-trainer/issues)
- **Discussions**: [GitHub Discussions](https://github.com/HelpingAI/llm-trainer/discussions)
- **Documentation**: [Read the Docs](https://github.com/HelpingAI/llm-trainer/tree/main/docs)