# LLM Trainer

<div align="center">

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)](https://opensource.org/licenses/Apache-2.0)
[![GitHub](https://img.shields.io/badge/GitHub-HelpingAI/llm--trainer-black.svg)](https://github.com/HelpingAI/llm-trainer)
[![SafeTensors](https://img.shields.io/badge/SafeTensors-Supported-brightgreen.svg)](https://github.com/huggingface/safetensors)
[![Version](https://img.shields.io/badge/version-0.2.3-blue.svg)](https://github.com/HelpingAI/llm-trainer/releases)

*A production-ready framework for training Large Language Models from scratch with modern PyTorch*

</div>

## What's New in v0.2.3

- **SafeTensors Support**: Secure model serialization with automatic sharding for large models
- **HuggingFace Integration**: Use any pretrained tokenizer via `HFTokenizerWrapper`
- **Accelerate Support**: Distributed training with `use_accelerate=true`
- **LoRA/PEFT**: Parameter-efficient fine-tuning with `use_peft=true`
- **Backward Compatible**: Existing PyTorch models continue to work

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

### Python API

```python
from llm_trainer.config import ModelConfig, TrainingConfig, DataConfig
from llm_trainer.models import TransformerLM
from llm_trainer.tokenizer import BPETokenizer
from llm_trainer.training import Trainer

# Create and train tokenizer
tokenizer = BPETokenizer()
tokenizer.train_from_dataset(
    dataset_name="wikitext",
    dataset_config="wikitext-2-raw-v1",
    vocab_size=32000
)

# Configure model
model_config = ModelConfig(
    vocab_size=tokenizer.vocab_size,
    d_model=512,
    n_heads=8,
    n_layers=6,
    max_seq_len=1024
)

# Create model
model = TransformerLM(model_config)

# Configure training
training_config = TrainingConfig(
    batch_size=16,
    learning_rate=1e-4,
    num_epochs=3,
    warmup_steps=1000,
    checkpoint_dir="./checkpoints"
)

# Configure data
data_config = DataConfig(
    dataset_name="wikitext",
    dataset_config="wikitext-2-raw-v1",
    max_length=1024
)

# Train the model
trainer = Trainer(model, tokenizer, training_config)
trainer.train_from_config(model_config, data_config)
```

### HuggingFace Integration

```python
from llm_trainer.tokenizer import HFTokenizerWrapper
from llm_trainer.models import HuggingFaceModelWrapper

# Load pretrained tokenizer and model
tokenizer = HFTokenizerWrapper("microsoft/DialoGPT-medium")
model = HuggingFaceModelWrapper("microsoft/DialoGPT-medium")

# Configure PEFT training
training_config = TrainingConfig(
    use_accelerate=True,
    use_peft=True,
    peft_type="lora",
    peft_r=8,
    peft_alpha=16
)

trainer = Trainer(model, tokenizer, training_config)
trainer.train_from_config(model_config, data_config)
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
│   │   ├── trainer.py            # Main training logic
│   │   ├── optimizer.py          # Optimizers
│   │   └── scheduler.py          # Learning rate schedulers
│   ├── utils/                    # Utilities
│   │   ├── generation.py         # Text generation
│   │   ├── inference.py          # Inference utilities
│   │   └── metrics.py            # Evaluation metrics
│   └── config/                   # Configuration
│       ├── model_config.py       # Model configuration
│       ├── training_config.py    # Training configuration
│       └── data_config.py        # Data configuration
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