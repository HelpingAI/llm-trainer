# 🚀 LLM Trainer: Train Large Language Models from Scratch

<div align="center">

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)](https://opensource.org/licenses/Apache-2.0)
[![GitHub](https://img.shields.io/badge/GitHub-HelpingAI/llm--trainer-black.svg)](https://github.com/HelpingAI/llm-trainer)
[![SafeTensors](https://img.shields.io/badge/SafeTensors-Supported-brightgreen.svg)](https://github.com/huggingface/safetensors)
[![Version](https://img.shields.io/badge/version-0.2.3-blue.svg)](https://github.com/HelpingAI/llm-trainer/releases)

*A comprehensive, production-ready framework for training Large Language Models from scratch with modern PyTorch*

</div>

---

## 🔔 What's New in v0.2.3

### 🛡️ SafeTensors Support (NEW!)
- **Secure Model Serialization**: Save and load models using SafeTensors format for enhanced security
- **Automatic Sharding**: Large models are automatically sharded with customizable shard sizes
- **Format Auto-Detection**: Seamlessly loads both SafeTensors and PyTorch formats
- **Backward Compatible**: Existing PyTorch models continue to work

### 🚀 Advanced Training Features
- **HuggingFace Integration**: Use any HF tokenizer via `HFTokenizerWrapper`
- **Accelerate Support**: Distributed training with `use_accelerate=true`
- **LoRA/PEFT**: Parameter-efficient fine-tuning with `use_peft=true`
- **Custom Architectures**: Implement `BaseLanguageModel` for your own models

### 💾 Enhanced Model Management
```python
# SafeTensors saving with sharding
model.save_pretrained("./my_model", safe_serialization=True, max_shard_size="2GB")

# Automatic format detection
model = TransformerLM.from_pretrained("./my_model")  # Auto-detects format
```

### ⚙️ Streamlined Configuration
```yaml
training:
  use_accelerate: true
  accelerate_mixed_precision: "fp16"
  use_peft: true
  peft_type: "lora"
  peft_r: 8
  peft_alpha: 16
```

## 📚 Table of Contents

- Features
- Requirements
- Installation
- Quick Start
  - Python API
  - Command Line
  - Complete Pipeline
  - Using a Hugging Face Tokenizer/Model
- Configuration
- Project Structure
- Documentation
- Development
- Contributing
- License
- Acknowledgments
- Support

## ✨ Core Features

### 🏗️ **Advanced Architecture**
- **Custom Transformer Implementation**: Multi-head attention, feed-forward networks, positional encodings
- **SafeTensors Integration**: Secure model serialization with automatic sharding for large models
- **Modular Design**: Easy to extend and customize for research and production

### 🔤 **Tokenization Excellence**
- **BPE Tokenizer**: From-scratch BPE with Unicode and emoji support
- **HuggingFace Integration**: Use any pretrained tokenizer (Mistral, Llama, GPT-2, etc.)
- **WordPiece Support**: Alternative tokenization strategies

### 📊 **Robust Data Pipeline**
- **HuggingFace Datasets**: Efficient loading with preprocessing and batching
- **Memory Optimization**: Smart sequence packing and data streaming
- **Multi-Processing**: Parallel data preprocessing for faster training

### 💻 **Flexible Training**
- **CPU/GPU Support**: Optimized configurations for both CPU and GPU training
- **Distributed Training**: Multi-GPU support via Accelerate and DeepSpeed
- **Parameter-Efficient**: LoRA/PEFT adapters for memory-efficient fine-tuning
- **Mixed Precision**: FP16/BF16 automatic mixed precision for faster training

### 🎯 **Advanced Inference**
- **Multiple Decoding Strategies**: Greedy, beam search, nucleus (top-p), top-k sampling
- **Interactive Generation**: Real-time text generation with customizable parameters
- **Batch Inference**: Efficient batch processing for production workloads

### 📈 **Comprehensive Monitoring**
- **TensorBoard Integration**: Real-time training metrics and visualizations
- **Weights & Biases**: Experiment tracking and hyperparameter optimization
- **Rich Logging**: Detailed progress tracking with configurable verbosity

### 🔧 **Production Ready**
- **Flexible Configuration**: YAML/JSON configs for reproducible experiments
- **Model Evaluation**: Comprehensive metrics including perplexity and generation quality
- **Checkpoint Management**: Automatic saving, loading, and resuming of training
- **Deployment Utilities**: Easy model export and serving preparation

## ✅ Requirements

- Python 3.8 or higher
- PyTorch 2.0 or higher
- GPU: CUDA-compatible GPU (recommended) or CPU-only mode
- Memory: 8GB RAM minimum (16GB+ recommended)

## 🧩 Installation

### Quick Install
```bash
# Clone the repository
git clone https://github.com/HelpingAI/llm-trainer.git
cd llm-trainer

# Create and activate virtual environment
python -m venv venv

# Windows
.\venv\Scripts\activate

# macOS/Linux
source venv/bin/activate

# Install the package
pip install -e .
```

### Optional Dependencies

```bash
# Development tools (tests, linters, type checking)
pip install -e ".[dev]"

# SafeTensors support (recommended for secure model saving)
pip install -e ".[safetensors]"

# Distributed training with DeepSpeed
pip install -e ".[distributed]"

# All features
pip install -e ".[full]"

# Individual extras
pip install peft  # For LoRA/PEFT support
pip install apex  # For NVIDIA mixed precision (optional)
```

### System Requirements
- **Python**: 3.8 or higher
- **PyTorch**: 2.0 or higher
- **Memory**: 8GB RAM minimum (16GB+ recommended)
- **GPU**: CUDA-compatible GPU (optional, CPU training supported)
- **Storage**: 2GB+ free space for model checkpoints

## 🚀 Quick Start

### 🐍 Option 1: Using the Python API

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

### 🤗 Using HuggingFace Pretrained Models

Seamlessly integrate with HuggingFace's ecosystem:

```python
from llm_trainer.tokenizer import HFTokenizerWrapper
from llm_trainer.models import HuggingFaceModelWrapper
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load pretrained tokenizer and model
tokenizer = HFTokenizerWrapper("microsoft/DialoGPT-medium")
model = HuggingFaceModelWrapper("microsoft/DialoGPT-medium")

# Or use with popular models
tokenizer = HFTokenizerWrapper("mistralai/Mistral-7B-Instruct-v0.2")
model = HuggingFaceModelWrapper("mistralai/Mistral-7B-Instruct-v0.2")

# Train with LLM Trainer
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

### 💻 Option 2: Using the Command Line

#### GPU Training (Faster)
```powershell
# Train a model using GPU configuration
python scripts/train.py --config configs/small_model.yaml --output_dir ./output

# Train medium model on GPU
python scripts/train.py --config configs/medium_model.yaml --output_dir ./output
```

#### CPU Training (Accessible - No GPU Required!)
```powershell
# Train small model on CPU (recommended for CPU)
python scripts/train.py --config configs/cpu_small_model.yaml --output_dir ./output/cpu_small

# Train medium model on CPU (slower but higher quality)
python scripts/train.py --config configs/cpu_medium_model.yaml --output_dir ./output/cpu_medium
```

#### Text Generation and Evaluation
```powershell
# Generate text interactively (works with both CPU and GPU trained models)
python scripts/generate.py --model_path ./output --prompts "The quick brown fox" --interactive

# Evaluate model performance
python scripts/evaluate.py --model_path ./output --dataset_config configs/eval_config.json
```

> Tip: New to LLM training? Start with `configs/cpu_small_model.yaml` for accessible CPU training, then move to `configs/small_model.yaml` when you have GPU access.

### 🔄 Option 3: Complete Pipeline Example

```powershell
# Run the complete pipeline (tokenizer + training + evaluation)
python examples/complete_pipeline.py

# Train a small model quickly
python examples/train_small_model.py
```

> The complete pipeline includes tokenizer training, model training, text generation, and evaluation metrics.

## 🔧 Configuration

The framework uses YAML/JSON configuration files for easy experimentation:

### Small Model (Quick Start)
```yaml
# configs/small_model.yaml
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
# configs/cpu_small_model.yaml
device: "cpu"
model:
  d_model: 256
  n_heads: 4
  n_layers: 4
  max_seq_len: 512

training:
  batch_size: 2
  use_amp: false  # Disabled for CPU
  gradient_accumulation_steps: 8
  dataloader_num_workers: 2
```

### Advanced Configuration with SafeTensors
```yaml
# Advanced training setup
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
  
# SafeTensors settings
save_format: "safetensors"
max_shard_size: "2GB"
```

> 💡 **Tip**: Start with CPU configs for experimentation, then scale to GPU for production training.

## 🗂️ Project Structure

```
llm-trainer/
├── 📦 src/llm_trainer/              # Main package
│   ├── 🏗️ models/                   # Model architectures
│   │   ├── base_model.py            # Base model interface
│   │   ├── transformer.py           # Custom Transformer implementation
│   │   ├── safetensors_utils.py     # SafeTensors utilities
│   │   └── attention.py             # Attention mechanisms
│   ├── 🔤 tokenizer/                # Tokenization
│   │   ├── bpe_tokenizer.py         # BPE implementation
│   │   ├── hf_tokenizer.py          # HuggingFace wrapper
│   │   └── wordpiece_tokenizer.py   # WordPiece implementation
│   ├── 📊 data/                     # Data pipeline
│   │   ├── dataset.py               # Dataset classes
│   │   ├── dataloader.py            # Data loading
│   │   └── preprocessing.py         # Data preprocessing
│   ├── 🚀 training/                 # Training infrastructure
│   │   ├── trainer.py               # Main training logic
│   │   ├── optimizer.py             # Optimizers
│   │   └── scheduler.py             # Learning rate schedulers
│   ├── 🛠️ utils/                    # Utilities
│   │   ├── generation.py            # Text generation
│   │   ├── inference.py             # Inference utilities
│   │   └── metrics.py               # Evaluation metrics
│   └── ⚙️ config/                   # Configuration
│       ├── model_config.py          # Model configuration
│       ├── training_config.py       # Training configuration
│       └── data_config.py           # Data configuration
├── 📜 scripts/                      # CLI tools
│   ├── train.py                     # Training script
│   ├── generate.py                  # Text generation
│   └── evaluate.py                  # Model evaluation
├── ⚙️ configs/                      # Pre-configured setups
│   ├── small_model.yaml             # Small GPU model
│   ├── medium_model.yaml            # Medium GPU model
│   ├── cpu_small_model.yaml         # CPU-optimized small
│   └── cpu_medium_model.yaml        # CPU-optimized medium
├── 📖 examples/                     # Usage examples
│   ├── complete_pipeline.py         # End-to-end example
│   ├── safetensors_example.py       # SafeTensors demo
│   └── train_small_model.py         # Quick start example
└── 📚 docs/                         # Documentation
```

## 📚 Documentation

- 📖 [Getting Started Guide](docs/getting_started.md) — Complete setup and first steps
- 🏗️ [Model Architecture](docs/architecture.md) — Transformer implementation details
- 🚀 [Training Guide](docs/training.md) — Comprehensive training tutorial (includes CPU training)
- 💻 [CPU Training Guide](docs/cpu_training.md) — Dedicated CPU training documentation
- 🔤 [Tokenizer Details](docs/tokenizer.md) — BPE tokenizer documentation
- 📋 [API Reference](docs/api.md) — Complete API documentation

## 🎯 Key Features Deep Dive

### 🛡️ SafeTensors Integration
- **Secure Serialization**: Protection against arbitrary code execution
- **Automatic Sharding**: Large models split across multiple files
- **Metadata Preservation**: Training statistics and model configuration
- **Format Auto-Detection**: Seamless loading of both formats

```python
# Save with SafeTensors (recommended)
model.save_pretrained("./model", safe_serialization=True)

# Load automatically detects format
model = TransformerLM.from_pretrained("./model")
```

### 🔤 Advanced Tokenization
- **BPE from Scratch**: Unicode, emoji, and multilingual support
- **HuggingFace Integration**: Use any pretrained tokenizer
- **WordPiece Support**: Alternative subword tokenization
- **Efficient Training**: Fast BPE with dataset streaming

### 🏗️ Modern Transformer Architecture
- **Multi-Head Attention**: Scaled dot-product with causal masking
- **Flexible Normalization**: Pre-norm/post-norm configurations
- **Positional Encodings**: Sinusoidal or learned embeddings
- **Memory Efficient**: Gradient checkpointing and optimizations

### 🚀 Production-Grade Training
- **Distributed Training**: Multi-GPU with Accelerate/DeepSpeed
- **Mixed Precision**: FP16/BF16 automatic mixed precision
- **Parameter-Efficient**: LoRA/PEFT adapters for fine-tuning
- **Robust Checkpointing**: Save/resume with full state recovery

### 🎯 Advanced Inference
- **Multiple Strategies**: Greedy, beam, nucleus (top-p), top-k
- **Batch Processing**: Efficient inference for production
- **Interactive Generation**: Real-time text generation
- **Customizable Sampling**: Temperature, repetition penalty, length control

### 📊 Comprehensive Evaluation
- **Language Modeling Metrics**: Perplexity, cross-entropy loss
- **Generation Quality**: Diversity, repetition, coherence metrics
- **Custom Metrics**: Extensible evaluation framework
- **Visualization**: TensorBoard and W&B integration

## 🛠️ Development

### Running Tests
```bash
# Install development dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/

# Run with coverage
pytest tests/ --cov=llm_trainer
```

### Code Formatting
```bash
# Format code
black src/ scripts/ examples/

# Check style
flake8 src/ scripts/ examples/

# Type checking
mypy src/
```

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Attention Is All You Need** - Original Transformer paper
- **Hugging Face** - For the excellent datasets and tokenizers library
- **PyTorch Team** - For the amazing deep learning framework

## 📞 Support & Community

- 🐛 **Bug Reports**: [GitHub Issues](https://github.com/HelpingAI/llm-trainer/issues)
- 💬 **Discussions**: [GitHub Discussions](https://github.com/HelpingAI/llm-trainer/discussions)
- 📧 **Contact**: [HelpingAI Team](mailto:helpingai5@gmail.com)
- 📖 **Documentation**: [Read the Docs](https://github.com/HelpingAI/llm-trainer/tree/main/docs)
- 🚀 **Feature Requests**: [GitHub Issues](https://github.com/HelpingAI/llm-trainer/issues/new)

## 🏆 Performance Benchmarks

| Model Size | Parameters | Training Time (GPU) | Memory Usage | CPU Training |
|------------|------------|-------------------|--------------|---------------|
| Small      | ~25M       | 2-4 hours         | 4GB VRAM     | ✅ Supported  |
| Medium     | ~100M      | 8-12 hours        | 8GB VRAM     | ✅ Supported  |
| Large      | ~350M      | 24-48 hours       | 16GB VRAM    | ⚠️ Limited     |

## 🔄 Migration Guide

### From v0.2.2 to v0.2.3
- **SafeTensors**: Models are now saved in SafeTensors format by default
- **Backward Compatible**: Existing PyTorch models continue to work
- **New Features**: Automatic sharding for large models

```python
# Old way (still works)
model.save_state_dict("model.pt")

# New way (recommended)
model.save_pretrained("./model", safe_serialization=True)
```

---

<div align="center">

**⭐ Star this repository if you find it helpful!**

**🙏 Built with ❤️ by the [HelpingAI](https://github.com/HelpingAI) team**

[![GitHub stars](https://img.shields.io/github/stars/HelpingAI/llm-trainer?style=social)](https://github.com/HelpingAI/llm-trainer/stargazers)
[![GitHub forks](https://img.shields.io/github/forks/HelpingAI/llm-trainer?style=social)](https://github.com/HelpingAI/llm-trainer/network)

</div>