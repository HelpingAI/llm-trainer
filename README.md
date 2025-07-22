# 🚀 LLM Trainer: Train Large Language Models from Scratch

<div align="center">

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)](https://opensource.org/licenses/Apache-2.0)
[![GitHub](https://img.shields.io/badge/GitHub-OEvortex/llm--trainer-black.svg)](https://github.com/OEvortex/llm-trainer)

*A comprehensive framework for training Large Language Models (LLMs) from scratch using PyTorch*

</div>

---

## ✨ Features

- 🏗️ **Custom Transformer Architecture**: Complete implementation from scratch with multi-head attention, feed-forward networks, and positional encoding
- 🔤 **BPE Tokenizer**: Byte Pair Encoding tokenizer implemented from scratch with Unicode and emoji support
- 📊 **Data Pipeline**: Efficient data loading from Hugging Face datasets with preprocessing and batching
- 🚀 **Training Infrastructure**: Distributed training support, gradient accumulation, and checkpointing
- 🎯 **Inference Engine**: Text generation with multiple decoding strategies (greedy, beam search, sampling)
- 📈 **Monitoring**: Integration with TensorBoard and Weights & Biases
- ⚡ **Performance**: Mixed precision training, gradient checkpointing, and memory optimization
- 🔧 **Flexible Configuration**: YAML/JSON configuration files for easy experimentation
- 📦 **Production Ready**: Model saving/loading, evaluation metrics, and deployment utilities

## 🚀 Quick Start

### 📋 Prerequisites

> [!IMPORTANT]
> - Python 3.8 or higher
> - PyTorch 2.0 or higher
> - CUDA-compatible GPU (recommended for training)
> - At least 8GB RAM (16GB+ recommended)

### 📦 Installation

```bash
# Clone the repository
git clone https://github.com/OEvortex/llm-trainer.git
cd llm-trainer

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install the package in development mode
pip install -e .
```

> [!NOTE]
> For distributed training, install DeepSpeed: `pip install deepspeed>=0.9.0`

> [!WARNING]
> Training large models requires significant computational resources. Start with the small model configuration for experimentation.

### 🐍 Option 1: Using Python API

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

### 💻 Option 2: Using Command Line

```bash
# Train a model using configuration file
python scripts/train.py --config configs/small_model.yaml --output_dir ./output

# Generate text interactively
python scripts/generate.py --model_path ./output --prompts "The quick brown fox" --interactive

# Evaluate model performance
python scripts/evaluate.py --model_path ./output --dataset_config configs/eval_config.json
```

> [!TIP]
> Use `configs/small_model.yaml` for quick experimentation and `configs/medium_model.yaml` for better quality results.

### 🔄 Option 3: Complete Pipeline Example

```bash
# Run the complete pipeline (tokenizer + training + evaluation)
python examples/complete_pipeline.py

# Train a small model quickly
python examples/train_small_model.py
```

> [!NOTE]
> The complete pipeline includes tokenizer training, model training, text generation, and evaluation metrics.

## 📁 Project Structure

```
llm-trainer/
├── 📦 src/llm_trainer/           # Main package
│   ├── 🏗️ models/                # Transformer architecture
│   ├── 🔤 tokenizer/             # BPE tokenizer implementation
│   ├── 📊 data/                  # Data loading and preprocessing
│   ├── 🚀 training/              # Training infrastructure
│   ├── 🛠️ utils/                 # Utility functions
│   └── ⚙️ config/                # Configuration classes
├── 📜 scripts/                   # Training and inference scripts
├── ⚙️ configs/                   # Configuration files
├── 🧪 tests/                     # Unit tests
├── 📖 examples/                  # Usage examples
└── 📚 docs/                      # Documentation
```

## 📚 Documentation

- 📖 [Getting Started Guide](docs/getting_started.md) - Complete setup and first steps
- 🏗️ [Model Architecture](docs/architecture.md) - Transformer implementation details
- 🚀 [Training Guide](docs/training.md) - Comprehensive training tutorial
- 🔤 [Tokenizer Details](docs/tokenizer.md) - BPE tokenizer documentation
- 📋 [API Reference](docs/api.md) - Complete API documentation

## 🔧 Configuration

The framework uses YAML/JSON configuration files for easy experimentation:

### Small Model (Quick Start)
```yaml
# configs/small_model.yaml
model:
  d_model: 256
  n_heads: 4
  n_layers: 4
  vocab_size: 32000

training:
  batch_size: 16
  learning_rate: 5e-4
  num_epochs: 5
```

### Medium Model (Better Quality)
```yaml
# configs/medium_model.yaml
model:
  d_model: 768
  n_heads: 12
  n_layers: 12
  vocab_size: 50000

training:
  batch_size: 8
  learning_rate: 1e-4
  gradient_accumulation_steps: 8
```

> [!CAUTION]
> Large models require significant GPU memory. Monitor your system resources during training.

## 🎯 Key Features Explained

### 🔤 Advanced BPE Tokenizer
- **Unicode Support**: Handles international characters, emojis, and symbols
- **Efficient Training**: Fast BPE algorithm with dataset streaming
- **Special Tokens**: Configurable special tokens (PAD, UNK, BOS, EOS)
- **Compatibility**: Works with Hugging Face datasets

### 🏗️ Transformer Architecture
- **Multi-Head Attention**: Scaled dot-product attention with causal masking
- **Feed-Forward Networks**: Position-wise feed-forward with configurable activation
- **Layer Normalization**: Pre-norm and post-norm support
- **Positional Encoding**: Sinusoidal and learned positional embeddings

### 🚀 Training Infrastructure
- **Distributed Training**: Multi-GPU support with DDP and DeepSpeed
- **Mixed Precision**: Automatic mixed precision with FP16/BF16
- **Gradient Accumulation**: Memory-efficient training for large batches
- **Checkpointing**: Automatic saving and resuming from checkpoints

### 🎯 Text Generation
- **Multiple Strategies**: Greedy, beam search, nucleus sampling, top-k sampling
- **Temperature Control**: Fine-tune randomness in generation
- **Repetition Penalty**: Reduce repetitive outputs
- **Interactive Mode**: Real-time text generation interface

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

## 📞 Support

- 🐛 **Bug Reports**: [GitHub Issues](https://github.com/OEvortex/llm-trainer/issues)
- 💬 **Discussions**: [GitHub Discussions](https://github.com/OEvortex/llm-trainer/discussions)
- 📧 **Contact**: [Your Email](mailto:abhay@helpingai.co)

---

<div align="center">

**⭐ Star this repository if you find it helpful!**

Made with ❤️ by [OEvortex](https://github.com/OEvortex)

</div>