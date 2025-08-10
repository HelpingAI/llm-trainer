# 🚀 LLM Trainer: Train Large Language Models from Scratch

<div align="center">

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)](https://opensource.org/licenses/Apache-2.0)
[![GitHub](https://img.shields.io/badge/GitHub-OEvortex/llm--trainer-black.svg)](https://github.com/OEvortex/llm-trainer)

*A comprehensive framework for training Large Language Models (LLMs) from scratch using PyTorch*

</div>

---

## 🔔 What's New

Optional compatibility with HF Transformers, Accelerate, and PEFT (LoRA):

- Use any Hugging Face tokenizer via `HFTokenizerWrapper`
- Wrap an HF Causal LM with `HuggingFaceModelWrapper` and train with this trainer
- Turn on Accelerate: set `use_accelerate=true` in `TrainingConfig`
- Apply LoRA adapters: set `use_peft=true` and `peft_*` fields in `TrainingConfig` (requires `peft`)

Minimal JSON training config:

```json
{
  "training": {
    "use_accelerate": true,
    "accelerate_mixed_precision": "fp16",
    "use_peft": true,
    "peft_type": "lora",
    "peft_r": 8,
    "peft_alpha": 16,
    "peft_dropout": 0.05
  }
}
```

Use your own architecture by implementing `BaseLanguageModel` (see `src/llm_trainer/models/base_model.py`) and passing it to `Trainer`.

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

## ✨ Features

- 🏗️ **Custom Transformer Architecture**: Multi-head attention, feed-forward networks, positional encodings
- 🔤 **BPE Tokenizer**: From-scratch BPE with Unicode and emoji support
- 📊 **Data Pipeline**: Efficient HF datasets loading with preprocessing and batching
- 💻 **CPU Training Support**: Optimized configs—no GPU required
- ⚙️ **Training Infrastructure**: Distributed support (GPU/CPU), grad accumulation, checkpointing
- 🎯 **Inference Engine**: Greedy/beam/nucleus/top-k decoding
- 📈 **Monitoring**: TensorBoard and Weights & Biases
- ⚡ **Performance**: Mixed precision (GPU), grad checkpointing, memory optimizations
- 🔧 **Flexible Configuration**: YAML/JSON configs for experiments
- 📦 **Production Ready**: Save/load, evaluation metrics, deployment utilities

## ✅ Requirements

- Python 3.8 or higher
- PyTorch 2.0 or higher
- GPU: CUDA-compatible GPU (recommended) or CPU-only mode
- Memory: 8GB RAM minimum (16GB+ recommended)

## 🧩 Installation

```powershell
# Clone the repository
git clone https://github.com/OEvortex/llm-trainer.git
cd llm-trainer

# Create and activate a virtual environment (Windows PowerShell)
python -m venv venv
./venv/Scripts/Activate.ps1

# Install the package (installs all core dependencies from setup.py)
pip install -e .

# Optional extras
# Development tools (tests, linters, type checking)
pip install -e ".[dev]"
# Distributed training (DeepSpeed)
pip install -e ".[distributed]"
# LoRA/PEFT support when using use_peft=true
pip install peft
```

> On macOS/Linux, activate the venv with: `source venv/bin/activate`.
>
> For mixed precision with NVIDIA Apex, install: `pip install -e ".[mixed-precision]"` (Apex must be available for your environment).

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

### 🤗 Using a Hugging Face Pretrained Tokenizer/Model

You can use a pretrained tokenizer from Hugging Face (e.g., Mistral, Llama, etc.) via `HFTokenizerWrapper`:

```python
from llm_trainer.tokenizer import HFTokenizerWrapper
from transformers import MistralConfig, MistralForCausalLM

# Load a pretrained tokenizer from Hugging Face
hf_tokenizer = HFTokenizerWrapper("mistralai/Mistral-7B-Instruct-v0.2")
hf_tokenizer.tokenizer.pad_token = hf_tokenizer.tokenizer.eos_token  # Set padding token if needed

# Configure your model (example: Mistral)
model_config = MistralConfig(
    vocab_size=hf_tokenizer.tokenizer.vocab_size,
    hidden_size=2048,
    intermediate_size=7168,
    num_hidden_layers=24,
    num_attention_heads=32,
    num_key_value_heads=8,
    hidden_act="silu",
    max_position_embeddings=4096,
    pad_token_id=hf_tokenizer.tokenizer.pad_token_id,
    bos_token_id=hf_tokenizer.tokenizer.bos_token_id,
    eos_token_id=hf_tokenizer.tokenizer.eos_token_id
)
model = MistralForCausalLM(model_config)

# Use hf_tokenizer in your Trainer or data pipeline as you would with any tokenizer
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

> Caution: Large models require significant GPU memory. Monitor your system resources during training.

## 🗂️ Project Structure

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

- 📖 [Getting Started Guide](docs/getting_started.md) — Complete setup and first steps
- 🏗️ [Model Architecture](docs/architecture.md) — Transformer implementation details
- 🚀 [Training Guide](docs/training.md) — Comprehensive training tutorial (includes CPU training)
- 💻 [CPU Training Guide](docs/cpu_training.md) — Dedicated CPU training documentation
- 🔤 [Tokenizer Details](docs/tokenizer.md) — BPE tokenizer documentation
- 📋 [API Reference](docs/api.md) — Complete API documentation

## 🎯 Key Features Explained

### 🔤 Advanced BPE Tokenizer
- Unicode support: international characters, emojis, and symbols
- Efficient training: fast BPE with dataset streaming
- Special tokens: PAD, UNK, BOS, EOS
- HF-compatible datasets

### 🏗️ Transformer Architecture
- Scaled dot-product attention with causal masking
- Position-wise feed-forward with configurable activation
- Pre-norm/post-norm layer normalization
- Sinusoidal or learned positional embeddings

### 🚀 Training Infrastructure
- Distributed training (DDP and DeepSpeed)
- Automatic mixed precision FP16/BF16
- Gradient accumulation for large batches
- Checkpointing (save/resume)

### ✍️ Text Generation
- Greedy, beam search, nucleus sampling, top-k sampling
- Temperature control and repetition penalty
- Interactive mode for real-time generation

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
- 📧 **Contact**: [Vortex](mailto:abhay@helpingai.co)

---

<div align="center">

**⭐ Star this repository if you find it helpful!**

Made with ❤️ by [OEvortex](https://github.com/OEvortex)

</div>