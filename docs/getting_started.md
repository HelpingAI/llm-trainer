# 🚀 Getting Started with LLM Trainer

Welcome to LLM Trainer! This comprehensive guide will walk you through everything you need to know to start training your own Large Language Models from scratch.

## 📋 Prerequisites & System Requirements

> [!IMPORTANT]
> **Minimum Requirements:**
> - Python 3.8 or higher
> - PyTorch 2.0 or higher
> - 8GB RAM (16GB+ recommended)
> - 10GB free disk space

> [!NOTE]
> **Recommended for Training:**
> - CUDA-compatible GPU with 8GB+ VRAM
> - 32GB+ system RAM
> - SSD storage for faster data loading

### 🖥️ Hardware Recommendations

| Model Size | GPU Memory | System RAM | Training Time* |
|------------|------------|------------|----------------|
| Small (25M) | 4GB+ | 8GB+ | 2-4 hours |
| Medium (117M) | 8GB+ | 16GB+ | 8-12 hours |
| Large (345M) | 16GB+ | 32GB+ | 24-48 hours |

*Approximate times on modern hardware

## 📦 Installation

### Step 1: Clone the Repository

```bash
# Clone from GitHub
git clone https://github.com/OEvortex/llm-trainer.git
cd llm-trainer

# Verify the installation
ls -la

### Step 2: Set Up Environment

```bash
# Create virtual environment (highly recommended)
python -m venv llm-trainer-env
source llm-trainer-env/bin/activate  # On Windows: llm-trainer-env\Scripts\activate

# Upgrade pip
pip install --upgrade pip
```

### Step 3: Install Dependencies

```bash
# Install core dependencies
pip install -r requirements.txt

# Install the package in development mode
pip install -e .

# Verify installation
python -c "import llm_trainer; print('✅ Installation successful!')"
```

> [!TIP]
> **Optional Dependencies:**
> ```bash
> # For distributed training
> pip install deepspeed>=0.9.0
> 
> # For advanced mixed precision
> pip install apex
> 
> # For development
> pip install -e ".[dev]"
> ```

### Optional Dependencies

For additional features, install these optional packages:

```bash
# For Weights & Biases logging
pip install wandb

# For ROUGE metrics
pip install rouge-score

# For semantic similarity evaluation
pip install sentence-transformers

# For distributed training
pip install deepspeed
```

## Quick Start

### 1. Train a Small Model

The fastest way to get started is to train a small model on a subset of data:

```python
from llm_trainer.config import ModelConfig, TrainingConfig, DataConfig
from llm_trainer.models import TransformerLM
from llm_trainer.tokenizer import BPETokenizer
from llm_trainer.training import Trainer

# Configure a small model
model_config = ModelConfig(
    vocab_size=32000,
    d_model=256,
    n_heads=4,
    n_layers=4,
    max_seq_len=512
)

# Configure training
training_config = TrainingConfig(
    batch_size=16,
    learning_rate=5e-4,
    num_epochs=3,
    checkpoint_dir="./checkpoints"
)

# Configure data
data_config = DataConfig(
    dataset_name="wikitext",
    dataset_config="wikitext-2-raw-v1",
    max_length=512
)

# Create and train tokenizer
tokenizer = BPETokenizer()
tokenizer.train_from_dataset(
    dataset_name="wikitext",
    dataset_config="wikitext-2-raw-v1",
    vocab_size=32000
)

# Create model
model = TransformerLM(model_config)

# Create trainer and train
trainer = Trainer(model, tokenizer, training_config)
trainer.train_from_config(model_config, data_config)
```

### 2. Using Command Line Scripts

You can also use the provided command-line scripts:

```bash
# Train a model
python scripts/train.py --config examples/training_config.json --output_dir ./output

# Generate text
python scripts/generate.py --model_path ./output --prompts "The quick brown fox" --interactive

# Evaluate model
python scripts/evaluate.py --model_path ./output --eval_config examples/evaluation_config.json
```

### 3. Text Generation

Once you have a trained model, you can generate text:

```python
from llm_trainer.utils.generation import TextGenerator, GenerationConfig

# Load model and tokenizer
model = TransformerLM.from_pretrained("./output")
tokenizer = BPETokenizer.from_pretrained("./output")

# Create generator
generator = TextGenerator(model, tokenizer)

# Configure generation
config = GenerationConfig(
    max_length=100,
    temperature=0.8,
    top_p=0.9,
    do_sample=True
)

# Generate text
generated_text = generator.generate("The quick brown fox", config)
print(generated_text[0])
```

## Project Structure

```
llm-trainer/
├── src/llm_trainer/           # Main package
│   ├── models/                # Transformer architecture
│   │   ├── transformer.py     # Main model class
│   │   ├── attention.py       # Multi-head attention
│   │   ├── layers.py          # Transformer layers
│   │   └── embeddings.py      # Token and positional embeddings
│   ├── tokenizer/             # BPE tokenizer implementation
│   │   ├── bpe_tokenizer.py   # Main tokenizer class
│   │   └── base_tokenizer.py  # Base tokenizer interface
│   ├── data/                  # Data loading and preprocessing
│   │   ├── dataset.py         # Dataset classes
│   │   ├── dataloader.py      # Data loading utilities
│   │   └── preprocessing.py   # Text preprocessing
│   ├── training/              # Training infrastructure
│   │   ├── trainer.py         # Main trainer class
│   │   ├── optimizer.py       # Optimizer utilities
│   │   ├── scheduler.py       # Learning rate schedulers
│   │   └── utils.py           # Training utilities
│   ├── utils/                 # Utility functions
│   │   ├── generation.py      # Text generation
│   │   ├── inference.py       # Inference engine
│   │   └── metrics.py         # Evaluation metrics
│   └── config/                # Configuration classes
│       ├── model_config.py    # Model configuration
│       ├── training_config.py # Training configuration
│       └── data_config.py     # Data configuration
├── scripts/                   # Command-line scripts
│   ├── train.py              # Training script
│   ├── generate.py           # Generation script
│   └── evaluate.py           # Evaluation script
├── configs/                   # Configuration files
├── examples/                  # Usage examples
└── docs/                     # Documentation
```

## Key Features

### 🏗️ Custom Transformer Architecture
- Complete implementation from scratch
- Multi-head attention with causal masking
- Feed-forward networks with various activations
- Layer normalization (pre-norm and post-norm)
- Positional encoding (sinusoidal and learned)

### 🔤 BPE Tokenizer
- Byte Pair Encoding implemented from scratch
- Vocabulary building from datasets
- Efficient encoding and decoding
- Special token handling

### 📊 Data Pipeline
- Hugging Face datasets integration
- Text preprocessing and filtering
- Sequence packing for efficiency
- Streaming support for large datasets

### 🚀 Training Infrastructure
- Distributed training support
- Mixed precision training
- Gradient accumulation
- Learning rate scheduling
- Checkpointing and resuming

### 🎯 Inference Engine
- Multiple decoding strategies (greedy, beam search, sampling)
- Streaming generation
- Batch processing
- Performance optimization

### 📈 Monitoring and Evaluation
- TensorBoard and Weights & Biases integration
- Comprehensive metrics (perplexity, BLEU, diversity)
- Generation quality evaluation

## Next Steps

- [Training Guide](training.md) - Detailed training instructions
- [Model Architecture](architecture.md) - Understanding the Transformer implementation
- [Configuration Reference](configuration.md) - All configuration options
- [API Reference](api.md) - Complete API documentation
- [Examples](../examples/) - More usage examples

## Common Issues

### CUDA Out of Memory
- Reduce batch size
- Enable gradient checkpointing
- Use gradient accumulation
- Enable mixed precision training

### Slow Training
- Use multiple GPUs with distributed training
- Enable model compilation (PyTorch 2.0+)
- Optimize data loading (more workers, pin memory)
- Use sequence packing

### Poor Generation Quality
- Train for more epochs
- Increase model size
- Tune generation parameters (temperature, top-p)
- Use better datasets

For more detailed information, see the specific documentation pages.
