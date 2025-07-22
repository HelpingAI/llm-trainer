# LLM Trainer: Train Large Language Models from Scratch

A comprehensive framework for training Large Language Models (LLMs) from scratch using PyTorch. This project implements a complete pipeline including custom tokenizer, Transformer architecture, training infrastructure, and inference capabilities.

## Features

- 🏗️ **Custom Transformer Architecture**: Complete implementation from scratch with multi-head attention, feed-forward networks, and positional encoding
- 🔤 **BPE Tokenizer**: Byte Pair Encoding tokenizer implemented from scratch
- 📊 **Data Pipeline**: Efficient data loading from Hugging Face datasets with preprocessing and batching
- 🚀 **Training Infrastructure**: Distributed training support, gradient accumulation, and checkpointing
- 🎯 **Inference Engine**: Text generation with multiple decoding strategies
- 📈 **Monitoring**: Integration with TensorBoard and Weights & Biases
- ⚡ **Performance**: Mixed precision training and memory optimization

## Quick Start

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd llm-trainer

# Install dependencies
pip install -r requirements.txt

# Install the package in development mode
pip install -e .
```

### Option 1: Using Python API

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

### Option 2: Using Command Line

```bash
# Train a model
python scripts/train.py --config examples/training_config.json --output_dir ./output

# Generate text
python scripts/generate.py --model_path ./output --prompts "The quick brown fox" --interactive

# Evaluate model
python scripts/evaluate.py --model_path ./output --eval_config examples/evaluation_config.json
```

### Option 3: Complete Pipeline Example

```bash
# Run the complete pipeline (tokenizer + training + evaluation)
python examples/complete_pipeline.py
```

## Project Structure

```
llm-trainer/
├── src/llm_trainer/           # Main package
│   ├── models/                # Transformer architecture
│   ├── tokenizer/             # BPE tokenizer implementation
│   ├── data/                  # Data loading and preprocessing
│   ├── training/              # Training infrastructure
│   ├── utils/                 # Utility functions
│   └── config/                # Configuration classes
├── scripts/                   # Training and inference scripts
├── configs/                   # Configuration files
├── tests/                     # Unit tests
├── examples/                  # Usage examples
└── docs/                      # Documentation
```

## Documentation

- [Model Architecture](docs/architecture.md)
- [Training Guide](docs/training.md)
- [Tokenizer Details](docs/tokenizer.md)
- [API Reference](docs/api.md)

## License

MIT License - see [LICENSE](LICENSE) file for details.
