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
> **Recommended for GPU Training:**
> - CUDA-compatible GPU with 8GB+ VRAM
> - 32GB+ system RAM
> - SSD storage for faster data loading

> [!TIP]
> **CPU Training Alternative:**
> - No GPU required - works on any modern CPU
> - 8GB+ RAM for small models, 16GB+ for medium models
> - Longer training times but accessible to everyone

### 🖥️ Hardware Recommendations

#### GPU Training (Recommended for Production)

| Model Size | GPU Memory | System RAM | Training Time* |
|------------|------------|------------|----------------|
| Small (25M) | 4GB+ | 8GB+ | 2-4 hours |
| Medium (117M) | 8GB+ | 16GB+ | 8-12 hours |
| Large (345M) | 16GB+ | 32GB+ | 24-48 hours |

#### CPU Training (Accessible Alternative)

| Model Size | CPU Cores | System RAM | Training Time* |
|------------|-----------|------------|----------------|
| Small (25M) | 4-8 cores | 8GB+ | 8-24 hours |
| Medium (117M) | 8+ cores | 16GB+ | 2-7 days |
| Large (345M) | 16+ cores | 32GB+ | 1-2 weeks |

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

### 1. Choose Your Training Approach

#### Option A: GPU Training (Faster)
For users with CUDA-compatible GPUs:

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

# Configure GPU training
training_config = TrainingConfig(
    device="auto",  # Automatically selects GPU if available
    batch_size=16,
    learning_rate=5e-4,
    num_epochs=3,
    use_amp=True,  # Mixed precision for faster training
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

#### Option B: CPU Training (Accessible)
For users without GPUs or for development/testing:

```python
from llm_trainer.config import ModelConfig, TrainingConfig, DataConfig
from llm_trainer.models import TransformerLM
from llm_trainer.tokenizer import BPETokenizer
from llm_trainer.training import Trainer

# Configure a CPU-optimized small model
model_config = ModelConfig(
    vocab_size=32000,
    d_model=256,
    n_heads=4,
    n_layers=4,
    max_seq_len=512,
    gradient_checkpointing=False  # Disabled for CPU
)

# Configure CPU training
training_config = TrainingConfig(
    device="cpu",  # Explicit CPU selection
    batch_size=2,  # Smaller batch for CPU memory
    learning_rate=8e-4,  # Slightly higher for smaller batch
    num_epochs=3,
    gradient_accumulation_steps=8,  # Effective batch size = 16
    use_amp=False,  # Not supported on CPU
    dataloader_num_workers=2,  # Reduced workers
    dataloader_pin_memory=False,  # Disabled for CPU
    checkpoint_dir="./checkpoints"
)

# Configure data
data_config = DataConfig(
    dataset_name="wikitext",
    dataset_config="wikitext-2-raw-v1",
    max_length=512,
    preprocessing_num_workers=2  # Reduced for CPU
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

> [!TIP]
> **Quick CPU Training**: Use the pre-configured [`configs/cpu_small_model.yaml`](configs/cpu_small_model.yaml:1) for optimal CPU training settings.

### 2. Using Command Line Scripts

#### GPU Training (Default)
```bash
# Train a model with GPU (if available)
python scripts/train.py --config configs/small_model.yaml --output_dir ./output

# Train medium model with GPU
python scripts/train.py --config configs/medium_model.yaml --output_dir ./output
```

#### CPU Training (Accessible)
```bash
# Train small model on CPU (recommended)
python scripts/train.py --config configs/cpu_small_model.yaml --output_dir ./output/cpu_small

# Train medium model on CPU (slower but higher quality)
python scripts/train.py --config configs/cpu_medium_model.yaml --output_dir ./output/cpu_medium
```

#### Text Generation and Evaluation
```bash
# Generate text (works with both CPU and GPU trained models)
python scripts/generate.py --model_path ./output --prompts "The quick brown fox" --interactive

# Evaluate model performance
python scripts/evaluate.py --model_path ./output --eval_config examples/evaluation_config.json
```

> [!NOTE]
> **CPU vs GPU Training**: CPU configurations use smaller batch sizes, disabled AMP, and optimized worker settings for better CPU performance.

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

### How to Choose Between CPU and GPU Training

#### Use CPU Training When:
- **No GPU Available**: Your system doesn't have a CUDA-compatible GPU
- **Development/Testing**: Quick experimentation and code validation
- **Small Models**: Training models with <50M parameters
- **Learning**: Understanding the training process without GPU costs
- **Resource Constraints**: Limited access to GPU resources

#### Use GPU Training When:
- **Production Models**: Training models for deployment
- **Large Models**: Models with >100M parameters
- **Time Constraints**: Need faster training iterations
- **Batch Experiments**: Running multiple training experiments
- **Fine-tuning**: Adapting pre-trained models

#### Migration Guide: GPU to CPU Training

If you have existing GPU training configurations, here's how to adapt them for CPU:

```python
# Original GPU config
gpu_config = TrainingConfig(
    device="cuda",
    batch_size=32,
    use_amp=True,
    dataloader_pin_memory=True,
    dataloader_num_workers=8
)

# CPU-adapted config
cpu_config = TrainingConfig(
    device="cpu",
    batch_size=2,  # Reduce batch size
    gradient_accumulation_steps=16,  # Maintain effective batch size
    use_amp=False,  # Disable AMP
    dataloader_pin_memory=False,  # Disable pin memory
    dataloader_num_workers=2  # Reduce workers
)
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
- **CPU and GPU Support**: Flexible device selection with optimized configurations
- Distributed training support (NCCL for GPU, Gloo for CPU)
- Mixed precision training (GPU only)
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
- [CPU Training Guide](cpu_training.md) - Comprehensive CPU training documentation
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
