# API Reference

This document provides a comprehensive API reference for all classes and functions in LLM Trainer.

## Core Components

### Configuration Classes

#### ModelConfig

Configuration class for Transformer model architecture.

```python
from llm_trainer.config import ModelConfig

config = ModelConfig(
    vocab_size=50000,           # Vocabulary size
    d_model=768,                # Model dimension
    n_heads=12,                 # Number of attention heads
    n_layers=12,                # Number of transformer layers
    d_ff=3072,                  # Feed-forward dimension
    max_seq_len=1024,           # Maximum sequence length
    dropout=0.1,                # Dropout rate
    attention_dropout=0.1,      # Attention dropout rate
    activation="gelu",          # Activation function
    pre_norm=True,              # Use pre-normalization
    use_learned_pos_emb=False,  # Use learned positional embeddings
    gradient_checkpointing=False # Enable gradient checkpointing
)
```

**Methods:**
- `save(path: str)` - Save configuration to file
- `load(path: str)` - Load configuration from file
- `to_dict()` - Convert to dictionary
- `from_dict(config_dict: dict)` - Create from dictionary

**Properties:**
- `d_head` - Dimension per attention head (d_model // n_heads)

#### TrainingConfig

Configuration class for training process.

```python
from llm_trainer.config import TrainingConfig

config = TrainingConfig(
    # Core training parameters
    batch_size=32,              # Batch size per device
    learning_rate=1e-4,         # Peak learning rate
    weight_decay=0.01,          # L2 regularization
    num_epochs=10,              # Number of training epochs
    max_steps=None,             # Maximum training steps
    
    # Learning rate scheduling
    lr_scheduler="cosine",      # Scheduler type
    warmup_steps=1000,          # Warmup steps
    warmup_ratio=0.1,           # Warmup ratio
    min_lr_ratio=0.1,           # Minimum LR ratio
    
    # Optimization
    optimizer="adamw",          # Optimizer type
    max_grad_norm=1.0,          # Gradient clipping
    gradient_accumulation_steps=1,
    
    # Mixed precision
    use_amp=True,               # Enable AMP
    amp_dtype="float16",        # Precision type
    
    # Checkpointing
    save_steps=1000,            # Save frequency
    checkpoint_dir="./checkpoints",
    
    # Evaluation
    eval_steps=500,             # Evaluation frequency
    eval_strategy="steps",      # Evaluation strategy
    
    # Logging
    logging_steps=100,          # Logging frequency
    report_to=["tensorboard"],  # Logging backends
    
    # Distributed training
    world_size=1,               # Number of processes
    local_rank=-1,              # Local process rank
)
```

**Properties:**
- `effective_batch_size` - Total effective batch size across all devices

#### DataConfig

Configuration class for data loading and preprocessing.

```python
from llm_trainer.config import DataConfig

config = DataConfig(
    # Dataset
    dataset_name="wikitext",
    dataset_config="wikitext-103-raw-v1",
    dataset_split="train",
    validation_split="validation",
    text_column="text",
    
    # Preprocessing
    max_length=1024,            # Maximum sequence length
    min_length=10,              # Minimum sequence length
    vocab_size=50000,           # Tokenizer vocabulary size
    
    # Optimization
    pack_sequences=True,        # Pack sequences for efficiency
    preprocessing_num_workers=4,
    dataloader_num_workers=4,
    
    # Custom files
    train_file=None,            # Custom training file
    validation_file=None,       # Custom validation file
)
```

### Model Classes

#### TransformerLM

Main Transformer language model class.

```python
from llm_trainer.models import TransformerLM
from llm_trainer.config import ModelConfig

# Create model
config = ModelConfig(vocab_size=50000, d_model=768, n_heads=12, n_layers=12)
model = TransformerLM(config)

# Model information
print(f"Parameters: {model.get_num_params():,}")
print(f"Model size: {model.get_model_size_mb():.1f} MB")
```

**Methods:**
- `forward(input_ids, attention_mask=None, labels=None)` - Forward pass
- `generate(input_ids, max_length=100, **kwargs)` - Text generation
- `get_num_params()` - Get parameter count
- `get_model_size_mb()` - Get model size in MB
- `save_pretrained(save_directory)` - Save model
- `from_pretrained(load_directory)` - Load model

**Generation Parameters:**
- `max_length` - Maximum generation length
- `temperature` - Sampling temperature
- `top_k` - Top-k sampling
- `top_p` - Nucleus sampling
- `do_sample` - Enable sampling
- `num_beams` - Beam search width
- `repetition_penalty` - Repetition penalty

### Tokenizer Classes

#### BPETokenizer

Byte Pair Encoding tokenizer implementation.

```python
from llm_trainer.tokenizer import BPETokenizer

# Create and train tokenizer
tokenizer = BPETokenizer()
tokenizer.train_from_dataset(
    dataset_name="wikitext",
    dataset_config="wikitext-2-raw-v1",
    vocab_size=32000
)

# Encode/decode
token_ids = tokenizer.encode("Hello world!")
text = tokenizer.decode(token_ids)
```

**Methods:**
- `train(texts, vocab_size=50000)` - Train on text corpus
- `train_from_dataset(dataset_name, vocab_size=50000, **kwargs)` - Train on HF dataset
- `encode(text, add_special_tokens=True)` - Encode text to IDs
- `decode(token_ids, skip_special_tokens=True)` - Decode IDs to text
- `save_pretrained(save_directory)` - Save tokenizer
- `from_pretrained(load_directory)` - Load tokenizer
- `get_vocab_stats()` - Get vocabulary statistics

**Properties:**
- `vocab_size` - Vocabulary size
- `vocab` - Token to ID mapping
- `special_tokens` - List of special tokens

### Training Classes

#### Trainer

Main trainer class for model training.

```python
from llm_trainer.training import Trainer
from llm_trainer.config import TrainingConfig

# Create trainer
trainer = Trainer(
    model=model,
    tokenizer=tokenizer,
    config=training_config,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset
)

# Train model
trainer.train()
```

**Methods:**
- `train()` - Start training loop
- `train_from_config(model_config, data_config)` - Train with configs
- `evaluate()` - Run evaluation
- `generate_text(prompt, max_length=100, **kwargs)` - Generate text
- `save_model(save_path)` - Save trained model
- `from_pretrained(model_path, tokenizer_path, config)` - Load pretrained

**Training Methods:**
- `_train_epoch()` - Train single epoch
- `_evaluate()` - Evaluate model
- `_save_checkpoint()` - Save checkpoint
- `_load_checkpoint()` - Load checkpoint

### Utility Classes

#### TextGenerator

Text generation utility with advanced decoding strategies.

```python
from llm_trainer.utils.generation import TextGenerator, GenerationConfig

# Create generator
generator = TextGenerator(model, tokenizer, device)

# Configure generation
config = GenerationConfig(
    max_length=100,
    temperature=0.8,
    top_k=50,
    top_p=0.9,
    do_sample=True,
    repetition_penalty=1.1
)

# Generate text
generated = generator.generate(
    prompt="The future of AI is",
    config=config
)
```

**Methods:**
- `generate(prompt, config=None)` - Generate text
- `batch_generate(prompts, config=None)` - Batch generation
- `interactive_generate()` - Interactive generation mode

#### InferenceEngine

Optimized inference engine for production use.

```python
from llm_trainer.utils.inference import InferenceEngine

# Create engine
engine = InferenceEngine(model, tokenizer, device)

# Optimize for inference
engine.optimize(
    enable_kv_cache=True,
    compile_model=True,
    quantize=False
)

# Run inference
result = engine.infer(
    prompt="Hello world",
    max_length=50
)
```

**Methods:**
- `infer(prompt, **kwargs)` - Single inference
- `batch_infer(prompts, **kwargs)` - Batch inference
- `optimize(**kwargs)` - Optimize for inference
- `benchmark(prompts, num_runs=10)` - Benchmark performance

### Data Classes

#### LanguageModelingDataset

Dataset class for language modeling.

```python
from llm_trainer.data import LanguageModelingDataset

# Create dataset
dataset = LanguageModelingDataset(
    texts=texts,
    tokenizer=tokenizer,
    max_length=1024,
    pack_sequences=True
)

# Use with DataLoader
from torch.utils.data import DataLoader
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
```

**Methods:**
- `__len__()` - Dataset length
- `__getitem__(idx)` - Get item by index
- `pack_sequences()` - Pack multiple sequences
- `get_stats()` - Get dataset statistics

#### create_dataloader

Utility function to create optimized dataloaders.

```python
from llm_trainer.data import create_dataloader

dataloader = create_dataloader(
    dataset=dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4,
    pin_memory=True,
    drop_last=True
)
```

### Metrics and Evaluation

#### Metrics Functions

```python
from llm_trainer.utils.metrics import (
    compute_perplexity,
    compute_bleu_score,
    compute_diversity_metrics,
    compute_repetition_metrics,
    evaluate_generation_quality
)

# Compute perplexity
perplexity = compute_perplexity(model, dataloader, device)

# Evaluate generation quality
quality_metrics = evaluate_generation_quality(
    generated_texts=generated_texts,
    reference_texts=reference_texts
)

# Diversity metrics
diversity = compute_diversity_metrics(generated_texts)
```

**Available Metrics:**
- `compute_perplexity(model, dataloader, device)` - Language model perplexity
- `compute_bleu_score(predictions, references)` - BLEU score
- `compute_diversity_metrics(texts)` - Distinct-1, Distinct-2
- `compute_repetition_metrics(texts)` - Repetition analysis
- `evaluate_generation_quality(generated, references)` - Comprehensive evaluation

### Optimization Classes

#### Optimizers

```python
from llm_trainer.training.optimizer import create_optimizer

# Create optimizer
optimizer = create_optimizer(
    model=model,
    optimizer_type="adamw",
    learning_rate=1e-4,
    weight_decay=0.01,
    betas=(0.9, 0.999),
    eps=1e-8
)
```

**Supported Optimizers:**
- `adamw` - AdamW optimizer (recommended)
- `adam` - Adam optimizer
- `sgd` - SGD with momentum

#### Learning Rate Schedulers

```python
from llm_trainer.training.scheduler import create_scheduler

# Create scheduler
scheduler = create_scheduler(
    optimizer=optimizer,
    scheduler_type="cosine",
    num_training_steps=total_steps,
    warmup_steps=warmup_steps,
    min_lr_ratio=0.1
)
```

**Supported Schedulers:**
- `cosine` - Cosine annealing (recommended)
- `linear` - Linear decay
- `constant` - Constant learning rate
- `polynomial` - Polynomial decay

### Utility Functions

#### Training Utilities

```python
from llm_trainer.training.utils import (
    set_seed,
    get_device,
    setup_logging,
    save_checkpoint,
    load_checkpoint,
    get_memory_usage
)

# Set random seed
set_seed(42)

# Get optimal device
device = get_device()

# Setup logging
setup_logging(level="info")

# Memory monitoring
memory_usage = get_memory_usage()
```

#### Generation Utilities

```python
from llm_trainer.utils.generation import (
    sample_top_k,
    sample_top_p,
    apply_repetition_penalty,
    create_causal_mask
)

# Sampling functions
logits = sample_top_k(logits, k=50)
logits = sample_top_p(logits, p=0.9)
logits = apply_repetition_penalty(logits, input_ids, penalty=1.1)
```

## Configuration File Formats

### YAML Configuration

```yaml
# model_config.yaml
model:
  vocab_size: 50000
  d_model: 768
  n_heads: 12
  n_layers: 12
  d_ff: 3072
  max_seq_len: 1024
  dropout: 0.1
  activation: "gelu"

training:
  batch_size: 32
  learning_rate: 1e-4
  num_epochs: 10
  lr_scheduler: "cosine"
  warmup_steps: 1000
  use_amp: true

data:
  dataset_name: "wikitext"
  dataset_config: "wikitext-103-raw-v1"
  max_length: 1024
  vocab_size: 50000
```

### JSON Configuration

```json
{
  "model": {
    "vocab_size": 50000,
    "d_model": 768,
    "n_heads": 12,
    "n_layers": 12,
    "d_ff": 3072,
    "max_seq_len": 1024,
    "dropout": 0.1,
    "activation": "gelu"
  },
  "training": {
    "batch_size": 32,
    "learning_rate": 1e-4,
    "num_epochs": 10,
    "lr_scheduler": "cosine",
    "warmup_steps": 1000,
    "use_amp": true
  },
  "data": {
    "dataset_name": "wikitext",
    "dataset_config": "wikitext-103-raw-v1",
    "max_length": 1024,
    "vocab_size": 50000
  }
}
```

## Command Line Interface

### Training Script

```bash
python scripts/train.py \
    --config configs/model_config.yaml \
    --output_dir ./output \
    --resume_from_checkpoint ./checkpoints/checkpoint-1000 \
    --force_retrain_tokenizer
```

**Arguments:**
- `--config` - Configuration file path
- `--output_dir` - Output directory
- `--resume_from_checkpoint` - Resume from checkpoint
- `--force_retrain_tokenizer` - Force tokenizer retraining
- `--log_level` - Logging level

### Generation Script

```bash
python scripts/generate.py \
    --model_path ./output \
    --prompts "The future of AI" "Once upon a time" \
    --max_length 100 \
    --temperature 0.8 \
    --top_p 0.9 \
    --interactive
```

**Arguments:**
- `--model_path` - Path to trained model
- `--prompts` - Input prompts
- `--max_length` - Maximum generation length
- `--temperature` - Sampling temperature
- `--top_k` - Top-k sampling
- `--top_p` - Nucleus sampling
- `--interactive` - Interactive mode

### Evaluation Script

```bash
python scripts/evaluate.py \
    --model_path ./output \
    --dataset_config configs/eval_config.json \
    --output_path evaluation_results.json \
    --max_batches 100
```

**Arguments:**
- `--model_path` - Path to trained model
- `--dataset_config` - Evaluation dataset config
- `--output_path` - Results output path
- `--max_batches` - Maximum evaluation batches

## Error Handling

### Common Exceptions

```python
from llm_trainer.exceptions import (
    TokenizerNotTrainedException,
    ModelConfigurationError,
    TrainingError,
    GenerationError
)

try:
    # Training code
    trainer.train()
except TrainingError as e:
    print(f"Training failed: {e}")
except ModelConfigurationError as e:
    print(f"Model configuration error: {e}")
```

### Validation

All configuration classes include validation:

```python
# This will raise an assertion error
config = ModelConfig(
    d_model=768,
    n_heads=5  # Error: d_model not divisible by n_heads
)
```

## Performance Considerations

### Memory Usage

```python
# Estimate memory usage
def estimate_memory_usage(config):
    """Estimate model memory usage."""
    params = calculate_parameters(config)
    
    # Model parameters (4 bytes each)
    model_memory = params * 4
    
    # Gradients (4 bytes each)
    gradient_memory = params * 4
    
    # Optimizer states (8 bytes each for AdamW)
    optimizer_memory = params * 8
    
    # Total in GB
    total_gb = (model_memory + gradient_memory + optimizer_memory) / (1024**3)
    
    return total_gb
```

### Speed Optimization

```python
# Optimize for speed
training_config = TrainingConfig(
    use_amp=True,                    # Mixed precision
    compile_model=True,              # PyTorch 2.0 compilation
    dataloader_num_workers=8,        # Parallel data loading
    gradient_accumulation_steps=4,   # Larger effective batch
)
```

## Examples and Tutorials

### Complete Training Example

```python
#!/usr/bin/env python3
"""Complete training example with all components."""

import os
from llm_trainer.config import ModelConfig, TrainingConfig, DataConfig
from llm_trainer.models import TransformerLM
from llm_trainer.tokenizer import BPETokenizer
from llm_trainer.training import Trainer

def main():
    # Configuration
    model_config = ModelConfig(
        vocab_size=32000,
        d_model=512,
        n_heads=8,
        n_layers=6,
        max_seq_len=1024
    )
    
    training_config = TrainingConfig(
        batch_size=16,
        learning_rate=1e-4,
        num_epochs=3,
        warmup_steps=1000,
        use_amp=True,
        checkpoint_dir="./checkpoints"
    )
    
    data_config = DataConfig(
        dataset_name="wikitext",
        dataset_config="wikitext-2-raw-v1",
        max_length=1024,
        vocab_size=32000
    )
    
    # Create tokenizer
    tokenizer = BPETokenizer()
    tokenizer.train_from_dataset(
        dataset_name=data_config.dataset_name,
        dataset_config=data_config.dataset_config,
        vocab_size=data_config.vocab_size
    )
    
    # Update model config
    model_config.vocab_size = tokenizer.vocab_size
    
    # Create model
    model = TransformerLM(model_config)
    
    # Create trainer
    trainer = Trainer(model, tokenizer, training_config)
    
    # Train
    trainer.train_from_config(model_config, data_config)
    
    # Save
    trainer.save_model("./final_model")

if __name__ == "__main__":
    main()
```

### Custom Dataset Example

```python
#!/usr/bin/env python3
"""Train on custom dataset."""

from llm_trainer.data import LanguageModelingDataset, create_dataloader

def train_on_custom_data():
    # Load custom texts
    texts = []
    for file_path in ["data1.txt", "data2.txt"]:
        with open(file_path, 'r') as f:
            texts.append(f.read())
    
    # Train tokenizer
    tokenizer = BPETokenizer()
    tokenizer.train(texts, vocab_size=32000)
    
    # Create dataset
    dataset = LanguageModelingDataset(
        texts=texts,
        tokenizer=tokenizer,
        max_length=512
    )
    
    # Create dataloader
    dataloader = create_dataloader(
        dataset=dataset,
        batch_size=32,
        shuffle=True
    )
    
    # Continue with training...
```

## Version Compatibility

### PyTorch Versions

- **PyTorch 2.0+**: Full support with compilation
- **PyTorch 1.13+**: Basic support without compilation
- **PyTorch <1.13**: Not supported

### Python Versions

- **Python 3.11**: Recommended
- **Python 3.10**: Full support
- **Python 3.9**: Full support
- **Python 3.8**: Basic support
- **Python <3.8**: Not supported

### CUDA Versions

- **CUDA 11.8+**: Recommended
- **CUDA 11.0+**: Supported
- **CUDA <11.0**: Limited support

---

For more examples and detailed usage, see the `examples/` directory and documentation files.