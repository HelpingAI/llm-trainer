# Training Guide

This comprehensive guide covers everything you need to know about training Large Language Models with the LLM Trainer framework, from basic concepts to advanced techniques.

## Training Process Overview

The training pipeline consists of several interconnected steps:

### Step-by-Step Process

1. **Data Preparation**: Load and preprocess text data from various sources
2. **Tokenizer Training**: Create a BPE tokenizer optimized for your dataset
3. **Model Creation**: Initialize the Transformer model with your configuration
4. **Training Loop**: Train the model using gradient descent with optimizations
5. **Evaluation**: Monitor training progress and model quality metrics
6. **Checkpointing**: Save model states for resuming training and deployment

> [!IMPORTANT]
> **Training Time Estimates:**
> - Small Model (25M): 2-4 hours on single GPU
> - Medium Model (117M): 8-12 hours on single GPU
> - Large Model (345M): 24-48 hours on single GPU or 6-12 hours on 4 GPUs

## Configuration System

Training is controlled through three main configuration classes that provide complete control over the training process:

> [!NOTE]
> **Configuration Files**: Use YAML or JSON files for easy experimentation and reproducibility.

### ModelConfig

Defines the Transformer architecture parameters:

```python
from llm_trainer.config import ModelConfig

model_config = ModelConfig(
    vocab_size=50000,        # Vocabulary size (set by tokenizer)
    d_model=768,             # Model dimension
    n_heads=12,              # Number of attention heads
    n_layers=12,             # Number of transformer layers
    d_ff=3072,               # Feed-forward dimension
    max_seq_len=1024,        # Maximum sequence length
    dropout=0.1,             # Dropout rate
    attention_dropout=0.1,   # Attention dropout rate
    activation="gelu",       # Activation function
    pre_norm=True,           # Use pre-normalization
    gradient_checkpointing=False  # Enable for memory efficiency
)
```

> [!TIP]
> **Architecture Guidelines:**
> - `d_ff` is typically 4x `d_model`
> - `d_model` must be divisible by `n_heads`
> - Enable `gradient_checkpointing` for models >100M parameters

### TrainingConfig

Controls the training process and optimization:

```python
from llm_trainer.config import TrainingConfig

training_config = TrainingConfig(
    # Core training parameters
    batch_size=32,           # Batch size per device
    learning_rate=1e-4,      # Peak learning rate
    weight_decay=0.01,       # L2 regularization
    num_epochs=10,           # Number of training epochs
    
    # Learning rate scheduling
    lr_scheduler="cosine",   # Scheduler type
    warmup_steps=1000,       # Warmup steps
    min_lr_ratio=0.1,        # Minimum LR as ratio of peak LR
    
    # Optimization
    optimizer="adamw",       # Optimizer type
    max_grad_norm=1.0,       # Gradient clipping
    gradient_accumulation_steps=1,  # Gradient accumulation
    
    # Mixed precision
    use_amp=True,            # Enable automatic mixed precision
    amp_dtype="float16",     # Precision type
    
    # Monitoring and saving
    logging_steps=100,       # Log every N steps
    eval_steps=500,          # Evaluate every N steps
    save_steps=1000,         # Save checkpoint every N steps
    checkpoint_dir="./checkpoints",
    
    # Distributed training
    world_size=1,            # Number of processes
    distributed_backend="nccl"
)
```

### DataConfig

Specifies dataset and preprocessing options:

```python
from llm_trainer.config import DataConfig

data_config = DataConfig(
    # Dataset
    dataset_name="wikitext",
    dataset_config="wikitext-103-raw-v1",
    text_column="text",
    
    # Preprocessing
    max_length=1024,         # Maximum sequence length
    min_length=10,           # Minimum sequence length
    vocab_size=50000,        # Tokenizer vocabulary size
    
    # Optimization
    pack_sequences=True,     # Pack multiple sequences per batch
    preprocessing_num_workers=4,
    dataloader_num_workers=4,
    
    # Caching
    cache_dir="./cache",
    use_streaming=False      # Enable for very large datasets
)
```

## Quick Start Examples

### Example 1: Train Small Model

Perfect for experimentation and learning:

```python
#!/usr/bin/env python3
"""Train a small model for quick experimentation."""

from llm_trainer.config import ModelConfig, TrainingConfig, DataConfig
from llm_trainer.models import TransformerLM
from llm_trainer.tokenizer import BPETokenizer
from llm_trainer.training import Trainer

# Small model configuration
model_config = ModelConfig(
    vocab_size=32000,
    d_model=256,
    n_heads=4,
    n_layers=4,
    d_ff=1024,
    max_seq_len=512,
    dropout=0.1
)

# Fast training configuration
training_config = TrainingConfig(
    batch_size=16,
    learning_rate=5e-4,
    num_epochs=3,
    warmup_steps=500,
    gradient_accumulation_steps=2,
    use_amp=True,
    checkpoint_dir="./checkpoints/small_model"
)

# Dataset configuration
data_config = DataConfig(
    dataset_name="wikitext",
    dataset_config="wikitext-2-raw-v1",
    max_length=512,
    vocab_size=32000
)

# Create and train tokenizer
print("Training tokenizer...")
tokenizer = BPETokenizer()
tokenizer.train_from_dataset(
    dataset_name=data_config.dataset_name,
    dataset_config=data_config.dataset_config,
    vocab_size=data_config.vocab_size,
    max_samples=10000  # Limit for faster training
)

# Update model config with actual vocab size
model_config.vocab_size = tokenizer.vocab_size

# Create model and trainer
model = TransformerLM(model_config)
trainer = Trainer(model, tokenizer, training_config)

# Start training
print("Starting training...")
trainer.train_from_config(model_config, data_config)

print("Training completed!")
```

### Example 2: Production Training

For serious model training with monitoring:

```python
#!/usr/bin/env python3
"""Production training with full monitoring."""

import os
from llm_trainer.config import ModelConfig, TrainingConfig, DataConfig
from llm_trainer.models import TransformerLM
from llm_trainer.tokenizer import BPETokenizer
from llm_trainer.training import Trainer

# Production model configuration
model_config = ModelConfig(
    vocab_size=50000,
    d_model=768,
    n_heads=12,
    n_layers=12,
    d_ff=3072,
    max_seq_len=1024,
    dropout=0.1,
    gradient_checkpointing=True  # Enable for memory efficiency
)

# Production training configuration
training_config = TrainingConfig(
    batch_size=8,
    learning_rate=1e-4,
    weight_decay=0.01,
    num_epochs=10,
    lr_scheduler="cosine",
    warmup_steps=2000,
    optimizer="adamw",
    gradient_accumulation_steps=8,
    use_amp=True,
    amp_dtype="float16",
    
    # Monitoring
    logging_steps=50,
    eval_steps=1000,
    save_steps=2000,
    save_total_limit=3,
    report_to=["tensorboard", "wandb"],
    
    # Checkpointing
    checkpoint_dir="./checkpoints/production_model",
    
    # Early stopping
    early_stopping_patience=5,
    early_stopping_threshold=0.01
)

# Dataset configuration
data_config = DataConfig(
    dataset_name="wikitext",
    dataset_config="wikitext-103-raw-v1",
    max_length=1024,
    vocab_size=50000,
    pack_sequences=True,
    preprocessing_num_workers=8,
    cache_dir="./cache"
)

# Setup output directory
output_dir = "./output/production_model"
os.makedirs(output_dir, exist_ok=True)

# Train or load tokenizer
tokenizer_path = os.path.join(output_dir, "tokenizer")
if os.path.exists(tokenizer_path):
    print("Loading existing tokenizer...")
    tokenizer = BPETokenizer.from_pretrained(tokenizer_path)
else:
    print("Training new tokenizer...")
    tokenizer = BPETokenizer()
    tokenizer.train_from_dataset(
        dataset_name=data_config.dataset_name,
        dataset_config=data_config.dataset_config,
        vocab_size=data_config.vocab_size
    )
    tokenizer.save_pretrained(tokenizer_path)

# Update model config
model_config.vocab_size = tokenizer.vocab_size

# Create model and trainer
model = TransformerLM(model_config)
trainer = Trainer(model, tokenizer, training_config)

# Save configurations
model_config.save(os.path.join(output_dir, "model_config.json"))
training_config.save(os.path.join(output_dir, "training_config.json"))
data_config.save(os.path.join(output_dir, "data_config.json"))

# Start training
print(f"Starting training with {model.get_num_params():,} parameters...")
trainer.train_from_config(model_config, data_config)

# Save final model
trainer.save_model(os.path.join(output_dir, "final_model"))
print(f"Training completed! Model saved to {output_dir}")
```

## Advanced Training Techniques

### Distributed Training

For training on multiple GPUs:

```bash
# Single node, multiple GPUs
torchrun --nproc_per_node=4 scripts/train.py \
    --config configs/medium_model.yaml \
    --output_dir ./output/distributed

# Multiple nodes
torchrun --nnodes=2 --nproc_per_node=4 \
    --rdzv_id=100 --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_ADDR:29400 \
    scripts/train.py --config configs/large_model.yaml
```

### DeepSpeed Integration

For very large models:

```python
# Enable DeepSpeed in training config
training_config = TrainingConfig(
    use_deepspeed=True,
    deepspeed_config="configs/deepspeed_config.json",
    # ... other parameters
)
```

DeepSpeed configuration file:
```json
{
    "train_batch_size": 64,
    "gradient_accumulation_steps": 8,
    "optimizer": {
        "type": "AdamW",
        "params": {
            "lr": 1e-4,
            "weight_decay": 0.01
        }
    },
    "fp16": {
        "enabled": true,
        "loss_scale": 0,
        "loss_scale_window": 1000,
        "hysteresis": 2,
        "min_loss_scale": 1
    },
    "zero_optimization": {
        "stage": 2,
        "allgather_partitions": true,
        "allgather_bucket_size": 2e8,
        "overlap_comm": true,
        "reduce_scatter": true,
        "reduce_bucket_size": 2e8,
        "contiguous_gradients": true
    }
}
```

### Memory Optimization

> [!WARNING]
> **Memory Issues**: Large models can easily exceed GPU memory. Use these techniques:

1. **Gradient Checkpointing**:
```python
model_config.gradient_checkpointing = True
```

2. **Gradient Accumulation**:
```python
training_config.gradient_accumulation_steps = 8  # Effective batch size = batch_size * 8
```

3. **Mixed Precision**:
```python
training_config.use_amp = True
training_config.amp_dtype = "float16"  # or "bfloat16"
```

4. **Sequence Packing**:
```python
data_config.pack_sequences = True
```

### Learning Rate Scheduling

Different scheduling strategies:

```python
# Cosine annealing (recommended)
training_config.lr_scheduler = "cosine"
training_config.warmup_steps = 2000
training_config.min_lr_ratio = 0.1

# Linear decay
training_config.lr_scheduler = "linear"
training_config.warmup_ratio = 0.1

# Polynomial decay
training_config.lr_scheduler = "polynomial"
training_config.warmup_steps = 1000
```

## Monitoring and Debugging

### TensorBoard Integration

```python
training_config.report_to = ["tensorboard"]
```

View training progress:
```bash
tensorboard --logdir ./checkpoints/your_model/logs
```

### Weights & Biases Integration

```python
training_config.report_to = ["wandb"]
```

Initialize wandb:
```python
import wandb
wandb.init(project="llm-trainer", name="my-experiment")
```

### Key Metrics to Monitor

> [!NOTE]
> **Important Metrics:**
> - **Training Loss**: Should decrease steadily
> - **Validation Loss**: Should track training loss (watch for overfitting)
> - **Learning Rate**: Should follow your schedule
> - **Gradient Norm**: Should be stable (watch for exploding gradients)
> - **Memory Usage**: Monitor GPU memory utilization

### Common Issues and Solutions

| Issue | Symptoms | Solution |
|-------|----------|----------|
| **Out of Memory** | CUDA OOM error | Reduce batch size, enable gradient checkpointing |
| **Exploding Gradients** | Loss becomes NaN | Lower learning rate, check gradient clipping |
| **Slow Convergence** | Loss plateaus early | Increase learning rate, check data quality |
| **Overfitting** | Val loss > train loss | Add dropout, reduce model size, more data |

## Evaluation and Validation

### Automatic Evaluation

The trainer automatically evaluates on validation data:

```python
training_config.eval_steps = 500  # Evaluate every 500 steps
training_config.eval_strategy = "steps"  # or "epoch"
```

### Manual Evaluation

```python
# Evaluate perplexity
perplexity = trainer.evaluate_perplexity()
print(f"Validation Perplexity: {perplexity:.2f}")

# Generate sample text
generated = trainer.generate_text(
    prompt="The future of AI is",
    max_length=100,
    temperature=0.8
)
print(f"Generated: {generated}")
```

### Evaluation Metrics

- **Perplexity**: Lower is better (measures prediction confidence)
- **BLEU Score**: For text generation quality
- **Diversity Metrics**: Distinct-1, Distinct-2 for generation diversity
- **Repetition Rate**: Measure of repetitive outputs

## Best Practices

### Data Preparation

> [!TIP]
> **Data Quality Tips:**
> - Clean your data thoroughly
> - Remove duplicates and low-quality text
> - Ensure diverse and representative content
> - Use appropriate train/validation splits

### Training Strategy

1. **Start Small**: Begin with a small model to validate your pipeline
2. **Gradual Scaling**: Increase model size progressively
3. **Monitor Closely**: Watch for overfitting and instability
4. **Save Frequently**: Regular checkpointing prevents data loss
5. **Experiment**: Try different hyperparameters and architectures

### Hyperparameter Tuning

Key hyperparameters to tune:

| Parameter | Typical Range | Impact |
|-----------|---------------|---------|
| Learning Rate | 1e-5 to 1e-3 | Training speed and stability |
| Batch Size | 8 to 128 | Memory usage and convergence |
| Warmup Steps | 500 to 5000 | Training stability |
| Dropout | 0.0 to 0.3 | Regularization |
| Weight Decay | 0.01 to 0.1 | Regularization |

### Production Considerations

> [!IMPORTANT]
> **For Production Models:**
> - Use version control for configurations
> - Implement proper logging and monitoring
> - Set up automated evaluation pipelines
> - Plan for model deployment and serving
> - Consider model compression techniques

## Troubleshooting

### Common Error Messages

**"CUDA out of memory"**
```bash
# Solutions:
# 1. Reduce batch size
# 2. Enable gradient checkpointing
# 3. Use gradient accumulation
# 4. Enable mixed precision training
```

**"Loss becomes NaN"**
```bash
# Solutions:
# 1. Lower learning rate
# 2. Check gradient clipping (max_grad_norm)
# 3. Verify data preprocessing
# 4. Use more stable optimizer settings
```

**"Training is very slow"**
```bash
# Solutions:
# 1. Enable mixed precision (use_amp=True)
# 2. Increase batch size if memory allows
# 3. Use more dataloader workers
# 4. Enable sequence packing
# 5. Use faster storage (SSD)
```

### Performance Optimization

```python
# Optimized configuration for speed
training_config = TrainingConfig(
    use_amp=True,                    # Mixed precision
    compile_model=True,              # PyTorch 2.0 compilation
    dataloader_num_workers=8,        # Parallel data loading
    dataloader_pin_memory=True,      # Faster GPU transfer
    gradient_accumulation_steps=4,   # Larger effective batch size
)

data_config = DataConfig(
    pack_sequences=True,             # Efficient sequence packing
    preprocessing_num_workers=8,     # Parallel preprocessing
    cache_dir="./cache",             # Cache preprocessed data
)
```

## Next Steps

After successful training:

1. **Evaluate Thoroughly**: Test on held-out datasets
2. **Generate Samples**: Assess text quality manually
3. **Fine-tune**: Adapt to specific tasks or domains
4. **Deploy**: Set up inference serving
5. **Monitor**: Track model performance in production

For more advanced topics, see:
- [Model Architecture](architecture.md) - Deep dive into the Transformer implementation
- [API Reference](api.md) - Complete API documentation
- [Tokenizer Details](tokenizer.md) - BPE tokenizer specifics

---

Happy training! 🚀