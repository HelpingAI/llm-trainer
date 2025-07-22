# Training Guide

This guide covers everything you need to know about training Large Language Models with the LLM Trainer framework.

## Training Process Overview

The training process consists of several key steps:

1. **Data Preparation**: Load and preprocess text data
2. **Tokenizer Training**: Create a BPE tokenizer from the training data
3. **Model Creation**: Initialize the Transformer model
4. **Training Loop**: Train the model with gradient descent
5. **Evaluation**: Monitor training progress and model quality
6. **Checkpointing**: Save model states for resuming training

## Configuration

Training is controlled through three main configuration classes:

### ModelConfig

Defines the Transformer architecture:

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
    pre_norm=True,           # Use pre-norm architecture
    gradient_checkpointing=False  # Enable for memory efficiency
)
```

### TrainingConfig

Controls the training process:

```python
from llm_trainer.config import TrainingConfig

training_config = TrainingConfig(
    batch_size=8,                    # Batch size per device
    learning_rate=1e-4,              # Learning rate
    weight_decay=0.01,               # Weight decay
    num_epochs=3,                    # Number of training epochs
    lr_scheduler="cosine",           # Learning rate scheduler
    warmup_steps=2000,               # Warmup steps
    optimizer="adamw",               # Optimizer type
    gradient_accumulation_steps=8,   # Gradient accumulation
    use_amp=True,                    # Mixed precision training
    max_grad_norm=1.0,               # Gradient clipping
    save_steps=2000,                 # Checkpoint saving frequency
    eval_steps=1000,                 # Evaluation frequency
    logging_steps=100,               # Logging frequency
    checkpoint_dir="./checkpoints",  # Checkpoint directory
    report_to=["tensorboard"]        # Monitoring tools
)
```

### DataConfig

Specifies data loading and preprocessing:

```python
from llm_trainer.config import DataConfig

data_config = DataConfig(
    dataset_name="wikitext",
    dataset_config="wikitext-103-raw-v1",
    dataset_split="train",
    validation_split="validation",
    text_column="text",
    max_length=1024,
    min_length=10,
    pack_sequences=True,             # Pack sequences for efficiency
    preprocessing_num_workers=8      # Parallel preprocessing
)
```

## Training Strategies

### Small Model (Quick Experimentation)

For quick experimentation and testing:

```python
# Small model configuration
model_config = ModelConfig(
    d_model=256,
    n_heads=4,
    n_layers=4,
    d_ff=1024,
    max_seq_len=512
)

training_config = TrainingConfig(
    batch_size=16,
    learning_rate=5e-4,
    num_epochs=3,
    gradient_accumulation_steps=2
)

data_config = DataConfig(
    dataset_name="wikitext",
    dataset_config="wikitext-2-raw-v1",  # Smaller dataset
    max_length=512
)
```

### Medium Model (GPT-2 Small Scale)

For more serious training:

```python
model_config = ModelConfig(
    d_model=768,
    n_heads=12,
    n_layers=12,
    d_ff=3072,
    max_seq_len=1024,
    gradient_checkpointing=True  # Save memory
)

training_config = TrainingConfig(
    batch_size=8,
    learning_rate=1e-4,
    gradient_accumulation_steps=8,  # Effective batch size: 64
    use_amp=True,
    num_epochs=3
)
```

### Large Model (Resource Intensive)

For large-scale training:

```python
model_config = ModelConfig(
    d_model=1024,
    n_heads=16,
    n_layers=24,
    d_ff=4096,
    max_seq_len=2048,
    gradient_checkpointing=True
)

training_config = TrainingConfig(
    batch_size=4,
    gradient_accumulation_steps=16,  # Effective batch size: 64
    use_amp=True,
    use_deepspeed=True,  # Enable DeepSpeed for large models
    num_epochs=1
)
```

## Memory Optimization

### Gradient Checkpointing

Trade computation for memory:

```python
model_config = ModelConfig(
    gradient_checkpointing=True  # Recompute activations during backward pass
)
```

### Mixed Precision Training

Use 16-bit floats to reduce memory usage:

```python
training_config = TrainingConfig(
    use_amp=True,
    amp_dtype="float16"  # or "bfloat16"
)
```

### Gradient Accumulation

Simulate larger batch sizes:

```python
training_config = TrainingConfig(
    batch_size=4,                   # Small batch per step
    gradient_accumulation_steps=16  # Accumulate 16 steps = effective batch size 64
)
```

### Sequence Packing

Pack multiple sequences into fixed-length chunks:

```python
data_config = DataConfig(
    pack_sequences=True,
    packing_strategy="greedy",  # or "first_fit"
    max_length=1024
)
```

## Distributed Training

### Multi-GPU Training

For training on multiple GPUs:

```bash
# Using torchrun (recommended)
torchrun --nproc_per_node=4 scripts/train.py --config config.json

# Using python -m torch.distributed.launch (older)
python -m torch.distributed.launch --nproc_per_node=4 scripts/train.py --config config.json
```

Configuration for distributed training:

```python
training_config = TrainingConfig(
    world_size=4,                    # Number of GPUs
    distributed_backend="nccl",      # Communication backend
    batch_size=8,                    # Per-device batch size
    gradient_accumulation_steps=2    # Total effective batch size: 4*8*2=64
)
```

### DeepSpeed Integration

For very large models:

```python
training_config = TrainingConfig(
    use_deepspeed=True,
    deepspeed_config="deepspeed_config.json"
)
```

Example DeepSpeed configuration:

```json
{
  "train_batch_size": 64,
  "train_micro_batch_size_per_gpu": 4,
  "gradient_accumulation_steps": 4,
  "optimizer": {
    "type": "AdamW",
    "params": {
      "lr": 1e-4,
      "weight_decay": 0.01
    }
  },
  "fp16": {
    "enabled": true
  },
  "zero_optimization": {
    "stage": 2
  }
}
```

## Monitoring and Logging

### TensorBoard

Monitor training with TensorBoard:

```python
training_config = TrainingConfig(
    report_to=["tensorboard"],
    logging_steps=100
)
```

View logs:
```bash
tensorboard --logdir ./checkpoints/logs
```

### Weights & Biases

For more advanced monitoring:

```python
training_config = TrainingConfig(
    report_to=["wandb"],
    logging_steps=100
)
```

### Custom Metrics

Track custom metrics during training:

```python
class CustomTrainer(Trainer):
    def _log_metrics(self, loss: float):
        super()._log_metrics(loss)
        
        # Add custom metrics
        custom_metrics = {
            "custom/metric1": compute_custom_metric(),
            "custom/metric2": compute_another_metric()
        }
        
        self.metrics_tracker.update(custom_metrics, self.current_step)
```

## Checkpointing and Resuming

### Automatic Checkpointing

Checkpoints are saved automatically:

```python
training_config = TrainingConfig(
    save_steps=2000,        # Save every 2000 steps
    save_total_limit=3,     # Keep only 3 most recent checkpoints
    checkpoint_dir="./checkpoints"
)
```

### Resuming Training

Resume from a checkpoint:

```python
training_config = TrainingConfig(
    resume_from_checkpoint="./checkpoints/checkpoint-10000.pt"
)
```

Or using command line:

```bash
python scripts/train.py --config config.json --resume_from_checkpoint ./checkpoints/checkpoint-10000.pt
```

### Best Model Saving

The best model (lowest validation loss) is automatically saved:

```python
# Best model saved to: {checkpoint_dir}/best_model/
```

## Troubleshooting

### Common Issues

**CUDA Out of Memory:**
- Reduce batch size
- Enable gradient checkpointing
- Use gradient accumulation
- Enable mixed precision

**Slow Training:**
- Use multiple GPUs
- Optimize data loading (more workers)
- Enable model compilation (PyTorch 2.0+)
- Use sequence packing

**NaN Loss:**
- Reduce learning rate
- Enable gradient clipping
- Check data for corrupted samples
- Use mixed precision carefully

**Poor Convergence:**
- Increase warmup steps
- Adjust learning rate schedule
- Check data quality
- Increase model capacity

### Performance Tips

1. **Data Loading**: Use multiple workers and pin memory
2. **Sequence Packing**: Pack sequences to reduce padding
3. **Mixed Precision**: Use AMP for faster training
4. **Gradient Checkpointing**: Trade compute for memory
5. **Model Compilation**: Use `torch.compile()` for PyTorch 2.0+

### Memory Estimation

Rough memory requirements:

- **Small Model (125M params)**: 4-8 GB
- **Medium Model (350M params)**: 8-16 GB  
- **Large Model (1.3B params)**: 16-32 GB
- **Very Large Model (6B+ params)**: 32+ GB (requires distributed training)

Memory usage includes:
- Model parameters
- Gradients
- Optimizer states
- Activations
- Data batches

Use gradient checkpointing and mixed precision to reduce memory usage.
