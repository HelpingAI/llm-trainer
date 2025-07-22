# 💻 CPU Training Guide

This comprehensive guide covers everything you need to know about training Large Language Models on CPU with the LLM Trainer framework.

## 🎯 Overview

CPU training enables you to train language models without requiring expensive GPU hardware. While slower than GPU training, it provides an accessible entry point for learning, development, and small-scale model training.

### Why CPU Training?

- **Accessibility**: No expensive GPU hardware required
- **Development**: Perfect for code testing and experimentation  
- **Learning**: Understand the training process without GPU costs
- **Resource Flexibility**: Use existing hardware effectively
- **Cost-Effective**: Leverage CPU resources you already have

## 🚀 Quick Start

### Minimal CPU Training Example

```bash
# Train a small model on CPU using pre-configured settings
python scripts/train.py --config configs/cpu_small_model.yaml --output_dir ./output/cpu_small
```

This single command will:
- Load CPU-optimized configuration
- Train a 25M parameter model
- Save checkpoints and final model
- Complete training in 8-24 hours on modern CPUs

## ⚙️ Configuration Deep Dive

### CPU-Optimized Configuration Files

#### [`configs/cpu_small_model.yaml`](../configs/cpu_small_model.yaml)

**Best for**: Learning, development, quick experiments

```yaml
device: "cpu"
model:
  vocab_size: 32000
  d_model: 256        # Smaller than typical 768
  n_heads: 4          # Fewer attention heads  
  n_layers: 4         # Fewer transformer layers
  max_seq_len: 512    # Shorter sequences
  gradient_checkpointing: false  # Disabled for CPU

training:
  batch_size: 2                    # Small for CPU memory
  gradient_accumulation_steps: 8   # Effective batch = 16
  learning_rate: 0.0008           # Adjusted for batch size
  use_amp: false                  # Not supported on CPU
  dataloader_num_workers: 2       # Reduced workers
  dataloader_pin_memory: false    # Disabled for CPU
```

#### [`configs/cpu_medium_model.yaml`](../configs/cpu_medium_model.yaml)

**Best for**: Higher quality results, production use

```yaml
device: "cpu"
model:
  vocab_size: 50000
  d_model: 768        # Full-size model
  n_heads: 12         # Standard attention heads
  n_layers: 12        # Full transformer depth
  max_seq_len: 1024   # Longer sequences

training:
  batch_size: 1                    # Minimal for large model
  gradient_accumulation_steps: 16  # Effective batch = 16
  learning_rate: 0.0002           # Lower for stability
  dataloader_num_workers: 1       # Minimal workers
```

### Custom CPU Configuration

```python
from llm_trainer.config import ModelConfig, TrainingConfig, DataConfig

# CPU-optimized model configuration
model_config = ModelConfig(
    vocab_size=32000,
    d_model=256,                    # Start small
    n_heads=4,                      # d_model must be divisible by n_heads
    n_layers=4,                     # Fewer layers = faster training
    d_ff=1024,                      # 4x d_model typically
    max_seq_len=512,                # Shorter sequences
    dropout=0.1,
    gradient_checkpointing=False    # Less beneficial on CPU
)

# CPU training configuration
training_config = TrainingConfig(
    device="cpu",                   # Explicit CPU selection
    batch_size=2,                   # Small batch for CPU memory
    learning_rate=8e-4,             # Adjusted for small batch
    num_epochs=5,
    gradient_accumulation_steps=8,  # Simulate larger batch
    use_amp=False,                  # Not supported on CPU
    
    # CPU-optimized data loading
    dataloader_num_workers=2,       # Avoid overhead
    dataloader_pin_memory=False,    # Not beneficial on CPU
    
    # Monitoring and checkpointing
    logging_steps=25,               # Frequent logging
    eval_steps=250,                 # Regular evaluation
    save_steps=500,                 # Save progress
    checkpoint_dir="./checkpoints/cpu_model"
)

# Data configuration
data_config = DataConfig(
    dataset_name="wikitext",
    dataset_config="wikitext-2-raw-v1",
    max_length=512,
    vocab_size=32000,
    pack_sequences=True,            # Efficient sequence packing
    preprocessing_num_workers=2     # Reduced for CPU
)
```

## 🔧 Device Selection Options

### Explicit Device Selection

```python
# Method 1: Explicit CPU selection
config = TrainingConfig(device="cpu")

# Method 2: Force CPU even if GPU available
config = TrainingConfig(device="cuda", force_cpu=True)

# Method 3: Automatic fallback to CPU
config = TrainingConfig(device="auto")  # Uses GPU if available, else CPU
```

### Device Detection

```python
import torch
from llm_trainer.config import TrainingConfig

# Check available devices
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CPU threads: {torch.get_num_threads()}")

# Create config with automatic device selection
config = TrainingConfig(device="auto")
device = config.get_effective_device()
print(f"Selected device: {device}")
```

## 📊 Performance Optimization

### Hardware Optimization

#### CPU Selection Guidelines

**Recommended CPUs for Training:**
- **Intel**: i7/i9 with 8+ cores, high single-thread performance
- **AMD**: Ryzen 7/9 with 8+ cores, good multi-threading
- **Server**: Xeon/EPYC with many cores for distributed training

#### Memory Recommendations

| Model Size | Minimum RAM | Recommended RAM | Optimal RAM |
|------------|-------------|-----------------|-------------|
| Small (25M) | 8GB | 16GB | 32GB |
| Medium (117M) | 16GB | 32GB | 64GB |
| Large (345M) | 32GB | 64GB | 128GB |

#### Storage Considerations

```python
# Use SSD for better I/O performance
data_config = DataConfig(
    cache_dir="/path/to/ssd/cache",  # Cache on fast storage
    preprocessing_num_workers=2      # Balance I/O and CPU
)
```

### Training Optimization Strategies

#### 1. Batch Size Tuning

```python
# Start small and increase gradually
training_configs = [
    TrainingConfig(batch_size=1, gradient_accumulation_steps=16),   # Conservative
    TrainingConfig(batch_size=2, gradient_accumulation_steps=8),    # Balanced  
    TrainingConfig(batch_size=4, gradient_accumulation_steps=4),    # Aggressive
]

# Monitor memory usage and adjust
import psutil
memory_percent = psutil.virtual_memory().percent
if memory_percent > 80:
    # Reduce batch size
    pass
```

#### 2. Threading Optimization

```python
import torch
import psutil

# Optimize PyTorch threading for your CPU
cpu_count = psutil.cpu_count(logical=False)  # Physical cores
torch.set_num_threads(cpu_count)
torch.set_num_interop_threads(1)  # Reduce thread contention

print(f"PyTorch threads: {torch.get_num_threads()}")
```

#### 3. Memory Management

```python
# Monitor memory usage during training
import psutil

def monitor_memory():
    memory = psutil.virtual_memory()
    print(f"Memory usage: {memory.percent:.1f}%")
    print(f"Available: {memory.available / (1024**3):.1f} GB")
    
    if memory.percent > 85:
        print("⚠️  High memory usage - consider reducing batch size")
```

#### 4. Data Loading Optimization

```python
# Optimize data loading for CPU
training_config = TrainingConfig(
    dataloader_num_workers=2,        # Start with 2, adjust based on CPU cores
    dataloader_pin_memory=False,     # Not beneficial on CPU
    dataloader_prefetch_factor=2,    # Reduce prefetching
)

# For systems with many cores, you can increase workers
cpu_cores = psutil.cpu_count()
if cpu_cores >= 8:
    training_config.dataloader_num_workers = min(4, cpu_cores // 2)
```

## 🏃‍♂️ Performance Expectations

### Training Time Estimates

#### Small Model (25M parameters)

| CPU Type | Cores | RAM | Dataset | Time |
|----------|-------|-----|---------|------|
| Intel i5-8400 | 6 | 16GB | WikiText-2 | 12-18 hours |
| Intel i7-10700K | 8 | 32GB | WikiText-2 | 8-12 hours |
| AMD Ryzen 7 3700X | 8 | 32GB | WikiText-2 | 8-12 hours |
| Intel Xeon E5-2680 | 14 | 64GB | WikiText-2 | 6-10 hours |

#### Medium Model (117M parameters)

| CPU Type | Cores | RAM | Dataset | Time |
|----------|-------|-----|---------|------|
| Intel i7-10700K | 8 | 32GB | WikiText-103 | 3-5 days |
| AMD Ryzen 9 3900X | 12 | 64GB | WikiText-103 | 2-3 days |
| Intel Xeon Gold 6248 | 20 | 128GB | WikiText-103 | 1-2 days |

### Performance Benchmarks

```python
# Benchmark your system
import time
import torch
from llm_trainer.models import TransformerLM
from llm_trainer.config import ModelConfig

def benchmark_cpu():
    # Create small model for benchmarking
    config = ModelConfig(
        vocab_size=1000,
        d_model=128,
        n_heads=4,
        n_layers=2,
        max_seq_len=256
    )
    
    model = TransformerLM(config)
    
    # Benchmark forward pass
    batch_size = 4
    seq_len = 256
    input_ids = torch.randint(0, 1000, (batch_size, seq_len))
    
    # Warmup
    for _ in range(5):
        _ = model(input_ids)
    
    # Benchmark
    start_time = time.time()
    for _ in range(100):
        _ = model(input_ids)
    end_time = time.time()
    
    avg_time = (end_time - start_time) / 100
    print(f"Average forward pass time: {avg_time:.4f}s")
    print(f"Throughput: {batch_size / avg_time:.1f} samples/second")

benchmark_cpu()
```

## 🔄 CPU vs GPU Training Comparison

### Feature Comparison

| Feature | CPU Training | GPU Training |
|---------|-------------|-------------|
| **Hardware Cost** | $500-2000 | $1000-5000+ |
| **Training Speed** | 1x (baseline) | 10-50x faster |
| **Memory Type** | System RAM | VRAM |
| **Memory Capacity** | 8-128GB+ | 4-80GB |
| **Power Consumption** | 65-200W | 200-400W+ |
| **Accessibility** | Universal | CUDA required |
| **Development** | Excellent | Excellent |
| **Production** | Small models | All models |

### When to Choose CPU Training

#### ✅ CPU Training is Great For:
- **Learning and Development**: Understanding LLM training concepts
- **Small Models**: <100M parameters for specific use cases
- **Prototyping**: Testing architectures and hyperparameters
- **Budget Constraints**: Limited hardware budget
- **Accessibility**: No GPU access
- **Long-running Experiments**: Where time isn't critical

#### ⚠️ Consider GPU Training For:
- **Large Models**: >100M parameters
- **Production Training**: Models for deployment
- **Time-sensitive Projects**: Need results quickly
- **Hyperparameter Tuning**: Many experiments to run
- **Fine-tuning**: Adapting pre-trained models

### Migration Strategies

#### From CPU to GPU

```python
# CPU configuration
cpu_config = TrainingConfig(
    device="cpu",
    batch_size=2,
    gradient_accumulation_steps=8,
    use_amp=False,
    dataloader_pin_memory=False
)

# GPU adaptation
gpu_config = TrainingConfig(
    device="cuda",
    batch_size=16,                  # Increase batch size
    gradient_accumulation_steps=1,  # Reduce accumulation
    use_amp=True,                   # Enable mixed precision
    dataloader_pin_memory=True      # Enable for GPU
)
```

#### From GPU to CPU

```python
# GPU configuration  
gpu_config = TrainingConfig(
    device="cuda",
    batch_size=32,
    use_amp=True,
    dataloader_pin_memory=True
)

# CPU adaptation
cpu_config = TrainingConfig(
    device="cpu",
    batch_size=2,                   # Reduce batch size significantly
    gradient_accumulation_steps=16, # Maintain effective batch size
    use_amp=False,                  # Disable AMP
    dataloader_pin_memory=False     # Disable pin memory
)
```

## 🔧 Distributed CPU Training

### Multi-Process CPU Training

```bash
# Distributed CPU training using gloo backend
torchrun --nproc_per_node=4 scripts/train.py \
    --config configs/cpu_small_model.yaml \
    --output_dir ./output/cpu_distributed
```

### Configuration for Distributed Training

```python
training_config = TrainingConfig(
    device="cpu",
    distributed_backend="gloo",     # Automatically set for CPU
    world_size=4,                   # Number of processes
    batch_size=1,                   # Per-process batch size
    gradient_accumulation_steps=4,  # Maintain effective batch size
)
```

### Multi-Node CPU Training

```bash
# Node 0 (master)
torchrun --nnodes=2 --nproc_per_node=4 \
    --rdzv_id=100 --rdzv_backend=c10d \
    --rdzv_endpoint=192.168.1.100:29400 \
    scripts/train.py --config configs/cpu_medium_model.yaml

# Node 1 (worker)  
torchrun --nnodes=2 --nproc_per_node=4 \
    --rdzv_id=100 --rdzv_backend=c10d \
    --rdzv_endpoint=192.168.1.100:29400 \
    scripts/train.py --config configs/cpu_medium_model.yaml
```

## 🐛 Troubleshooting

### Common Issues and Solutions

#### High Memory Usage

**Problem**: System becomes unresponsive due to high memory usage

**Solutions**:
```python
# 1. Reduce batch size
training_config.batch_size = 1

# 2. Disable sequence packing temporarily
data_config.pack_sequences = False

# 3. Reduce model size
model_config.d_model = 128
model_config.n_layers = 2

# 4. Monitor memory usage
import psutil
def check_memory():
    mem = psutil.virtual_memory()
    if mem.percent > 85:
        print("⚠️  High memory usage!")
```

#### Slow Training Performance

**Problem**: Training is extremely slow

**Solutions**:
```python
# 1. Optimize PyTorch threading
import torch
torch.set_num_threads(psutil.cpu_count())

# 2. Reduce data loading overhead
training_config.dataloader_num_workers = 1
data_config.preprocessing_num_workers = 1

# 3. Use smaller sequences
data_config.max_length = 256

# 4. Disable unnecessary features
model_config.gradient_checkpointing = False
```

#### Process Hanging

**Problem**: Training process becomes unresponsive

**Solutions**:
```python
# 1. Disable multiprocessing
training_config.dataloader_num_workers = 0

# 2. Reduce prefetch factor
training_config.dataloader_prefetch_factor = 1

# 3. Check data loading
# Add timeout to data loading operations
```

#### Poor Convergence

**Problem**: Loss doesn't decrease or converges slowly

**Solutions**:
```python
# 1. Increase learning rate for smaller batches
training_config.learning_rate = 1e-3  # Higher than GPU default

# 2. Adjust warmup steps
training_config.warmup_steps = 500

# 3. Check effective batch size
effective_batch = training_config.batch_size * training_config.gradient_accumulation_steps
print(f"Effective batch size: {effective_batch}")

# 4. Verify data quality
# Ensure tokenizer is properly trained
```

### Error Messages and Solutions

#### "RuntimeError: Attempting to use AMP on CPU"

```python
# Solution: Disable AMP for CPU training
training_config.use_amp = False
```

#### "RuntimeError: NCCL backend not available"

```python
# Solution: Use gloo backend for CPU distributed training
training_config.distributed_backend = "gloo"
```

#### "DataLoader worker process died"

```python
# Solution: Reduce or disable workers
training_config.dataloader_num_workers = 0
data_config.preprocessing_num_workers = 1
```

#### "Out of memory" (System RAM)

```python
# Solution: Reduce memory usage
training_config.batch_size = 1
model_config.max_seq_len = 256
data_config.pack_sequences = False
```

## 📈 Monitoring CPU Training

### System Monitoring

```python
import psutil
import time
import matplotlib.pyplot as plt

class CPUTrainingMonitor:
    def __init__(self):
        self.cpu_usage = []
        self.memory_usage = []
        self.timestamps = []
    
    def log_system_stats(self):
        """Log current system statistics."""
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        
        self.cpu_usage.append(cpu_percent)
        self.memory_usage.append(memory.percent)
        self.timestamps.append(time.time())
        
        print(f"CPU: {cpu_percent:.1f}% | Memory: {memory.percent:.1f}% | Available: {memory.available/(1024**3):.1f}GB")
    
    def plot_usage(self):
        """Plot CPU and memory usage over time."""
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(self.timestamps, self.cpu_usage)
        plt.title('CPU Usage Over Time')
        plt.ylabel('CPU %')
        
        plt.subplot(1, 2, 2)
        plt.plot(self.timestamps, self.memory_usage)
        plt.title('Memory Usage Over Time')
        plt.ylabel('Memory %')
        
        plt.tight_layout()
        plt.show()

# Usage during training
monitor = CPUTrainingMonitor()
# Call monitor.log_system_stats() periodically during training
```

### Training Progress Monitoring

```python
# Enhanced logging for CPU training
training_config = TrainingConfig(
    device="cpu",
    logging_steps=25,           # Frequent logging
    eval_steps=250,             # Regular evaluation
    save_steps=500,             # Save progress frequently
    
    # Enable detailed logging
    report_to=["tensorboard"],  # TensorBoard logging
    logging_dir="./logs/cpu_training"
)

# Monitor training metrics
def log_training_stats(trainer, step):
    """Log detailed training statistics."""
    print(f"Step {step}:")
    print(f"  Loss: {trainer.state.log_history[-1]['train_loss']:.4f}")
    print(f"  Learning Rate: {trainer.state.log_history[-1]['learning_rate']:.2e}")
    print(f"  Steps/sec: {trainer.state.log_history[-1]['train_samples_per_second']:.2f}")
    
    # System stats
    memory = psutil.virtual_memory()
    print(f"  Memory: {memory.percent:.1f}%")
    print(f"  CPU: {psutil.cpu_percent():.1f}%")
```

## 🎯 Advanced CPU Training Topics

### Model Architecture Considerations

#### CPU-Optimized Architectures

```python
# Efficient architecture for CPU training
model_config = ModelConfig(
    # Smaller dimensions reduce computation
    d_model=256,                    # vs 768 for GPU
    n_heads=4,                      # vs 12 for GPU
    n_layers=4,                     # vs 12 for GPU
    
    # Shorter sequences reduce memory
    max_seq_len=512,                # vs 1024+ for GPU
    
    # Optimizations
    activation="relu",              # Faster than GELU/SwiGLU
    pre_norm=True,                  # Slightly more stable
    gradient_checkpointing=False    # Less beneficial on CPU
)
```

#### Memory-Efficient Techniques

```python
# Gradient accumulation for effective large batches
training_config = TrainingConfig(
    batch_size=1,                   # Minimal memory per step
    gradient_accumulation_steps=32, # Effective batch = 32
    
    # Memory optimizations
    dataloader_pin_memory=False,    # Save memory
    max_grad_norm=1.0,             # Prevent gradient explosion
)
```

### Custom Training Loops

```python
# Custom training loop with CPU optimizations
import torch
from torch.utils.data import DataLoader
from llm_trainer.models import TransformerLM
from llm_trainer.tokenizer import BPETokenizer

def train_cpu_optimized(model, tokenizer, dataset, config):
    """CPU-optimized training loop."""
    
    # Setup
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    dataloader = DataLoader(
        dataset, 
        batch_size=config.batch_size,
        num_workers=0,  # Single-threaded for stability
        pin_memory=False
    )
    
    # Training loop
    step = 0
    accumulated_loss = 0
    
    for epoch in range(config.num_epochs):
        for batch_idx, batch in enumerate(dataloader):
            # Forward pass
            outputs = model(**batch)
            loss = outputs.loss / config.gradient_accumulation_steps
            
            # Backward pass
            loss.backward()
            accumulated_loss += loss.item()
            
            # Update weights
            if (batch_idx + 1) % config.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad()
                
                step += 1
                
                # Logging
                if step % config.logging_steps == 0:
                    avg_loss = accumulated_loss / config.gradient_accumulation_steps
                    print(f"Step {step}: Loss = {avg_loss:.4f}")
                    accumulated_loss = 0
                
                # Save checkpoint
                if step % config.save_steps == 0:
                    torch.save({
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'step': step,
                        'loss': avg_loss,
                    }, f"checkpoint_step_{step}.pt")
```

### Hyperparameter Tuning for CPU

```python
# CPU-specific hyperparameter ranges
cpu_hyperparameters = {
    'learning_rate': [5e-4, 8e-4, 1e-3, 2e-3],      # Higher than GPU
    'batch_size': [1, 2, 4],                         # Small sizes
    'gradient_accumulation_steps': [8, 16, 32],      # Maintain effective batch
    'warmup_steps': [200, 500, 1000],               # Shorter warmup
    'weight_decay': [0.01, 0.05, 0.1],              # Standard range
}

def tune_cpu_hyperparameters(base_config, param_grid, num_trials=10):
    """Simple hyperparameter tuning for CPU training."""
    import itertools
    import random
    
    # Generate parameter combinations
    keys = param_grid.keys()
    values = param_grid.values()
    combinations = list(itertools.product(*values))
    
    # Random sample for efficiency
    selected_combinations = random.sample(combinations, min(num_trials, len(combinations)))
    
    results = []
    for i, params in enumerate(selected_combinations):
        print(f"Trial {i+1}/{len(selected_combinations)}")
        
        # Create config with these parameters
        config = base_config.copy()
        for key, value in zip(keys, params):
            setattr(config, key, value)
        
        # Train and evaluate (simplified)
        # result = train_and_evaluate(config)
        # results.append((params, result))
        
    return results
```

## 📚 Additional Resources

### Example Scripts

- [`examples/train_small_model.py`](../examples/train_small_model.py) - Complete CPU training example
- [`examples/complete_pipeline.py`](../examples/complete_pipeline.py) - End-to-end pipeline with CPU support
- [`test_cpu_training.py`](../test_cpu_training.py) - Comprehensive test suite

### Configuration Templates

- [`configs/cpu_small_model.yaml`](../configs/cpu_small_model.yaml) - Small model for CPU
- [`configs/cpu_medium_model.yaml`](../configs/cpu_medium_model.yaml) - Medium model for CPU

### Related Documentation

- [Training Guide](training.md) - General training documentation
- [Getting Started](getting_started.md) - Setup and first steps
- [Configuration Reference](api.md) - Complete configuration options

## 🤝 Community and Support

### Getting Help

- **GitHub Issues**: [Report bugs and request features](https://github.com/OEvortex/llm-trainer/issues)
- **Discussions**: [Ask questions and share experiences](https://github.com/OEvortex/llm-trainer/discussions)
- **Documentation**: [Browse all documentation](README.md)

### Contributing

We welcome contributions to improve CPU training support:

1. **Performance Optimizations**: Better CPU utilization strategies
2. **Documentation**: Improved guides and examples
3. **Testing**: More comprehensive CPU training tests
4. **Configuration**: Additional pre-tuned configurations

---

Happy CPU training! 🚀💻