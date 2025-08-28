# SafeTensors Support in LLM Trainer

LLM Trainer now supports SafeTensors format for model saving and loading, providing better performance, safety, and compatibility with the Hugging Face ecosystem.

## Overview

SafeTensors is a simple, safe, and fast format for storing tensors. It provides several advantages over PyTorch's default pickle-based format:

- **Safety**: Prevents arbitrary code execution during loading
- **Performance**: Faster loading times and better memory efficiency
- **Sharding**: Automatic support for large model sharding
- **Cross-platform**: Better compatibility across different systems
- **Memory mapping**: Efficient memory usage for large models

## Installation

SafeTensors support is optional. To enable it:

```bash
# Install SafeTensors separately
pip install safetensors

# Or install with the safetensors extra
pip install -e ".[safetensors]"

# Or install all optional dependencies
pip install -e ".[full]"
```

## Basic Usage

### Saving Models

```python
from llm_trainer.config import ModelConfig
from llm_trainer.models import TransformerLM

# Create and configure model
config = ModelConfig(vocab_size=32000, d_model=512, n_heads=8, n_layers=6)
model = TransformerLM(config)

# Save with SafeTensors (default)
model.save_pretrained("./my_model", safe_serialization=True)

# Save with PyTorch format
model.save_pretrained("./my_model", safe_serialization=False)
```

### Loading Models

```python
# Load model (automatically detects format)
model = TransformerLM.from_pretrained("./my_model")

# The loader will try SafeTensors first, then fallback to PyTorch format
```

## Sharding Support

For large models, SafeTensors automatically handles sharding:

```python
# Save large model with custom shard size
model.save_pretrained(
    "./large_model",
    safe_serialization=True,
    max_shard_size="2GB"  # Each shard will be max 2GB
)
```

### Generated Files

For sharded models, you'll see files like:
```
large_model/
├── config.json
├── model.safetensors.index.json  # Index file mapping parameters to shards
├── model-00001-of-00003.safetensors
├── model-00002-of-00003.safetensors
└── model-00003-of-00003.safetensors
```

For single-file models:
```
my_model/
├── config.json
└── model.safetensors
```

## Advanced Features

### Utility Functions

```python
from llm_trainer.models import (
    save_model_safetensors,
    load_model_safetensors,
    convert_pytorch_to_safetensors,
    get_safetensors_metadata,
    list_safetensors_tensors,
    is_safetensors_available
)

# Check if SafeTensors is available
if is_safetensors_available():
    print("SafeTensors is ready to use!")

# Convert existing PyTorch model to SafeTensors
convert_pytorch_to_safetensors(
    "model.bin", 
    "model.safetensors"
)

# Inspect SafeTensors metadata
metadata = get_safetensors_metadata("model.safetensors")
print("Model metadata:", metadata)

# List all tensors in the file
tensors = list_safetensors_tensors("model.safetensors")
print(f"Found {len(tensors)} tensors")
```

### Custom Metadata

```python
# Save with custom metadata
model.save_pretrained(
    "./my_model",
    safe_serialization=True,
    metadata={
        "training_dataset": "custom_dataset",
        "training_steps": "10000",
        "custom_field": "value"
    }
)
```

## Configuration Options

### save_pretrained() Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `save_directory` | str | Required | Directory to save the model |
| `safe_serialization` | bool | `True` | Whether to use SafeTensors format |
| `max_shard_size` | str/int | `"5GB"` | Maximum size per shard |
| `push_to_hub` | bool | `False` | Push to HF Hub (not implemented) |

### Shard Size Examples

```python
# Different shard size formats
model.save_pretrained("./model", max_shard_size="1GB")    # 1 gigabyte
model.save_pretrained("./model", max_shard_size="500MB")  # 500 megabytes
model.save_pretrained("./model", max_shard_size=1073741824)  # 1GB in bytes
```

## Performance Comparison

| Format | Loading Speed | Memory Usage | Safety | Sharding |
|--------|---------------|--------------|--------|----------|
| PyTorch (.bin) | Baseline | Higher | ⚠️ | Manual |
| SafeTensors | 2-5x faster | Lower | ✅ | Automatic |

## Migration Guide

### From PyTorch Format

If you have existing models saved in PyTorch format:

```python
# Method 1: Convert existing files
from llm_trainer.models import convert_pytorch_to_safetensors

convert_pytorch_to_safetensors(
    "old_model/pytorch_model.bin",
    "old_model/model.safetensors"
)

# Method 2: Load and re-save
model = TransformerLM.from_pretrained("old_model")  # Loads pytorch_model.bin
model.save_pretrained("new_model", safe_serialization=True)  # Saves as SafeTensors
```

### Backward Compatibility

The framework maintains full backward compatibility:

- New models default to SafeTensors format
- Loading automatically detects and handles both formats
- If SafeTensors is not available, automatically falls back to PyTorch format

## Troubleshooting

### SafeTensors Not Available

```python
# Check availability
from llm_trainer.models import is_safetensors_available

if not is_safetensors_available():
    print("Install SafeTensors: pip install safetensors")
```

### Loading Issues

```python
# Force PyTorch format loading
model.save_pretrained("./model", safe_serialization=False)

# Manual SafeTensors loading
from llm_trainer.models import load_model_safetensors
load_model_safetensors(model, "./model")
```

### Memory Issues with Large Models

```python
# Use smaller shard sizes for very large models
model.save_pretrained(
    "./huge_model",
    max_shard_size="1GB"  # Smaller shards for limited memory systems
)
```

## Integration with Hugging Face

SafeTensors models are fully compatible with Hugging Face:

```python
# Save in HF-compatible format
model.save_pretrained("./hf_compatible_model")

# Files created are compatible with:
# - transformers.AutoModel.from_pretrained()
# - Hugging Face Hub
# - Hugging Face inference endpoints
```

## Best Practices

1. **Default to SafeTensors**: Use `safe_serialization=True` (default) for new models
2. **Appropriate Sharding**: Use 2-5GB shard sizes for optimal performance
3. **Include Metadata**: Add useful metadata for model tracking
4. **Test Loading**: Always test model loading after saving
5. **Monitor Memory**: Use smaller shards on memory-constrained systems

## Example: Complete Workflow

```python
from llm_trainer.config import ModelConfig
from llm_trainer.models import TransformerLM

# 1. Create model
config = ModelConfig(
    vocab_size=50000,
    d_model=768,
    n_heads=12,
    n_layers=12
)
model = TransformerLM(config)

# 2. Train model (your training code here)
# trainer.train(model, ...)

# 3. Save with SafeTensors
model.save_pretrained(
    "./trained_model",
    safe_serialization=True,
    max_shard_size="2GB"
)

# 4. Load for inference
loaded_model = TransformerLM.from_pretrained("./trained_model")

# 5. Generate text
output = loaded_model.generate(input_ids, max_length=100)
```

## See Also

- [SafeTensors Documentation](https://huggingface.co/docs/safetensors/)
- [Hugging Face Model Hub](https://huggingface.co/models)
- [LLM Trainer Examples](../examples/safetensors_example.py)