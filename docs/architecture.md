# Model Architecture

This document provides a comprehensive overview of the Transformer architecture implemented in LLM Trainer, built from scratch in PyTorch.

## Overview

Our implementation follows the standard Transformer decoder architecture optimized for autoregressive language modeling. The model is designed to be:

- **Scalable**: From 25M to 1B+ parameters
- **Efficient**: Memory-optimized with gradient checkpointing
- **Flexible**: Configurable components and hyperparameters
- **Modern**: Incorporates latest architectural improvements

### Key Features

- Multi-head self-attention with causal masking
- Feed-forward networks with configurable activation functions
- Layer normalization (pre-norm and post-norm support)
- Positional encoding (sinusoidal and learned embeddings)
- Gradient checkpointing for memory efficiency
- Mixed precision training support
- Rotary Position Embedding (RoPE) option

## Architecture Components

### Model Structure

```
Input Tokens
    ↓
Token Embedding + Positional Encoding
    ↓
┌─────────────────────────────────────┐
│ Transformer Block 1                │
│ ┌─────────────────────────────────┐ │
│ │ Multi-Head Self-Attention      │ │
│ │ + Residual + LayerNorm         │ │
│ └─────────────────────────────────┘ │
│ ┌─────────────────────────────────┐ │
│ │ Feed-Forward Network           │ │
│ │ + Residual + LayerNorm         │ │
│ └─────────────────────────────────┘ │
└─────────────────────────────────────┘
    ↓
... (N layers) ...
    ↓
Final Layer Normalization
    ↓
Language Modeling Head
    ↓
Output Logits
```

### 1. Token Embeddings

**Purpose**: Convert discrete tokens to continuous vector representations

```python
# Configuration
vocab_size: int = 50000      # Vocabulary size
d_model: int = 768           # Model dimension
```

**Features**:
- Learnable embedding matrix: `[vocab_size, d_model]`
- Shared weights with output projection (optional)
- Dropout for regularization

### 2. Positional Encoding

**Purpose**: Inject sequence position information

> [!NOTE]
> **Two Types Supported:**
> - **Sinusoidal**: Fixed mathematical encoding (no parameters)
> - **Learned**: Trainable position embeddings

```python
# Sinusoidal encoding formula
PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```

### 3. Multi-Head Self-Attention

**Purpose**: Allow model to attend to different positions and representation subspaces

```python
# Configuration
n_heads: int = 12            # Number of attention heads
d_head: int = d_model // n_heads  # Dimension per head
attention_dropout: float = 0.1     # Attention dropout rate
```

**Mechanism**:
1. **Linear Projections**: Q, K, V = Linear(x)
2. **Scaled Dot-Product**: Attention(Q,K,V) = softmax(QK^T/√d_k)V
3. **Causal Masking**: Prevent attending to future positions
4. **Multi-Head**: Concatenate multiple attention heads

> [!IMPORTANT]
> **Causal Masking**: Essential for autoregressive generation - ensures position i can only attend to positions ≤ i

### 4. Feed-Forward Network

**Purpose**: Apply position-wise transformations

```python
# Configuration
d_ff: int = 3072             # Feed-forward dimension (usually 4 * d_model)
activation: str = "gelu"     # Activation function
dropout: float = 0.1         # Dropout rate
```

**Structure**:
```python
FFN(x) = Linear2(Dropout(Activation(Linear1(x))))
```

**Supported Activations**:
- **GELU**: Gaussian Error Linear Unit (default)
- **ReLU**: Rectified Linear Unit
- **SwiGLU**: Swish-Gated Linear Unit (advanced)

### 5. Layer Normalization

**Purpose**: Stabilize training and improve convergence

> [!TIP]
> **Pre-norm vs Post-norm:**
> - **Pre-norm**: LayerNorm before attention/FFN (default, more stable)
> - **Post-norm**: LayerNorm after attention/FFN (original Transformer)

```python
# Pre-norm (recommended)
x = x + Attention(LayerNorm(x))
x = x + FFN(LayerNorm(x))

# Post-norm
x = LayerNorm(x + Attention(x))
x = LayerNorm(x + FFN(x))
```

### 6. Residual Connections

**Purpose**: Enable gradient flow and training stability

- Applied around attention and feed-forward blocks
- Critical for training deep networks
- Combined with layer normalization

## Model Configurations

### Small Model (25M parameters)
```yaml
d_model: 256
n_heads: 4
n_layers: 4
d_ff: 1024
max_seq_len: 512
```

### Medium Model (117M parameters)
```yaml
d_model: 768
n_heads: 12
n_layers: 12
d_ff: 3072
max_seq_len: 1024
```

### Large Model (345M parameters)
```yaml
d_model: 1024
n_heads: 16
n_layers: 24
d_ff: 4096
max_seq_len: 2048
```

## Advanced Features

### Gradient Checkpointing

**Purpose**: Trade computation for memory efficiency

> [!WARNING]
> Increases training time by ~20% but reduces memory usage significantly

```python
gradient_checkpointing: bool = True  # Enable for large models
```

### Mixed Precision Training

**Purpose**: Accelerate training and reduce memory usage

```python
use_amp: bool = True
amp_dtype: str = "float16"  # or "bfloat16"
```

### Attention Optimizations

- **Flash Attention**: Memory-efficient attention computation
- **Attention Dropout**: Regularization during training
- **Key-Value Caching**: Efficient inference for generation

## Memory and Compute Analysis

### Parameter Count Formula

```python
def calculate_parameters(vocab_size, d_model, n_layers, d_ff):
    # Embeddings
    embedding_params = vocab_size * d_model
    
    # Each transformer layer
    attention_params = 4 * d_model * d_model  # Q, K, V, O projections
    ffn_params = 2 * d_model * d_ff           # Two linear layers
    layer_params = attention_params + ffn_params
    
    # Total
    total_params = embedding_params + n_layers * layer_params
    return total_params
```

### Memory Requirements

| Component | Memory Usage |
|-----------|--------------|
| Model Parameters | 4 bytes × num_parameters |
| Gradients | 4 bytes × num_parameters |
| Optimizer States | 8 bytes × num_parameters (AdamW) |
| Activations | Depends on batch_size × seq_len × d_model |

> [!NOTE]
> **Memory Estimation**: Total ≈ 16 × num_parameters + activation_memory

## Implementation Details

### Attention Mechanism

```python
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.o_proj = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        batch_size, seq_len = x.shape[:2]
        
        # Project to Q, K, V
        q = self.q_proj(x).view(batch_size, seq_len, self.n_heads, self.d_head)
        k = self.k_proj(x).view(batch_size, seq_len, self.n_heads, self.d_head)
        v = self.v_proj(x).view(batch_size, seq_len, self.n_heads, self.d_head)
        
        # Transpose for attention computation
        q = q.transpose(1, 2)  # [batch, n_heads, seq_len, d_head]
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_head)
        
        if mask is not None:
            scores.masked_fill_(mask == 0, -1e9)
        
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        attn_output = torch.matmul(attn_weights, v)
        
        # Reshape and project
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, self.d_model)
        
        return self.o_proj(attn_output)
```

### Causal Mask Generation

```python
def create_causal_mask(seq_len, device):
    """Create lower triangular causal mask."""
    mask = torch.tril(torch.ones(seq_len, seq_len, device=device))
    return mask.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, seq_len]
```

## Performance Considerations

### Training Tips

> [!TIP]
> **For Optimal Performance:**
> - Use gradient checkpointing for models >100M parameters
> - Enable mixed precision training
> - Use appropriate batch size for your hardware
> - Consider sequence packing for efficiency

### Inference Optimizations

- **KV-Cache**: Store key-value pairs for efficient generation
- **Batched Generation**: Process multiple sequences simultaneously
- **Model Quantization**: Reduce precision for deployment

## Customization Options

The architecture is highly configurable through the `ModelConfig` class:

```python
from llm_trainer.config import ModelConfig

config = ModelConfig(
    vocab_size=50000,
    d_model=768,
    n_heads=12,
    n_layers=12,
    d_ff=3072,
    max_seq_len=1024,
    dropout=0.1,
    attention_dropout=0.1,
    activation="gelu",
    pre_norm=True,
    use_learned_pos_emb=False,
    gradient_checkpointing=False
)
```

## References

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) - Original Transformer paper
- [Language Models are Unsupervised Multitask Learners](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) - GPT-2 paper
- [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864) - RoPE
- [GLU Variants Improve Transformer](https://arxiv.org/abs/2002.05202) - SwiGLU activation

---

For implementation details, see the source code in `src/llm_trainer/models/`.