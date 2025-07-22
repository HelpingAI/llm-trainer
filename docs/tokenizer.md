# Tokenizer Details

This document provides comprehensive coverage of the Byte Pair Encoding (BPE) tokenizer implemented in LLM Trainer, built from scratch with modern features.

## Key Features

- **BPE Algorithm**: Complete implementation from scratch with optimizations
- **Unicode Support**: Full Unicode support including international characters
- **Emoji Handling**: Proper tokenization of emojis and symbols
- **Special Tokens**: Configurable special tokens (PAD, UNK, BOS, EOS)
- **Efficient Processing**: Fast encoding/decoding with caching
- **Dataset Integration**: Seamless integration with Hugging Face datasets
- **Streaming Support**: Handle large datasets without memory issues
- **Regex Pre-tokenization**: Advanced pattern matching for better tokenization

> [!NOTE]
> Our BPE implementation follows the GPT-2 tokenization strategy with enhancements for better Unicode and emoji support.

## Quick Start

### Basic Usage

```python
from llm_trainer.tokenizer import BPETokenizer

# Create tokenizer
tokenizer = BPETokenizer()

# Train on dataset
tokenizer.train_from_dataset(
    dataset_name="wikitext",
    dataset_config="wikitext-2-raw-v1",
    vocab_size=32000,
    max_samples=100000,  # Limit for faster training
    verbose=True
)

# Encode text to token IDs
text = "Hello world! This is a test with emojis and symbols."
token_ids = tokenizer.encode(text)
print(f"Token IDs: {token_ids}")

# Decode back to text
decoded_text = tokenizer.decode(token_ids)
print(f"Decoded: {decoded_text}")

# Get vocabulary statistics
stats = tokenizer.get_vocab_stats()
print(f"Vocabulary size: {stats['total_tokens']}")
```

### Advanced Configuration

```python
# Custom special tokens
tokenizer = BPETokenizer()
tokenizer.special_tokens = ["<pad>", "<unk>", "<bos>", "<eos>", "<mask>"]

# Train with custom parameters
tokenizer.train_from_dataset(
    dataset_name="your_dataset",
    vocab_size=50000,
    max_samples=None,  # Use entire dataset
    min_frequency=2,   # Minimum token frequency
    verbose=True
)
```

> [!TIP]
> **Performance Tips:**
> - Use `max_samples` for faster experimentation
> - Increase `vocab_size` for better text representation
> - Cache trained tokenizers to avoid retraining

## BPE Algorithm Overview

### What is Byte Pair Encoding?

BPE is a subword tokenization algorithm that:

1. **Starts with characters**: Initial vocabulary contains all characters
2. **Finds frequent pairs**: Identifies most common adjacent character/token pairs
3. **Merges iteratively**: Combines frequent pairs into new tokens
4. **Builds vocabulary**: Continues until reaching desired vocabulary size

### Algorithm Steps

```python
# Simplified BPE algorithm
def train_bpe(texts, vocab_size):
    # 1. Initialize with character-level tokens
    vocab = set(char for text in texts for char in text)
    
    # 2. Get word frequencies
    word_freqs = get_word_frequencies(texts)
    
    # 3. Iteratively merge most frequent pairs
    merges = []
    while len(vocab) < vocab_size:
        # Find most frequent pair
        pairs = get_pair_frequencies(word_freqs)
        if not pairs:
            break
        
        best_pair = max(pairs, key=pairs.get)
        
        # Merge the pair
        word_freqs = merge_pair(word_freqs, best_pair)
        merges.append(best_pair)
        vocab.add(''.join(best_pair))
    
    return vocab, merges
```

### Pre-tokenization Patterns

Our tokenizer uses sophisticated regex patterns for pre-tokenization:

```python
# Regex pattern components
patterns = [
    r"'(?:[sdmt]|ll|ve|re|d|m)",  # Contractions
    r" ?[\u0041-\u005A\u0061-\u007A]+",  # Letters
    r" ?[0-9]+",  # Numbers
    r" ?[^\s\w]+",  # Punctuation and symbols
    r"\s+(?!\S)",  # Whitespace
    r"\s+"  # Remaining whitespace
]
```

> [!IMPORTANT]
> **Pre-tokenization Benefits:**
> - Better handling of punctuation
> - Consistent treatment of contractions
> - Improved emoji and symbol tokenization
> - More efficient vocabulary usage

## Implementation Details

### Core Classes

#### BPETokenizer

Main tokenizer class with full BPE implementation:

```python
class BPETokenizer(BaseTokenizer):
    def __init__(self):
        super().__init__()
        self.merges = []  # List of merge rules
        self.cache = {}   # Encoding cache for speed
        
    def train(self, texts, vocab_size=50000):
        """Train BPE on text corpus."""
        # Implementation details...
        
    def encode(self, text, add_special_tokens=True):
        """Encode text to token IDs."""
        # Implementation details...
        
    def decode(self, token_ids, skip_special_tokens=True):
        """Decode token IDs to text."""
        # Implementation details...
```

#### BaseTokenizer

Abstract base class providing common functionality:

```python
class BaseTokenizer:
    def __init__(self):
        self.vocab = {}  # token -> id mapping
        self.id_to_token = {}  # id -> token mapping
        self.special_tokens = ["<pad>", "<unk>", "<bos>", "<eos>"]
        
    @property
    def vocab_size(self):
        return len(self.vocab)
        
    def save_pretrained(self, save_directory):
        """Save tokenizer to directory."""
        
    @classmethod
    def from_pretrained(cls, load_directory):
        """Load tokenizer from directory."""
```

### Training Process

#### Step 1: Text Preprocessing

```python
def preprocess_text(text):
    """Clean and normalize text."""
    # Unicode normalization
    text = unicodedata.normalize('NFC', text)
    
    # Remove control characters
    text = ''.join(char for char in text if not unicodedata.category(char).startswith('C'))
    
    return text
```

#### Step 2: Word Frequency Counting

```python
def get_word_frequencies(texts):
    """Count word frequencies with character splitting."""
    word_freqs = defaultdict(int)
    
    for text in texts:
        # Pre-tokenize using regex
        words = re.findall(self.pat, text)
        
        for word in words:
            # Split into characters with end-of-word marker
            word_tokens = list(word) + ['</w>']
            word_freqs[tuple(word_tokens)] += 1
    
    return word_freqs
```

#### Step 3: Pair Frequency Analysis

```python
def get_pair_frequencies(word_freqs):
    """Count frequencies of adjacent token pairs."""
    pairs = defaultdict(int)
    
    for word, freq in word_freqs.items():
        for i in range(len(word) - 1):
            pair = (word[i], word[i + 1])
            pairs[pair] += freq
    
    return pairs
```

#### Step 4: Vocabulary Building

```python
def build_vocabulary(self):
    """Build final vocabulary with special tokens."""
    # Add special tokens first
    for token in self.special_tokens:
        self.vocab[token] = len(self.vocab)
    
    # Add learned tokens
    for token in self.learned_tokens:
        if token not in self.vocab:
            self.vocab[token] = len(self.vocab)
    
    # Create reverse mapping
    self.id_to_token = {id: token for token, id in self.vocab.items()}
```

## Advanced Features

### Unicode and Emoji Support

Our tokenizer handles complex Unicode scenarios:

```python
# Examples of supported text
texts = [
    "Hello world! 👋",  # Basic emoji
    "Café naïve résumé",  # Accented characters
    "北京大学",  # Chinese characters
    "🏳️‍🌈 🏳️‍⚧️",  # Complex emoji sequences
    "🧑‍💻👨‍👩‍👧‍👦",  # Multi-person emojis
    "Москва",  # Cyrillic
    "العربية",  # Arabic
]

for text in texts:
    tokens = tokenizer.encode(text)
    decoded = tokenizer.decode(tokens)
    assert text == decoded  # Perfect reconstruction
```

### Caching System

For improved performance during encoding:

```python
class BPETokenizer:
    def __init__(self):
        self.cache = {}  # word -> tokens mapping
        
    def _apply_bpe(self, word):
        """Apply BPE with caching."""
        if word in self.cache:
            return self.cache[word]
        
        # Apply BPE algorithm
        tokens = self._bpe_encode(word)
        
        # Cache result
        self.cache[word] = tokens
        return tokens
```

### Streaming Dataset Support

For large datasets that don't fit in memory:

```python
def train_from_dataset(self, dataset_name, vocab_size=50000, streaming=True):
    """Train tokenizer with streaming support."""
    if streaming:
        # Load dataset in streaming mode
        dataset = load_dataset(dataset_name, streaming=True)
        
        # Process in chunks
        for chunk in self._chunk_dataset(dataset, chunk_size=10000):
            self._update_frequencies(chunk)
    else:
        # Load entire dataset
        dataset = load_dataset(dataset_name)
        self._train_on_full_dataset(dataset)
```

## Configuration Options

### Vocabulary Size Guidelines

| Use Case | Recommended Size | Notes |
|----------|------------------|-------|
| Experimentation | 8K - 16K | Fast training, basic quality |
| Small Models | 16K - 32K | Good balance of speed/quality |
| Medium Models | 32K - 50K | Standard for most applications |
| Large Models | 50K - 100K | Best quality, slower training |

> [!WARNING]
> **Memory Considerations**: Larger vocabularies require more memory for embedding layers.

### Special Token Configuration

```python
# Default special tokens
default_special_tokens = ["<pad>", "<unk>", "<bos>", "<eos>"]

# Extended special tokens for specific tasks
extended_special_tokens = [
    "<pad>",    # Padding token
    "<unk>",    # Unknown token
    "<bos>",    # Beginning of sequence
    "<eos>",    # End of sequence
    "<mask>",   # Masked language modeling
    "<sep>",    # Separator token
    "<cls>",    # Classification token
]

# Custom domain-specific tokens
domain_tokens = [
    "<code>", "</code>",  # Code blocks
    "<math>", "</math>",  # Mathematical expressions
    "<url>",              # URL placeholder
    "<email>",            # Email placeholder
]
```

### Training Parameters

```python
# Comprehensive training configuration
tokenizer.train_from_dataset(
    dataset_name="wikitext",
    dataset_config="wikitext-103-raw-v1",
    vocab_size=50000,
    
    # Performance options
    max_samples=None,           # Use entire dataset
    min_frequency=2,            # Minimum token frequency
    
    # Processing options
    lowercase=False,            # Preserve case
    strip_accents=False,        # Preserve accents
    
    # Output options
    verbose=True,               # Show progress
    save_path="./tokenizer",    # Auto-save location
)
```

## Usage Examples

### Training on Custom Dataset

```python
# Train on your own text files
def train_on_custom_data():
    tokenizer = BPETokenizer()
    
    # Read custom text files
    texts = []
    for file_path in ["data1.txt", "data2.txt", "data3.txt"]:
        with open(file_path, 'r', encoding='utf-8') as f:
            texts.append(f.read())
    
    # Train tokenizer
    tokenizer.train(texts, vocab_size=32000)
    
    # Save for later use
    tokenizer.save_pretrained("./my_tokenizer")
    
    return tokenizer
```

### Loading Pre-trained Tokenizer

```python
# Load previously trained tokenizer
tokenizer = BPETokenizer.from_pretrained("./my_tokenizer")

# Use for encoding
text = "This is a test sentence."
token_ids = tokenizer.encode(text)
print(f"Tokens: {[tokenizer.id_to_token[id] for id in token_ids]}")
```

### Batch Processing

```python
# Efficient batch encoding
def batch_encode(tokenizer, texts, max_length=512):
    """Encode multiple texts efficiently."""
    batch_ids = []
    
    for text in texts:
        ids = tokenizer.encode(text, add_special_tokens=True)
        
        # Truncate or pad to max_length
        if len(ids) > max_length:
            ids = ids[:max_length]
        else:
            ids.extend([tokenizer.vocab["<pad>"]] * (max_length - len(ids)))
        
        batch_ids.append(ids)
    
    return batch_ids

# Usage
texts = ["First text", "Second text", "Third text"]
batch_ids = batch_encode(tokenizer, texts)
```

## Performance Optimization

### Memory Efficiency

```python
# Memory-efficient training for large datasets
def memory_efficient_training():
    tokenizer = BPETokenizer()
    
    # Use streaming with limited samples
    tokenizer.train_from_dataset(
        dataset_name="large_dataset",
        vocab_size=50000,
        max_samples=1000000,  # Limit memory usage
        streaming=True,       # Stream data
        cache_dir="./cache"   # Cache preprocessed data
    )
    
    return tokenizer
```

### Speed Optimization

```python
# Speed optimizations
class OptimizedBPETokenizer(BPETokenizer):
    def __init__(self):
        super().__init__()
        self.cache_size_limit = 10000  # Limit cache size
        
    def _apply_bpe(self, word):
        # Check cache first
        if word in self.cache:
            return self.cache[word]
        
        # Apply BPE
        tokens = super()._apply_bpe(word)
        
        # Manage cache size
        if len(self.cache) >= self.cache_size_limit:
            # Remove oldest entries
            self.cache.clear()
        
        self.cache[word] = tokens
        return tokens
```

## Evaluation and Analysis

### Tokenization Quality Metrics

```python
def evaluate_tokenizer(tokenizer, test_texts):
    """Evaluate tokenizer quality."""
    total_chars = 0
    total_tokens = 0
    
    for text in test_texts:
        tokens = tokenizer.encode(text, add_special_tokens=False)
        total_chars += len(text)
        total_tokens += len(tokens)
    
    # Compression ratio
    compression_ratio = total_chars / total_tokens
    
    # Vocabulary utilization
    used_tokens = set()
    for text in test_texts:
        tokens = tokenizer.encode(text, add_special_tokens=False)
        used_tokens.update(tokens)
    
    vocab_utilization = len(used_tokens) / tokenizer.vocab_size
    
    return {
        "compression_ratio": compression_ratio,
        "vocab_utilization": vocab_utilization,
        "avg_tokens_per_char": 1 / compression_ratio
    }
```

### Vocabulary Analysis

```python
def analyze_vocabulary(tokenizer):
    """Analyze learned vocabulary."""
    stats = tokenizer.get_vocab_stats()
    
    # Token length distribution
    token_lengths = [len(token) for token in tokenizer.vocab.keys()]
    
    # Character coverage
    chars_in_vocab = set()
    for token in tokenizer.vocab.keys():
        chars_in_vocab.update(token)
    
    return {
        "vocab_size": stats["total_tokens"],
        "avg_token_length": sum(token_lengths) / len(token_lengths),
        "max_token_length": max(token_lengths),
        "character_coverage": len(chars_in_vocab)
    }
```

## Integration with Training Pipeline

### Seamless Integration

```python
from llm_trainer.config import DataConfig
from llm_trainer.tokenizer import BPETokenizer
from llm_trainer.training import Trainer

# Configure data with tokenizer
data_config = DataConfig(
    dataset_name="wikitext",
    dataset_config="wikitext-103-raw-v1",
    vocab_size=50000,
    max_length=1024
)

# Train or load tokenizer
tokenizer = BPETokenizer()
if not os.path.exists("./tokenizer"):
    tokenizer.train_from_dataset(
        dataset_name=data_config.dataset_name,
        dataset_config=data_config.dataset_config,
        vocab_size=data_config.vocab_size
    )
    tokenizer.save_pretrained("./tokenizer")
else:
    tokenizer = BPETokenizer.from_pretrained("./tokenizer")

# Use in training
trainer = Trainer(model, tokenizer, training_config)
```

## Best Practices

### Training Recommendations

> [!TIP]
> **Best Practices:**
> - Train on diverse, high-quality text
> - Use appropriate vocabulary size for your model
> - Save tokenizers for reproducibility
> - Validate tokenization quality before training
> - Consider domain-specific vocabularies

### Common Pitfalls

> [!WARNING]
> **Avoid These Mistakes:**
> - Training on low-quality or biased data
> - Using vocabulary size that's too small/large
> - Ignoring Unicode normalization
> - Not handling special tokens properly
> - Forgetting to save trained tokenizers

### Production Considerations

```python
# Production-ready tokenizer setup
def setup_production_tokenizer():
    # Load pre-trained tokenizer
    tokenizer = BPETokenizer.from_pretrained("./production_tokenizer")
    
    # Validate tokenizer
    test_texts = ["Hello world!", "Test with emojis 🚀", "Unicode: café naïve"]
    for text in test_texts:
        tokens = tokenizer.encode(text)
        decoded = tokenizer.decode(tokens)
        assert text == decoded, f"Tokenization failed for: {text}"
    
    # Log statistics
    stats = tokenizer.get_vocab_stats()
    print(f"Loaded tokenizer with {stats['total_tokens']} tokens")
    
    return tokenizer
```

## Troubleshooting

### Common Issues

| Issue | Symptoms | Solution |
|-------|----------|----------|
| **Poor Compression** | Too many tokens per character | Increase vocab size, improve data quality |
| **Memory Error** | OOM during training | Use streaming, reduce max_samples |
| **Slow Training** | Long training time | Use max_samples, enable caching |
| **Unicode Issues** | Garbled text output | Check text encoding, enable Unicode support |

### Debugging Tools

```python
# Debug tokenization
def debug_tokenization(tokenizer, text):
    """Debug tokenization process."""
    print(f"Input text: {repr(text)}")
    
    # Show pre-tokenization
    import re
    pre_tokens = re.findall(tokenizer.pat, text)
    print(f"Pre-tokens: {pre_tokens}")
    
    # Show final tokens
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    tokens = [tokenizer.id_to_token[id] for id in token_ids]
    print(f"Final tokens: {tokens}")
    
    # Show reconstruction
    decoded = tokenizer.decode(token_ids, skip_special_tokens=True)
    print(f"Decoded: {repr(decoded)}")
    
    # Check if perfect reconstruction
    if text == decoded:
        print("✅ Perfect reconstruction")
    else:
        print("❌ Reconstruction failed")
```

## References

- [Neural Machine Translation of Rare Words with Subword Units](https://arxiv.org/abs/1508.07909) - Original BPE paper
- [Language Models are Unsupervised Multitask Learners](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) - GPT-2 tokenization
- [SentencePiece: A simple and language independent subword tokenizer](https://arxiv.org/abs/1808.06226) - Alternative approach
- [Hugging Face Tokenizers](https://huggingface.co/docs/tokenizers/) - Modern tokenization library

---

For implementation details, see `src/llm_trainer/tokenizer/bpe_tokenizer.py`.