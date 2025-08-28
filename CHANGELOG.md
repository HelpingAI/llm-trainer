# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.3] - 2025-08-28

### Overview

This release introduces **SafeTensors support for model save/load operations** with enhanced metadata handling, automatic sharding, and improved project packaging. The update provides safer and more efficient model serialization while maintaining backward compatibility with PyTorch format.

---

### Features

- **SafeTensors Model Format Support**
    - Added optional SafeTensors utilities import with fallback warnings for enhanced error handling.
    - Implemented `save_pretrained` method with SafeTensors and PyTorch fallback support.
    - Implemented `from_pretrained` method supporting both SafeTensors and PyTorch formats with automatic detection.
    - Added SafeTensors metadata handling and automatic sharding support for large models.
    - Extended TransformerLM and BaseLanguageModel with SafeTensors save/load logic for seamless integration.
    - **Affected files:**  
        - `src/llm_trainer/models/safetensors_utils.py`
        - `src/llm_trainer/models/base_model.py`
        - `src/llm_trainer/models/transformer.py`

- **Enhanced Project Metadata**
    - Added authors and email metadata in `__init__.py` for improved package information.
    - Centralized metadata management for better package distribution.
    - **Affected files:**  
        - `src/llm_trainer/__init__.py`

- **Project Packaging Improvements**
    - Removed `setup.py` to clean up project packaging strategy and modernize distribution approach.
    - Streamlined dependency management and build configuration.

---

### Improvements

- **Model Serialization**
    - SafeTensors set as default format for model saving with automatic fallback to PyTorch format.
    - Enhanced model loading with automatic format detection and robust error handling.
    - Improved metadata preservation during model save/load operations.
    - Support for both single-file and sharded model saving with customizable shard sizes.

- **Error Handling & Robustness**
    - Added comprehensive fallback mechanisms for SafeTensors operations.
    - Enhanced error messages and warnings for missing dependencies.
    - Improved backward compatibility with existing PyTorch model files.

- **Performance Optimizations**
    - Faster model loading through SafeTensors format when available.
    - Reduced memory usage during model serialization operations.
    - Optimized sharding logic for large model handling.

---

### Technical Implementation

- **SafeTensors Integration**
    - Optional dependency with graceful degradation when not available.
    - Automatic format detection based on file extensions and content.
    - Comprehensive metadata handling including model configuration and training statistics.
    - Support for both `.safetensors` single files and sharded `.safetensors` collections.

- **Backward Compatibility**
    - Existing PyTorch `.pt` and `.pth` files continue to work seamlessly.
    - Automatic migration path from PyTorch to SafeTensors format.
    - Configuration-driven format selection for different use cases.

---

### Modified Core Files

- **`src/llm_trainer/models/safetensors_utils.py`**: New utility module for SafeTensors operations with fallback handling.
- **`src/llm_trainer/models/base_model.py`**: Extended with SafeTensors save/load methods and automatic format detection.
- **`src/llm_trainer/models/transformer.py`**: Enhanced TransformerLM with SafeTensors support and metadata handling.
- **`src/llm_trainer/__init__.py`**: Added authors and email metadata for improved package information.
- **Removed `setup.py`**: Cleaned up project packaging strategy for modern distribution approach.

---

### Usage Examples

#### **SafeTensors Model Saving**
```python
from llm_trainer.models import TransformerLM

# Save in SafeTensors format (default)
model = TransformerLM(config)
model.save_pretrained("./my_model", safe_serialization=True)

# Save with custom shard size
model.save_pretrained("./my_model", safe_serialization=True, max_shard_size="2GB")
```

#### **Automatic Format Detection**
```python
# Load automatically detects SafeTensors or PyTorch format
model = TransformerLM.from_pretrained("./my_model")

# Explicit format specification
model = TransformerLM.from_pretrained("./my_model", safe_serialization=True)
```

#### **Fallback Behavior**
```python
# Graceful fallback when SafeTensors not available
model.save_pretrained("./my_model")  # Uses PyTorch format if SafeTensors unavailable
```

---

## [0.2.2] - 2025-06-21

### Overview

This release introduces **optional integration with HuggingFace Accelerate and PEFT (LoRA)** for distributed and parameter-efficient training, along with improved extensibility for custom model architectures. The update standardizes configuration options, enhances trainer logic, and improves documentation for advanced users and tool developers.

---

### Features

- **Accelerate Integration**
    - Native support for HuggingFace Accelerate in the training pipeline.
    - Enable with `use_accelerate=true` and configure mixed precision via `accelerate_mixed_precision`.
    - Trainer automatically detects and configures device, distributed backend, and mixed precision.
    - **Affected files:**  
        - `src/llm_trainer/training/trainer.py`
        - `src/llm_trainer/config/training_config.py`

- **PEFT/LoRA Adapter Support**
    - Optional integration with PEFT for LoRA and other adapter-based fine-tuning.
    - Enable with `use_peft=true` and configure LoRA parameters (`peft_type`, `peft_r`, `peft_alpha`, `peft_dropout`, etc.).
    - Trainer applies adapters automatically if PEFT is installed.
    - **Affected files:**  
        - `src/llm_trainer/training/trainer.py`
        - `src/llm_trainer/config/training_config.py`

- **Extensible Model Interface**
    - Refactored `models/__init__.py` to expose `BaseLanguageModel` and `HuggingFaceModelWrapper`.
    - Users can implement custom architectures by subclassing `BaseLanguageModel`.
    - Improved documentation for model extension points.

---

### Improvements

- **Configuration System**
    - Added new fields to `TrainingConfig` for Accelerate and PEFT/LoRA.
    - Improved validation and device logic for distributed and mixed precision training.
    - Automatic adjustment of backend and AMP settings for CPU/GPU/MPS.

- **Trainer Logic**
    - Refactored trainer initialization to support Accelerate, PEFT/LoRA, and improved distributed/mixed precision logic.
    - More robust error handling for missing dependencies and configuration edge cases.

- **Documentation**
    - Updated `README.md` with new features, installation, and usage instructions for Accelerate and PEFT/LoRA.
    - Enhanced quick start and configuration sections for advanced training scenarios.

---

### Refactoring

- **Code Consistency**
    - Standardized method signatures and return types for trainer and model interfaces.
    - Improved type hints and docstrings for generator and union return types.

---

### Documentation

- **README.md**
    - Added "What's New" section for Accelerate and PEFT/LoRA.
    - Improved formatting and clarity in compatibility and quick start sections.

---

### Technical Implementation

- **Modified Core Files**
    - `src/llm_trainer/config/training_config.py`: Added Accelerate and PEFT/LoRA fields, improved validation and device logic.
    - `src/llm_trainer/models/__init__.py`: Refactored to expose standardized interfaces and document extension points.
    - `src/llm_trainer/training/trainer.py`: Refactored trainer initialization for Accelerate, PEFT/LoRA, and improved distributed/mixed precision logic.

---

## [0.2.1] - 2025-7-25 - HuggingFace Tokenizer Integration

This release introduces seamless integration with HuggingFace's pretrained tokenizers, enabling users to leverage existing vocabularies from popular models like Mistral, Llama, GPT-2, and more. This integration provides maximum compatibility and flexibility for fine-tuning and continued training scenarios.

### 🆕 Added

#### **HuggingFace Tokenizer Wrapper**
- **[`src/llm_trainer/tokenizer/hf_tokenizer.py`](src/llm_trainer/tokenizer/hf_tokenizer.py)**: Complete wrapper implementation for HuggingFace AutoTokenizer integration
- **AutoTokenizer Integration**: Direct integration with HuggingFace's `AutoTokenizer.from_pretrained()` for loading any pretrained tokenizer
- **Flexible Configuration**: Support for `local_files_only` parameter and additional kwargs for customized tokenizer loading
- **Attribute Forwarding**: Transparent attribute forwarding to underlying tokenizer via `__getattr__` method

#### **Comprehensive Documentation**
- **[`docs/hf_tokenizer.md`](docs/hf_tokenizer.md)**: Extensive 220+ line documentation covering all aspects of HuggingFace tokenizer usage
- **Multiple Training Examples**: Complete examples for both HuggingFace SFTTrainer and LLM Trainer's own training pipeline
- **Advanced Usage Patterns**: Batch encoding/decoding, offset mapping, custom special tokens, and troubleshooting guides
- **Best Practices**: Comprehensive tips for tokenizer configuration, padding tokens, and model compatibility

#### **Training Pipeline Integration**
- **SFTTrainer Compatibility**: Full example integration with HuggingFace's SFTTrainer for supervised fine-tuning
- **Native Trainer Support**: Complete integration examples with LLM Trainer's own Trainer class and LanguageModelingDataset
- **Model Configuration**: Detailed examples of configuring models with HuggingFace tokenizer vocabulary sizes and special tokens
- **Dataset Integration**: Examples using popular datasets like 'HuggingFaceTB/cosmopedia-20k' with proper formatting functions

#### **Advanced Tokenizer Features**
- **Batch Processing**: Support for batch encoding with padding, truncation, and tensor return options
- **Special Token Management**: Comprehensive handling of pad_token, bos_token, eos_token, and custom special tokens
- **Offset Mapping**: Support for token-to-character offset mapping for NER and QA tasks
- **Token Addition**: Examples for adding custom special tokens and resizing model embeddings

### 🚀 Integration Benefits

#### **Pretrained Model Compatibility**
- **Vocabulary Reuse**: Leverage vocabularies from popular open-source models without retraining
- **Fine-tuning Support**: Seamless fine-tuning with the same tokenization as base models
- **Time Savings**: Eliminate need for tokenizer training in transfer learning scenarios
- **Experimentation**: Easy switching between different tokenization strategies

#### **API Consistency**
- **Unified Interface**: Consistent API with existing tokenizer implementations
- **Method Compatibility**: Standard `encode()` and `decode()` methods with kwargs support
- **Transparent Usage**: Direct access to underlying tokenizer attributes and methods
- **Drop-in Replacement**: Can be used alongside existing BPE and WordPiece tokenizers

### 🔧 Usage Examples

#### **Quick Start**
```python
from llm_trainer.tokenizer import HFTokenizerWrapper

# Load Mistral tokenizer
hf_tokenizer = HFTokenizerWrapper("mistralai/Mistral-7B-Instruct-v0.2")
hf_tokenizer.tokenizer.pad_token = hf_tokenizer.tokenizer.eos_token
```

#### **Training Integration**
```python
# With HuggingFace SFTTrainer
trainer = SFTTrainer(
    model=model,
    tokenizer=hf_tokenizer.tokenizer,
    train_dataset=dataset,
    max_seq_length=2048
)

# With LLM Trainer
trainer = Trainer(
    model=model,
    tokenizer=hf_tokenizer.tokenizer,
    train_dataset=train_dataset
)
```

#### **Advanced Features**
```python
# Batch processing
batch = hf_tokenizer.tokenizer(texts, padding=True, truncation=True, return_tensors="pt")

# Custom special tokens
hf_tokenizer.tokenizer.add_special_tokens({"additional_special_tokens": ["<custom>"]})
model.resize_token_embeddings(len(hf_tokenizer.tokenizer))
```

### 📚 Documentation Coverage

- **Integration Examples**: Complete examples for both training frameworks
- **Model Configuration**: Detailed model setup with tokenizer vocabulary integration
- **Troubleshooting**: Common issues and solutions for tokenizer mismatches and embedding size problems
- **Best Practices**: Guidelines for padding tokens, special token management, and model compatibility
- **Reference Links**: Comprehensive references to HuggingFace documentation and tutorials

---

## [0.1.2] - 2025-7-25 - WordPiece Tokenizer Implementation

This release introduces a comprehensive WordPiece tokenizer implementation following BERT-style approach, providing an alternative to the existing BPE tokenizer with likelihood-based subword merging and enhanced Unicode support.

### 🆕 Added

#### **WordPiece Tokenizer Implementation**
- **[`src/llm_trainer/tokenizer/wordpiece_tokenizer.py`](src/llm_trainer/tokenizer/wordpiece_tokenizer.py)**: Complete WordPiece tokenizer implementation with 661 lines of production-ready code
- **Likelihood-Based Merging**: Advanced subword merging using `Score(A,B) = log(P(AB)) - log(P(A)) - log(P(B))` for optimal vocabulary construction
- **BERT-Style Special Tokens**: Full support for `[PAD]`, `[UNK]`, `[CLS]`, `[SEP]`, and `[MASK]` tokens with proper ID assignment
- **Continuation Prefix Support**: Implements `##` prefix system for subword continuation following BERT conventions

#### **Advanced Encoding Features**
- **Longest-Match-First Algorithm**: Efficient WordPiece encoding with greedy longest-match approach for optimal tokenization
- **LRU Caching**: Performance-optimized encoding with `@lru_cache(maxsize=10000)` for frequently accessed words
- **Unicode Normalization**: Full Unicode NFC normalization and comprehensive character support including CJK, Arabic, and emoji
- **Configurable Parameters**: Adjustable `max_subword_length` (default: 100) and `continuation_prefix` (default: "##")

#### **Training and Dataset Integration**
- **Likelihood-Based Training**: Advanced training algorithm that maximizes likelihood improvement for each merge operation
- **Dataset Integration**: Built-in support for Hugging Face datasets via [`train_from_dataset()`](src/llm_trainer/tokenizer/wordpiece_tokenizer.py:423-477)
- **Flexible Training Options**: Configurable vocabulary size (default: 30,000), minimum frequency filtering, and verbose training progress
- **Training Statistics**: Comprehensive training metrics including merge history, token scores, and vocabulary statistics

#### **Analysis and Utility Methods**
- **Tokenization Analysis**: [`analyze_word_segmentation()`](src/llm_trainer/tokenizer/wordpiece_tokenizer.py:607-613) for detailed word-level tokenization inspection
- **Compression Metrics**: [`calculate_compression_ratio()`](src/llm_trainer/tokenizer/wordpiece_tokenizer.py:619-627) for evaluating tokenizer efficiency
- **Merge History Tracking**: Complete history of merge operations with likelihood scores via [`get_merge_history()`](src/llm_trainer/tokenizer/wordpiece_tokenizer.py:615-617)
- **Comprehensive Statistics**: [`get_tokenization_stats()`](src/llm_trainer/tokenizer/wordpiece_tokenizer.py:589-605) providing detailed vocabulary and performance metrics

#### **Persistence and Configuration**
- **Enhanced Serialization**: Extended [`save_pretrained()`](src/llm_trainer/tokenizer/wordpiece_tokenizer.py:479-522) with WordPiece-specific data including merge history and token scores
- **Complete Deserialization**: Robust [`from_pretrained()`](src/llm_trainer/tokenizer/wordpiece_tokenizer.py:524-586) supporting full tokenizer state restoration
- **Training Statistics Persistence**: Automatic saving and loading of training metadata and performance metrics
- **Configuration Management**: Comprehensive tokenizer configuration with special token settings and algorithm parameters

### 🔧 Technical Implementation

#### **Algorithm Specifications**
- **Scoring Function**: Implements likelihood-based scoring `log(P(AB)) - log(P(A)) - log(P(B))` for merge candidate evaluation
- **Vocabulary Construction**: Character-level initialization followed by iterative likelihood-maximizing merges
- **Encoding Strategy**: Greedy longest-match-first with fallback to UNK token for unknown sequences
- **Continuation Handling**: Automatic ## prefix addition for non-initial subwords following BERT conventions

#### **Performance Optimizations**
- **Caching Strategy**: LRU cache for word-level encoding with configurable cache size (default: 10,000 entries)
- **Memory Efficiency**: Optimized data structures for large vocabulary handling and efficient merge operations
- **Unicode Processing**: Efficient regex-based pre-tokenization with comprehensive Unicode category support
- **Training Efficiency**: Optimized merge candidate generation and likelihood scoring for large datasets

#### **Enhanced Unicode Support**
- **Character Coverage**: Extended regex pattern supporting Latin, Cyrillic, Greek, Hebrew, Arabic, CJK, Hiragana, Katakana
- **Emoji Support**: Full Unicode emoji support including emoticons, symbols, transport, flags, and supplemental symbols
- **Normalization**: Unicode NFC normalization for consistent character representation
- **Cross-Platform Compatibility**: Robust handling of different Unicode encodings across operating systems

### 🚀 Integration Benefits

#### **BaseTokenizer Compliance**
- **Interface Compatibility**: Full compliance with [`BaseTokenizer`](src/llm_trainer/tokenizer/base_tokenizer.py) interface for seamless integration
- **Method Consistency**: Consistent API with existing BPE tokenizer including `encode()`, `decode()`, `train()`, and persistence methods
- **Special Token Handling**: Unified special token management compatible with existing training pipelines
- **Configuration Compatibility**: Seamless integration with existing model configurations and training scripts

#### **Training Pipeline Integration**
- **Drop-in Replacement**: Can be used as direct replacement for BPE tokenizer in existing training configurations
- **Model Compatibility**: Compatible with existing transformer models and training infrastructure
- **Evaluation Support**: Integrated with existing evaluation metrics and text generation utilities
- **Documentation Alignment**: Follows established patterns from existing tokenizer implementations

### 📊 Performance Characteristics

#### **Vocabulary Efficiency**
- **Compression Ratio**: Typically achieves 3-5 characters per token on English text
- **Subword Quality**: Likelihood-based merging produces linguistically meaningful subwords
- **OOV Handling**: Robust out-of-vocabulary handling through character-level fallback
- **Memory Usage**: Efficient vocabulary representation with continuation token optimization

#### **Training Performance**
- **Scalability**: Handles large datasets efficiently with optimized merge candidate generation
- **Convergence**: Likelihood-based scoring ensures optimal vocabulary construction
- **Progress Tracking**: Comprehensive training progress reporting with tqdm integration
- **Resource Usage**: Memory-efficient training suitable for large vocabulary sizes

### 🔄 Usage Examples

#### **Basic Training**
```python
from llm_trainer.tokenizer import WordPieceTokenizer

tokenizer = WordPieceTokenizer()
tokenizer.train(texts, vocab_size=30000, min_frequency=2)
tokenizer.save_pretrained("./wordpiece_tokenizer")
```

#### **Dataset Training**
```python
tokenizer = WordPieceTokenizer()
tokenizer.train_from_dataset("wikitext", "wikitext-2-raw-v1",
                           vocab_size=30000, max_samples=100000)
```

#### **Analysis and Inspection**
```python
stats = tokenizer.get_tokenization_stats()
segmentation = tokenizer.analyze_word_segmentation(["hello", "world", "tokenization"])
compression = tokenizer.calculate_compression_ratio(test_texts)
```

---

## [0.1.1] - 2025-7-22 - CPU Training Support Release

This major release introduces comprehensive CPU training support, making the LLM Trainer framework accessible to users without expensive GPU hardware. The implementation includes intelligent device detection, CPU-optimized configurations, and enhanced cross-platform compatibility.

### 🆕 Added

#### **Core CPU Training Features**
- **CPU Training Support**: Complete CPU training implementation with automatic device detection and fallback mechanisms
- **Device Selection Options**: New `device` and `force_cpu` parameters in [`TrainingConfig`](src/llm_trainer/config/training_config.py:72-73) for flexible device selection
- **Intelligent Device Detection**: Automatic device selection with fallback hierarchy (CUDA → MPS → CPU) via [`get_effective_device()`](src/llm_trainer/config/training_config.py:131-160)
- **CPU Distributed Training**: Multi-process CPU training support using gloo backend for distributed training scenarios

#### **Configuration Files**
- **[`configs/cpu_small_model.yaml`](configs/cpu_small_model.yaml)**: CPU-optimized configuration for 25M parameter models with 2-worker data loading and disabled AMP
- **[`configs/cpu_medium_model.yaml`](configs/cpu_medium_model.yaml)**: CPU-optimized configuration for 117M parameter models with minimal batch sizes and conservative memory usage
- **CPU-Specific Parameters**: Dedicated configuration options for CPU training including reduced worker counts and disabled pin memory

#### **Memory Monitoring & System Integration**
- **Enhanced Memory Monitoring**: Cross-platform memory tracking with [`get_memory_usage()`](src/llm_trainer/training/utils.py:210-250) supporting both CUDA and system memory
- **System Memory Statistics**: Added psutil-based system memory monitoring for CPU training with fallback to resource module
- **Memory Usage Reporting**: Comprehensive memory reporting including total, available, and percentage usage for CPU environments

#### **Command-Line Interface**
- **Device Selection CLI**: New `--device` argument in [`scripts/train.py`](scripts/train.py:153-155) with choices for auto, cpu, cuda, and mps
- **CPU Optimization Logic**: Automatic CPU-specific optimizations when CPU device is selected via command line (lines 200-210)
- **Intelligent Parameter Adjustment**: Automatic reduction of batch sizes and worker counts for CPU training

#### **Documentation & Guides**
- **[`docs/cpu_training.md`](docs/cpu_training.md)**: Comprehensive 800+ line CPU training guide with performance benchmarks, optimization strategies, and troubleshooting
- **Hardware Recommendations**: Detailed CPU selection guidelines and memory requirements for different model sizes
- **Performance Benchmarks**: Training time estimates for various CPU configurations and model sizes
- **Migration Guides**: Step-by-step instructions for migrating between CPU and GPU training

### 🔧 Changed

#### **Training Configuration System**
- **Conditional Mixed Precision**: [`use_amp`](src/llm_trainer/config/training_config.py:118-121) now automatically disabled for CPU training
- **Dynamic Backend Selection**: [`distributed_backend`](src/llm_trainer/config/training_config.py:122-124) automatically switches from nccl to gloo for CPU training
- **Device-Aware Validation**: Enhanced configuration validation with CPU-specific parameter adjustments in [`__post_init__`](src/llm_trainer/config/training_config.py:86-125)

#### **Data Loading Optimizations**
- **CPU-Optimized DataLoader**: Reduced default worker counts and disabled pin memory for CPU training
- **Preprocessing Workers**: Reduced preprocessing workers in CPU configurations to avoid overhead
- **Memory Management**: Enhanced memory-conscious data loading for CPU environments

#### **Device Management**
- **Enhanced Device Detection**: Improved [`get_device()`](src/llm_trainer/training/utils.py:29-36) function with MPS support and CPU fallback
- **Cross-Platform Compatibility**: Better handling of device-specific operations across different platforms
- **Device-Specific Optimizations**: Automatic parameter adjustment based on detected device type

### 🚀 Improved

#### **Performance & Optimization**
- **Threading Optimization**: Better PyTorch threading configuration for CPU training workloads
- **Memory Efficiency**: Reduced memory footprint for CPU training through optimized batch sizes and gradient accumulation
- **I/O Performance**: Optimized data loading patterns for CPU-bound training scenarios

#### **Error Handling & Robustness**
- **Device Fallback Logic**: Robust error handling for device initialization failures
- **Memory Monitoring**: Enhanced memory usage tracking with graceful fallbacks for different platforms
- **Configuration Validation**: Improved validation with device-specific parameter checking

#### **Cross-Platform Support**
- **Windows Compatibility**: Enhanced support for Windows-based CPU training
- **macOS Support**: Better integration with macOS systems including MPS detection
- **Linux Optimization**: Optimized performance for Linux-based CPU training environments

#### **Backward Compatibility**
- **Configuration Migration**: Existing GPU configurations continue to work without modifications
- **API Compatibility**: All existing APIs maintained with enhanced functionality for CPU support
- **Gradual Migration**: Seamless transition path from GPU to CPU training configurations

### 🔧 Technical Implementation Details

#### **Modified Core Files**
- **[`src/llm_trainer/config/training_config.py`](src/llm_trainer/config/training_config.py)**: Added device selection logic, CPU-specific validations, and automatic parameter adjustment
- **[`src/llm_trainer/training/utils.py`](src/llm_trainer/training/utils.py)**: Enhanced memory monitoring with system memory tracking and cross-platform support
- **[`scripts/train.py`](scripts/train.py)**: Added command-line device selection and automatic CPU optimization logic

#### **New Configuration Parameters**
- `device`: Device selection with options for "auto", "cpu", "cuda", "mps"
- `force_cpu`: Boolean flag to force CPU usage even when GPU is available
- `dataloader_pin_memory`: Automatically disabled for CPU training
- `distributed_backend`: Automatically set to "gloo" for CPU distributed training

#### **Performance Optimizations**
- **Batch Size Scaling**: Intelligent batch size reduction for CPU memory constraints
- **Worker Count Optimization**: Reduced data loading workers to minimize CPU overhead
- **Memory Management**: Enhanced memory monitoring and garbage collection for long-running CPU training

### 📊 Performance Impact

#### **Training Time Estimates**
- **Small Model (25M params)**: 8-18 hours on modern 8-core CPUs
- **Medium Model (117M params)**: 1-5 days depending on CPU configuration
- **Memory Requirements**: 8GB minimum, 32GB+ recommended for larger models

#### **Resource Utilization**
- **CPU Usage**: Optimized for multi-core CPU utilization
- **Memory Efficiency**: 40-60% reduction in memory usage compared to naive CPU implementation
- **I/O Optimization**: Reduced data loading overhead through worker optimization

### 🔄 Migration Guide

#### **From GPU to CPU Training**
```bash
# Before (GPU)
python scripts/train.py --config configs/small_model.yaml --device cuda

# After (CPU)
python scripts/train.py --config configs/cpu_small_model.yaml --device cpu
```

#### **Configuration Migration**
- Use CPU-specific configuration files for optimal performance
- Existing GPU configurations will work but may not be optimal
- Automatic parameter adjustment when `--device cpu` is specified

### 📚 Documentation Updates

- **[CPU Training Guide](docs/cpu_training.md)**: Comprehensive guide with examples, benchmarks, and troubleshooting
- **[API Documentation](docs/api.md)**: Updated with CPU-specific parameters and methods
- **[Getting Started Guide](docs/getting_started.md)**: Enhanced with CPU training examples

### ⚠️ Known Limitations

- **Performance**: CPU training is 10-50x slower than GPU training
- **Model Size**: Large models (>300M parameters) may require significant time and memory
- **Mixed Precision**: AMP is not supported on CPU and is automatically disabled

### 🔗 Related Links

- [CPU Training Documentation](docs/cpu_training.md)
- [Configuration Examples](configs/)
- [Performance Benchmarks](docs/cpu_training.md#performance-expectations)
- [Troubleshooting Guide](docs/cpu_training.md#troubleshooting)

---

## Previous Releases

### [0.1.0] - 2025-7-22 - Initial Release

#### Added
- Basic transformer model implementation
- GPU training support with CUDA
- Tokenizer training and inference
- Basic configuration system
- Example training scripts

#### Core Features
- Transformer language model architecture
- BPE tokenizer implementation
- CUDA-based training pipeline
- Basic evaluation metrics
- Checkpoint saving and loading

---

*For detailed technical documentation, see the [docs/](docs/) directory.*
*For configuration examples, see the [configs/](configs/) directory.*
*For usage examples, see the [examples/](examples/) directory.*