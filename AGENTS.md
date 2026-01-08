# Agent Rules for LLM Trainer Codebase

This document provides comprehensive rules and guidelines for AI agents working with the LLM Trainer codebase. Follow these rules strictly to maintain code quality and consistency.

## Tool Preferences

**ALWAYS use `uv` for package management and tooling:**

- ✅ `uv run <command>` - Run commands in the project environment
- ✅ `uvx <tool>` - Run tools via uvx (e.g., `uvx ty check`)
- ✅ `uv pip install` - Install packages
- ✅ `uv sync` - Sync dependencies

- ❌ NEVER use `pip` directly
- ❌ NEVER use `python -m` for tools (use `uv run` instead)
- ❌ NEVER use `npx` or other package managers

## Code Quality Standards

### Type Checking

**Always run type checking before committing:**
```bash
uvx ty check
```

**Key rules:**
- Use proper type hints: `Callable` from `typing`, not `callable`
- Handle `Optional` types properly - always check for `None` before use
- Use `Union` for multiple types, `Optional[T]` for `Union[T, None]`
- Provide default values for optional parameters when calling functions

### Linting

**Always run linting:**
```bash
uv run ruff check . --select E,F,W
uv run ruff check . --select E,F,W --fix  # Auto-fix when possible
```

**Rules:**
- Follow PEP 8 style guidelines
- Maximum line length: 88 characters (Black default)
- Use `ruff` for all linting - it's fast and comprehensive
- Fix all errors before committing

## Project Structure

```
llm-trainer/
├── src/llm_trainer/          # Main package source
│   ├── config/               # Configuration classes
│   ├── data/                 # Data loading and preprocessing
│   ├── kernels/              # Kernel optimizations
│   ├── models/              # Model architectures
│   ├── tokenizer/           # Tokenizer implementations
│   ├── training/            # Training infrastructure
│   └── utils/               # Utility functions
├── scripts/                 # CLI scripts
├── examples/                # Python examples
├── notebooks/               # Jupyter notebooks
├── configs/                 # YAML configuration files
└── docs/                    # Documentation (essential only)
```

## Code Patterns

### Tokenizer Creation

**ALWAYS use the factory function:**
```python
from llm_trainer.tokenizer import create_tokenizer

# ✅ Correct
tokenizer = create_tokenizer("bpe")
tokenizer = create_tokenizer("bpe", pretrained_path="./tokenizer")

# ❌ Wrong - don't import tokenizer classes directly
from llm_trainer.tokenizer import BPETokenizer  # Avoid
```

### Model Creation

**Use ModelConfig for configuration:**
```python
from llm_trainer.config import ModelConfig
from llm_trainer.models import TransformerLM

model_config = ModelConfig(
    vocab_size=32000,
    d_model=512,
    n_heads=8,
    n_layers=6
)
model = TransformerLM(model_config)
```

### Training Setup

**Use TrainingConfig and Trainer:**
```python
from llm_trainer.config import TrainingConfig
from llm_trainer.training import Trainer

training_config = TrainingConfig(
    batch_size=16,
    learning_rate=1e-4,
    num_epochs=3
)

trainer = Trainer(model, tokenizer, training_config)
```

### ❌ Avoided Patterns

- **Don't use monkey-patching** - Use proper inheritance or composition
- **Don't create unnecessary MD files** - Only essential documentation
- **Don't use `pip` directly** - Always use `uv`
- **Don't import from `__init__` unnecessarily** - Use direct imports when possible

### Notebooks

**All notebooks go in `notebooks/` directory:**
- Use descriptive names: `01_`, `02_`, etc. for ordering
- Keep notebooks focused and small
- Include markdown cells with explanations
- Make notebooks runnable standalone

## Type Safety Rules

### Required Type Hints

**Always provide type hints for:**
- Function parameters
- Return types
- Class attributes (when possible)
- Module-level variables

**Example:**
```python
from typing import List, Optional, Dict, Callable

def process_text(
    texts: List[str],
    tokenizer: BaseTokenizer,
    callback: Optional[Callable[[str], str]] = None
) -> List[int]:
    ...
```

### Handling Optional Types

**Always check for None:**
```python
# ✅ Correct
if dataset_name:
    dataset = load_dataset(dataset_name)

# ❌ Wrong
dataset = load_dataset(dataset_name)  # dataset_name might be None
```

### Default Values

**Provide sensible defaults:**
```python
# ✅ Correct
pad_token_id = getattr(tokenizer, "pad_token_id", None) or 0
eos_token_id = getattr(tokenizer, "eos_token_id", None) or 3

# ❌ Wrong
pad_token_id = getattr(tokenizer, "pad_token_id", None)  # Might be None
```

## Error Handling

### Required Checks

**Always validate:**
- None values before use
- Empty lists/collections before indexing
- Required attributes exist before access
- Dataloaders are not None before iteration

**Example:**
```python
if pbar is None:
    raise ValueError("train_dataloader is None. Please provide training data.")
```

## Import Organization

### Standard Import Order

1. Standard library imports
2. Third-party imports
3. Local application imports

**Example:**
```python
# Standard library
import os
from typing import List, Optional

# Third-party
import torch
from datasets import load_dataset

# Local
from llm_trainer.tokenizer import create_tokenizer
from llm_trainer.models import TransformerLM
```

## Testing and Validation

### Before Committing

**Always run:**
1. Type checking: `uvx ty check`
2. Linting: `uv run ruff check . --select E,F,W`
3. Fix auto-fixable issues: `uv run ruff check . --select E,F,W --fix`

### Common Issues to Fix

1. **Type errors:**
   - Use `Callable` not `callable`
   - Handle `Optional` types properly
   - Provide defaults for None values

2. **Import errors:**
   - These are expected if packages aren't installed
   - Only fix actual code errors, not missing package warnings

3. **Attribute errors:**
   - Check if attributes exist before access
   - Use `getattr()` with defaults

## Code Style

### Function Naming

- Use descriptive, clear names
- Prefer verbs for functions: `create_tokenizer()`, `train_model()`
- Use nouns for classes: `TransformerLM`, `BPETokenizer`

### Variable Naming

- Use snake_case for variables and functions
- Use PascalCase for classes
- Use UPPER_CASE for constants

### Documentation

- Add docstrings to all public functions and classes
- Use Google-style docstrings
- Include type information in docstrings

## Tokenizer Guidelines

### Available Tokenizers

- `"bpe"` - Byte Pair Encoding (recommended default)
- `"wordpiece"` - WordPiece (BERT-style)
- `"sentencepiece"` - SentencePiece Unigram
- `"char"` - Character-level
- `"bytebpe"` - Byte-level BPE (GPT-2 style)
- `"simple"` - Simple whitespace
- `"hf"` - HuggingFace pretrained

### Tokenizer Usage

```python
# Create new tokenizer
tokenizer = create_tokenizer("bpe")
tokenizer.train("wikitext", vocab_size=32000)

# Load pretrained
tokenizer = create_tokenizer("bpe", pretrained_path="./tokenizer")

# HuggingFace tokenizer
tokenizer = create_tokenizer("hf", pretrained_path="gpt2")
```

## Model Guidelines

### Model Configuration

- Always use `ModelConfig` for configuration
- Update `vocab_size` to match tokenizer
- Use appropriate model sizes for available resources

### Training Configuration

- Use `TrainingConfig` for all training settings
- Set appropriate batch sizes for CPU/GPU
- Use gradient accumulation for effective batch size
- Enable AMP only for GPU training

## Notebook Guidelines

### Structure

1. **Markdown cell** - Title and description
2. **Code cells** - Step-by-step execution
3. **Comments** - Explain what each step does
4. **Output** - Show results clearly

### Best Practices

- Keep notebooks focused on one topic
- Use descriptive cell outputs
- Include error handling
- Make notebooks runnable from top to bottom

## Common Tasks

### Adding a New Tokenizer

1. Create tokenizer class in `src/llm_trainer/tokenizer/`
2. Inherit from `BaseTokenizer`
3. Implement required methods: `train()`, `encode()`, `decode()`
4. Add to `TOKENIZER_REGISTRY` in `factory.py`
5. Export in `tokenizer/__init__.py`
6. Add tests and examples

### Adding a New Feature

1. Create feature in appropriate module
2. Add type hints
3. Add docstrings
4. Update exports in `__init__.py`
5. Add examples/notebooks
6. Update documentation

### Fixing Type Errors

1. Run `uvx ty check` to identify issues
2. Fix actual type errors (not missing import warnings)
3. Add proper type hints
4. Handle None values correctly
5. Provide default values where needed

## Environment Setup

### Using uv

```bash
# Install dependencies
uv sync

# Run commands
uv run python script.py
uv run ruff check .
uvx ty check

# Install package in development mode
uv pip install -e .
```

### Virtual Environment

- uv manages virtual environments automatically
- No need to manually create venv
- Use `uv run` to execute in project environment

## Debugging

### Common Issues

1. **Import errors** - Usually means package not installed
2. **Type errors** - Check type hints and None handling
3. **Attribute errors** - Check if attribute exists
4. **TOML errors** - Validate syntax, check for unsupported features

### Tools

- `uvx ty check` - Type checking
- `uv run ruff check` - Linting
- `uv run ruff check --fix` - Auto-fix issues
- `uv run python -m pytest` - Run tests (if available)

## Summary

**Key Principles:**
1. ✅ Always use `uv` for package management
2. ✅ Use `create_tokenizer()` factory function
3. ✅ Provide proper type hints
4. ✅ Handle None values correctly
5. ✅ Run type checking and linting before committing
6. ✅ Keep documentation minimal and essential
7. ✅ Follow project structure conventions
8. ❌ Never use removed features (patching, recommend_tokenizer)
9. ❌ Never use `pip` directly
10. ❌ Never create unnecessary documentation files
11. Use #codebase-retrieval (it is a context engine/indexing) tool for codebase navigation and finding information

**Remember:** This is a professional codebase. Keep code clean, typed, and well-documented. Use `uv` for all tooling and package management.
