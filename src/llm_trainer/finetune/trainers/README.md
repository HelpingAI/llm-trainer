# LLM Trainer - Clean TRL/Unsloth-Style Trainers

This directory contains simplified, focused trainers that follow TRL and Unsloth patterns, emphasizing chat templates and proper dataset formatting.

## 🚀 Available Trainers

### SFTTrainer (Supervised Fine-Tuning)
- **Purpose**: Instruction-following fine-tuning with automatic chat template handling
- **Features**:
  - Automatic chat template application (like TRL/Unsloth)
  - Support for `messages` and `text` formats
  - Clean, simple API
- **Best for**: Instruction following, chat models

### DPOTrainer (Direct Preference Optimization)
- **Purpose**: Preference-based alignment training
- **Features**:
  - Automatic reference model creation
  - Support for `prompt`/`chosen`/`rejected` format
  - TRL-compatible API
- **Best for**: Model alignment, preference learning

## 🎯 Key Features

### Chat Template Focus
- **Automatic Detection**: Automatically applies chat templates to `messages` format
- **Fallback Formatting**: Simple fallback for models without chat templates
- **TRL/Unsloth Style**: Follows the same patterns as leading libraries

### Simple Dataset Support
- **Messages Format**: `{"messages": [{"role": "user", "content": "..."}, ...]}`
- **Text Format**: `{"text": "Human: ... Assistant: ..."}`
- **DPO Format**: `{"prompt": "...", "chosen": "...", "rejected": "..."}`

## 🛠 Usage Examples

### Basic SFT Training (TRL/Unsloth Style)
```python
from llm_trainer.finetune.trainers import SFTTrainer, SFTConfig

# Simple configuration
config = SFTConfig(
    output_dir="./results",
    num_train_epochs=1,
    per_device_train_batch_size=1,
    learning_rate=5e-5,
    max_seq_length=512,
)

# Create trainer - chat templates applied automatically
trainer = SFTTrainer(
    model="microsoft/DialoGPT-small",
    args=config,
    train_dataset=dataset,  # with 'messages' or 'text' format
)

trainer.train()
```

### DPO Training (TRL Style)
```python
from llm_trainer.finetune.trainers import DPOTrainer, DPOConfig

# Simple DPO configuration
config = DPOConfig(
    output_dir="./dpo_results",
    beta=0.1,
    num_train_epochs=1,
    per_device_train_batch_size=1,
    learning_rate=1e-6,
)

# Create trainer - reference model created automatically
trainer = DPOTrainer(
    model="microsoft/DialoGPT-small",
    args=config,
    train_dataset=preference_dataset,  # with prompt/chosen/rejected
)

trainer.train()
```

## 📊 Dataset Formats

### Messages Format (Preferred)
```python
{
    "messages": [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello!"},
        {"role": "assistant", "content": "Hello! How can I help you?"}
    ]
}
```

### Text Format
```python
{
    "text": "Human: Hello!\nAssistant: Hello! How can I help you?"
}
```

### DPO Format
```python
{
    "prompt": "What is AI?",
    "chosen": "AI stands for Artificial Intelligence...",
    "rejected": "AI is just a buzzword..."
}
```

## 🎯 Design Philosophy

This implementation focuses on:
- **Simplicity**: Clean, minimal API like TRL and Unsloth
- **Chat Templates**: Automatic handling of conversational data
- **Core Functionality**: Essential features without complexity
- **TRL Compatibility**: Familiar patterns for TRL users

## 🚀 Getting Started

1. **Prepare your dataset** in `messages`, `text`, or DPO format
2. **Choose your trainer** (SFT or DPO)
3. **Configure training** with simple config classes
4. **Train your model** with automatic chat template handling

For complete examples, see `examples/trl_style_training_example.py`.
