# LLM Trainer - Complete TRL-Style Trainer Ecosystem

This directory contains a comprehensive trainer ecosystem that mirrors HuggingFace TRL's structure while maintaining LLM Trainer's performance optimizations and clean architecture.

## 🚀 Available Trainers

### Core Trainers

#### SFTTrainer (Supervised Fine-Tuning)
- **Purpose**: Instruction-following fine-tuning with automatic chat template handling
- **Features**:
  - Automatic chat template application (like TRL/Unsloth)
  - Support for `messages` and `text` formats
  - Clean, simple API
- **Best for**: Instruction following, chat models

#### DPOTrainer (Direct Preference Optimization)
- **Purpose**: Preference-based alignment training
- **Features**:
  - Automatic reference model creation
  - Support for `prompt`/`chosen`/`rejected` format
  - TRL-compatible API
- **Best for**: Model alignment, preference learning

#### PPOTrainer (Proximal Policy Optimization)
- **Purpose**: RLHF training with reward models
- **Features**:
  - Policy optimization with clipping
  - KL divergence penalty
  - Reward model integration
- **Best for**: RLHF, complex alignment tasks

#### RewardTrainer (Reward Model Training)
- **Purpose**: Training reward models for RLHF
- **Features**:
  - Ranking loss optimization
  - Preference pair training
  - Automatic reward head addition
- **Best for**: Building reward models for PPO

### Advanced Trainers

#### ORPOTrainer (Odds Ratio Preference Optimization)
- **Purpose**: Alternative to DPO with odds ratio formulation
- **Best for**: Preference learning without reference models

#### KTOTrainer (Kahneman-Tversky Optimization)
- **Purpose**: Preference optimization based on prospect theory
- **Best for**: Human-aligned preference learning

#### GRPOTrainer (Group Relative Policy Optimization)
- **Purpose**: Group-based policy optimization
- **Best for**: Multi-group preference alignment

#### RLOOTrainer (REINFORCE Leave One Out)
- **Purpose**: REINFORCE with leave-one-out baseline
- **Best for**: Policy gradient methods

#### XPOTrainer (eXpected Policy Optimization)
- **Purpose**: Expected policy optimization
- **Best for**: Robust policy learning

#### CPOTrainer (Conservative Policy Optimization)
- **Purpose**: Conservative policy updates
- **Best for**: Safe policy learning

#### BCOTrainer (Binary Classifier Optimization)
- **Purpose**: Binary classification for preferences
- **Best for**: Simple preference classification

#### PRMTrainer (Process Reward Model)
- **Purpose**: Training process-based reward models
- **Best for**: Step-by-step reasoning tasks

#### NashMDTrainer (Nash Mirror Descent)
- **Purpose**: Game-theoretic optimization
- **Best for**: Multi-agent scenarios

#### OnlineDPOTrainer (Online Direct Preference Optimization)
- **Purpose**: Online version of DPO
- **Best for**: Streaming preference data

## 🛠 Supporting Modules

### Callbacks
- **BEMACallback**: Beta Exponential Moving Average for model parameters
- **LogCompletionsCallback**: Log model completions during training
- **MergeModelCallback**: Merge PEFT adapters with base model
- **RichProgressCallback**: Enhanced progress logging with rich formatting
- **SyncRefModelCallback**: Synchronize reference models in preference learning
- **WinRateCallback**: Track win rates in preference learning

### Utilities
- **RunningMoments**: Compute running statistics using Welford's algorithm
- **compute_accuracy**: Calculate accuracy between predictions and labels
- **disable_dropout_in_model**: Disable dropout in all model modules
- **empty_cache**: Clear GPU cache if CUDA is available
- **peft_module_casting_to_bf16**: Cast PEFT modules to bfloat16
- **get_model_param_count**: Count model parameters
- **freeze_model_layers**: Freeze specified model layers

### Judges
- **BaseJudge**: Abstract base class for all judges
- **BaseBinaryJudge**: Binary classification judges
- **BasePairwiseJudge**: Pairwise comparison judges
- **BaseRankJudge**: Ranking judges
- **AllTrueJudge**: Always returns True (for testing)
- **HfPairwiseJudge**: Pairwise judge using HuggingFace models
- **OpenAIPairwiseJudge**: Pairwise judge using OpenAI API
- **PairRMJudge**: Pairwise judge using reward models

## 🎯 Key Features

### TRL Compatibility
- **Lazy Loading**: TRL-style lazy module loading for efficiency
- **Consistent API**: Familiar patterns for TRL users
- **Complete Structure**: All TRL trainers and supporting modules

### Chat Template Focus
- **Automatic Detection**: Automatically applies chat templates to `messages` format
- **Fallback Formatting**: Simple fallback for models without chat templates
- **TRL/Unsloth Style**: Follows the same patterns as leading libraries

### Dataset Support
- **Messages Format**: `{"messages": [{"role": "user", "content": "..."}, ...]}`
- **Text Format**: `{"text": "Human: ... Assistant: ..."}`
- **DPO Format**: `{"prompt": "...", "chosen": "...", "rejected": "..."}`
- **Reward Format**: Preference pairs for reward model training

## 🛠 Usage Examples

### Basic SFT Training (TRL/Unsloth Style)
```python
from llm_trainer.finetune.trainers import SFTTrainer, SFTConfig, RichProgressCallback

# Simple configuration
config = SFTConfig(
    output_dir="./results",
    num_train_epochs=1,
    per_device_train_batch_size=1,
    learning_rate=5e-5,
    max_seq_length=512,
)

# Create trainer with enhanced callbacks
trainer = SFTTrainer(
    model="microsoft/DialoGPT-small",
    args=config,
    train_dataset=dataset,  # with 'messages' or 'text' format
    callbacks=[RichProgressCallback()],
)

trainer.train()
```

### Reward Model Training
```python
from llm_trainer.finetune.trainers import RewardTrainer, RewardConfig

config = RewardConfig(
    output_dir="./reward_model",
    num_train_epochs=1,
    per_device_train_batch_size=1,
    learning_rate=1e-5,
    loss_type="ranking",
    margin=0.1,
)

trainer = RewardTrainer(
    model="microsoft/DialoGPT-small",
    args=config,
    train_dataset=preference_dataset,  # with prompt/chosen/rejected
)

trainer.train()
```

### DPO Training with Judges
```python
from llm_trainer.finetune.trainers import DPOTrainer, DPOConfig, HfPairwiseJudge

# Create judge for evaluation
judge = HfPairwiseJudge("microsoft/DialoGPT-small")

config = DPOConfig(
    output_dir="./dpo_results",
    beta=0.1,
    num_train_epochs=1,
    per_device_train_batch_size=1,
    learning_rate=1e-6,
)

trainer = DPOTrainer(
    model="microsoft/DialoGPT-small",
    args=config,
    train_dataset=preference_dataset,
)

trainer.train()
```

### PPO Training (RLHF)
```python
from llm_trainer.finetune.trainers import PPOTrainer, PPOConfig, BEMACallback

config = PPOConfig(
    output_dir="./ppo_results",
    learning_rate=1e-5,
    clip_range=0.2,
    kl_coef=0.1,
)

trainer = PPOTrainer(
    model="microsoft/DialoGPT-small",
    reward_model="./reward_model/final",
    args=config,
    callbacks=[BEMACallback(beta=0.999)],
)

# PPO training loop
queries = ["What is AI?", "How does ML work?"]
stats = trainer.step(queries)
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

### Quick Start
1. **Prepare your dataset** in supported formats (`messages`, `text`, preference pairs)
2. **Choose your trainer** based on your use case:
   - **SFT**: For instruction following and chat models
   - **Reward**: For training reward models from preferences
   - **DPO**: For preference-based alignment without reward models
   - **PPO**: For RLHF with reward models
3. **Configure training** with trainer-specific config classes
4. **Add callbacks and utilities** for enhanced training experience
5. **Train your model** with automatic optimizations

### Complete Examples
- **Basic Usage**: `examples/trl_style_training_example.py`
- **Full Ecosystem**: `examples/complete_trainer_ecosystem_example.py`

### Architecture Benefits
- **TRL Compatibility**: Drop-in replacement for TRL trainers
- **Performance Optimized**: LLM Trainer's kernel optimizations
- **Clean API**: Simple, intuitive interface
- **Extensible**: Easy to add custom trainers and callbacks
- **Production Ready**: Comprehensive error handling and logging

The LLM Trainer ecosystem now provides a complete, professional fine-tuning solution that rivals TRL while maintaining superior performance and clean architecture! 🎉
