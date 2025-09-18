#!/usr/bin/env python3
"""
Clean TRL/Unsloth-Style Training Example

This example demonstrates the simplified LLM Trainer fine-tuning system
that focuses on chat templates and proper dataset formatting like TRL and Unsloth.

Key features:
- Automatic chat template application
- Simple dataset format support (text, messages)
- Clean TRL-style API
- Focus on core functionality
"""

import os
from datasets import Dataset

# Import the clean trainers
from llm_trainer.finetune.trainers import (
    SFTTrainer,
    SFTConfig,
    DPOTrainer,
    DPOConfig,
)


def create_sample_sft_dataset():
    """Create a sample SFT dataset for demonstration."""
    data = [
        {
            "messages": [
                {"role": "user", "content": "What is the capital of France?"},
                {"role": "assistant", "content": "The capital of France is Paris."}
            ]
        },
        {
            "messages": [
                {"role": "user", "content": "Explain quantum computing in simple terms."},
                {"role": "assistant", "content": "Quantum computing uses quantum mechanics principles to process information in ways that classical computers cannot, potentially solving certain problems much faster."}
            ]
        },
        {
            "text": "Human: How do you make coffee?\nAssistant: To make coffee, you'll need coffee beans, water, and a brewing method like a coffee maker, French press, or pour-over. Grind the beans, add hot water, and let it brew for the appropriate time."
        }
    ]
    return Dataset.from_list(data)


def create_sample_dpo_dataset():
    """Create a sample DPO dataset for demonstration."""
    data = [
        {
            "prompt": "What is the best programming language?",
            "chosen": "The best programming language depends on your specific needs and use case. Python is great for beginners and data science, JavaScript for web development, and C++ for performance-critical applications.",
            "rejected": "Python is definitely the best programming language and everyone should use it for everything."
        },
        {
            "prompt": "How should I invest my money?",
            "chosen": "Investment decisions should be based on your financial goals, risk tolerance, and time horizon. Consider consulting with a financial advisor and diversifying your portfolio across different asset classes.",
            "rejected": "Just put all your money in cryptocurrency, it's guaranteed to make you rich!"
        }
    ]
    return Dataset.from_list(data)


def example_sft_training():
    """Minimal SFT training example (silent)."""
    train_dataset = create_sample_sft_dataset()
    sft_config = SFTConfig(
        output_dir="./results/sft-example",
        num_train_epochs=1,
        per_device_train_batch_size=1,
        learning_rate=5e-5,
        max_seq_length=512,
    )
    trainer = SFTTrainer(
        model="microsoft/DialoGPT-small",
        args=sft_config,
        train_dataset=train_dataset,
    )
    trainer.train()
    trainer.save_model("./results/sft-example/final")


def example_dpo_training():
    """Minimal DPO training example (silent)."""
    train_dataset = create_sample_dpo_dataset()
    dpo_config = DPOConfig(
        output_dir="./results/dpo-example",
        num_train_epochs=1,
        per_device_train_batch_size=1,
        learning_rate=1e-6,
        max_length=512,
        beta=0.1,
    )
    trainer = DPOTrainer(
        model="microsoft/DialoGPT-small",
        args=dpo_config,
        train_dataset=train_dataset,
    )
    trainer.train()
    trainer.save_model("./results/dpo-example/final")




def main():
    """Run minimal SFT and DPO examples (silent)."""
    example_sft_training()
    example_dpo_training()


if __name__ == "__main__":
    main()
