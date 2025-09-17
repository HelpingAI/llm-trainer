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
    """Example of Supervised Fine-Tuning using clean TRL/Unsloth-style API."""
    print("🚀 Starting SFT Training Example")
    print("=" * 50)

    # Create sample dataset with chat format
    train_dataset = create_sample_sft_dataset()

    # Simple TRL-style configuration
    sft_config = SFTConfig(
        output_dir="./results/sft-example",
        num_train_epochs=1,
        per_device_train_batch_size=1,
        learning_rate=5e-5,
        max_seq_length=512,
        logging_steps=1,
        save_steps=10,
    )

    # Initialize SFT trainer (TRL/Unsloth style)
    trainer = SFTTrainer(
        model="microsoft/DialoGPT-small",
        args=sft_config,
        train_dataset=train_dataset,
    )

    print(f"📊 Trainer initialized")
    print(f"📈 Training dataset size: {len(train_dataset)}")
    print(f"🎯 Chat template will be automatically applied to 'messages' format")

    # Print trainable parameters
    trainer.print_trainable_parameters()

    # Start training
    print("\n🏃 Starting training...")
    try:
        trainer.train()
        print("✅ SFT training completed successfully!")

        # Save the model
        trainer.save_model("./results/sft-example/final")
        print("💾 Model saved")

    except Exception as e:
        print(f"❌ Training failed: {e}")

    print("=" * 50)


def example_dpo_training():
    """Example of Direct Preference Optimization using clean TRL-style API."""
    print("🎯 Starting DPO Training Example")
    print("=" * 50)

    # Create sample preference dataset
    train_dataset = create_sample_dpo_dataset()

    # Simple DPO configuration
    dpo_config = DPOConfig(
        output_dir="./results/dpo-example",
        num_train_epochs=1,
        per_device_train_batch_size=1,
        learning_rate=1e-6,
        max_length=512,
        beta=0.1,  # DPO beta parameter
        logging_steps=1,
        save_steps=10,
    )

    # Initialize DPO trainer (TRL style)
    trainer = DPOTrainer(
        model="microsoft/DialoGPT-small",
        args=dpo_config,
        train_dataset=train_dataset,
    )

    print(f"📊 DPO Trainer initialized")
    print(f"📈 Training dataset size: {len(train_dataset)}")
    print(f"🎯 Beta parameter: {trainer.beta}")
    print(f"🔄 Reference model will be auto-created")

    # Print trainable parameters
    trainer.print_trainable_parameters()

    # Start training
    print("\n🏃 Starting training...")
    try:
        trainer.train()
        print("✅ DPO training completed successfully!")

        # Save the model
        trainer.save_model("./results/dpo-example/final")
        print("💾 Model saved")

    except Exception as e:
        print(f"❌ Training failed: {e}")

    print("=" * 50)


def example_dataset_formats():
    """Demonstrate different dataset formats supported by the trainers."""
    print("📋 Dataset Format Examples")
    print("=" * 50)

    # Messages format (ChatML style) - preferred format
    messages_data = [
        {
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Hello!"},
                {"role": "assistant", "content": "Hello! How can I help you today?"}
            ]
        }
    ]

    # Plain text format
    text_data = [
        {
            "text": "This is a sample text for language modeling."
        }
    ]

    # DPO format
    dpo_data = [
        {
            "prompt": "What is AI?",
            "chosen": "AI stands for Artificial Intelligence, which refers to computer systems that can perform tasks typically requiring human intelligence.",
            "rejected": "AI is just a buzzword that doesn't mean anything real."
        }
    ]

    print("✅ Messages format (ChatML):", len(messages_data), "examples - PREFERRED")
    print("✅ Plain text format:", len(text_data), "examples")
    print("✅ DPO format:", len(dpo_data), "examples")

    print("\n🔄 Use chat templates or formatting functions for best results!")
    print("=" * 50)


def main():
    """Main function to run all examples."""
    print("🎉 LLM Trainer - Clean TRL/Unsloth-Style Examples")
    print("=" * 60)
    print("Simplified trainers focusing on chat templates and")
    print("proper dataset formatting like TRL and Unsloth.")
    print("=" * 60)

    # Show supported dataset formats
    example_dataset_formats()

    # Run SFT example
    example_sft_training()

    # Run DPO example
    example_dpo_training()

    print("\n🎊 Examples completed!")
    print("Clean, focused trainers with TRL/Unsloth-style API")
    print("and automatic chat template handling!")


if __name__ == "__main__":
    main()
