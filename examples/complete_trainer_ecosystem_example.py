#!/usr/bin/env python3
"""
Complete LLM Trainer Ecosystem Example

This example demonstrates the full TRL-style trainer ecosystem with:
- SFT (Supervised Fine-Tuning)
- Reward Model Training
- DPO (Direct Preference Optimization)
- PPO (Proximal Policy Optimization)
- Enhanced callbacks, utils, and judges

The implementation follows TRL patterns while maintaining LLM Trainer's
performance optimizations and clean architecture.
"""

import os
import logging
from datasets import Dataset

# Import the complete trainer ecosystem
from llm_trainer.finetune.trainers import (
    # Core trainers
    SFTTrainer, SFTConfig,
    DPOTrainer, DPOConfig,
    RewardTrainer, RewardConfig,
    PPOTrainer, PPOConfig,
    
    # Supporting modules
    BEMACallback, LogCompletionsCallback, RichProgressCallback,
    RunningMoments, compute_accuracy, disable_dropout_in_model,
    AllTrueJudge, HfPairwiseJudge, PairRMJudge,
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_sft_dataset():
    """Create sample SFT dataset with messages format."""
    data = [
        {
            "messages": [
                {"role": "user", "content": "What is machine learning?"},
                {"role": "assistant", "content": "Machine learning is a subset of AI that enables computers to learn and improve from experience without being explicitly programmed."}
            ]
        },
        {
            "messages": [
                {"role": "user", "content": "Explain neural networks briefly."},
                {"role": "assistant", "content": "Neural networks are computing systems inspired by biological neural networks, consisting of interconnected nodes that process information."}
            ]
        },
    ]
    return Dataset.from_list(data)


def create_preference_dataset():
    """Create sample preference dataset for reward training and DPO."""
    data = [
        {
            "prompt": "What is the best programming language?",
            "chosen": "The best programming language depends on your specific needs and use case. Python is great for beginners and data science, while JavaScript is essential for web development.",
            "rejected": "Python is definitely the best programming language for everything and everyone should use it."
        },
        {
            "prompt": "How do you learn to code?",
            "chosen": "Learning to code involves practice, starting with basics, building projects, and gradually tackling more complex challenges. Online resources, courses, and coding communities can be very helpful.",
            "rejected": "Just memorize all the syntax and you'll be a great programmer immediately."
        },
    ]
    return Dataset.from_list(data)


def example_sft_training():
    """Demonstrate SFT training with enhanced callbacks."""
    logger.info("🚀 Starting SFT Training Example")
    
    # Create dataset and config
    train_dataset = create_sft_dataset()
    config = SFTConfig(
        output_dir="./results/sft-enhanced",
        num_train_epochs=1,
        per_device_train_batch_size=1,
        learning_rate=5e-5,
        max_seq_length=512,
        logging_steps=1,
        save_steps=10,
    )
    
    # Create trainer with enhanced callbacks
    trainer = SFTTrainer(
        model="microsoft/DialoGPT-small",
        args=config,
        train_dataset=train_dataset,
        callbacks=[
            RichProgressCallback(),
            LogCompletionsCallback(log_freq=5),
        ],
    )
    
    # Train and save
    trainer.train()
    trainer.save_model("./results/sft-enhanced/final")
    logger.info("✅ SFT training completed")


def example_reward_training():
    """Demonstrate reward model training."""
    logger.info("🎯 Starting Reward Model Training Example")
    
    # Create preference dataset
    train_dataset = create_preference_dataset()
    config = RewardConfig(
        output_dir="./results/reward-model",
        num_train_epochs=1,
        per_device_train_batch_size=1,
        learning_rate=1e-5,
        loss_type="ranking",
        margin=0.1,
    )
    
    # Create reward trainer
    trainer = RewardTrainer(
        model="microsoft/DialoGPT-small",
        args=config,
        train_dataset=train_dataset,
    )
    
    # Train and save
    trainer.train()
    trainer.save_model("./results/reward-model/final")
    logger.info("✅ Reward model training completed")


def example_dpo_training():
    """Demonstrate DPO training with judges."""
    logger.info("🔄 Starting DPO Training Example")
    
    # Create preference dataset and judge
    train_dataset = create_preference_dataset()
    judge = HfPairwiseJudge("microsoft/DialoGPT-small")
    
    config = DPOConfig(
        output_dir="./results/dpo-enhanced",
        num_train_epochs=1,
        per_device_train_batch_size=1,
        learning_rate=1e-6,
        beta=0.1,
        max_length=512,
    )
    
    # Create DPO trainer
    trainer = DPOTrainer(
        model="microsoft/DialoGPT-small",
        args=config,
        train_dataset=train_dataset,
    )
    
    # Train and save
    trainer.train()
    trainer.save_model("./results/dpo-enhanced/final")
    logger.info("✅ DPO training completed")


def example_ppo_training():
    """Demonstrate PPO training setup."""
    logger.info("🎮 Starting PPO Training Example")
    
    # Create queries for PPO
    queries = [
        "What is artificial intelligence?",
        "How does machine learning work?",
        "Explain deep learning concepts.",
    ]
    
    config = PPOConfig(
        output_dir="./results/ppo-enhanced",
        learning_rate=1e-5,
        clip_range=0.2,
        kl_coef=0.1,
    )
    
    # Create PPO trainer
    trainer = PPOTrainer(
        model="microsoft/DialoGPT-small",
        reward_model="./results/reward-model/final",  # Use trained reward model
        args=config,
        callbacks=[BEMACallback(beta=0.999)],
    )
    
    # Perform PPO steps
    for i, query_batch in enumerate([queries]):
        stats = trainer.step(query_batch)
        logger.info(f"PPO Step {i+1}: {stats}")
    
    logger.info("✅ PPO training demonstration completed")


def example_utils_and_judges():
    """Demonstrate utility functions and judges."""
    logger.info("🔧 Testing Utils and Judges")
    
    # Test running moments
    moments = RunningMoments()
    moments.update([1.0, 2.0, 3.0, 4.0, 5.0])
    mean, var = moments.get_mean_var()
    logger.info(f"Running moments - Mean: {mean:.2f}, Variance: {var:.2f}")
    
    # Test judges
    judges = [
        AllTrueJudge(),
        HfPairwiseJudge(),
        PairRMJudge(),
    ]
    
    prompts = ["What is AI?"]
    responses = ["AI is artificial intelligence", "AI is machine learning"]
    
    for judge in judges:
        score = judge.evaluate(prompts, responses)
        logger.info(f"{judge.name}: Score = {score:.3f}")
    
    logger.info("✅ Utils and judges testing completed")


def main():
    """Run the complete trainer ecosystem demonstration."""
    logger.info("🎉 LLM Trainer - Complete TRL-Style Ecosystem Demo")
    logger.info("=" * 60)
    
    try:
        # Run training examples
        example_sft_training()
        example_reward_training()
        example_dpo_training()
        example_ppo_training()
        
        # Test utilities
        example_utils_and_judges()
        
        logger.info("\n🎊 Complete ecosystem demonstration finished!")
        logger.info("Your LLM Trainer now has a full TRL-compatible ecosystem")
        logger.info("with enhanced trainers, callbacks, utilities, and judges!")
        
    except Exception as e:
        logger.error(f"❌ Demo failed: {e}")
        logger.info("Note: This is a demonstration - some features require actual models and data")


if __name__ == "__main__":
    main()
