#!/usr/bin/env python3
"""Example script for training a small language model."""

import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from llm_trainer.config import ModelConfig, TrainingConfig, DataConfig
from llm_trainer.models import TransformerLM
from llm_trainer.tokenizer import BPETokenizer
from llm_trainer.training import Trainer


def main():
    """Train a small language model on WikiText-2."""
    
    # Model configuration (small model for quick training)
    model_config = ModelConfig(
        vocab_size=32000,  # Will be updated based on tokenizer
        d_model=256,
        n_heads=4,
        n_layers=4,
        d_ff=1024,
        max_seq_len=512,
        dropout=0.1,
        attention_dropout=0.1,
        activation="gelu",
        pre_norm=True
    )
    
    # Training configuration
    training_config = TrainingConfig(
        batch_size=16,
        learning_rate=5e-4,
        weight_decay=0.01,
        num_epochs=3,
        lr_scheduler="cosine",
        warmup_steps=500,
        optimizer="adamw",
        gradient_accumulation_steps=2,
        use_amp=True,
        save_steps=1000,
        eval_steps=500,
        logging_steps=50,
        checkpoint_dir="./checkpoints/small_model",
        report_to=["tensorboard"]
    )
    
    # Data configuration
    data_config = DataConfig(
        dataset_name="wikitext",
        dataset_config="wikitext-2-raw-v1",
        dataset_split="train",
        validation_split="validation",
        text_column="text",
        max_length=512,
        min_length=10,
        vocab_size=32000,
        pack_sequences=True,
        preprocessing_num_workers=4
    )
    
    print("Creating tokenizer...")
    # Create and train tokenizer
    tokenizer = BPETokenizer()
    tokenizer.train_from_dataset(
        dataset_name=data_config.dataset_name,
        dataset_config=data_config.dataset_config,
        vocab_size=data_config.vocab_size,
        max_samples=10000,  # Limit for faster training
        verbose=True
    )
    
    # Update model config with actual vocab size
    model_config.vocab_size = tokenizer.vocab_size
    
    print("Creating model...")
    # Create model
    model = TransformerLM(model_config)
    print(f"Model created with {model.get_num_params():,} parameters")
    
    print("Creating trainer...")
    # Create trainer
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        config=training_config
    )
    
    print("Starting training...")
    # Train from config
    trainer.train_from_config(model_config, data_config)
    
    print("Training completed!")
    
    # Save final model
    output_dir = "./output/small_model"
    os.makedirs(output_dir, exist_ok=True)
    trainer.save_model(output_dir)
    
    print(f"Model saved to {output_dir}")
    
    # Test generation
    print("\nTesting text generation...")
    generated_text = trainer.generate_text(
        prompt="The quick brown fox",
        max_length=50,
        temperature=0.8,
        do_sample=True
    )
    print(f"Generated: {generated_text}")


if __name__ == "__main__":
    main()
