#!/usr/bin/env python3
"""Example script for training a small language model with device selection."""

import argparse
import os
import sys
import torch
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from llm_trainer.config import ModelConfig, TrainingConfig, DataConfig
from llm_trainer.models import TransformerLM
from llm_trainer.tokenizer import BPETokenizer
from llm_trainer.training import Trainer


def get_device_config(device: str = "auto"):
    """Get device-specific configuration adjustments."""
    if device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    
    # Device-specific optimizations
    if device == "cpu":
        return {
            "batch_size": 2,
            "learning_rate": 8e-4,
            "gradient_accumulation_steps": 8,
            "use_amp": False,
            "dataloader_num_workers": 2,
            "dataloader_pin_memory": False,
            "save_steps": 500,
            "eval_steps": 250,
            "logging_steps": 25
        }
    else:
        return {
            "batch_size": 16,
            "learning_rate": 5e-4,
            "gradient_accumulation_steps": 2,
            "use_amp": True,
            "dataloader_num_workers": 4,
            "dataloader_pin_memory": True,
            "save_steps": 1000,
            "eval_steps": 500,
            "logging_steps": 50
        }


def main():
    """Train a small language model on WikiText-2."""
    
    parser = argparse.ArgumentParser(description="Train small language model")
    parser.add_argument("--device", type=str, default="auto",
                       choices=["auto", "cpu", "cuda", "mps"],
                       help="Device to use for training")
    parser.add_argument("--use_cpu_config", action="store_true",
                       help="Use CPU-optimized configuration file")
    
    args = parser.parse_args()
    
    # Get device-specific configuration
    device_config = get_device_config(args.device)
    
    print(f"Using device: {args.device}")
    print(f"Device config: {device_config}")
    
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
    
    # Training configuration with device-specific settings
    training_config = TrainingConfig(
        batch_size=device_config["batch_size"],
        learning_rate=device_config["learning_rate"],
        weight_decay=0.01,
        num_epochs=3,
        lr_scheduler="cosine",
        warmup_steps=500,
        optimizer="adamw",
        gradient_accumulation_steps=device_config["gradient_accumulation_steps"],
        use_amp=device_config["use_amp"],
        save_steps=device_config["save_steps"],
        eval_steps=device_config["eval_steps"],
        logging_steps=device_config["logging_steps"],
        checkpoint_dir="./checkpoints/small_model",
        report_to=["tensorboard"],
        device=args.device
    )
    
    # Data configuration with device-specific settings
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
        preprocessing_num_workers=device_config["dataloader_num_workers"]
    )
    
    # Alternative: Load from CPU-optimized config file
    if args.use_cpu_config:
        print("Loading CPU-optimized configuration from file...")
        import yaml
        config_path = Path(__file__).parent.parent / "configs" / "cpu_small_model.yaml"
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Override with config file settings
        model_config = ModelConfig.from_dict(config["model"])
        training_config = TrainingConfig.from_dict(config["training"])
        data_config = DataConfig.from_dict(config["data"])
        training_config.checkpoint_dir = "./checkpoints/small_model"
        training_config.report_to = ["tensorboard"]
    
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
