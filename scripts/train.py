#!/usr/bin/env python3
"""Training script for language models."""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
from llm_trainer.models import TransformerLM
from llm_trainer.tokenizer import BPETokenizer
from llm_trainer.config import ModelConfig, TrainingConfig, DataConfig
from llm_trainer.training import Trainer
from llm_trainer.data import LanguageModelingDataset


def setup_logging(log_level: str = "info"):
    """Setup logging configuration."""
    level_map = {
        "debug": logging.DEBUG,
        "info": logging.INFO,
        "warning": logging.WARNING,
        "error": logging.ERROR
    }
    
    logging.basicConfig(
        level=level_map.get(log_level.lower(), logging.INFO),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def create_or_load_tokenizer(tokenizer_config: dict, force_retrain: bool = False) -> BPETokenizer:
    """Create or load tokenizer."""
    tokenizer_path = tokenizer_config.get("save_path", "tokenizer")
    
    # Try to load existing tokenizer
    if not force_retrain and os.path.exists(tokenizer_path):
        try:
            logging.info(f"Loading existing tokenizer from {tokenizer_path}")
            tokenizer = BPETokenizer.from_pretrained(tokenizer_path)
            return tokenizer
        except Exception as e:
            logging.warning(f"Failed to load tokenizer: {e}. Training new one.")
    
    # Train new tokenizer
    logging.info("Training new tokenizer...")
    tokenizer = BPETokenizer()
    
    # Train from dataset
    dataset_name = tokenizer_config["dataset_name"]
    dataset_config = tokenizer_config.get("dataset_config")
    vocab_size = tokenizer_config.get("vocab_size", 50000)
    max_samples = tokenizer_config.get("max_samples")
    
    tokenizer.train_from_dataset(
        dataset_name=dataset_name,
        dataset_config=dataset_config,
        vocab_size=vocab_size,
        max_samples=max_samples,
        verbose=True
    )
    
    # Save tokenizer
    os.makedirs(tokenizer_path, exist_ok=True)
    tokenizer.save_pretrained(tokenizer_path)
    logging.info(f"Tokenizer saved to {tokenizer_path}")
    
    return tokenizer


def create_model(model_config: ModelConfig, tokenizer: BPETokenizer) -> TransformerLM:
    """Create model from configuration."""
    # Update vocab size to match tokenizer
    model_config.vocab_size = tokenizer.vocab_size
    
    # Create model
    model = TransformerLM(model_config)
    
    logging.info(f"Created model with {model.get_num_params():,} parameters")
    return model


def create_datasets(data_config: DataConfig, tokenizer: BPETokenizer) -> tuple:
    """Create training and evaluation datasets."""
    logging.info("Creating datasets...")
    
    # Training dataset
    train_dataset = LanguageModelingDataset(
        dataset_name=data_config.dataset_name,
        dataset_config=data_config.dataset_config,
        split=data_config.dataset_split,
        tokenizer=tokenizer,
        text_column=data_config.text_column,
        max_length=data_config.max_length,
        preprocessing_config={
            "min_length": data_config.min_length,
            "filter_empty": data_config.filter_empty_lines,
            "remove_duplicates": data_config.remove_duplicates
        },
        pack_sequences=data_config.pack_sequences,
        packing_strategy=data_config.packing_strategy,
        num_proc=data_config.preprocessing_num_workers
    )
    
    # Evaluation dataset
    eval_dataset = None
    if data_config.validation_split:
        eval_dataset = LanguageModelingDataset(
            dataset_name=data_config.dataset_name,
            dataset_config=data_config.dataset_config,
            split=data_config.validation_split,
            tokenizer=tokenizer,
            text_column=data_config.text_column,
            max_length=data_config.max_length,
            preprocessing_config={
                "min_length": data_config.min_length,
                "filter_empty": data_config.filter_empty_lines,
                "remove_duplicates": data_config.remove_duplicates
            },
            pack_sequences=data_config.pack_sequences,
            packing_strategy=data_config.packing_strategy,
            num_proc=data_config.preprocessing_num_workers
        )
    
    logging.info(f"Training dataset size: {len(train_dataset)}")
    if eval_dataset:
        logging.info(f"Evaluation dataset size: {len(eval_dataset)}")
    
    return train_dataset, eval_dataset


def main():
    parser = argparse.ArgumentParser(description="Train language model")
    parser.add_argument("--config", type=str, required=True,
                       help="Path to training configuration file")
    parser.add_argument("--model_config", type=str,
                       help="Path to model configuration file (overrides config)")
    parser.add_argument("--data_config", type=str,
                       help="Path to data configuration file (overrides config)")
    parser.add_argument("--training_config", type=str,
                       help="Path to training configuration file (overrides config)")
    parser.add_argument("--output_dir", type=str, default="./output",
                       help="Output directory for model and checkpoints")
    parser.add_argument("--resume_from_checkpoint", type=str,
                       help="Path to checkpoint to resume from")
    parser.add_argument("--force_retrain_tokenizer", action="store_true",
                       help="Force retraining of tokenizer")
    parser.add_argument("--device", type=str, default=None,
                       choices=["auto", "cpu", "cuda", "mps"],
                       help="Device to use for training (overrides config file)")
    parser.add_argument("--log_level", type=str, default="info",
                       choices=["debug", "info", "warning", "error"],
                       help="Logging level")
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    
    # Load main configuration
    with open(args.config, 'r') as f:
        config = json.load(f)
    
    # Load individual configs if provided
    if args.model_config:
        with open(args.model_config, 'r') as f:
            config["model"] = json.load(f)
    
    if args.data_config:
        with open(args.data_config, 'r') as f:
            config["data"] = json.load(f)
    
    if args.training_config:
        with open(args.training_config, 'r') as f:
            config["training"] = json.load(f)
    
    # Create configuration objects
    model_config = ModelConfig.from_dict(config["model"])
    data_config = DataConfig.from_dict(config["data"])
    training_config = TrainingConfig.from_dict(config["training"])
    
    # Override checkpoint directory
    training_config.checkpoint_dir = args.output_dir
    
    # Override resume checkpoint
    if args.resume_from_checkpoint:
        training_config.resume_from_checkpoint = args.resume_from_checkpoint
    
    # Override device if provided via command line
    if args.device:
        training_config.device = args.device
        logging.info(f"Device overridden via command line: {args.device}")
        
        # Apply CPU-specific optimizations if device is CPU
        if args.device == "cpu":
            logging.info("Applying CPU optimizations...")
            training_config.use_amp = False
            training_config.dataloader_pin_memory = False
            # Reduce workers to avoid overhead on CPU
            if hasattr(training_config, 'dataloader_num_workers'):
                training_config.dataloader_num_workers = min(training_config.dataloader_num_workers, 2)
            # Reduce batch size if it's too large for CPU
            if training_config.batch_size > 4:
                logging.info(f"Reducing batch size from {training_config.batch_size} to 4 for CPU training")
                training_config.batch_size = 4
    
    # Create or load tokenizer
    tokenizer_config = config.get("tokenizer", {})
    tokenizer_config.setdefault("dataset_name", data_config.dataset_name)
    tokenizer_config.setdefault("dataset_config", data_config.dataset_config)
    tokenizer_config.setdefault("vocab_size", model_config.vocab_size)
    tokenizer_config.setdefault("save_path", os.path.join(args.output_dir, "tokenizer"))
    
    tokenizer = create_or_load_tokenizer(tokenizer_config, args.force_retrain_tokenizer)
    
    # Create model
    model = create_model(model_config, tokenizer)
    
    # Create datasets
    train_dataset, eval_dataset = create_datasets(data_config, tokenizer)
    
    # Create trainer
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        config=training_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_config=data_config
    )
    
    # Save configurations
    os.makedirs(args.output_dir, exist_ok=True)
    model_config.save(os.path.join(args.output_dir, "model_config.json"))
    data_config.save(os.path.join(args.output_dir, "data_config.json"))
    training_config.save(os.path.join(args.output_dir, "training_config.json"))
    
    # Start training
    logging.info("Starting training...")
    trainer.train()
    
    # Save final model
    final_model_path = os.path.join(args.output_dir, "final_model")
    trainer.save_model(final_model_path)
    
    logging.info(f"Training completed. Model saved to {final_model_path}")


if __name__ == "__main__":
    main()
