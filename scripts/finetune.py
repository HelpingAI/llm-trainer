#!/usr/bin/env python3
"""
Fine-tuning script for language models using LLM Trainer's comprehensive system.

This script provides a complete fine-tuning solution inspired by Unsloth, TRL, and Axolotl,
with support for various training methods, quantization, LoRA, and export formats.

Usage:
    # Using YAML config (recommended)
    python scripts/finetune.py --config configs/finetune_example.yaml

    # Using command line arguments
    python scripts/finetune.py --model microsoft/DialoGPT-medium --dataset alpaca --output-dir ./output
"""

import argparse
import os
import sys
import yaml
import json
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import logging

from llm_trainer.finetune import (
    create_finetune_trainer,
    create_trainer_from_config,
    export_model
)


def load_config(config_path: str) -> dict:
    """Load configuration from YAML or JSON file."""
    config_path = Path(config_path)

    with open(config_path, 'r') as f:
        if config_path.suffix.lower() in ['.yaml', '.yml']:
            return yaml.safe_load(f)
        else:
            return json.load(f)


def create_default_config(args) -> dict:
    """Create default configuration from command line arguments."""
    return {
        "model": {
            "model_name_or_path": args.model,
            "max_seq_length": args.max_length,
            "torch_dtype": args.dtype,
            "quantization_bits": args.quantization_bits,
            "use_gradient_checkpointing": True,
            "lora": {
                "r": args.lora_r,
                "alpha": args.lora_alpha,
                "dropout": args.lora_dropout,
                "target_modules": "auto"
            } if args.use_lora else None
        },
        "data": {
            "dataset_name": args.dataset,
            "dataset_config": args.dataset_config,
            "split": args.split,
            "validation_split": args.validation_split,
            "eval_size": args.eval_size
        },
        "training": {
            "output_dir": args.output_dir,
            "num_train_epochs": args.epochs,
            "per_device_train_batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "warmup_steps": args.warmup_steps,
            "logging_steps": args.logging_steps,
            "save_steps": args.save_steps,
            "evaluation_strategy": "steps" if args.eval_steps else "no",
            "eval_steps": args.eval_steps,
            "bf16": args.bf16,
            "fp16": args.fp16,
            "gradient_checkpointing": True,
            "seed": args.seed
        }
    }


def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune language models with LLM Trainer",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Configuration
    parser.add_argument("--config", help="Configuration file (YAML or JSON)")

    # Model arguments (for CLI mode)
    model_group = parser.add_argument_group("Model Configuration")
    model_group.add_argument("--model", help="Model name or path")
    model_group.add_argument("--dtype", default="auto", choices=["auto", "float16", "bfloat16", "float32"])
    model_group.add_argument("--quantization-bits", type=int, choices=[4, 8])
    model_group.add_argument("--max-length", type=int, default=2048)

    # LoRA arguments
    lora_group = parser.add_argument_group("LoRA Configuration")
    lora_group.add_argument("--use-lora", action="store_true", help="Use LoRA fine-tuning")
    lora_group.add_argument("--lora-r", type=int, default=16)
    lora_group.add_argument("--lora-alpha", type=int, default=32)
    lora_group.add_argument("--lora-dropout", type=float, default=0.05)

    # Data arguments
    data_group = parser.add_argument_group("Data Configuration")
    data_group.add_argument("--dataset", help="Dataset name or path")
    data_group.add_argument("--dataset-config", help="Dataset configuration")
    data_group.add_argument("--split", default="train")
    data_group.add_argument("--validation-split", help="Validation split name")
    data_group.add_argument("--eval-size", type=float, default=0.1)

    # Training arguments
    training_group = parser.add_argument_group("Training Configuration")
    training_group.add_argument("--trainer-type", default="sft", choices=["sft", "dpo", "ppo", "reward"])
    training_group.add_argument("--epochs", type=int, default=3)
    training_group.add_argument("--batch-size", type=int, default=4)
    training_group.add_argument("--learning-rate", type=float, default=2e-4)
    training_group.add_argument("--weight-decay", type=float, default=0.01)
    training_group.add_argument("--warmup-steps", type=int, default=100)
    training_group.add_argument("--logging-steps", type=int, default=10)
    training_group.add_argument("--save-steps", type=int, default=500)
    training_group.add_argument("--eval-steps", type=int)

    # Mixed precision
    precision_group = parser.add_argument_group("Mixed Precision")
    precision_group.add_argument("--bf16", action="store_true")
    precision_group.add_argument("--fp16", action="store_true")

    # Export options
    export_group = parser.add_argument_group("Export Options")
    export_group.add_argument("--export-formats", nargs="+", default=["huggingface"],
                             choices=["huggingface", "gguf", "ollama"])
    export_group.add_argument("--merge-lora", action="store_true", default=True)
    export_group.add_argument("--push-to-hub", action="store_true")
    export_group.add_argument("--repo-id", help="Repository ID for Hub upload")

    # General arguments
    parser.add_argument("--output-dir", help="Output directory")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args()

    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)

    # Determine mode: config file or CLI arguments
    if args.config:
        logger.info(f"📄 Loading configuration from {args.config}")
        cfg = load_config(args.config)

        # Extract configurations
        model_cfg = cfg.get("model", {})
        data_cfg = cfg.get("data", {})
        train_cfg = cfg.get("training", {})
        output_dir = cfg.get("output_dir", args.output_dir or "./output/finetune")
        trainer_type = cfg.get("trainer_type", "sft")

        # Create trainer from config
        trainer = create_trainer_from_config(args.config, trainer_type)

    else:
        # CLI mode - require essential arguments
        if not args.model or not args.dataset or not args.output_dir:
            parser.error("--model, --dataset, and --output-dir are required when not using --config")

        logger.info("⚙️  Using command line configuration")
        cfg = create_default_config(args)

        model_cfg = cfg["model"]
        data_cfg = cfg["data"]
        train_cfg = cfg["training"]
        output_dir = args.output_dir
        trainer_type = args.trainer_type

        # Create trainer
        trainer = create_finetune_trainer(
            model_config=model_cfg,
            data_config=data_cfg,
            training_config=train_cfg,
            trainer_type=trainer_type
        )

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    logger.info("🚀 Starting LLM Trainer Fine-tuning")
    logger.info(f"   Model: {model_cfg.get('model_name_or_path', 'Unknown')}")
    logger.info(f"   Dataset: {data_cfg.get('dataset_name', 'Unknown')}")
    logger.info(f"   Trainer: {trainer_type.upper()}")
    logger.info(f"   Output: {output_dir}")

    # Train
    logger.info("🏋️  Starting training...")
    try:
        trainer.train()
        logger.info("✅ Training completed successfully!")
    except Exception as e:
        logger.error(f"❌ Training failed: {e}")
        raise

    # Save model
    final_dir = os.path.join(output_dir, "final_model")
    logger.info(f"💾 Saving model to {final_dir}")
    trainer.save_model(final_dir)

    # Export to additional formats
    export_formats = getattr(args, 'export_formats', ['huggingface'])
    if len(export_formats) > 1 or export_formats[0] != "huggingface":
        logger.info(f"📦 Exporting to formats: {export_formats}")
        export_model(
            trainer.model,
            trainer.tokenizer,
            Path(output_dir) / "exports",
            formats=export_formats,
            model_name="fine-tuned-model",
            merge_lora=getattr(args, 'merge_lora', True),
            push_to_hub=getattr(args, 'push_to_hub', False),
            repo_id=getattr(args, 'repo_id', None)
        )

    logger.info("🎉 Fine-tuning completed successfully!")
    logger.info(f"📁 Model saved to: {final_dir}")


if __name__ == "__main__":
    main()

