#!/usr/bin/env python3
"""Example: Train LLM from scratch with custom tokenizer (like in reference notebook)."""

import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from llm_trainer.config import ModelConfig, TrainingConfig, DataConfig
from llm_trainer.models import TransformerLM
from llm_trainer.tokenizer import CustomTokenizerWrapper
from llm_trainer.training import Trainer


def main():
    """Train LLM with custom tokenizer like in the reference notebook."""
    
    print("🚀 Training LLM with Custom Tokenizer (Mistral-style)")
    print("=" * 60)
    
    # Configuration similar to reference notebook
    output_dir = "./output/custom_tokenizer_mistral"
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n📋 Step 1: Load Custom Tokenizer")
    print("-" * 30)
    
    # Load Mistral tokenizer like in the reference notebook
    tokenizer = CustomTokenizerWrapper(tokenizer_name_or_path="mistralai/Mistral-7B-Instruct-v0.2")
    print(f"Loaded Mistral tokenizer with vocab size: {tokenizer.vocab_size}")
    print(f"Pad token: {tokenizer.pad_token} (ID: {tokenizer.pad_token_id})")
    print(f"EOS token: {tokenizer.eos_token} (ID: {tokenizer.eos_token_id})")
    
    print("\n🏗️ Step 2: Configure Model (Mistral-style)")
    print("-" * 30)
    
    # Model configuration similar to reference notebook
    model_config = ModelConfig(
        vocab_size=tokenizer.vocab_size,  # Use tokenizer's vocab size
        d_model=2048,  # hidden_size from reference
        n_heads=32,    # num_attention_heads from reference  
        n_layers=24,   # num_hidden_layers from reference
        d_ff=7168,     # intermediate_size from reference
        max_seq_len=4096,  # max_position_embeddings from reference
        dropout=0.1,
        activation="silu",  # hidden_act from reference
        pre_norm=True
    )
    
    print(f"Model: {model_config.n_layers} layers, {model_config.d_model} dim")
    print(f"Vocab size: {model_config.vocab_size}")
    print(f"Max sequence length: {model_config.max_seq_len}")
    
    print("\n⚙️ Step 3: Configure Training")
    print("-" * 30)
    
    # Training configuration similar to reference notebook
    training_config = TrainingConfig(
        batch_size=2,  # per_device_train_batch_size from reference
        learning_rate=1e-4,  # learning_rate from reference
        weight_decay=0.01,
        num_epochs=1,  # Reduced for demo
        max_steps=2000,  # max_steps from reference
        lr_scheduler="cosine",
        warmup_steps=2,  # warmup_steps from reference
        optimizer="paged_adamw_32bit",  # optim from reference (fallback to adamw)
        gradient_accumulation_steps=1,  # gradient_accumulation_steps from reference
        use_amp=True,
        save_steps=1000,  # save_steps from reference
        eval_steps=500,
        logging_steps=1,  # logging_steps from reference
        checkpoint_dir=os.path.join(output_dir, "checkpoints"),
        report_to=["none"]  # report_to from reference
    )
    
    print(f"Batch size: {training_config.batch_size}")
    print(f"Learning rate: {training_config.learning_rate}")
    print(f"Max steps: {training_config.max_steps}")
    print(f"Optimizer: {training_config.optimizer}")
    
    print("\n📊 Step 4: Configure Data (Cosmopedia-style)")
    print("-" * 30)
    
    # Data configuration similar to reference notebook
    data_config = DataConfig(
        dataset_name="HuggingFaceTB/cosmopedia-20k",  # Dataset from reference
        dataset_config=None,
        dataset_split="train",
        validation_split=None,  # No validation in reference
        text_column="text",
        max_length=2048,  # max_seq_length from reference
        min_length=10,
        pack_sequences=True,
        preprocessing_num_workers=4
    )
    
    print(f"Dataset: {data_config.dataset_name}")
    print(f"Max length: {data_config.max_length}")
    
    print("\n🏗️ Step 5: Create Model")
    print("-" * 30)
    
    # Create model
    model = TransformerLM(model_config)
    print(f"Model created with {model.get_num_params():,} parameters")
    
    print("\n🎯 Step 6: Create Trainer")
    print("-" * 30)
    
    # Create trainer
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        config=training_config
    )
    
    print("Trainer created successfully!")
    
    print("\n🚀 Step 7: Start Training")
    print("-" * 30)
    
    # Save configurations
    model_config.save(os.path.join(output_dir, "model_config.json"))
    training_config.save(os.path.join(output_dir, "training_config.json"))
    data_config.save(os.path.join(output_dir, "data_config.json"))
    
    try:
        print("Starting training with custom tokenizer...")
        trainer.train_from_config(model_config, data_config)
        
        # Save final model
        final_model_path = os.path.join(output_dir, "final_model")
        trainer.save_model(final_model_path)
        print(f"Final model saved to {final_model_path}")
        
    except Exception as e:
        print(f"Training error: {e}")
        print("This is expected if datasets/transformers are not fully set up")
        print("The example shows how to configure custom tokenizers!")
    
    print("\n🎯 Step 8: Text Generation Test")
    print("-" * 30)
    
    # Test text generation
    test_prompt = "The quick brown fox"
    print(f"Testing generation with prompt: '{test_prompt}'")
    
    try:
        generated_text = trainer.generate_text(
            prompt=test_prompt,
            max_length=100,
            temperature=0.8,
            do_sample=True
        )
        print(f"Generated: {generated_text}")
    except Exception as e:
        print(f"Generation test error: {e}")
        print("This is expected without full training")
    
    print("\n✅ Example Complete!")
    print("-" * 30)
    print("Key features demonstrated:")
    print("✓ Loading custom tokenizer (Mistral)")
    print("✓ Configuring model with tokenizer vocab size")
    print("✓ Setting up training similar to reference notebook")
    print("✓ Using Cosmopedia dataset")
    print("✓ Compatible with existing training pipeline")
    
    print(f"\nFiles saved to: {output_dir}")
    print("This example shows how to use any custom tokenizer!")


if __name__ == "__main__":
    main()