#!/usr/bin/env python3
"""Complete pipeline example: tokenizer training, model training, and evaluation."""

import os
import sys
import json
from pathlib import Path
from typing import Dict, Any

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from llm_trainer.config import ModelConfig, TrainingConfig, DataConfig
from llm_trainer.models import TransformerLM
from llm_trainer.tokenizer import BPETokenizer
from llm_trainer.training import Trainer
from llm_trainer.utils.generation import TextGenerator, GenerationConfig
from llm_trainer.utils.metrics import compute_perplexity


def main() -> Any:
    """Complete pipeline from tokenizer training to model evaluation."""
    
    print("🚀 Starting Complete LLM Training Pipeline")
    print("=" * 60)
    
    # Configuration
    output_dir = "./output/complete_pipeline"
    os.makedirs(output_dir, exist_ok=True)
    
    # Step 1: Configure the pipeline
    print("\n📋 Step 1: Configuration")
    print("-" * 30)
    
    model_config = ModelConfig(
        vocab_size=32000,  # Will be updated by tokenizer
        d_model=512,
        n_heads=8,
        n_layers=6,
        d_ff=2048,
        max_seq_len=1024,
        dropout=0.1,
        activation="gelu",
        pre_norm=True
    )
    
    training_config = TrainingConfig(
        batch_size=8,
        learning_rate=1e-4,
        weight_decay=0.01,
        num_epochs=2,
        lr_scheduler="cosine",
        warmup_steps=1000,
        optimizer="adamw",
        gradient_accumulation_steps=4,
        fp16=True,
        save_steps=1000,
        eval_steps=500,
        logging_steps=100,
        checkpoint_dir=os.path.join(output_dir, "checkpoints"),
        report_to=["tensorboard"]
    )
    
    data_config = DataConfig(
        dataset_name="wikitext",
        dataset_config="wikitext-2-raw-v1",
        dataset_split="train",
        validation_split="validation",
        text_column="text",
        max_length=1024,
        min_length=10,
        vocab_size=32000,
        pack_sequences=True,
        preprocessing_num_workers=4
    )
    
    print(f"Model: {model_config.n_layers} layers, {model_config.d_model} dim")
    print(f"Training: {training_config.num_epochs} epochs, batch size {training_config.batch_size}")
    print(f"Data: {data_config.dataset_name}-{data_config.dataset_config}")
    
    # Step 2: Train or load tokenizer
    print("\n🔤 Step 2: Tokenizer Training")
    print("-" * 30)
    
    tokenizer_path = os.path.join(output_dir, "tokenizer")
    
    if os.path.exists(tokenizer_path):
        print(f"Loading existing tokenizer from {tokenizer_path}")
        tokenizer = BPETokenizer.from_pretrained(tokenizer_path)
    else:
        print("Training new BPE tokenizer...")
        tokenizer = BPETokenizer()
        tokenizer.train(
            data_config.dataset_name,
            dataset_config=data_config.dataset_config,
            vocab_size=data_config.vocab_size,
            max_samples=50000,  # Limit for faster training
            verbose=True
        )
        
        # Save tokenizer
        os.makedirs(tokenizer_path, exist_ok=True)
        tokenizer.save_pretrained(tokenizer_path)
        print(f"Tokenizer saved to {tokenizer_path}")
    
    print(f"Tokenizer vocabulary size: {tokenizer.vocab_size}")
    
    # Update model config with actual vocab size
    model_config.vocab_size = tokenizer.vocab_size
    
    # Step 3: Create and train model
    print("\n🏗️ Step 3: Model Creation and Training")
    print("-" * 30)
    
    model = TransformerLM(model_config)
    print(f"Model created with {model.get_num_params():,} parameters")
    
    # Create trainer
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        config=training_config
    )
    
    # Save configurations
    model_config.save(os.path.join(output_dir, "model_config.json"))
    training_config.save(os.path.join(output_dir, "training_config.json"))
    data_config.save(os.path.join(output_dir, "data_config.json"))
    
    print("Starting training...")
    trainer.train_from_config(model_config, data_config)
    
    # Save final model
    final_model_path = os.path.join(output_dir, "final_model")
    trainer.save_model(final_model_path)
    print(f"Final model saved to {final_model_path}")
    
    # Step 4: Text Generation
    print("\n🎯 Step 4: Text Generation")
    print("-" * 30)
    
    generator = TextGenerator(model, tokenizer)
    
    test_prompts = [
        "The quick brown fox",
        "In a world where artificial intelligence",
        "Once upon a time",
        "The future of technology"
    ]
    
    generation_config = GenerationConfig(
        max_length=100,
        temperature=0.8,
        top_p=0.9,
        do_sample=True,
        num_return_sequences=2
    )
    
    generated_results = []
    
    for prompt in test_prompts:
        print(f"\nPrompt: {prompt}")
        print("-" * 40)
        
        generated_texts = generator.generate(prompt, generation_config)
        
        for i, text in enumerate(generated_texts):
            print(f"Generated {i+1}: {text}")
            generated_results.append({
                "prompt": prompt,
                "generated_text": text
            })
    
    # Save generation results
    with open(os.path.join(output_dir, "generation_results.json"), 'w') as f:
        json.dump(generated_results, f, indent=2)
    
    # Step 5: Evaluation
    print("\n📊 Step 5: Model Evaluation")
    print("-" * 30)
    
    # Evaluate generation quality
    generated_texts = [result["generated_text"] for result in generated_results]
    
    # Compute diversity metrics
    from llm_trainer.utils.metrics import compute_diversity_metrics, compute_repetition_metrics
    
    diversity_metrics = compute_diversity_metrics(generated_texts)
    repetition_metrics = compute_repetition_metrics(generated_texts)
    
    print("Generation Quality Metrics:")
    print(f"  Distinct-1: {diversity_metrics['distinct_1']:.3f}")
    print(f"  Distinct-2: {diversity_metrics['distinct_2']:.3f}")
    print(f"  Entropy: {diversity_metrics['entropy']:.3f}")
    print(f"  Repetition Rate: {repetition_metrics['repetition_rate']:.3f}")
    
    # Evaluate perplexity on validation set
    print("\nEvaluating perplexity on validation set...")
    
    try:
        # Create validation dataset
        from llm_trainer.data import LanguageModelingDataset, create_dataloader
        
        val_dataset = LanguageModelingDataset(
            dataset_name=data_config.dataset_name,
            dataset_config=data_config.dataset_config,
            split="validation",
            tokenizer=tokenizer,
            max_length=data_config.max_length,
            preprocessing_config={
                "min_length": data_config.min_length,
                "filter_empty": True
            }
        )
        
        val_dataloader = create_dataloader(
            dataset=val_dataset,
            batch_size=8,
            shuffle=False,
            pad_token_id=tokenizer.pad_token_id
        )
        
        import torch
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        
        perplexity = compute_perplexity(model, val_dataloader, device, max_batches=50)
        print(f"Validation Perplexity: {perplexity:.2f}")
        
    except Exception as e:
        print(f"Could not compute perplexity: {e}")
    
    # Step 6: Summary
    print("\n✅ Step 6: Pipeline Summary")
    print("-" * 30)
    
    summary: Dict[str, Any] = {
        "model_parameters": model.get_num_params(),
        "tokenizer_vocab_size": tokenizer.vocab_size,
        "training_epochs": training_config.num_epochs,
        "generation_metrics": {
            "distinct_1": diversity_metrics['distinct_1'],
            "distinct_2": diversity_metrics['distinct_2'],
            "repetition_rate": repetition_metrics['repetition_rate']
        },
        "output_directory": output_dir
    }
    
    # Save summary
    with open(os.path.join(output_dir, "pipeline_summary.json"), 'w') as f:
        json.dump(summary, f, indent=2)
    
    print("Pipeline completed successfully! 🎉")
    print(f"All outputs saved to: {output_dir}")
    print("\nSummary:")
    print(f"  Model Parameters: {summary['model_parameters']:,}")
    print(f"  Vocabulary Size: {summary['tokenizer_vocab_size']:,}")
    print(f"  Training Epochs: {summary['training_epochs']}")
    print(f"  Generation Quality: Distinct-1={summary['generation_metrics']['distinct_1']:.3f}")  # type: ignore
    
    print("\nNext steps:")
    print("  1. Experiment with different generation parameters")
    print("  2. Train on larger datasets for better quality")
    print("  3. Scale up the model size for improved performance")
    print("  4. Fine-tune on specific domains or tasks")


if __name__ == "__main__":
    main()
