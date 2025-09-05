#!/usr/bin/env python3
"""Example demonstrating enhanced training with TRL-style API and memory optimizations."""

import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from llm_trainer.config import ModelConfig, TrainingConfig
from llm_trainer.models import TransformerLM
from llm_trainer.tokenizer import BPETokenizer
from llm_trainer.training import Trainer


def create_sample_model_and_tokenizer():
    """Create a small sample model and tokenizer for demonstration."""
    # Model configuration
    model_config = ModelConfig(
        vocab_size=32000,
        d_model=256,
        n_heads=4,
        n_layers=4,
        d_ff=1024,
        max_seq_len=512,
        dropout=0.1
    )
    
    # Create model
    model = TransformerLM(model_config)
    
    # Create tokenizer
    tokenizer = BPETokenizer()
    
    return model, tokenizer


def demonstrate_trl_style_api():
    """Demonstrate TRL-style API."""
    print("🚀 Demonstrating TRL-Style API")
    print("=" * 50)
    
    # Create model and tokenizer
    model, tokenizer = create_sample_model_and_tokenizer()
    
    # Create training configuration with TRL-style parameters
    training_config = TrainingConfig(
        per_device_train_batch_size=4,
        gradient_accumulation_steps=2,
        learning_rate=2e-5,
        num_train_epochs=3,
        logging_steps=10,
        save_steps=100,
        evaluation_strategy="steps",
        eval_steps=50,
        optim="adamw"  # TRL-style parameter
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        config=training_config
    )
    
    print(f"✅ Trainer created with TRL-style config:")
    print(f"   - Batch size: {training_config.per_device_train_batch_size}")
    print(f"   - Learning rate: {training_config.learning_rate}")
    print(f"   - Epochs: {training_config.num_train_epochs}")
    
    # Show trainable parameters
    trainer.print_trainable_parameters()
    
    print("\n💡 To actually train, you would call:")
    print("   trainer.train()")
    print("   or")
    print("   trainer.sft_train()")
    
    return trainer


def demonstrate_peft_integration():
    """Demonstrate PEFT integration."""
    print("\n🚀 Demonstrating PEFT Integration")
    print("=" * 50)
    
    try:
        from peft import LoraConfig, TaskType
        
        # Create model and tokenizer
        model, tokenizer = create_sample_model_and_tokenizer()
        
        # Create PEFT configuration
        peft_config = LoraConfig(
            r=8,
            lora_alpha=16,
            lora_dropout=0.05,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
            target_modules=["q_proj", "v_proj"]  # Example target modules
        )
        
        # Create trainer with PEFT config
        trainer = Trainer(
            model=model,
            tokenizer=tokenizer,
            config=TrainingConfig(),  # Default config
            peft_config=peft_config
        )
        
        print("✅ PEFT configuration applied:")
        print(f"   - LoRA r: {peft_config.r}")
        print(f"   - LoRA alpha: {peft_config.lora_alpha}")
        print(f"   - LoRA dropout: {peft_config.lora_dropout}")
        
        # Show trainable parameters
        trainer.print_trainable_parameters()
        
        # Prepare for k-bit training
        trainer.prepare_model_for_kbit_training()
        
        print("\n💡 Model prepared for k-bit training")
        
    except ImportError:
        print("❌ PEFT not installed. Install with: pip install peft")
        print("Skipping PEFT demonstration.")
    except Exception as e:
        print(f"❌ Error in PEFT demonstration: {e}")


def demonstrate_huggingface_style_api():
    """Demonstrate HuggingFace-style API."""
    print("\n🚀 Demonstrating HuggingFace-Style API")
    print("=" * 50)
    
    # Create model and tokenizer
    model, tokenizer = create_sample_model_and_tokenizer()
    
    # Create trainer
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        config=TrainingConfig()
    )
    
    # Create output directory
    output_dir = "./output/enhanced_example"
    os.makedirs(output_dir, exist_ok=True)
    
    print("💡 HuggingFace-style APIs available:")
    print(f"   trainer.save_model('{output_dir}')")
    print(f"   trainer.save_pretrained('{output_dir}')")
    
    # Show trainable parameters
    trainer.print_trainable_parameters()
    
    print("\n💡 To load a pretrained model, you would call:")
    print("   trainer = Trainer.from_pretrained('model_path', 'tokenizer_path', config)")
    
    return trainer


def main():
    """Main function to demonstrate all enhanced features."""
    print("🎯 LLM Trainer Enhanced Features Examples")
    print("=" * 60)
    
    # Demonstrate TRL-style API
    trl_trainer = demonstrate_trl_style_api()
    
    # Demonstrate PEFT integration
    demonstrate_peft_integration()
    
    # Demonstrate HuggingFace-style API
    hf_trainer = demonstrate_huggingface_style_api()
    
    print("\n🎉 All enhanced features demonstrated!")
    print("\n🔧 Key Features:")
    print("   • TRL-style .train(), .sft_train(), .dpo_train() methods")
    print("   • HuggingFace-style .save_model(), .save_pretrained(), .from_pretrained()")
    print("   • PEFT integration for parameter-efficient training")
    print("   • .print_trainable_parameters() method")
    print("   • Support for all HuggingFace model architectures")
    print("   • Memory-efficient training techniques")
    
    print("\n📚 Next Steps:")
    print("   1. Install required dependencies: pip install peft transformers")
    print("   2. Replace sample model with actual HuggingFace model")
    print("   3. Add real datasets for training")
    print("   4. Configure training parameters for your use case")


if __name__ == "__main__":
    main()