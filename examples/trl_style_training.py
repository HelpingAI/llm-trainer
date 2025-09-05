#!/usr/bin/env python3
"""Example demonstrating TRL-style API for training with SFT, DPO, and other methods."""

import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from llm_trainer.config import SFTConfig, DPOConfig
from llm_trainer.training import SFTTrainer, DPOTrainer
from llm_trainer.models import TransformerLM
from llm_trainer.config import ModelConfig
from llm_trainer.tokenizer import BPETokenizer


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


def demonstrate_sft_training():
    """Demonstrate SFT training with TRL-style API."""
    print("🚀 Demonstrating SFT Training with TRL-style API")
    print("=" * 60)
    
    # Create model and tokenizer
    model, tokenizer = create_sample_model_and_tokenizer()
    
    # Create SFT configuration
    sft_config = SFTConfig(
        per_device_train_batch_size=4,
        gradient_accumulation_steps=2,
        learning_rate=2e-5,
        num_train_epochs=3,
        logging_steps=10,
        save_steps=100,
        evaluation_strategy="steps",
        eval_steps=50,
        max_seq_length=512,
        packing=True,
        dataset_text_field="text"
    )
    
    # Create SFT trainer (TRL-style)
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        config=sft_config
    )
    
    print(f"✅ SFT Trainer created with config:")
    print(f"   - Batch size: {sft_config.per_device_train_batch_size}")
    print(f"   - Learning rate: {sft_config.learning_rate}")
    print(f"   - Epochs: {sft_config.num_train_epochs}")
    
    # Show trainable parameters
    trainer.print_trainable_parameters()
    
    print("\n💡 To actually train, you would call:")
    print("   trainer.train()")
    print("   or")
    print("   trainer.sft_train()")
    
    return trainer


def demonstrate_dpo_training():
    """Demonstrate DPO training with TRL-style API."""
    print("\n🚀 Demonstrating DPO Training with TRL-style API")
    print("=" * 60)
    
    # Create model and tokenizer
    model, tokenizer = create_sample_model_and_tokenizer()
    
    # Create DPO configuration
    dpo_config = DPOConfig(
        per_device_train_batch_size=4,
        gradient_accumulation_steps=2,
        learning_rate=5e-6,
        num_train_epochs=2,
        beta=0.1,
        loss_type="sigmoid",
        logging_steps=10,
        save_steps=100,
        evaluation_strategy="steps",
        eval_steps=50,
        max_length=512,
        max_prompt_length=256
    )
    
    # Create DPO trainer (TRL-style)
    trainer = DPOTrainer(
        model=model,
        tokenizer=tokenizer,
        config=dpo_config
    )
    
    print(f"✅ DPO Trainer created with config:")
    print(f"   - Batch size: {dpo_config.per_device_train_batch_size}")
    print(f"   - Learning rate: {dpo_config.learning_rate}")
    print(f"   - Beta: {dpo_config.beta}")
    print(f"   - Loss type: {dpo_config.loss_type}")
    
    # Show trainable parameters
    trainer.print_trainable_parameters()
    
    print("\n💡 To actually train, you would call:")
    print("   trainer.train()")
    print("   or")
    print("   trainer.dpo_train()")
    
    return trainer


def demonstrate_peft_integration():
    """Demonstrate PEFT integration."""
    print("\n🚀 Demonstrating PEFT Integration")
    print("=" * 60)
    
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
        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
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


def demonstrate_model_saving_and_loading():
    """Demonstrate model saving and loading with HuggingFace-style API."""
    print("\n🚀 Demonstrating Model Saving and Loading")
    print("=" * 60)
    
    # Create model and tokenizer
    model, tokenizer = create_sample_model_and_tokenizer()
    
    # Create trainer
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer
    )
    
    # Create output directory
    output_dir = "./output/trl_style_example"
    os.makedirs(output_dir, exist_ok=True)
    
    print("💡 To save the model, you would call:")
    print(f"   trainer.save_model('{output_dir}')")
    print(f"   or")
    print(f"   trainer.save_pretrained('{output_dir}')")
    
    # Show trainable parameters
    trainer.print_trainable_parameters()
    
    print("\n💡 To load a pretrained model, you would call:")
    print("   trainer = SFTTrainer.from_pretrained('model_name')")
    
    return trainer


def main():
    """Main function to demonstrate all TRL-style training methods."""
    print("🎯 LLM Trainer TRL-style API Examples")
    print("=" * 60)
    
    # Demonstrate SFT training
    sft_trainer = demonstrate_sft_training()
    
    # Demonstrate DPO training
    dpo_trainer = demonstrate_dpo_training()
    
    # Demonstrate PEFT integration
    demonstrate_peft_integration()
    
    # Demonstrate model saving and loading
    save_trainer = demonstrate_model_saving_and_loading()
    
    print("\n🎉 All TRL-style API examples completed!")
    print("\n🔧 Key Features Demonstrated:")
    print("   • SFTConfig and DPOConfig classes following TRL conventions")
    print("   • SFTTrainer, DPOTrainer classes with .train() method")
    print("   • PEFT integration for LoRA and other adapters")
    print("   • HuggingFace-style .save_model(), .save_pretrained(), .from_pretrained()")
    print("   • .print_trainable_parameters() method")
    print("   • Support for all HuggingFace model architectures")
    
    print("\n📚 Next Steps:")
    print("   1. Install required dependencies: pip install peft transformers")
    print("   2. Replace sample model with actual HuggingFace model")
    print("   3. Add real datasets for training")
    print("   4. Configure training parameters for your use case")


if __name__ == "__main__":
    main()