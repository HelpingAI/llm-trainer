#!/usr/bin/env python3
"""Example demonstrating SafeTensors model saving and loading with sharding support."""

import os
import sys
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from llm_trainer.config import ModelConfig
from llm_trainer.models import TransformerLM, is_safetensors_available
from llm_trainer.models import (
    save_model_safetensors, load_model_safetensors,
    convert_pytorch_to_safetensors, get_safetensors_metadata,
    list_safetensors_tensors
)


def main():
    """Demonstrate SafeTensors saving and loading with different model sizes."""
    
    print("🔒 SafeTensors Model Saving and Loading Example")
    print("=" * 60)
    
    # Check SafeTensors availability
    if not is_safetensors_available():
        print("❌ SafeTensors not available. Install with: pip install safetensors")
        print("This example will demonstrate the functionality once SafeTensors is installed.")
        return
    
    print("✅ SafeTensors is available!")
    
    # Create output directory if it doesn't exist
    os.makedirs("./output", exist_ok=True)
    
    # Example 1: Small model (single file)
    print("\n📦 Example 1: Small Model (Single File)")
    print("-" * 40)
    
    small_config = ModelConfig(
        vocab_size=32000,
        d_model=256,
        n_heads=4,
        n_layers=4,
        d_ff=1024,
        max_seq_len=512
    )
    
    small_model = TransformerLM(small_config)
    small_output_dir = "./output/small_model_safetensors"
    
    print(f"Model parameters: {small_model.get_num_params():,}")
    
    # Save with SafeTensors
    print("Saving small model with SafeTensors...")
    small_model.save_pretrained(small_output_dir, safe_serialization=True)
    
    # Load and verify
    print("Loading small model...")
    loaded_small = TransformerLM.from_pretrained(small_output_dir)
    print("✅ Small model loaded successfully!")
    
    # Check files created
    print("Files created:")
    if os.path.exists(small_output_dir):
        for file in os.listdir(small_output_dir):
            file_path = os.path.join(small_output_dir, file)
            size = os.path.getsize(file_path) / (1024 * 1024)
            print(f"  - {file} ({size:.2f} MB)")
    
    # Example 2: Medium model (potentially sharded)
    print("\n📦 Example 2: Medium Model (Potentially Sharded)")
    print("-" * 40)
    
    medium_config = ModelConfig(
        vocab_size=50000,
        d_model=768,
        n_heads=12,
        n_layers=12,
        d_ff=3072,
        max_seq_len=2048
    )
    
    medium_model = TransformerLM(medium_config)
    medium_output_dir = "./output/medium_model_safetensors"
    
    print(f"Model parameters: {medium_model.get_num_params():,}")
    
    # Save with smaller shard size to demonstrate sharding
    print("Saving medium model with SafeTensors (max_shard_size='100MB')...")
    medium_model.save_pretrained(
        medium_output_dir, 
        safe_serialization=True,
        max_shard_size="100MB"  # Small shard size to force sharding
    )
    
    # Load and verify
    print("Loading medium model...")
    loaded_medium = TransformerLM.from_pretrained(medium_output_dir)
    print("✅ Medium model loaded successfully!")
    
    # Check files created
    print("Files created:")
    if os.path.exists(medium_output_dir):
        for file in sorted(os.listdir(medium_output_dir)):
            file_path = os.path.join(medium_output_dir, file)
            size = os.path.getsize(file_path) / (1024 * 1024)
            print(f"  - {file} ({size:.2f} MB)")
    
    # Example 3: Convert existing PyTorch model to SafeTensors
    print("\n🔄 Example 3: Convert PyTorch to SafeTensors")
    print("-" * 40)
    
    # First save as PyTorch format
    pytorch_dir = "./output/pytorch_model"
    small_model.save_pretrained(pytorch_dir, safe_serialization=False)
    
    # Try to convert to SafeTensors if function exists
    try:
        pytorch_path = os.path.join(pytorch_dir, "pytorch_model.bin")
        safetensors_path = os.path.join(pytorch_dir, "model.safetensors")
        
        if os.path.exists(pytorch_path):
            print(f"Converting {pytorch_path} to SafeTensors...")
            convert_pytorch_to_safetensors(pytorch_path, safetensors_path)
            print("✅ Conversion completed!")
            
            # Show file sizes
            pytorch_size = os.path.getsize(pytorch_path) / (1024 * 1024)
            safetensors_size = os.path.getsize(safetensors_path) / (1024 * 1024)
            
            print(f"PyTorch file size: {pytorch_size:.2f} MB")
            print(f"SafeTensors file size: {safetensors_size:.2f} MB")
            print(f"Size difference: {((safetensors_size - pytorch_size) / pytorch_size * 100):+.1f}%")
        else:
            print(f"❌ PyTorch model file not found at {pytorch_path}")
            safetensors_path = None
    except ImportError:
        print("❌ convert_pytorch_to_safetensors function not available")
        safetensors_path = None
    
    # Example 4: Inspect SafeTensors metadata
    print("\n🔍 Example 4: Inspect SafeTensors Metadata")
    print("-" * 40)
    
    try:
        if safetensors_path and os.path.exists(safetensors_path):
            print("SafeTensors metadata:")
            metadata = get_safetensors_metadata(safetensors_path)
            for key, value in metadata.items():
                print(f"  {key}: {value}")
            
            print("\nTensor names:")
            tensor_names = list_safetensors_tensors(safetensors_path)
            for i, name in enumerate(tensor_names[:10]):  # Show first 10
                print(f"  {i+1}. {name}")
            if len(tensor_names) > 10:
                print(f"  ... and {len(tensor_names) - 10} more tensors")
        else:
            print("❌ SafeTensors file not available for inspection")
    except ImportError:
        print("❌ SafeTensors metadata functions not available")
    
    # Performance comparison
    print("\n⚡ Example 5: Performance Comparison")
    print("-" * 40)
    
    # Test loading speed
    print("Testing loading speed...")
    
    try:
        # SafeTensors loading
        start_time = time.time()
        _ = TransformerLM.from_pretrained(small_output_dir)
        safetensors_time = time.time() - start_time
        
        # PyTorch loading
        start_time = time.time()
        _ = TransformerLM.from_pretrained(pytorch_dir)
        pytorch_time = time.time() - start_time
        
        print(f"SafeTensors loading time: {safetensors_time:.3f}s")
        print(f"PyTorch loading time: {pytorch_time:.3f}s")
        
        if pytorch_time > 0:
            print(f"SafeTensors speedup: {pytorch_time / safetensors_time:.1f}x")
        else:
            print("PyTorch loading time too small to measure speedup accurately")
    except Exception as e:
        print(f"❌ Error during performance comparison: {e}")
    
    print("\n🎉 All examples completed successfully!")
    print("\n💡 Benefits of SafeTensors:")
    print("  - Faster loading times")
    print("  - Safer format (prevents arbitrary code execution)")
    print("  - Better memory efficiency")
    print("  - Automatic sharding for large models")
    print("  - Cross-platform compatibility")
    print("  - Hugging Face ecosystem compatibility")


if __name__ == "__main__":
    main()
