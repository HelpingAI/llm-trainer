#!/usr/bin/env python3
"""Example script for text generation with different strategies."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
from llm_trainer.models import TransformerLM
from llm_trainer.tokenizer import BPETokenizer
from llm_trainer.utils.generation import TextGenerator, GenerationConfig
from llm_trainer.utils.inference import InferenceEngine


def load_model_and_tokenizer(model_path: str):
    """Load model and tokenizer."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load model
    model = TransformerLM.from_pretrained(model_path)
    model = model.to(device)
    model.eval()
    
    # Load tokenizer
    tokenizer = BPETokenizer.from_pretrained(model_path)
    
    return model, tokenizer, device


def demonstrate_generation_strategies(generator: TextGenerator, prompt: str):
    """Demonstrate different generation strategies."""
    print(f"Prompt: {prompt}")
    print("=" * 60)
    
    # Greedy decoding
    print("\n1. Greedy Decoding:")
    print("-" * 30)
    greedy_config = GenerationConfig(
        max_length=100,
        do_sample=False,
        temperature=1.0
    )
    greedy_text = generator.generate(prompt, greedy_config)[0]
    print(greedy_text)
    
    # Temperature sampling
    print("\n2. Temperature Sampling (temp=0.8):")
    print("-" * 30)
    temp_config = GenerationConfig(
        max_length=100,
        do_sample=True,
        temperature=0.8
    )
    temp_text = generator.generate(prompt, temp_config)[0]
    print(temp_text)
    
    # Top-k sampling
    print("\n3. Top-k Sampling (k=50):")
    print("-" * 30)
    topk_config = GenerationConfig(
        max_length=100,
        do_sample=True,
        temperature=0.8,
        top_k=50
    )
    topk_text = generator.generate(prompt, topk_config)[0]
    print(topk_text)
    
    # Top-p (nucleus) sampling
    print("\n4. Top-p Sampling (p=0.9):")
    print("-" * 30)
    topp_config = GenerationConfig(
        max_length=100,
        do_sample=True,
        temperature=0.8,
        top_p=0.9
    )
    topp_text = generator.generate(prompt, topp_config)[0]
    print(topp_text)
    
    # Beam search
    print("\n5. Beam Search (beams=5):")
    print("-" * 30)
    beam_config = GenerationConfig(
        max_length=100,
        num_beams=5,
        do_sample=False,
        early_stopping=True
    )
    beam_text = generator.generate(prompt, beam_config)[0]
    print(beam_text)
    
    # Multiple sequences
    print("\n6. Multiple Sequences (n=3, temp=0.8):")
    print("-" * 30)
    multi_config = GenerationConfig(
        max_length=80,
        do_sample=True,
        temperature=0.8,
        num_return_sequences=3
    )
    multi_texts = generator.generate(prompt, multi_config)
    for i, text in enumerate(multi_texts):
        print(f"Sequence {i+1}: {text}")
        print()


def demonstrate_streaming_generation(engine: InferenceEngine, prompt: str):
    """Demonstrate streaming generation."""
    print(f"\nStreaming Generation:")
    print("=" * 60)
    print(f"Prompt: {prompt}")
    print("-" * 30)
    
    config = GenerationConfig(
        max_length=100,
        temperature=0.8,
        do_sample=True
    )
    
    print(prompt, end="", flush=True)
    for token in engine.generate_streaming(prompt, config):
        print(token, end="", flush=True)
    print("\n")


def demonstrate_batch_generation(generator: TextGenerator, prompts: list):
    """Demonstrate batch generation."""
    print(f"\nBatch Generation:")
    print("=" * 60)
    
    config = GenerationConfig(
        max_length=80,
        temperature=0.8,
        do_sample=True
    )
    
    results = generator.generate_batch(prompts, config)
    
    for i, (prompt, generated_texts) in enumerate(zip(prompts, results)):
        print(f"\nPrompt {i+1}: {prompt}")
        print("-" * 40)
        for j, text in enumerate(generated_texts):
            print(f"Generated {j+1}: {text}")


def main():
    """Main function to demonstrate text generation."""
    # Model path (update this to your trained model path)
    model_path = "./output/small_model"
    
    try:
        # Load model and tokenizer
        print("Loading model and tokenizer...")
        model, tokenizer, device = load_model_and_tokenizer(model_path)
        print(f"Model loaded on {device}")
        
        # Create generator and inference engine
        generator = TextGenerator(model, tokenizer, device)
        engine = InferenceEngine(model, tokenizer, device)
        
        # Test prompts
        prompts = [
            "The quick brown fox",
            "In a world where artificial intelligence",
            "Once upon a time in a distant galaxy",
            "The future of technology will be"
        ]
        
        # Demonstrate different generation strategies
        for prompt in prompts[:2]:  # Use first 2 prompts
            demonstrate_generation_strategies(generator, prompt)
            print("\n" + "="*80 + "\n")
        
        # Demonstrate streaming generation
        demonstrate_streaming_generation(engine, prompts[0])
        
        # Demonstrate batch generation
        demonstrate_batch_generation(generator, prompts)
        
        # Performance benchmark
        print(f"\nPerformance Benchmark:")
        print("=" * 60)
        
        benchmark_config = GenerationConfig(
            max_length=50,
            temperature=0.8,
            do_sample=True
        )
        
        results = engine.benchmark(prompts[:2], benchmark_config, num_runs=3)
        print(f"Average time: {results['average_time']:.2f}s")
        print(f"Tokens per second: {results['tokens_per_second']:.1f}")
        
    except FileNotFoundError:
        print(f"Model not found at {model_path}")
        print("Please train a model first using the training example or update the model_path.")
        print("You can train a small model by running: python examples/train_small_model.py")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
