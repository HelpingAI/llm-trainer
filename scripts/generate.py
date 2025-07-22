#!/usr/bin/env python3
"""Text generation script for trained language models."""

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
from llm_trainer.utils.generation import TextGenerator, GenerationConfig
from llm_trainer.utils.inference import InferenceEngine


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


def load_model_and_tokenizer(model_path: str, device: torch.device):
    """Load model and tokenizer from path."""
    logging.info(f"Loading model from {model_path}")
    
    # Load model
    model = TransformerLM.from_pretrained(model_path)
    model = model.to(device)
    model.eval()
    
    # Load tokenizer
    tokenizer = BPETokenizer.from_pretrained(model_path)
    
    logging.info(f"Model loaded with {model.get_num_params():,} parameters")
    
    return model, tokenizer


def generate_from_prompts(generator: TextGenerator, prompts: list, 
                         generation_config: GenerationConfig, output_file: str = None):
    """Generate text from a list of prompts."""
    results = []
    
    for i, prompt in enumerate(prompts):
        print(f"\n{'='*60}")
        print(f"Prompt {i+1}: {prompt}")
        print('='*60)
        
        # Generate text
        generated_texts = generator.generate(prompt, generation_config)
        
        for j, text in enumerate(generated_texts):
            print(f"\nGenerated {j+1}:")
            print("-" * 40)
            print(text)
            
            results.append({
                "prompt": prompt,
                "generated_text": text,
                "prompt_index": i,
                "generation_index": j
            })
    
    # Save results if output file specified
    if output_file:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {output_file}")
    
    return results


def interactive_generation(engine: InferenceEngine, generation_config: GenerationConfig):
    """Interactive text generation session."""
    print("\n" + "="*60)
    print("INTERACTIVE TEXT GENERATION")
    print("="*60)
    print("Enter prompts to generate text. Type 'quit' to exit.")
    print("Commands:")
    print("  'quit' or 'exit' - Exit the session")
    print("  'config' - Show current generation config")
    print("  'stats' - Show generation statistics")
    print("  'streaming' - Toggle streaming mode")
    print("="*60)
    
    streaming_mode = False
    
    while True:
        try:
            prompt = input("\nEnter prompt: ").strip()
            
            if prompt.lower() in ['quit', 'exit', 'q']:
                break
            elif prompt.lower() == 'config':
                print("\nCurrent generation config:")
                print(json.dumps(generation_config.__dict__, indent=2))
                continue
            elif prompt.lower() == 'stats':
                stats = engine.get_stats()
                print("\nGeneration statistics:")
                print(json.dumps(stats, indent=2))
                continue
            elif prompt.lower() == 'streaming':
                streaming_mode = not streaming_mode
                print(f"Streaming mode: {'ON' if streaming_mode else 'OFF'}")
                continue
            elif not prompt:
                print("Please enter a prompt or 'quit' to exit.")
                continue
            
            print("\nGenerating...")
            
            if streaming_mode:
                print("Generated text (streaming):")
                print("-" * 40)
                print(prompt, end="", flush=True)
                
                for token in engine.generate_streaming(prompt, generation_config):
                    print(token, end="", flush=True)
                print("\n")
            else:
                results = engine.generate(prompt, generation_config, return_stats=True)
                generated_texts = results["generated_text"]
                stats = results["stats"]
                
                for i, text in enumerate(generated_texts):
                    print(f"\nGenerated text {i+1}:")
                    print("-" * 40)
                    print(text)
                
                print(f"\nGeneration stats:")
                print(f"Time: {stats['generation_time']:.2f}s")
                print(f"Tokens/sec: {stats['tokens_per_second']:.1f}")
        
        except KeyboardInterrupt:
            print("\nSession interrupted by user")
            break
        except Exception as e:
            print(f"Error during generation: {e}")
    
    print("\nSession ended.")


def benchmark_generation(engine: InferenceEngine, prompts: list, 
                        generation_config: GenerationConfig, num_runs: int = 3):
    """Benchmark generation performance."""
    print(f"\nBenchmarking generation with {len(prompts)} prompts, {num_runs} runs...")
    
    # Warmup
    engine.warmup()
    
    # Run benchmark
    results = engine.benchmark(prompts, generation_config, num_runs)
    
    print("\nBenchmark Results:")
    print("="*40)
    print(f"Average time: {results['average_time']:.2f}s")
    print(f"Average tokens: {results['average_tokens']:.0f}")
    print(f"Tokens per second: {results['tokens_per_second']:.1f}")
    print(f"Min time: {results['min_time']:.2f}s")
    print(f"Max time: {results['max_time']:.2f}s")
    print(f"Number of runs: {results['num_runs']}")
    print(f"Number of prompts: {results['num_prompts']}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Generate text with trained language model")
    parser.add_argument("--model_path", type=str, required=True,
                       help="Path to trained model directory")
    parser.add_argument("--prompts", type=str, nargs="+",
                       help="Text prompts for generation")
    parser.add_argument("--prompts_file", type=str,
                       help="File containing prompts (one per line)")
    parser.add_argument("--output_file", type=str,
                       help="Output file to save generated text")
    parser.add_argument("--interactive", action="store_true",
                       help="Start interactive generation session")
    parser.add_argument("--streaming", action="store_true",
                       help="Use streaming generation")
    parser.add_argument("--benchmark", action="store_true",
                       help="Run generation benchmark")
    parser.add_argument("--num_runs", type=int, default=3,
                       help="Number of benchmark runs")
    
    # Generation parameters
    parser.add_argument("--max_length", type=int, default=100,
                       help="Maximum generation length")
    parser.add_argument("--temperature", type=float, default=1.0,
                       help="Sampling temperature")
    parser.add_argument("--top_k", type=int, default=None,
                       help="Top-k sampling parameter")
    parser.add_argument("--top_p", type=float, default=None,
                       help="Top-p (nucleus) sampling parameter")
    parser.add_argument("--num_return_sequences", type=int, default=1,
                       help="Number of sequences to generate per prompt")
    parser.add_argument("--num_beams", type=int, default=1,
                       help="Number of beams for beam search")
    parser.add_argument("--repetition_penalty", type=float, default=1.0,
                       help="Repetition penalty")
    parser.add_argument("--do_sample", action="store_true", default=True,
                       help="Use sampling instead of greedy decoding")
    
    # System parameters
    parser.add_argument("--device", type=str, default="auto",
                       help="Device to use (cuda, cpu, or auto)")
    parser.add_argument("--compile_model", action="store_true",
                       help="Compile model for faster inference")
    parser.add_argument("--log_level", type=str, default="info",
                       choices=["debug", "info", "warning", "error"],
                       help="Logging level")
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    
    # Setup device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    
    logging.info(f"Using device: {device}")
    
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(args.model_path, device)
    
    # Create generation config
    generation_config = GenerationConfig(
        max_length=args.max_length,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        num_return_sequences=args.num_return_sequences,
        num_beams=args.num_beams,
        repetition_penalty=args.repetition_penalty,
        do_sample=args.do_sample,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id
    )
    
    # Create inference engine
    engine = InferenceEngine(model, tokenizer, device, args.compile_model)
    
    # Interactive mode
    if args.interactive:
        interactive_generation(engine, generation_config)
        return
    
    # Load prompts
    prompts = []
    if args.prompts:
        prompts.extend(args.prompts)
    
    if args.prompts_file:
        with open(args.prompts_file, 'r') as f:
            file_prompts = [line.strip() for line in f if line.strip()]
            prompts.extend(file_prompts)
    
    if not prompts:
        prompts = ["The quick brown fox"]
        print("No prompts provided, using default prompt.")
    
    # Benchmark mode
    if args.benchmark:
        benchmark_generation(engine, prompts, generation_config, args.num_runs)
        return
    
    # Generate text
    generator = TextGenerator(model, tokenizer, device)
    generate_from_prompts(generator, prompts, generation_config, args.output_file)


if __name__ == "__main__":
    main()
