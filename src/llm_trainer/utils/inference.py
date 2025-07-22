"""Inference engine for efficient text generation."""

import torch
import torch.nn.functional as F
from typing import List, Optional, Dict, Any, Union
import time
import logging
from contextlib import contextmanager

from .generation import TextGenerator, GenerationConfig


class InferenceEngine:
    """Optimized inference engine for text generation."""
    
    def __init__(self, 
                 model,
                 tokenizer,
                 device: Optional[torch.device] = None,
                 compile_model: bool = False,
                 use_cache: bool = True):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.use_cache = use_cache
        
        # Move model to device
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Compile model for faster inference (PyTorch 2.0+)
        if compile_model:
            try:
                self.model = torch.compile(self.model)
                logging.info("Model compiled for faster inference")
            except Exception as e:
                logging.warning(f"Failed to compile model: {e}")
        
        # Initialize text generator
        self.generator = TextGenerator(self.model, self.tokenizer, self.device)
        
        # Cache for KV pairs (if supported by model)
        self.kv_cache = {}
        
        # Performance metrics
        self.generation_stats = {
            "total_generations": 0,
            "total_tokens": 0,
            "total_time": 0.0
        }
    
    @contextmanager
    def inference_mode(self):
        """Context manager for optimized inference."""
        with torch.inference_mode():
            yield
    
    def generate(self, 
                prompt: Union[str, List[str]],
                config: Optional[GenerationConfig] = None,
                return_stats: bool = False,
                **kwargs) -> Union[List[str], Dict[str, Any]]:
        """Generate text with performance tracking."""
        start_time = time.time()
        
        # Handle single prompt or batch
        if isinstance(prompt, str):
            prompts = [prompt]
            single_prompt = True
        else:
            prompts = prompt
            single_prompt = False
        
        # Generate text
        with self.inference_mode():
            if single_prompt:
                results = self.generator.generate(prompts[0], config, **kwargs)
            else:
                results = self.generator.generate_batch(prompts, config, **kwargs)
        
        # Update statistics
        end_time = time.time()
        generation_time = end_time - start_time
        
        # Count tokens
        total_tokens = 0
        if single_prompt:
            for text in results:
                total_tokens += len(self.tokenizer.encode(text))
        else:
            for batch_results in results:
                for text in batch_results:
                    total_tokens += len(self.tokenizer.encode(text))
        
        self.generation_stats["total_generations"] += len(prompts)
        self.generation_stats["total_tokens"] += total_tokens
        self.generation_stats["total_time"] += generation_time
        
        if return_stats:
            stats = {
                "generation_time": generation_time,
                "tokens_generated": total_tokens,
                "tokens_per_second": total_tokens / generation_time if generation_time > 0 else 0,
                "prompts_processed": len(prompts)
            }
            
            if single_prompt:
                return {"generated_text": results, "stats": stats}
            else:
                return {"generated_text": results, "stats": stats}
        
        return results
    
    def generate_streaming(self, 
                          prompt: str,
                          config: Optional[GenerationConfig] = None,
                          **kwargs):
        """Generate text with streaming output (yields tokens as they're generated)."""
        if config is None:
            config = GenerationConfig(**kwargs)
        
        # Encode prompt
        input_ids = self.tokenizer.encode(prompt, add_special_tokens=True)
        input_ids = torch.tensor([input_ids], device=self.device)
        
        current_ids = input_ids.clone()
        
        with self.inference_mode():
            for _ in range(config.max_length - input_ids.shape[1]):
                # Forward pass
                outputs = self.model(current_ids)
                logits = outputs["logits"]
                
                # Get next token
                next_token_logits = logits[:, -1, :] / config.temperature
                
                # Apply filtering
                if config.top_k is not None:
                    next_token_logits = self.generator._top_k_filtering(next_token_logits, config.top_k)
                
                if config.top_p is not None:
                    next_token_logits = self.generator._top_p_filtering(next_token_logits, config.top_p)
                
                # Sample next token
                if config.do_sample:
                    probs = F.softmax(next_token_logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                else:
                    next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                
                # Decode and yield token
                token_text = self.tokenizer.decode([next_token.item()], skip_special_tokens=True)
                yield token_text
                
                # Update sequence
                current_ids = torch.cat([current_ids, next_token], dim=1)
                
                # Check for EOS
                if next_token.item() == config.eos_token_id:
                    break
    
    def batch_generate(self, 
                      prompts: List[str],
                      config: Optional[GenerationConfig] = None,
                      batch_size: int = 8,
                      **kwargs) -> List[List[str]]:
        """Generate text for large batches with automatic batching."""
        if config is None:
            config = GenerationConfig(**kwargs)
        
        results = []
        
        # Process in batches
        for i in range(0, len(prompts), batch_size):
            batch_prompts = prompts[i:i + batch_size]
            batch_results = self.generator.generate_batch(batch_prompts, config)
            results.extend(batch_results)
        
        return results
    
    def interactive_generation(self, 
                              initial_prompt: str = "",
                              config: Optional[GenerationConfig] = None,
                              **kwargs):
        """Interactive text generation session."""
        if config is None:
            config = GenerationConfig(**kwargs)
        
        conversation_history = initial_prompt
        
        print("Interactive Generation Session (type 'quit' to exit)")
        print("=" * 50)
        
        if initial_prompt:
            print(f"Initial prompt: {initial_prompt}")
        
        while True:
            try:
                user_input = input("\nEnter prompt (or 'quit' to exit): ")
                
                if user_input.lower() in ['quit', 'exit', 'q']:
                    break
                
                # Add user input to conversation
                if conversation_history:
                    full_prompt = conversation_history + " " + user_input
                else:
                    full_prompt = user_input
                
                # Generate response
                print("\nGenerating...")
                generated_texts = self.generate(full_prompt, config)
                
                # Display results
                for i, text in enumerate(generated_texts):
                    print(f"\nGenerated text {i+1}:")
                    print("-" * 30)
                    print(text)
                
                # Update conversation history
                if generated_texts:
                    conversation_history = generated_texts[0]
                
            except KeyboardInterrupt:
                print("\nSession interrupted by user")
                break
            except Exception as e:
                print(f"Error during generation: {e}")
    
    def benchmark(self, 
                 prompts: List[str],
                 config: Optional[GenerationConfig] = None,
                 num_runs: int = 3,
                 **kwargs) -> Dict[str, float]:
        """Benchmark generation performance."""
        if config is None:
            config = GenerationConfig(**kwargs)
        
        times = []
        token_counts = []
        
        for run in range(num_runs):
            start_time = time.time()
            
            results = self.generate(prompts, config)
            
            end_time = time.time()
            run_time = end_time - start_time
            times.append(run_time)
            
            # Count tokens
            total_tokens = 0
            if isinstance(results[0], list):  # Batch results
                for batch_results in results:
                    for text in batch_results:
                        total_tokens += len(self.tokenizer.encode(text))
            else:  # Single result
                for text in results:
                    total_tokens += len(self.tokenizer.encode(text))
            
            token_counts.append(total_tokens)
        
        # Calculate statistics
        avg_time = sum(times) / len(times)
        avg_tokens = sum(token_counts) / len(token_counts)
        tokens_per_second = avg_tokens / avg_time if avg_time > 0 else 0
        
        return {
            "average_time": avg_time,
            "average_tokens": avg_tokens,
            "tokens_per_second": tokens_per_second,
            "min_time": min(times),
            "max_time": max(times),
            "num_runs": num_runs,
            "num_prompts": len(prompts)
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get generation statistics."""
        stats = self.generation_stats.copy()
        
        if stats["total_time"] > 0:
            stats["average_tokens_per_second"] = stats["total_tokens"] / stats["total_time"]
        else:
            stats["average_tokens_per_second"] = 0
        
        if stats["total_generations"] > 0:
            stats["average_tokens_per_generation"] = stats["total_tokens"] / stats["total_generations"]
            stats["average_time_per_generation"] = stats["total_time"] / stats["total_generations"]
        else:
            stats["average_tokens_per_generation"] = 0
            stats["average_time_per_generation"] = 0
        
        return stats
    
    def reset_stats(self):
        """Reset generation statistics."""
        self.generation_stats = {
            "total_generations": 0,
            "total_tokens": 0,
            "total_time": 0.0
        }
    
    def warmup(self, num_warmup_steps: int = 3):
        """Warm up the model for consistent benchmarking."""
        warmup_prompt = "This is a warmup prompt for the model."
        warmup_config = GenerationConfig(max_length=20, do_sample=False)
        
        for _ in range(num_warmup_steps):
            self.generate(warmup_prompt, warmup_config)
        
        # Reset stats after warmup
        self.reset_stats()
