#!/usr/bin/env python3
"""Evaluation script for trained language models."""

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
from llm_trainer.data import LanguageModelingDataset, create_dataloader
from llm_trainer.utils.metrics import (
    compute_perplexity, compute_bleu_score, compute_diversity_metrics,
    compute_repetition_metrics, evaluate_generation_quality
)
from llm_trainer.utils.generation import TextGenerator, GenerationConfig
from llm_trainer.config import DataConfig


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


def evaluate_perplexity(model, tokenizer, dataset_config: dict, device: torch.device, 
                       max_batches: int = None) -> float:
    """Evaluate perplexity on a dataset."""
    logging.info("Evaluating perplexity...")
    
    # Create dataset
    dataset = LanguageModelingDataset(
        dataset_name=dataset_config["dataset_name"],
        dataset_config=dataset_config.get("dataset_config"),
        split=dataset_config.get("split", "test"),
        tokenizer=tokenizer,
        text_column=dataset_config.get("text_column", "text"),
        max_length=dataset_config.get("max_length", 1024),
        preprocessing_config=dataset_config.get("preprocessing_config", {})
    )
    
    # Create dataloader
    dataloader = create_dataloader(
        dataset=dataset,
        batch_size=dataset_config.get("batch_size", 8),
        shuffle=False,
        num_workers=0,
        pad_token_id=tokenizer.pad_token_id
    )
    
    # Compute perplexity
    perplexity = compute_perplexity(model, dataloader, device, tokenizer, max_batches)
    
    logging.info(f"Perplexity: {perplexity:.2f}")
    return perplexity


def evaluate_generation(model, tokenizer, prompts: list, generation_config: dict, 
                       device: torch.device) -> dict:
    """Evaluate text generation quality."""
    logging.info("Evaluating text generation...")
    
    # Create text generator
    generator = TextGenerator(model, tokenizer, device)
    
    # Create generation config
    gen_config = GenerationConfig(**generation_config)
    
    # Generate text for each prompt
    generated_texts = []
    for prompt in prompts:
        generated = generator.generate(prompt, gen_config)
        generated_texts.extend(generated)
    
    # Compute diversity and repetition metrics
    diversity_metrics = compute_diversity_metrics(generated_texts)
    repetition_metrics = compute_repetition_metrics(generated_texts)
    
    results = {
        "generated_texts": generated_texts,
        "diversity_metrics": diversity_metrics,
        "repetition_metrics": repetition_metrics,
        "num_generated": len(generated_texts)
    }
    
    logging.info(f"Generated {len(generated_texts)} texts")
    logging.info(f"Distinct-1: {diversity_metrics['distinct_1']:.3f}")
    logging.info(f"Distinct-2: {diversity_metrics['distinct_2']:.3f}")
    logging.info(f"Repetition rate: {repetition_metrics['repetition_rate']:.3f}")
    
    return results


def evaluate_with_references(predictions: list, references: list) -> dict:
    """Evaluate generation quality with reference texts."""
    logging.info("Evaluating with reference texts...")
    
    # Compute BLEU scores
    bleu_scores = compute_bleu_score(predictions, references)
    
    # Compute comprehensive metrics
    quality_metrics = evaluate_generation_quality(
        predictions=predictions,
        references=references,
        compute_bleu=True,
        compute_diversity=True,
        compute_repetition=True
    )
    
    logging.info(f"BLEU-1: {bleu_scores['bleu_1']:.3f}")
    logging.info(f"BLEU-2: {bleu_scores['bleu_2']:.3f}")
    logging.info(f"BLEU-4: {bleu_scores.get('bleu', 0.0):.3f}")
    
    return quality_metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained language model")
    parser.add_argument("--model_path", type=str, required=True,
                       help="Path to trained model directory")
    parser.add_argument("--eval_config", type=str, required=True,
                       help="Path to evaluation configuration file")
    parser.add_argument("--output_path", type=str, default="evaluation_results.json",
                       help="Path to save evaluation results")
    parser.add_argument("--device", type=str, default="auto",
                       help="Device to use (cuda, cpu, or auto)")
    parser.add_argument("--log_level", type=str, default="info",
                       choices=["debug", "info", "warning", "error"],
                       help="Logging level")
    parser.add_argument("--max_batches", type=int, default=None,
                       help="Maximum number of batches for perplexity evaluation")
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    
    # Setup device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    
    logging.info(f"Using device: {device}")
    
    # Load evaluation configuration
    with open(args.eval_config, 'r') as f:
        eval_config = json.load(f)
    
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(args.model_path, device)
    
    results = {}
    
    # Evaluate perplexity
    if "perplexity" in eval_config:
        perplexity = evaluate_perplexity(
            model, tokenizer, eval_config["perplexity"], device, args.max_batches
        )
        results["perplexity"] = perplexity
    
    # Evaluate generation
    if "generation" in eval_config:
        gen_config = eval_config["generation"]
        prompts = gen_config.get("prompts", ["The quick brown fox"])
        generation_config = gen_config.get("config", {})
        
        generation_results = evaluate_generation(
            model, tokenizer, prompts, generation_config, device
        )
        results["generation"] = generation_results
    
    # Evaluate with references
    if "references" in eval_config:
        ref_config = eval_config["references"]
        predictions = ref_config["predictions"]
        references = ref_config["references"]
        
        reference_results = evaluate_with_references(predictions, references)
        results["reference_evaluation"] = reference_results
    
    # Save results
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    with open(args.output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    logging.info(f"Evaluation results saved to {args.output_path}")
    
    # Print summary
    print("\n" + "="*50)
    print("EVALUATION SUMMARY")
    print("="*50)
    
    if "perplexity" in results:
        print(f"Perplexity: {results['perplexity']:.2f}")
    
    if "generation" in results:
        gen_results = results["generation"]
        print(f"Generated texts: {gen_results['num_generated']}")
        print(f"Distinct-1: {gen_results['diversity_metrics']['distinct_1']:.3f}")
        print(f"Distinct-2: {gen_results['diversity_metrics']['distinct_2']:.3f}")
        print(f"Repetition rate: {gen_results['repetition_metrics']['repetition_rate']:.3f}")
    
    if "reference_evaluation" in results:
        ref_results = results["reference_evaluation"]
        if "bleu_1" in ref_results:
            print(f"BLEU-1: {ref_results['bleu_1']:.3f}")
        if "bleu_2" in ref_results:
            print(f"BLEU-2: {ref_results['bleu_2']:.3f}")
        if "bleu" in ref_results:
            print(f"BLEU-4: {ref_results['bleu']:.3f}")


if __name__ == "__main__":
    main()
