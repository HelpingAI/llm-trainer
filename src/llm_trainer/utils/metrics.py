"""Evaluation metrics for language models."""

import torch
import math
from typing import List, Dict, Any, Optional
from collections import Counter
import numpy as np


def compute_perplexity(model, dataloader, device: torch.device,
                      tokenizer=None, max_batches: Optional[int] = None) -> float:
    """Compute perplexity on a dataset."""
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    num_batches = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if max_batches and batch_idx >= max_batches:
                break

            # Move batch to device
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()}

            # Forward pass
            outputs = model(**batch)
            loss = outputs["loss"]

            # Count tokens (excluding padding)
            if "attention_mask" in batch:
                num_tokens = batch["attention_mask"].sum().item()
            else:
                num_tokens = batch["input_ids"].numel()

            total_loss += loss.item() * num_tokens
            total_tokens += num_tokens
            num_batches += 1

    if total_tokens == 0:
        return float('inf')

    avg_loss = total_loss / total_tokens
    perplexity = math.exp(avg_loss) if avg_loss < 10 else float('inf')

    return perplexity


def compute_bleu_score(predictions: List[str],
                      references: List[str],
                      max_n: int = 4,
                      smooth: bool = True) -> Dict[str, float]:
    """Compute BLEU score for generated text."""
    assert len(predictions) == len(references), "Number of predictions and references must match"

    if len(predictions) == 0:
        return {f"bleu_{i}": 0.0 for i in range(1, max_n + 1)}

    # Tokenize sentences
    pred_tokens = [pred.split() for pred in predictions]
    ref_tokens = [ref.split() for ref in references]

    bleu_scores = {}

    for n in range(1, max_n + 1):
        total_precision = 0.0
        total_possible = 0

        for pred, ref in zip(pred_tokens, ref_tokens):
            # Get n-grams
            pred_ngrams = get_ngrams(pred, n)
            ref_ngrams = get_ngrams(ref, n)

            if len(pred_ngrams) == 0:
                continue

            # Count matches
            matches = 0
            for ngram in pred_ngrams:
                if ngram in ref_ngrams:
                    matches += min(pred_ngrams[ngram], ref_ngrams[ngram])

            # Add smoothing for higher-order n-grams
            if smooth and matches == 0 and n > 1:
                matches = 1
                total_possible += len(pred_ngrams) + 1
            else:
                total_possible += len(pred_ngrams)

            total_precision += matches

        # Calculate precision
        if total_possible > 0:
            precision = total_precision / total_possible
        else:
            precision = 0.0

        bleu_scores[f"bleu_{n}"] = precision

    # Calculate geometric mean (BLEU-4)
    if max_n >= 4:
        bleu_4_components = [bleu_scores[f"bleu_{i}"] for i in range(1, 5)]
        if all(score > 0 for score in bleu_4_components):
            bleu_scores["bleu"] = math.exp(sum(math.log(score) for score in bleu_4_components) / 4)
        else:
            bleu_scores["bleu"] = 0.0

    return bleu_scores


def get_ngrams(tokens: List[str], n: int) -> Counter:
    """Get n-grams from a list of tokens."""
    ngrams = Counter()
    for i in range(len(tokens) - n + 1):
        ngram = tuple(tokens[i:i + n])
        ngrams[ngram] += 1
    return ngrams


def compute_rouge_score(predictions: List[str],
                       references: List[str],
                       rouge_types: List[str] = ["rouge1", "rouge2", "rougeL"]) -> Dict[str, float]:
    """Compute ROUGE scores for generated text."""
    try:
        from rouge_score import rouge_scorer  # ty:ignore[unresolved-import]
    except ImportError:
        raise ImportError("Please install rouge-score: pip install rouge-score")

    scorer = rouge_scorer.RougeScorer(rouge_types, use_stemmer=True)

    scores = {rouge_type: {"precision": [], "recall": [], "fmeasure": []}
              for rouge_type in rouge_types}

    for pred, ref in zip(predictions, references):
        rouge_scores = scorer.score(ref, pred)

        for rouge_type in rouge_types:
            scores[rouge_type]["precision"].append(rouge_scores[rouge_type].precision)
            scores[rouge_type]["recall"].append(rouge_scores[rouge_type].recall)
            scores[rouge_type]["fmeasure"].append(rouge_scores[rouge_type].fmeasure)

    # Calculate averages
    avg_scores = {}
    for rouge_type in rouge_types:
        avg_scores[f"{rouge_type}_precision"] = np.mean(scores[rouge_type]["precision"])
        avg_scores[f"{rouge_type}_recall"] = np.mean(scores[rouge_type]["recall"])
        avg_scores[f"{rouge_type}_fmeasure"] = np.mean(scores[rouge_type]["fmeasure"])

    return avg_scores


def compute_diversity_metrics(generated_texts: List[str]) -> Dict[str, float]:
    """Compute diversity metrics for generated text."""
    if not generated_texts:
        return {"distinct_1": 0.0, "distinct_2": 0.0, "entropy": 0.0}

    # Tokenize all texts
    all_tokens = []
    for text in generated_texts:
        tokens = text.split()
        all_tokens.extend(tokens)

    if not all_tokens:
        return {"distinct_1": 0.0, "distinct_2": 0.0, "entropy": 0.0}

    # Distinct-1: ratio of unique unigrams to total unigrams
    unique_unigrams = set(all_tokens)
    distinct_1 = len(unique_unigrams) / len(all_tokens)

    # Distinct-2: ratio of unique bigrams to total bigrams
    bigrams = []
    for text in generated_texts:
        tokens = text.split()
        for i in range(len(tokens) - 1):
            bigrams.append((tokens[i], tokens[i + 1]))

    if bigrams:
        unique_bigrams = set(bigrams)
        distinct_2 = len(unique_bigrams) / len(bigrams)
    else:
        distinct_2 = 0.0

    # Entropy
    token_counts = Counter(all_tokens)
    total_tokens = len(all_tokens)
    entropy = 0.0

    for count in token_counts.values():
        prob = count / total_tokens
        entropy -= prob * math.log2(prob)

    return {
        "distinct_1": distinct_1,
        "distinct_2": distinct_2,
        "entropy": entropy,
        "unique_tokens": len(unique_unigrams),
        "total_tokens": len(all_tokens)
    }


def compute_repetition_metrics(generated_texts: List[str]) -> Dict[str, float]:
    """Compute repetition metrics for generated text."""
    if not generated_texts:
        return {"repetition_rate": 0.0, "avg_repetition_length": 0.0}

    total_repetitions = 0
    total_tokens = 0
    repetition_lengths = []

    for text in generated_texts:
        tokens = text.split()
        total_tokens += len(tokens)

        # Find repetitions
        i = 0
        while i < len(tokens):
            # Look for repetitions starting at position i
            max_rep_length = 0

            for rep_length in range(1, (len(tokens) - i) // 2 + 1):
                pattern = tokens[i:i + rep_length]

                # Check if pattern repeats
                j = i + rep_length
                rep_count = 1

                while j + rep_length <= len(tokens) and tokens[j:j + rep_length] == pattern:
                    rep_count += 1
                    j += rep_length

                if rep_count > 1:
                    max_rep_length = rep_length * rep_count

            if max_rep_length > 0:
                total_repetitions += max_rep_length
                repetition_lengths.append(max_rep_length)
                i += max_rep_length
            else:
                i += 1

    repetition_rate = float(total_repetitions / total_tokens) if total_tokens > 0 else 0.0
    avg_repetition_length = float(np.mean(repetition_lengths)) if repetition_lengths else 0.0

    return {
        "repetition_rate": repetition_rate,
        "avg_repetition_length": avg_repetition_length,
        "num_repetitions": len(repetition_lengths)
    }


def compute_semantic_similarity(predictions: List[str],
                               references: List[str],
                               model_name: str = "all-MiniLM-L6-v2") -> float:
    """Compute semantic similarity using sentence embeddings."""
    try:
        from sentence_transformers import SentenceTransformer  # ty:ignore[unresolved-import]
    except ImportError:
        # Return 0.0 if sentence-transformers is not installed
        return 0.0

    # Load model
    model = SentenceTransformer(model_name)

    # Encode sentences
    pred_embeddings = model.encode(predictions)
    ref_embeddings = model.encode(references)

    # Compute cosine similarity
    similarities = []
    for pred_emb, ref_emb in zip(pred_embeddings, ref_embeddings):
        similarity = np.dot(pred_emb, ref_emb) / (np.linalg.norm(pred_emb) * np.linalg.norm(ref_emb))
        similarities.append(similarity)

    return float(np.mean(similarities)) if similarities else 0.0


def evaluate_generation_quality(predictions: List[str],
                               references: Optional[List[str]] = None,
                               compute_bleu: bool = True,
                               compute_rouge: bool = False,
                               compute_diversity: bool = True,
                               compute_repetition: bool = True,
                               compute_semantic: bool = False) -> Dict[str, Any]:
    """Comprehensive evaluation of generation quality."""
    results = {}

    # Reference-based metrics
    if references is not None and len(references) == len(predictions):
        if compute_bleu:
            bleu_scores = compute_bleu_score(predictions, references)
            results.update(bleu_scores)

        if compute_rouge:
            rouge_scores = compute_rouge_score(predictions, references)
            results.update(rouge_scores)

        if compute_semantic:
            semantic_sim = compute_semantic_similarity(predictions, references)
            results["semantic_similarity"] = semantic_sim

    # Reference-free metrics
    if compute_diversity:
        diversity_metrics = compute_diversity_metrics(predictions)
        results.update(diversity_metrics)

    if compute_repetition:
        repetition_metrics = compute_repetition_metrics(predictions)
        results.update(repetition_metrics)

    return results
