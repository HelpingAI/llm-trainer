"""Text generation utilities with various decoding strategies."""

import torch
import torch.nn.functional as F
from typing import List, Optional
from dataclasses import dataclass


@dataclass
class GenerationConfig:
    """Configuration for text generation."""

    max_length: int = 100
    min_length: int = 1
    temperature: float = 1.0
    top_k: Optional[int] = None
    top_p: Optional[float] = None
    repetition_penalty: float = 1.0
    length_penalty: float = 1.0
    num_beams: int = 1
    num_return_sequences: int = 1
    do_sample: bool = True
    early_stopping: bool = True
    pad_token_id: int = 0
    bos_token_id: int = 2
    eos_token_id: int = 3

    def __post_init__(self):
        """Validate configuration."""
        assert self.max_length > 0, "max_length must be positive"
        assert self.min_length >= 0, "min_length must be non-negative"
        assert self.min_length <= self.max_length, "min_length must be <= max_length"
        assert self.temperature > 0, "temperature must be positive"
        assert self.repetition_penalty > 0, "repetition_penalty must be positive"
        assert self.num_beams > 0, "num_beams must be positive"
        assert self.num_return_sequences > 0, "num_return_sequences must be positive"

        if self.num_beams > 1:
            assert not self.do_sample, "Cannot use sampling with beam search"


class TextGenerator:
    """Text generator with multiple decoding strategies."""

    def __init__(self, model, tokenizer, device: Optional[torch.device] = None):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Move model to device
        self.model = self.model.to(self.device)
        self.model.eval()

    def generate(self,
                prompt: str,
                config: Optional[GenerationConfig] = None,
                **kwargs) -> List[str]:
        """Generate text from a prompt."""
        if config is None:
            config = GenerationConfig(**kwargs)

        # Encode prompt
        input_ids = self.tokenizer.encode(prompt, add_special_tokens=True)
        input_ids = torch.tensor([input_ids], device=self.device)

        # Generate based on strategy
        if config.num_beams > 1:
            generated_ids = self._beam_search(input_ids, config)
        else:
            generated_ids = self._sampling_generate(input_ids, config)

        # Decode generated sequences
        generated_texts = []
        for ids in generated_ids:
            text = self.tokenizer.decode(ids.tolist(), skip_special_tokens=True)
            generated_texts.append(text)

        return generated_texts

    def _sampling_generate(self,
                          input_ids: torch.Tensor,
                          config: GenerationConfig) -> List[torch.Tensor]:
        """Generate using sampling strategies (greedy, top-k, top-p, nucleus)."""
        generated_sequences = []

        for _ in range(config.num_return_sequences):
            current_ids = input_ids.clone()

            with torch.no_grad():
                for _ in range(config.max_length - input_ids.shape[1]):
                    # Forward pass
                    outputs = self.model(current_ids)
                    logits = outputs["logits"]

                    # Get logits for next token
                    next_token_logits = logits[:, -1, :] / config.temperature

                    # Apply repetition penalty
                    if config.repetition_penalty != 1.0:
                        next_token_logits = self._apply_repetition_penalty(
                            next_token_logits, current_ids, config.repetition_penalty
                        )

                    # Apply top-k filtering
                    if config.top_k is not None:
                        next_token_logits = self._top_k_filtering(next_token_logits, config.top_k)

                    # Apply top-p (nucleus) filtering
                    if config.top_p is not None:
                        next_token_logits = self._top_p_filtering(next_token_logits, config.top_p)

                    # Sample next token
                    if config.do_sample:
                        probs = F.softmax(next_token_logits, dim=-1)
                        next_token = torch.multinomial(probs, num_samples=1)
                    else:
                        next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

                    # Append to sequence
                    current_ids = torch.cat([current_ids, next_token], dim=1)

                    # Check for EOS token
                    if next_token.item() == config.eos_token_id:
                        break

                    # Check minimum length
                    if current_ids.shape[1] - input_ids.shape[1] >= config.min_length:
                        if next_token.item() == config.eos_token_id:
                            break

            generated_sequences.append(current_ids[0])

        return generated_sequences

    def _beam_search(self,
                    input_ids: torch.Tensor,
                    config: GenerationConfig) -> List[torch.Tensor]:
        """Generate using beam search."""
        batch_size = input_ids.shape[0]
        assert batch_size == 1, "Beam search currently supports batch_size=1"

        # Initialize beams
        beams = [(input_ids[0], 0.0)]  # (sequence, score)
        completed_sequences = []

        with torch.no_grad():
            for step in range(config.max_length - input_ids.shape[1]):
                candidates = []

                for sequence, score in beams:
                    # Skip if sequence is already completed
                    if sequence[-1].item() == config.eos_token_id:
                        completed_sequences.append((sequence, score))
                        continue

                    # Forward pass
                    sequence_input = sequence.unsqueeze(0)
                    outputs = self.model(sequence_input)
                    logits = outputs["logits"]

                    # Get logits for next token
                    next_token_logits = logits[0, -1, :] / config.temperature

                    # Apply repetition penalty
                    if config.repetition_penalty != 1.0:
                        next_token_logits = self._apply_repetition_penalty(
                            next_token_logits.unsqueeze(0),
                            sequence.unsqueeze(0),
                            config.repetition_penalty
                        )[0]

                    # Get top-k candidates
                    log_probs = F.log_softmax(next_token_logits, dim=-1)
                    top_k_probs, top_k_indices = torch.topk(log_probs, config.num_beams)

                    # Add candidates
                    for i in range(config.num_beams):
                        token_id = top_k_indices[i]
                        token_score = top_k_probs[i].item()

                        new_sequence = torch.cat([sequence, token_id.unsqueeze(0)])
                        new_score = score + token_score

                        # Apply length penalty
                        if config.length_penalty != 1.0:
                            length_penalty = ((5 + len(new_sequence)) / 6) ** config.length_penalty
                            new_score = new_score / length_penalty

                        candidates.append((new_sequence, new_score))

                # Select top beams
                candidates.sort(key=lambda x: x[1], reverse=True)
                beams = candidates[:config.num_beams]

                # Check if all beams are completed
                if len(completed_sequences) >= config.num_beams and config.early_stopping:
                    break

        # Add remaining beams to completed sequences
        completed_sequences.extend(beams)

        # Sort by score and return top sequences
        completed_sequences.sort(key=lambda x: x[1], reverse=True)

        result_sequences = []
        for i in range(min(config.num_return_sequences, len(completed_sequences))):
            result_sequences.append(completed_sequences[i][0])

        return result_sequences

    def _apply_repetition_penalty(self,
                                 logits: torch.Tensor,
                                 input_ids: torch.Tensor,
                                 penalty: float) -> torch.Tensor:
        """Apply repetition penalty to logits."""
        batch_size, vocab_size = logits.shape

        for i in range(batch_size):
            for token_id in set(input_ids[i].tolist()):
                if logits[i, token_id] < 0:
                    logits[i, token_id] *= penalty
                else:
                    logits[i, token_id] /= penalty

        return logits

    def _top_k_filtering(self, logits: torch.Tensor, top_k: int) -> torch.Tensor:
        """Apply top-k filtering to logits."""
        top_k = min(top_k, logits.size(-1))
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = float('-inf')
        return logits

    def _top_p_filtering(self, logits: torch.Tensor, top_p: float) -> torch.Tensor:
        """Apply top-p (nucleus) filtering to logits."""
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        logits[indices_to_remove] = float('-inf')
        return logits

    def generate_batch(self,
                      prompts: List[str],
                      config: Optional[GenerationConfig] = None,
                      **kwargs) -> List[List[str]]:
        """Generate text for a batch of prompts."""
        if config is None:
            config = GenerationConfig(**kwargs)

        results = []
        for prompt in prompts:
            generated_texts = self.generate(prompt, config)
            results.append(generated_texts)

        return results
