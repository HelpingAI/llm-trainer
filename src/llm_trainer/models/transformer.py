"""Main Transformer Language Model implementation."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Union

from ..config import ModelConfig
from .embeddings import CombinedEmbedding
from .layers import TransformerStack
from .attention import create_causal_mask


class TransformerLM(nn.Module):
    """Transformer Language Model for autoregressive text generation."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        # Embeddings
        self.embeddings = CombinedEmbedding(
            vocab_size=config.vocab_size,
            d_model=config.d_model,
            max_seq_len=config.max_seq_len,
            padding_idx=0,  # Assuming pad token ID is 0
            use_learned_pos=config.use_learned_pos_emb,
            dropout=config.dropout
        )

        # Transformer stack
        self.transformer = TransformerStack(
            n_layers=config.n_layers,
            d_model=config.d_model,
            n_heads=config.n_heads,
            d_ff=config.d_ff,
            dropout=config.dropout,
            attention_dropout=config.attention_dropout,
            activation=config.activation,
            use_bias=config.use_bias,
            layer_norm_eps=config.layer_norm_eps,
            pre_norm=config.pre_norm,
            gradient_checkpointing=config.gradient_checkpointing
        )

        # Language modeling head
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Initialize weights
        self.apply(self._init_weights)

        # Tie embeddings and output weights (optional)
        if hasattr(config, 'tie_weights') and config.tie_weights:
            self.lm_head.weight = self.embeddings.token_embedding.embedding.weight

    def _init_weights(self, module):
        """Initialize model weights."""
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.init_std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.init_std)
            if module.padding_idx is not None:
                nn.init.zeros_(module.weight[module.padding_idx])
        elif isinstance(module, nn.LayerNorm):
            nn.init.zeros_(module.bias)
            nn.init.ones_(module.weight)

    def forward(self, input_ids: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                labels: Optional[torch.Tensor] = None,
                return_attention: bool = False,
                return_hidden_states: bool = False) -> Dict[str, torch.Tensor]:
        """
        Forward pass of the Transformer LM.
        
        Args:
            input_ids: Token IDs of shape (batch_size, seq_len)
            attention_mask: Attention mask of shape (batch_size, seq_len)
            labels: Target labels for language modeling loss (batch_size, seq_len)
            return_attention: Whether to return attention weights
            return_hidden_states: Whether to return hidden states
            
        Returns:
            Dictionary containing:
                - logits: Output logits of shape (batch_size, seq_len, vocab_size)
                - loss: Language modeling loss (if labels provided)
                - hidden_states: Hidden states (if return_hidden_states=True)
                - attention_weights: Attention weights (if return_attention=True)
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        # Create causal mask
        causal_mask = create_causal_mask(seq_len, device)

        # Create padding mask if attention_mask is provided
        key_padding_mask = None
        if attention_mask is not None:
            key_padding_mask = ~attention_mask.bool()

        # Get embeddings
        hidden_states = self.embeddings(input_ids)

        # Pass through transformer
        hidden_states, all_attention_weights = self.transformer(
            hidden_states,
            attention_mask=causal_mask,
            key_padding_mask=key_padding_mask,
            return_attention=return_attention
        )

        # Language modeling head
        logits = self.lm_head(hidden_states)

        # Prepare output
        output = {"logits": logits}

        # Compute loss if labels are provided
        if labels is not None:
            # Shift labels for next token prediction
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            # Flatten for cross entropy
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100  # Ignore padding tokens
            )
            output["loss"] = loss

        # Add optional outputs
        if return_hidden_states:
            output["hidden_states"] = hidden_states

        if return_attention:
            output["attention_weights"] = all_attention_weights

        return output

    def generate(self, input_ids: torch.Tensor, max_length: int = 100,
                 temperature: float = 1.0, top_k: Optional[int] = None,
                 top_p: Optional[float] = None, do_sample: bool = True,
                 pad_token_id: int = 0, eos_token_id: int = 3) -> torch.Tensor:
        """
        Generate text using the model.
        
        Args:
            input_ids: Input token IDs of shape (batch_size, seq_len)
            max_length: Maximum length of generated sequence
            temperature: Sampling temperature
            top_k: Top-k sampling parameter
            top_p: Nucleus sampling parameter
            do_sample: Whether to use sampling or greedy decoding
            pad_token_id: Padding token ID
            eos_token_id: End-of-sequence token ID
            
        Returns:
            generated_ids: Generated token IDs of shape (batch_size, max_length)
        """
        self.eval()
        batch_size = input_ids.shape[0]
        device = input_ids.device

        # Initialize generated sequence with input
        generated = input_ids.clone()

        with torch.no_grad():
            for _ in range(max_length - input_ids.shape[1]):
                # Forward pass
                outputs = self.forward(generated)
                logits = outputs["logits"]

                # Get logits for next token
                next_token_logits = logits[:, -1, :] / temperature

                # Apply top-k filtering
                if top_k is not None:
                    top_k = min(top_k, logits.size(-1))
                    indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                    next_token_logits[indices_to_remove] = float('-inf')

                # Apply top-p (nucleus) filtering
                if top_p is not None:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

                    # Remove tokens with cumulative probability above the threshold
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0

                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    next_token_logits[indices_to_remove] = float('-inf')

                # Sample next token
                if do_sample:
                    probs = F.softmax(next_token_logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                else:
                    next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

                # Append to generated sequence
                generated = torch.cat([generated, next_token], dim=1)

                # Check for EOS token
                if (next_token == eos_token_id).all():
                    break

        return generated

    def get_num_params(self, non_embedding: bool = False) -> int:
        """Get number of parameters in the model."""
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            # Use getattr for type safety with dynamically accessed attributes
            token_emb = getattr(self.embeddings, 'token_embedding')
            embedding = getattr(token_emb, 'embedding')
            n_params -= embedding.weight.numel()
            
            pos_emb = getattr(self.embeddings, 'pos_embedding')
            if hasattr(pos_emb, 'pos_embedding'):
                pos_embedding_layer = getattr(pos_emb, 'pos_embedding')
                if hasattr(pos_embedding_layer, 'weight'):
                    n_params -= pos_embedding_layer.weight.numel()
        return n_params

    @classmethod
    def from_config(cls, config: ModelConfig) -> 'TransformerLM':
        """Create model from configuration."""
        return cls(config)

    def save_pretrained(self, save_directory: str,
                       safe_serialization: bool = True,
                       max_shard_size: Union[str, int] = "5GB",
                       push_to_hub: bool = False,
                       **kwargs) -> None:
        """
        Save model and configuration.
        
        Args:
            save_directory: Directory to save the model
            safe_serialization: Whether to use SafeTensors format (default: True)
            max_shard_size: Maximum size per shard for large models (e.g., "5GB", "500MB")
            push_to_hub: Whether to push to Hugging Face Hub (not implemented)
            **kwargs: Additional arguments
        """
        import os
        os.makedirs(save_directory, exist_ok=True)

        # Try to save with SafeTensors first if requested
        if safe_serialization:
            try:
                from .safetensors_utils import save_model_safetensors, is_safetensors_available

                if is_safetensors_available():
                    # Prepare metadata
                    metadata = {
                        "model_type": "transformer",
                        "framework": "pytorch",
                        "d_model": str(self.config.d_model),
                        "n_layers": str(self.config.n_layers),
                        "n_heads": str(self.config.n_heads),
                        "vocab_size": str(self.config.vocab_size),
                    }

                    # Save with SafeTensors (with automatic sharding for large models)
                    save_model_safetensors(self, save_directory, max_shard_size, metadata)
                else:
                    print("Warning: SafeTensors not available, falling back to PyTorch format")
                    safe_serialization = False
            except Exception as e:
                print(f"Warning: Failed to save with SafeTensors ({e}), falling back to PyTorch format")
                safe_serialization = False

        # Fallback to PyTorch format or if SafeTensors failed
        if not safe_serialization:
            model_path = os.path.join(save_directory, "pytorch_model.bin")
            torch.save(self.state_dict(), model_path)
            print(f"Model saved in PyTorch format to {model_path}")

        # Save configuration
        config_path = os.path.join(save_directory, "config.json")
        self.config.save(config_path)

    @classmethod
    def from_pretrained(cls, load_directory: str, **kwargs) -> 'TransformerLM':
        """
        Load model from directory with support for both SafeTensors and PyTorch formats.
        
        Args:
            load_directory: Directory containing the saved model
            **kwargs: Additional arguments
            
        Returns:
            Loaded TransformerLM model
        """
        import os

        # Load configuration
        config_path = os.path.join(load_directory, "config.json")
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")

        config = ModelConfig.load(config_path)

        # Create model
        model = cls(config)

        # Try to load from SafeTensors first, then fallback to PyTorch
        safetensors_path = os.path.join(load_directory, "model.safetensors")
        safetensors_index_path = os.path.join(load_directory, "model.safetensors.index.json")
        pytorch_path = os.path.join(load_directory, "pytorch_model.bin")

        if os.path.exists(safetensors_path) or os.path.exists(safetensors_index_path):
            try:
                from .safetensors_utils import load_model_safetensors, is_safetensors_available

                if is_safetensors_available():
                    load_model_safetensors(model, load_directory)
                    print("Model loaded from SafeTensors format")
                else:
                    raise ImportError("SafeTensors not available")
            except Exception as e:
                print(f"Warning: Failed to load SafeTensors ({e}), trying PyTorch format")
                if os.path.exists(pytorch_path):
                    state_dict = torch.load(pytorch_path, map_location='cpu')
                    model.load_state_dict(state_dict)
                    print("Model loaded from PyTorch format")
                else:
                    raise FileNotFoundError(f"No loadable model found in {load_directory}")
        elif os.path.exists(pytorch_path):
            # Load PyTorch format
            state_dict = torch.load(pytorch_path, map_location='cpu')
            model.load_state_dict(state_dict)
            print("Model loaded from PyTorch format")
        else:
            raise FileNotFoundError(
                f"No model files found in {load_directory}. "
                f"Expected 'model.safetensors', 'model.safetensors.index.json', or 'pytorch_model.bin'"
            )

        return model

    def push_to_hub(self, repo_id: str, **kwargs) -> None:
        """
        Push model to Hugging Face Hub.
        
        Args:
            repo_id: Repository ID on Hugging Face Hub
            **kwargs: Additional arguments passed to push_to_hub
        """
        try:
            from huggingface_hub import HfApi
            api = HfApi()
            
            # Save model locally first
            import tempfile
            with tempfile.TemporaryDirectory() as temp_dir:
                self.save_pretrained(temp_dir)
                api.upload_folder(
                    folder_path=temp_dir,
                    repo_id=repo_id,
                    repo_type="model",
                    **kwargs
                )
        except ImportError:
            raise ImportError("huggingface_hub is required to push to Hub. Install with: pip install huggingface_hub")
        except Exception as e:
            raise RuntimeError(f"Failed to push to Hub: {e}")
