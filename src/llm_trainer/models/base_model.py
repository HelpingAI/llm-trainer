"""Base model interfaces for supporting both custom and HuggingFace models."""

import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Union, Tuple
from dataclasses import dataclass

from ..config import ModelConfig


@dataclass
class BaseArchitectureConfig:
    """Base configuration class for different model architectures."""
    
    model_type: str
    vocab_size: int
    hidden_size: int
    max_position_embeddings: int
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return self.__dict__.copy()
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'BaseArchitectureConfig':
        """Create from dictionary."""
        return cls(**config_dict)


class BaseLanguageModel(nn.Module, ABC):
    """Abstract base class for all language models in the framework."""
    
    def __init__(self, config: Union[ModelConfig, BaseArchitectureConfig]):
        super().__init__()
        self.config = config
        
    @abstractmethod
    def forward(self, input_ids: torch.Tensor, 
                attention_mask: Optional[torch.Tensor] = None,
                labels: Optional[torch.Tensor] = None,
                **kwargs) -> Dict[str, torch.Tensor]:
        """
        Forward pass of the model.
        
        Args:
            input_ids: Token IDs of shape (batch_size, seq_len)
            attention_mask: Attention mask of shape (batch_size, seq_len)
            labels: Target labels for language modeling loss (batch_size, seq_len)
            **kwargs: Additional model-specific arguments
            
        Returns:
            Dictionary containing at least 'logits' and optionally 'loss'
        """
        pass
    
    @abstractmethod
    def generate(self, input_ids: torch.Tensor, max_length: int = 100,
                 temperature: float = 1.0, top_k: Optional[int] = None,
                 top_p: Optional[float] = None, do_sample: bool = True,
                 **kwargs) -> torch.Tensor:
        """
        Generate text using the model.
        
        Args:
            input_ids: Input token IDs of shape (batch_size, seq_len)
            max_length: Maximum length of generated sequence
            temperature: Sampling temperature
            top_k: Top-k sampling parameter
            top_p: Nucleus sampling parameter
            do_sample: Whether to use sampling or greedy decoding
            **kwargs: Additional generation parameters
            
        Returns:
            Generated token IDs of shape (batch_size, max_length)
        """
        pass
    
    @abstractmethod
    def get_num_params(self, non_embedding: bool = False) -> int:
        """Get number of parameters in the model."""
        pass
    
    def get_input_embeddings(self) -> nn.Module:
        """Get the input embeddings layer."""
        if hasattr(self, 'embeddings'):
            if hasattr(self.embeddings, 'token_embedding'):
                return self.embeddings.token_embedding
            elif hasattr(self.embeddings, 'word_embeddings'):
                return self.embeddings.word_embeddings
        elif hasattr(self, 'embed_tokens'):
            return self.embed_tokens
        else:
            raise NotImplementedError("Input embeddings not found")
    
    def set_input_embeddings(self, value: nn.Module) -> None:
        """Set the input embeddings layer."""
        if hasattr(self, 'embeddings'):
            if hasattr(self.embeddings, 'token_embedding'):
                self.embeddings.token_embedding = value
            elif hasattr(self.embeddings, 'word_embeddings'):
                self.embeddings.word_embeddings = value
        elif hasattr(self, 'embed_tokens'):
            self.embed_tokens = value
        else:
            raise NotImplementedError("Cannot set input embeddings")
    
    def get_output_embeddings(self) -> Optional[nn.Module]:
        """Get the output embeddings layer (language modeling head)."""
        if hasattr(self, 'lm_head'):
            return self.lm_head
        elif hasattr(self, 'output_projection'):
            return self.output_projection
        else:
            return None
    
    def set_output_embeddings(self, value: nn.Module) -> None:
        """Set the output embeddings layer."""
        if hasattr(self, 'lm_head'):
            self.lm_head = value
        elif hasattr(self, 'output_projection'):
            self.output_projection = value
        else:
            raise NotImplementedError("Cannot set output embeddings")
    
    def resize_token_embeddings(self, new_num_tokens: int) -> nn.Module:
        """Resize token embeddings to accommodate new vocabulary size."""
        # Get current embeddings
        old_embeddings = self.get_input_embeddings()
        old_num_tokens, embedding_dim = old_embeddings.weight.shape
        
        # Create new embeddings
        new_embeddings = nn.Embedding(new_num_tokens, embedding_dim)
        
        # Copy old weights
        num_tokens_to_copy = min(old_num_tokens, new_num_tokens)
        new_embeddings.weight.data[:num_tokens_to_copy] = old_embeddings.weight.data[:num_tokens_to_copy]
        
        # Set new embeddings
        self.set_input_embeddings(new_embeddings)
        
        # Also resize output embeddings if they exist
        output_embeddings = self.get_output_embeddings()
        if output_embeddings is not None:
            old_lm_head_weights = output_embeddings.weight.data
            new_lm_head = nn.Linear(embedding_dim, new_num_tokens, bias=output_embeddings.bias is not None)
            
            # Copy old weights
            new_lm_head.weight.data[:num_tokens_to_copy] = old_lm_head_weights[:num_tokens_to_copy]
            if output_embeddings.bias is not None:
                new_lm_head.bias.data[:num_tokens_to_copy] = output_embeddings.bias.data[:num_tokens_to_copy]
            
            self.set_output_embeddings(new_lm_head)
        
        # Update config if it has vocab_size
        if hasattr(self.config, 'vocab_size'):
            self.config.vocab_size = new_num_tokens
        
        return new_embeddings
    
    @classmethod
    @abstractmethod
    def from_config(cls, config: Union[ModelConfig, BaseArchitectureConfig]) -> 'BaseLanguageModel':
        """Create model from configuration."""
        pass
    
    def save_pretrained(self, save_directory: str) -> None:
        """Save model and configuration."""
        import os
        os.makedirs(save_directory, exist_ok=True)
        
        # Save model state dict
        model_path = os.path.join(save_directory, "pytorch_model.bin")
        torch.save(self.state_dict(), model_path)
        
        # Save configuration
        config_path = os.path.join(save_directory, "config.json")
        if hasattr(self.config, 'save'):
            self.config.save(config_path)
        else:
            import json
            with open(config_path, 'w') as f:
                json.dump(self.config.to_dict(), f, indent=2)
    
    @classmethod
    def from_pretrained(cls, model_name_or_path: str, **kwargs) -> 'BaseLanguageModel':
        """
        Load model from pretrained weights.
        This is a base implementation that should be overridden by subclasses.
        """
        import os
        
        if os.path.isdir(model_name_or_path):
            # Load from local directory
            config_path = os.path.join(model_name_or_path, "config.json")
            if hasattr(cls, '_config_class'):
                config = cls._config_class.load(config_path)
            else:
                # Fallback to ModelConfig
                from ..config import ModelConfig
                config = ModelConfig.load(config_path)
            
            # Create model
            model = cls.from_config(config)
            
            # Load state dict
            model_path = os.path.join(model_name_or_path, "pytorch_model.bin")
            state_dict = torch.load(model_path, map_location='cpu')
            model.load_state_dict(state_dict)
            
            return model
        else:
            raise NotImplementedError("Loading from HuggingFace Hub should be implemented by subclasses")


class HuggingFaceModelWrapper(BaseLanguageModel):
    """Wrapper for HuggingFace models to conform to our interface."""
    
    def __init__(self, hf_model, config: Optional[BaseArchitectureConfig] = None):
        # Don't call super().__init__ as we're wrapping an existing model
        nn.Module.__init__(self)
        self.model = hf_model
        self.config = config or self._extract_config_from_hf_model(hf_model)
        
    def _extract_config_from_hf_model(self, hf_model) -> BaseArchitectureConfig:
        """Extract configuration from HuggingFace model."""
        hf_config = hf_model.config
        
        return BaseArchitectureConfig(
            model_type=getattr(hf_config, 'model_type', 'unknown'),
            vocab_size=getattr(hf_config, 'vocab_size', 50000),
            hidden_size=getattr(hf_config, 'hidden_size', getattr(hf_config, 'd_model', 512)),
            max_position_embeddings=getattr(hf_config, 'max_position_embeddings', 
                                          getattr(hf_config, 'max_seq_len', 1024))
        )
    
    def forward(self, input_ids: torch.Tensor, 
                attention_mask: Optional[torch.Tensor] = None,
                labels: Optional[torch.Tensor] = None,
                **kwargs) -> Dict[str, torch.Tensor]:
        """Forward pass through HuggingFace model."""
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            **kwargs
        )
        
        # Convert HF outputs to our format
        result = {"logits": outputs.logits}
        if hasattr(outputs, 'loss') and outputs.loss is not None:
            result["loss"] = outputs.loss
        if hasattr(outputs, 'hidden_states') and outputs.hidden_states is not None:
            result["hidden_states"] = outputs.hidden_states
        if hasattr(outputs, 'attentions') and outputs.attentions is not None:
            result["attention_weights"] = outputs.attentions
            
        return result
    
    def generate(self, input_ids: torch.Tensor, max_length: int = 100,
                 temperature: float = 1.0, top_k: Optional[int] = None,
                 top_p: Optional[float] = None, do_sample: bool = True,
                 **kwargs) -> torch.Tensor:
        """Generate text using HuggingFace model."""
        generation_kwargs = {
            'max_length': max_length,
            'temperature': temperature,
            'do_sample': do_sample,
            **kwargs
        }
        
        if top_k is not None:
            generation_kwargs['top_k'] = top_k
        if top_p is not None:
            generation_kwargs['top_p'] = top_p
            
        return self.model.generate(input_ids, **generation_kwargs)
    
    def get_num_params(self, non_embedding: bool = False) -> int:
        """Get number of parameters in the wrapped model."""
        if hasattr(self.model, 'num_parameters'):
            return self.model.num_parameters(exclude_embeddings=non_embedding)
        else:
            n_params = sum(p.numel() for p in self.model.parameters())
            if non_embedding and hasattr(self.model, 'get_input_embeddings'):
                embed_params = sum(p.numel() for p in self.model.get_input_embeddings().parameters())
                n_params -= embed_params
            return n_params
    
    def get_input_embeddings(self) -> nn.Module:
        """Get input embeddings from HuggingFace model."""
        return self.model.get_input_embeddings()
    
    def set_input_embeddings(self, value: nn.Module) -> None:
        """Set input embeddings in HuggingFace model."""
        self.model.set_input_embeddings(value)
    
    def get_output_embeddings(self) -> Optional[nn.Module]:
        """Get output embeddings from HuggingFace model."""
        if hasattr(self.model, 'get_output_embeddings'):
            return self.model.get_output_embeddings()
        return None
    
    def set_output_embeddings(self, value: nn.Module) -> None:
        """Set output embeddings in HuggingFace model."""
        if hasattr(self.model, 'set_output_embeddings'):
            self.model.set_output_embeddings(value)
    
    def resize_token_embeddings(self, new_num_tokens: int) -> nn.Module:
        """Resize token embeddings in HuggingFace model."""
        return self.model.resize_token_embeddings(new_num_tokens)
    
    @classmethod
    def from_config(cls, config: BaseArchitectureConfig) -> 'HuggingFaceModelWrapper':
        """Create wrapper from configuration (not typically used)."""
        raise NotImplementedError("HuggingFaceModelWrapper should be created with an existing HF model")
    
    def save_pretrained(self, save_directory: str) -> None:
        """Save the wrapped HuggingFace model."""
        self.model.save_pretrained(save_directory)
    
    @classmethod
    def from_pretrained(cls, model_name_or_path: str, **kwargs) -> 'HuggingFaceModelWrapper':
        """Load HuggingFace model and wrap it."""
        try:
            from transformers import AutoModelForCausalLM
            hf_model = AutoModelForCausalLM.from_pretrained(model_name_or_path, **kwargs)
            return cls(hf_model)
        except ImportError:
            raise ImportError("transformers library is required to load HuggingFace models")