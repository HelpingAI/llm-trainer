"""Tokenizer factory utilities."""

from typing import Union, Optional, Dict, Any
from ..config import TokenizerConfig
from ..tokenizer import BPETokenizer, CustomTokenizerWrapper


def create_tokenizer(config: Union[TokenizerConfig, Dict[str, Any], str]) -> Union[BPETokenizer, CustomTokenizerWrapper]:
    """Create a tokenizer based on configuration.
    
    Args:
        config: TokenizerConfig object, dict with tokenizer config, or tokenizer type string
        
    Returns:
        Configured tokenizer instance
        
    Examples:
        # Using TokenizerConfig
        tokenizer_config = TokenizerConfig(type="custom", name_or_path="mistralai/Mistral-7B-Instruct-v0.2")
        tokenizer = create_tokenizer(tokenizer_config)
        
        # Using dict  
        tokenizer = create_tokenizer({
            "type": "custom",
            "name_or_path": "mistralai/Mistral-7B-Instruct-v0.2"
        })
        
        # Using string shorthand
        tokenizer = create_tokenizer("bpe")  # Creates BPE tokenizer
    """
    
    # Handle different input types
    if isinstance(config, str):
        # Simple string specification
        if config.lower() == "bpe":
            return BPETokenizer()
        else:
            # Assume it's a model name/path for custom tokenizer
            return CustomTokenizerWrapper(tokenizer_name_or_path=config)
    
    elif isinstance(config, dict):
        # Dictionary configuration
        config = TokenizerConfig.from_dict(config)
    
    elif not isinstance(config, TokenizerConfig):
        raise ValueError(f"config must be TokenizerConfig, dict, or str, got {type(config)}")
    
    # Create tokenizer based on type
    if config.type == "bpe":
        tokenizer = BPETokenizer()
        
        # Apply BPE-specific configurations
        if hasattr(tokenizer, 'min_frequency'):
            tokenizer.min_frequency = config.min_frequency
            
        return tokenizer
        
    elif config.type == "custom":
        if config.name_or_path is None:
            raise ValueError("name_or_path is required for custom tokenizers")
            
        # Create custom tokenizer wrapper
        tokenizer = CustomTokenizerWrapper(tokenizer_name_or_path=config.name_or_path)
        
        return tokenizer
        
    else:
        raise ValueError(f"Unsupported tokenizer type: {config.type}")


def load_tokenizer_from_config_file(config_file: str, 
                                   tokenizer_section: str = "tokenizer") -> Union[BPETokenizer, CustomTokenizerWrapper]:
    """Load tokenizer from a configuration file.
    
    Args:
        config_file: Path to YAML or JSON configuration file
        tokenizer_section: Section name in config file containing tokenizer config
        
    Returns:
        Configured tokenizer instance
    """
    
    if config_file.endswith(('.yaml', '.yml')):
        import yaml
        with open(config_file, 'r') as f:
            full_config = yaml.safe_load(f)
    elif config_file.endswith('.json'):
        import json
        with open(config_file, 'r') as f:
            full_config = json.load(f)
    else:
        raise ValueError("Config file must be .yaml, .yml, or .json")
    
    # Extract tokenizer configuration
    tokenizer_config = full_config.get(tokenizer_section, {})
    
    # Set defaults if not specified
    if not tokenizer_config:
        tokenizer_config = {"type": "bpe"}
    
    return create_tokenizer(tokenizer_config)


def get_tokenizer_for_model(model_name: str) -> CustomTokenizerWrapper:
    """Get the appropriate tokenizer for a specific model.
    
    Args:
        model_name: Name of the model (e.g., "mistral", "llama", "gpt")
        
    Returns:
        CustomTokenizerWrapper configured for the model
    """
    
    model_tokenizer_map = {
        "mistral": "mistralai/Mistral-7B-Instruct-v0.2",
        "mistral-7b": "mistralai/Mistral-7B-Instruct-v0.2", 
        "llama": "meta-llama/Llama-2-7b-hf",
        "llama-2": "meta-llama/Llama-2-7b-hf",
        "llama2": "meta-llama/Llama-2-7b-hf",
        "gpt2": "gpt2",
        "gpt-2": "gpt2",
    }
    
    model_name_lower = model_name.lower()
    
    if model_name_lower in model_tokenizer_map:
        tokenizer_path = model_tokenizer_map[model_name_lower]
        return CustomTokenizerWrapper(tokenizer_name_or_path=tokenizer_path)
    else:
        # Assume it's a direct model name/path
        return CustomTokenizerWrapper(tokenizer_name_or_path=model_name)