"""Tokenizer factory for easy tokenizer creation."""

from typing import Optional, Dict, cast
import logging

from .base_tokenizer import BaseTokenizer
from .bpe_tokenizer import BPETokenizer
from .wordpiece_tokenizer import WordPieceTokenizer
from .hf_tokenizer import HFTokenizerWrapper
from .sentencepiece_tokenizer import SentencePieceTokenizer
from .char_tokenizer import CharTokenizer
from .byte_bpe_tokenizer import ByteBPETokenizer
from .simple_tokenizer import SimpleTokenizer

logger = logging.getLogger(__name__)

# Tokenizer type registry
TOKENIZER_REGISTRY = {
    "bpe": BPETokenizer,
    "wordpiece": WordPieceTokenizer,
    "sentencepiece": SentencePieceTokenizer,
    "unigram": SentencePieceTokenizer,  # Alias
    "char": CharTokenizer,
    "character": CharTokenizer,  # Alias
    "bytebpe": ByteBPETokenizer,
    "byte_bpe": ByteBPETokenizer,
    "simple": SimpleTokenizer,
    "whitespace": SimpleTokenizer,  # Alias
    "hf": HFTokenizerWrapper,
    "huggingface": HFTokenizerWrapper,  # Alias
}


def create_tokenizer(
    tokenizer_type: str = "bpe",
    pretrained_path: Optional[str] = None,
    **kwargs
) -> BaseTokenizer:
    """
    Create a tokenizer instance easily.
    
    This is the recommended way for beginners to create tokenizers.
    It provides a simple, unified interface for all tokenizer types.
    
    Args:
        tokenizer_type: Type of tokenizer to create. Options:
            - "bpe": Byte Pair Encoding (default, recommended)
            - "wordpiece": WordPiece (BERT-style)
            - "sentencepiece" or "unigram": SentencePiece Unigram
            - "char" or "character": Character-level
            - "bytebpe" or "byte_bpe": Byte-level BPE (GPT-2 style)
            - "simple" or "whitespace": Simple whitespace-based
            - "hf" or "huggingface": HuggingFace pretrained tokenizer
        pretrained_path: Path to pretrained tokenizer (for loading saved tokenizers
                        or HuggingFace model name for HF tokenizers)
        **kwargs: Additional arguments passed to tokenizer constructor
    
    Returns:
        Tokenizer instance
    
    Examples:
        >>> # Create a BPE tokenizer (most common)
        >>> tokenizer = create_tokenizer("bpe")
        >>> 
        >>> # Create a simple tokenizer for beginners
        >>> tokenizer = create_tokenizer("simple")
        >>> 
        >>> # Load a pretrained tokenizer
        >>> tokenizer = create_tokenizer("bpe", pretrained_path="./my_tokenizer")
        >>> 
        >>> # Use HuggingFace tokenizer
        >>> tokenizer = create_tokenizer("hf", pretrained_path="gpt2")
        >>> 
        >>> # Create character-level tokenizer
        >>> tokenizer = create_tokenizer("char")
    
    Raises:
        ValueError: If tokenizer_type is not recognized
        FileNotFoundError: If pretrained_path is provided but doesn't exist
    """
    tokenizer_type = tokenizer_type.lower()

    if tokenizer_type not in TOKENIZER_REGISTRY:
        available = ", ".join(TOKENIZER_REGISTRY.keys())
        raise ValueError(
            f"Unknown tokenizer type: '{tokenizer_type}'. "
            f"Available types: {available}"
        )

    tokenizer_class = TOKENIZER_REGISTRY[tokenizer_type]

    # Handle HuggingFace tokenizer separately
    if tokenizer_type in ("hf", "huggingface"):
        if pretrained_path is None:
            raise ValueError(
                "HuggingFace tokenizer requires 'pretrained_path' argument "
                "(e.g., 'gpt2', 'bert-base-uncased', etc.)"
            )
        logger.info(f"Loading HuggingFace tokenizer: {pretrained_path}")
        return cast(BaseTokenizer, tokenizer_class(pretrained_path, **kwargs))

    # Handle loading pretrained tokenizers
    if pretrained_path:
        import os
        if not os.path.exists(pretrained_path):
            raise FileNotFoundError(
                f"Tokenizer path not found: {pretrained_path}"
            )
        logger.info(f"Loading pretrained tokenizer from: {pretrained_path}")
        return cast(BaseTokenizer, tokenizer_class.from_pretrained(pretrained_path))  # type: ignore

    # Create new tokenizer instance
    logger.info(f"Creating new {tokenizer_type} tokenizer")
    return cast(BaseTokenizer, tokenizer_class(**kwargs))


def get_available_tokenizers() -> Dict[str, str]:
    """
    Get list of available tokenizer types with descriptions.
    
    Returns:
        Dictionary mapping tokenizer types to descriptions
    """
    return {
        "bpe": "Byte Pair Encoding - Most common, good balance of efficiency and quality",
        "wordpiece": "WordPiece - BERT-style, good for masked language modeling",
        "sentencepiece": "SentencePiece Unigram - Good for multilingual text",
        "char": "Character-level - Simplest, one character per token",
        "bytebpe": "Byte-level BPE - GPT-2 style, handles any Unicode",
        "simple": "Simple whitespace - Basic word splitting, perfect for beginners",
        "hf": "HuggingFace pretrained - Use existing tokenizers from HuggingFace",
    }


