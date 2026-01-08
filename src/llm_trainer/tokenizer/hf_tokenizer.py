import logging
from typing import Any

from transformers import AutoTokenizer

class HFTokenizerWrapper:
    """
    Wrapper for HuggingFace AutoTokenizer to integrate with llm_trainer pipelines.
    """
    def __init__(self, pretrained_model_name_or_path, local_files_only=False, **kwargs):
        self._logger = logging.getLogger(__name__)
        self.tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path,
            local_files_only=local_files_only,
            **kwargs
        )
        self._ensure_pad_token()

    def encode(self, text, **kwargs):
        return self.tokenizer.encode(text, **kwargs)

    def decode(self, token_ids, **kwargs):
        return self.tokenizer.decode(token_ids, **kwargs)

    def __getattr__(self, item):
        # Forward any other attribute access to the underlying tokenizer
        return getattr(self.tokenizer, item)

    def save_pretrained(self, *args: Any, **kwargs: Any):
        return self.tokenizer.save_pretrained(*args, **kwargs)

    def _ensure_pad_token(self) -> None:
        """Ensure a pad token is available for batching operations."""
        if self.tokenizer.pad_token is not None:
            return

        fallback_token = getattr(self.tokenizer, "eos_token", None) or getattr(self.tokenizer, "bos_token", None)
        if fallback_token:
            self.tokenizer.pad_token = fallback_token
            self._logger.warning(
                "Tokenizer '%s' has no pad_token; reusing %s as pad_token.",
                self.tokenizer.name_or_path,
                "eos_token" if fallback_token == getattr(self.tokenizer, "eos_token", None) else "bos_token",
            )
            return

        new_pad = "<|pad|>"
        self.tokenizer.add_special_tokens({"pad_token": new_pad})
        self.tokenizer.pad_token = new_pad
        self._logger.warning(
            "Tokenizer '%s' had no pad_token; added '%s'. Remember to resize model embeddings if needed.",
            self.tokenizer.name_or_path,
            new_pad,
        )
