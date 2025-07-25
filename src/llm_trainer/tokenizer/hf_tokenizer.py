from transformers import AutoTokenizer

class HFTokenizerWrapper:
    """
    Wrapper for HuggingFace AutoTokenizer to integrate with llm_trainer pipelines.
    """
    def __init__(self, pretrained_model_name_or_path, local_files_only=False, **kwargs):
        self.tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path,
            local_files_only=local_files_only,
            **kwargs
        )

    def encode(self, text, **kwargs):
        return self.tokenizer.encode(text, **kwargs)

    def decode(self, token_ids, **kwargs):
        return self.tokenizer.decode(token_ids, **kwargs)

    def __getattr__(self, item):
        # Forward any other attribute access to the underlying tokenizer
        return getattr(self.tokenizer, item) 