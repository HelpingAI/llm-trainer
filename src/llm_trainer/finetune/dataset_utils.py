"""
Dataset utilities for fine-tuning with support for various formats and templates.

This module provides comprehensive dataset handling inspired by Axolotl's flexible
dataset processing capabilities.
"""

import json
from typing import Dict, Any, List, Optional, Callable, Union
import warnings

try:
    from datasets import Dataset, load_dataset
    _DATASETS_AVAILABLE = True
except ImportError:
    _DATASETS_AVAILABLE = False
    warnings.warn("Datasets library not available")

try:
    from transformers import PreTrainedTokenizer
    _TRANSFORMERS_AVAILABLE = True
except ImportError:
    _TRANSFORMERS_AVAILABLE = False


class DatasetFormatter:
    """Handles formatting of different dataset types for fine-tuning."""
    
    def __init__(self, tokenizer: "PreTrainedTokenizer", max_length: int = 2048):
        """
        Initialize dataset formatter.
        
        Args:
            tokenizer: Tokenizer to use
            max_length: Maximum sequence length
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def format_alpaca(self, example: Dict[str, Any]) -> str:
        """
        Format Alpaca-style instruction dataset.
        
        Args:
            example: Dataset example with instruction, input, output
            
        Returns:
            Formatted string
        """
        instruction = example.get("instruction", "")
        input_text = example.get("input", "")
        output = example.get("output", "")
        
        if input_text:
            prompt = f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n{output}"
        else:
            prompt = f"### Instruction:\n{instruction}\n\n### Response:\n{output}"
        
        return prompt
    
    def format_chat(self, example: Dict[str, Any]) -> str:
        """
        Format chat-style dataset using tokenizer's chat template.
        
        Args:
            example: Dataset example with messages
            
        Returns:
            Formatted string
        """
        messages = example.get("messages", [])
        
        if hasattr(self.tokenizer, 'apply_chat_template'):
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False
            )
        else:
            # Fallback formatting
            formatted_messages = []
            for message in messages:
                role = message.get("role", "user")
                content = message.get("content", "")
                formatted_messages.append(f"<|{role}|>\n{content}")
            return "\n".join(formatted_messages)
    
    def format_sharegpt(self, example: Dict[str, Any]) -> str:
        """
        Format ShareGPT-style conversations.
        
        Args:
            example: Dataset example with conversations
            
        Returns:
            Formatted string
        """
        conversations = example.get("conversations", [])
        
        # Convert to messages format
        messages = []
        for conv in conversations:
            role = "assistant" if conv.get("from") == "gpt" else "user"
            content = conv.get("value", "")
            messages.append({"role": role, "content": content})
        
        return self.format_chat({"messages": messages})
    
    def format_openai(self, example: Dict[str, Any]) -> str:
        """
        Format OpenAI-style messages.
        
        Args:
            example: Dataset example with messages in OpenAI format
            
        Returns:
            Formatted string
        """
        return self.format_chat(example)
    
    def format_custom(self, example: Dict[str, Any], template: str) -> str:
        """
        Format using custom template.
        
        Args:
            example: Dataset example
            template: Template string with {field} placeholders
            
        Returns:
            Formatted string
        """
        return template.format(**example)


class DatasetProcessor:
    """Processes datasets for different training scenarios."""
    
    def __init__(
        self,
        tokenizer: "PreTrainedTokenizer",
        max_length: int = 2048,
        format_type: str = "auto"
    ):
        """
        Initialize dataset processor.
        
        Args:
            tokenizer: Tokenizer to use
            max_length: Maximum sequence length
            format_type: Dataset format type
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.format_type = format_type
        self.formatter = DatasetFormatter(tokenizer, max_length)
    
    def process_sft_dataset(
        self,
        dataset: Dataset,
        text_field: str = "text",
        formatting_func: Optional[Callable] = None
    ) -> Dataset:
        """
        Process dataset for supervised fine-tuning.
        
        Args:
            dataset: Input dataset
            text_field: Field containing text data
            formatting_func: Custom formatting function
            
        Returns:
            Processed dataset
        """
        if formatting_func is not None:
            # Use custom formatting function
            dataset = dataset.map(
                lambda x: {"text": formatting_func(x)},
                remove_columns=[col for col in dataset.column_names if col != "text"]
            )
        elif text_field in dataset.column_names:
            # Use existing text field
            if text_field != "text":
                dataset = dataset.rename_column(text_field, "text")
        else:
            # Auto-detect format and apply appropriate formatter
            dataset = self._auto_format_dataset(dataset)
        
        # Tokenize
        def tokenize_function(examples):
            return self.tokenizer(
                examples["text"],
                truncation=True,
                padding=False,
                max_length=self.max_length,
                return_overflowing_tokens=False,
            )
        
        dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=dataset.column_names
        )
        
        return dataset
    
    def process_dpo_dataset(
        self,
        dataset: Dataset,
        prompt_field: str = "prompt",
        chosen_field: str = "chosen",
        rejected_field: str = "rejected"
    ) -> Dataset:
        """
        Process dataset for DPO training.
        
        Args:
            dataset: Input dataset
            prompt_field: Field containing prompts
            chosen_field: Field containing chosen responses
            rejected_field: Field containing rejected responses
            
        Returns:
            Processed dataset
        """
        def tokenize_dpo(examples):
            prompts = examples[prompt_field]
            chosen = examples[chosen_field]
            rejected = examples[rejected_field]
            
            # Combine prompt with responses
            chosen_texts = [p + c for p, c in zip(prompts, chosen)]
            rejected_texts = [p + r for p, r in zip(prompts, rejected)]
            
            # Tokenize
            chosen_tokens = self.tokenizer(
                chosen_texts,
                truncation=True,
                max_length=self.max_length,
                padding=False
            )
            rejected_tokens = self.tokenizer(
                rejected_texts,
                truncation=True,
                max_length=self.max_length,
                padding=False
            )
            
            return {
                "chosen_input_ids": chosen_tokens["input_ids"],
                "chosen_attention_mask": chosen_tokens["attention_mask"],
                "rejected_input_ids": rejected_tokens["input_ids"],
                "rejected_attention_mask": rejected_tokens["attention_mask"],
            }
        
        return dataset.map(
            tokenize_dpo,
            batched=True,
            remove_columns=dataset.column_names
        )
    
    def _auto_format_dataset(self, dataset: Dataset) -> Dataset:
        """Auto-detect dataset format and apply appropriate formatting."""
        columns = dataset.column_names
        
        # Check for different formats
        if "messages" in columns:
            # Chat format
            format_func = self.formatter.format_chat
        elif "conversations" in columns:
            # ShareGPT format
            format_func = self.formatter.format_sharegpt
        elif all(col in columns for col in ["instruction", "output"]):
            # Alpaca format
            format_func = self.formatter.format_alpaca
        else:
            # Default to first text-like column
            text_columns = [col for col in columns if any(
                keyword in col.lower() for keyword in ["text", "content", "response", "output"]
            )]
            if text_columns:
                return dataset.rename_column(text_columns[0], "text")
            else:
                raise ValueError(f"Cannot auto-detect format for columns: {columns}")
        
        return dataset.map(
            lambda x: {"text": format_func(x)},
            remove_columns=columns
        )


def load_and_process_dataset(
    dataset_name: str,
    tokenizer: "PreTrainedTokenizer",
    dataset_config: Optional[str] = None,
    split: str = "train",
    max_length: int = 2048,
    format_type: str = "auto",
    text_field: str = "text",
    formatting_func: Optional[Callable] = None,
    **kwargs
) -> Dataset:
    """
    Load and process a dataset for fine-tuning.
    
    Args:
        dataset_name: Name of dataset to load
        tokenizer: Tokenizer to use
        dataset_config: Dataset configuration
        split: Dataset split to load
        max_length: Maximum sequence length
        format_type: Dataset format type
        text_field: Field containing text data
        formatting_func: Custom formatting function
        **kwargs: Additional arguments for load_dataset
        
    Returns:
        Processed dataset ready for training
    """
    if not _DATASETS_AVAILABLE:
        raise ImportError("datasets library required for dataset loading")
    
    # Load dataset
    dataset = load_dataset(dataset_name, dataset_config, split=split, **kwargs)
    
    # Process dataset
    processor = DatasetProcessor(tokenizer, max_length, format_type)
    return processor.process_sft_dataset(dataset, text_field, formatting_func)


def create_chat_template(template_name: str = "chatml") -> str:
    """
    Create a chat template string.
    
    Args:
        template_name: Name of template to create
        
    Returns:
        Chat template string
    """
    templates = {
        "chatml": (
            "{% for message in messages %}"
            "<|im_start|>{{ message['role'] }}\n{{ message['content'] }}<|im_end|>\n"
            "{% endfor %}"
            "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
        ),
        "llama": (
            "{% for message in messages %}"
            "{% if message['role'] == 'user' %}"
            "[INST] {{ message['content'] }} [/INST]"
            "{% elif message['role'] == 'assistant' %}"
            " {{ message['content'] }} </s>"
            "{% endif %}"
            "{% endfor %}"
        ),
        "alpaca": (
            "{% for message in messages %}"
            "{% if message['role'] == 'user' %}"
            "### Instruction:\n{{ message['content'] }}\n\n### Response:\n"
            "{% elif message['role'] == 'assistant' %}"
            "{{ message['content'] }}"
            "{% endif %}"
            "{% endfor %}"
        )
    }
    
    return templates.get(template_name, templates["chatml"])


def apply_chat_template_to_dataset(
    dataset: Dataset,
    tokenizer: "PreTrainedTokenizer",
    template_name: str = "chatml"
) -> Dataset:
    """
    Apply chat template to a dataset.
    
    Args:
        dataset: Dataset with messages
        tokenizer: Tokenizer
        template_name: Chat template to use
        
    Returns:
        Dataset with formatted text
    """
    # Set chat template if not present
    if not hasattr(tokenizer, 'chat_template') or tokenizer.chat_template is None:
        tokenizer.chat_template = create_chat_template(template_name)
    
    def format_messages(example):
        messages = example.get("messages", [])
        formatted_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False
        )
        return {"text": formatted_text}
    
    return dataset.map(format_messages, remove_columns=dataset.column_names)


def filter_dataset_by_length(
    dataset: Dataset,
    tokenizer: "PreTrainedTokenizer",
    min_length: int = 10,
    max_length: int = 2048,
    text_field: str = "text"
) -> Dataset:
    """
    Filter dataset by text length.
    
    Args:
        dataset: Input dataset
        tokenizer: Tokenizer for length calculation
        min_length: Minimum token length
        max_length: Maximum token length
        text_field: Field containing text
        
    Returns:
        Filtered dataset
    """
    def filter_by_length(example):
        text = example[text_field]
        tokens = tokenizer.encode(text, add_special_tokens=False)
        length = len(tokens)
        return min_length <= length <= max_length
    
    return dataset.filter(filter_by_length)


def pack_dataset(
    dataset: Dataset,
    tokenizer: "PreTrainedTokenizer",
    max_length: int = 2048,
    text_field: str = "text"
) -> Dataset:
    """
    Pack multiple short examples into single sequences.
    
    Args:
        dataset: Input dataset
        tokenizer: Tokenizer
        max_length: Maximum sequence length
        text_field: Field containing text
        
    Returns:
        Packed dataset
    """
    # Tokenize all examples
    def tokenize(examples):
        return tokenizer(
            examples[text_field],
            add_special_tokens=False,
            return_attention_mask=False
        )
    
    tokenized = dataset.map(tokenize, batched=True, remove_columns=dataset.column_names)
    
    # Pack sequences
    packed_input_ids = []
    current_sequence = []
    
    for example in tokenized:
        tokens = example["input_ids"]
        
        # Add EOS token
        tokens = tokens + [tokenizer.eos_token_id]
        
        if len(current_sequence) + len(tokens) <= max_length:
            current_sequence.extend(tokens)
        else:
            # Save current sequence and start new one
            if current_sequence:
                # Pad to max_length
                current_sequence.extend([tokenizer.pad_token_id] * (max_length - len(current_sequence)))
                packed_input_ids.append(current_sequence)
            current_sequence = tokens
    
    # Add final sequence
    if current_sequence:
        current_sequence.extend([tokenizer.pad_token_id] * (max_length - len(current_sequence)))
        packed_input_ids.append(current_sequence)
    
    # Create new dataset
    return Dataset.from_dict({"input_ids": packed_input_ids})
