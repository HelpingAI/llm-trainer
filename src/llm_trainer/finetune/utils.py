"""
Utility functions for fine-tuning operations.
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any, List, Union
import warnings

try:
    from transformers import PreTrainedModel, PreTrainedTokenizer
    _TRANSFORMERS_AVAILABLE = True
except ImportError:
    _TRANSFORMERS_AVAILABLE = False

try:
    from peft import prepare_model_for_kbit_training as peft_prepare_model_for_kbit_training
    _PEFT_AVAILABLE = True
except ImportError:
    _PEFT_AVAILABLE = False


def print_trainable_parameters(model: nn.Module) -> None:
    """
    Print the number of trainable parameters in the model.
    
    Args:
        model: PyTorch model
    """
    trainable_params = 0
    all_param = 0
    
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    
    percentage = 100 * trainable_params / all_param if all_param > 0 else 0
    
    print(f"Trainable params: {trainable_params:,} || "
          f"All params: {all_param:,} || "
          f"Trainable%: {percentage:.4f}%")


def get_model_max_length(model: nn.Module) -> int:
    """
    Get the maximum sequence length for a model.
    
    Args:
        model: Model to check
        
    Returns:
        Maximum sequence length
    """
    if hasattr(model, 'config'):
        config = model.config
        
        # Try different attribute names
        for attr in ['max_position_embeddings', 'max_seq_len', 'n_positions', 'seq_length']:
            if hasattr(config, attr):
                return getattr(config, attr)
    
    # Default fallback
    return 2048


def prepare_model_for_kbit_training(
    model: nn.Module,
    use_gradient_checkpointing: bool = True,
    gradient_checkpointing_kwargs: Optional[Dict[str, Any]] = None
) -> nn.Module:
    """
    Prepare a model for k-bit training.
    
    Args:
        model: Model to prepare
        use_gradient_checkpointing: Whether to enable gradient checkpointing
        gradient_checkpointing_kwargs: Additional gradient checkpointing arguments
        
    Returns:
        Prepared model
    """
    if _PEFT_AVAILABLE:
        model = peft_prepare_model_for_kbit_training(
            model, 
            use_gradient_checkpointing=use_gradient_checkpointing,
            gradient_checkpointing_kwargs=gradient_checkpointing_kwargs or {}
        )
    else:
        warnings.warn("PEFT not available. Using basic preparation.")
        if use_gradient_checkpointing and hasattr(model, 'gradient_checkpointing_enable'):
            model.gradient_checkpointing_enable()
    
    return model


def apply_chat_template(
    tokenizer: "PreTrainedTokenizer",
    messages: List[Dict[str, str]],
    add_generation_prompt: bool = False,
    **kwargs
) -> str:
    """
    Apply chat template to messages.
    
    Args:
        tokenizer: Tokenizer with chat template
        messages: List of message dictionaries with 'role' and 'content'
        add_generation_prompt: Whether to add generation prompt
        **kwargs: Additional arguments
        
    Returns:
        Formatted chat string
    """
    if not _TRANSFORMERS_AVAILABLE:
        raise ImportError("Transformers is required for chat template functionality")
    
    if hasattr(tokenizer, 'apply_chat_template'):
        return tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=add_generation_prompt,
            tokenize=False,
            **kwargs
        )
    else:
        # Fallback for tokenizers without chat template
        formatted_messages = []
        for message in messages:
            role = message.get('role', 'user')
            content = message.get('content', '')
            formatted_messages.append(f"{role}: {content}")
        
        result = "\n".join(formatted_messages)
        if add_generation_prompt:
            result += "\nassistant:"
        
        return result


def format_instruction_dataset(
    example: Dict[str, Any],
    instruction_key: str = "instruction",
    input_key: str = "input", 
    output_key: str = "output",
    template: Optional[str] = None
) -> str:
    """
    Format an instruction dataset example.
    
    Args:
        example: Dataset example
        instruction_key: Key for instruction
        input_key: Key for input
        output_key: Key for output
        template: Custom template string
        
    Returns:
        Formatted string
    """
    instruction = example.get(instruction_key, "")
    input_text = example.get(input_key, "")
    output = example.get(output_key, "")
    
    if template is not None:
        return template.format(
            instruction=instruction,
            input=input_text,
            output=output
        )
    
    # Default Alpaca-style template
    if input_text:
        prompt = f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n{output}"
    else:
        prompt = f"### Instruction:\n{instruction}\n\n### Response:\n{output}"
    
    return prompt


def format_chat_dataset(
    example: Dict[str, Any],
    messages_key: str = "messages",
    tokenizer: Optional["PreTrainedTokenizer"] = None
) -> str:
    """
    Format a chat dataset example.
    
    Args:
        example: Dataset example
        messages_key: Key for messages
        tokenizer: Tokenizer for chat template
        
    Returns:
        Formatted string
    """
    messages = example.get(messages_key, [])
    
    if tokenizer is not None and hasattr(tokenizer, 'apply_chat_template'):
        return apply_chat_template(tokenizer, messages)
    
    # Fallback formatting
    formatted_messages = []
    for message in messages:
        role = message.get('role', 'user')
        content = message.get('content', '')
        formatted_messages.append(f"<|{role}|>\n{content}")
    
    return "\n".join(formatted_messages)


def get_target_modules_for_model(model_name: str) -> List[str]:
    """
    Get recommended target modules for LoRA based on model name.
    
    Args:
        model_name: Model name or path
        
    Returns:
        List of target module names
    """
    model_name_lower = model_name.lower()
    
    # Llama family
    if any(x in model_name_lower for x in ['llama', 'alpaca', 'vicuna']):
        return ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    
    # Mistral family  
    elif any(x in model_name_lower for x in ['mistral', 'mixtral']):
        return ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    
    # Qwen family
    elif 'qwen' in model_name_lower:
        return ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    
    # Gemma family
    elif 'gemma' in model_name_lower:
        return ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    
    # Phi family
    elif 'phi' in model_name_lower:
        return ["q_proj", "k_proj", "v_proj", "dense", "fc1", "fc2"]
    
    # GPT-2 family
    elif 'gpt2' in model_name_lower:
        return ["c_attn", "c_proj", "c_fc"]
    
    # GPT-NeoX family
    elif 'gpt-neox' in model_name_lower or 'pythia' in model_name_lower:
        return ["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"]
    
    # Default fallback
    else:
        return ["q_proj", "v_proj"]


def estimate_memory_usage(
    model_name: str,
    sequence_length: int = 2048,
    batch_size: int = 1,
    dtype: torch.dtype = torch.float16
) -> Dict[str, float]:
    """
    Estimate memory usage for a model.
    
    Args:
        model_name: Model name
        sequence_length: Sequence length
        batch_size: Batch size
        dtype: Model dtype
        
    Returns:
        Dictionary with memory estimates in GB
    """
    # Rough parameter counts for common models (in millions)
    param_counts = {
        '1b': 1000,
        '3b': 3000, 
        '7b': 7000,
        '8b': 8000,
        '13b': 13000,
        '30b': 30000,
        '70b': 70000,
    }
    
    # Extract size from model name
    model_size = None
    model_name_lower = model_name.lower()
    for size_key in param_counts:
        if size_key in model_name_lower:
            model_size = param_counts[size_key]
            break
    
    if model_size is None:
        model_size = 7000  # Default to 7B
    
    # Bytes per parameter based on dtype
    bytes_per_param = {
        torch.float32: 4,
        torch.float16: 2,
        torch.bfloat16: 2,
        torch.int8: 1,
    }.get(dtype, 2)
    
    # Model weights
    model_memory = (model_size * 1e6 * bytes_per_param) / (1024**3)  # GB
    
    # Activation memory (rough estimate)
    activation_memory = (batch_size * sequence_length * model_size * 0.001 * bytes_per_param) / (1024**3)
    
    # Optimizer states (AdamW uses 2x model params)
    optimizer_memory = model_memory * 2
    
    # Gradient memory
    gradient_memory = model_memory
    
    total_memory = model_memory + activation_memory + optimizer_memory + gradient_memory
    
    return {
        'model_weights': model_memory,
        'activations': activation_memory,
        'optimizer_states': optimizer_memory,
        'gradients': gradient_memory,
        'total': total_memory
    }
