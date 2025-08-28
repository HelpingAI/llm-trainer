"""SafeTensors utilities for model saving and loading with sharding support."""

import os
import json
import math
from typing import Dict, List, Optional, Union, Any
import torch
import torch.nn as nn


def is_safetensors_available() -> bool:
    """Check if SafeTensors is available."""
    try:
        import safetensors
        return True
    except ImportError:
        return False


def get_model_size_in_bytes(state_dict: Dict[str, torch.Tensor]) -> int:
    """Calculate the total size of model parameters in bytes."""
    total_size = 0
    for tensor in state_dict.values():
        total_size += tensor.numel() * tensor.element_size()
    return total_size


def should_shard_model(state_dict: Dict[str, torch.Tensor], 
                      max_shard_size: Union[int, str] = "5GB") -> bool:
    """Determine if the model should be sharded based on size."""
    if isinstance(max_shard_size, str):
        # Convert string like "5GB" to bytes
        if max_shard_size.endswith("GB"):
            max_shard_size = int(max_shard_size[:-2]) * 1024 * 1024 * 1024
        elif max_shard_size.endswith("MB"):
            max_shard_size = int(max_shard_size[:-2]) * 1024 * 1024
        else:
            raise ValueError(f"Unsupported size format: {max_shard_size}")
    
    model_size = get_model_size_in_bytes(state_dict)
    return model_size > max_shard_size


def create_model_index(sharded_files: List[str], 
                      state_dict: Dict[str, torch.Tensor]) -> Dict[str, Any]:
    """Create model index file for sharded models."""
    if not is_safetensors_available():
        raise ImportError("safetensors is required for sharding. Install with: pip install safetensors")
    
    from safetensors.torch import save_file
    
    # Create temporary files to get metadata
    weight_map = {}
    total_size = 0
    
    # Calculate which parameters go in which shard
    current_shard = 0
    current_shard_dict = {}
    shard_size = 0
    max_shard_size = 5 * 1024 * 1024 * 1024  # 5GB default
    
    for name, tensor in state_dict.items():
        tensor_size = tensor.numel() * tensor.element_size()
        
        # If adding this tensor would exceed max shard size, move to next shard
        if shard_size + tensor_size > max_shard_size and current_shard_dict:
            current_shard += 1
            current_shard_dict = {}
            shard_size = 0
        
        current_shard_dict[name] = tensor
        shard_size += tensor_size
        total_size += tensor_size
        
        # Map parameter name to shard file
        shard_filename = f"model-{current_shard+1:05d}-of-{len(sharded_files):05d}.safetensors"
        weight_map[name] = shard_filename
    
    return {
        "metadata": {
            "total_size": total_size,
            "format": "safetensors"
        },
        "weight_map": weight_map
    }


def shard_state_dict(state_dict: Dict[str, torch.Tensor], 
                    max_shard_size: Union[int, str] = "5GB") -> List[Dict[str, torch.Tensor]]:
    """Shard a state dictionary into multiple smaller dictionaries."""
    if isinstance(max_shard_size, str):
        if max_shard_size.endswith("GB"):
            max_shard_size = int(max_shard_size[:-2]) * 1024 * 1024 * 1024
        elif max_shard_size.endswith("MB"):
            max_shard_size = int(max_shard_size[:-2]) * 1024 * 1024
        else:
            raise ValueError(f"Unsupported size format: {max_shard_size}")
    
    shards = []
    current_shard = {}
    current_size = 0
    
    for name, tensor in state_dict.items():
        tensor_size = tensor.numel() * tensor.element_size()
        
        # If adding this tensor would exceed max shard size, start a new shard
        if current_size + tensor_size > max_shard_size and current_shard:
            shards.append(current_shard)
            current_shard = {}
            current_size = 0
        
        current_shard[name] = tensor
        current_size += tensor_size
    
    # Add the last shard if it has any tensors
    if current_shard:
        shards.append(current_shard)
    
    return shards


def save_model_safetensors(model: nn.Module, 
                          save_directory: str,
                          max_shard_size: Union[int, str] = "5GB",
                          metadata: Optional[Dict[str, str]] = None) -> None:
    """
    Save model using SafeTensors format with optional sharding.
    
    Args:
        model: PyTorch model to save
        save_directory: Directory to save the model
        max_shard_size: Maximum size per shard (e.g., "5GB", "500MB")
        metadata: Optional metadata to include in SafeTensors file
    """
    if not is_safetensors_available():
        raise ImportError("safetensors is required. Install with: pip install safetensors")
    
    from safetensors.torch import save_file
    
    os.makedirs(save_directory, exist_ok=True)
    state_dict = model.state_dict()
    
    # Prepare metadata
    if metadata is None:
        metadata = {}
    metadata.update({
        "format": "pt",
        "pytorch_version": torch.__version__,
    })
    
    # Check if we need to shard the model
    if should_shard_model(state_dict, max_shard_size):
        # Shard the model
        shards = shard_state_dict(state_dict, max_shard_size)
        num_shards = len(shards)
        
        # Save each shard
        sharded_files = []
        weight_map = {}
        total_size = 0
        
        for i, shard in enumerate(shards):
            shard_filename = f"model-{i+1:05d}-of-{num_shards:05d}.safetensors"
            shard_path = os.path.join(save_directory, shard_filename)
            
            # Save shard
            save_file(shard, shard_path, metadata=metadata)
            sharded_files.append(shard_filename)
            
            # Update weight map and total size
            for name in shard.keys():
                weight_map[name] = shard_filename
                total_size += shard[name].numel() * shard[name].element_size()
        
        # Create model index file
        index = {
            "metadata": {
                "total_size": total_size,
                "format": "safetensors"
            },
            "weight_map": weight_map
        }
        
        index_path = os.path.join(save_directory, "model.safetensors.index.json")
        with open(index_path, 'w') as f:
            json.dump(index, f, indent=2)
            
        print(f"Model saved in {num_shards} shards with total size: {total_size / (1024**3):.2f} GB")
        
    else:
        # Save as single file
        model_path = os.path.join(save_directory, "model.safetensors")
        save_file(state_dict, model_path, metadata=metadata)
        
        total_size = get_model_size_in_bytes(state_dict)
        print(f"Model saved as single file with size: {total_size / (1024**3):.2f} GB")


def load_model_safetensors(model: nn.Module, 
                          load_directory: str,
                          strict: bool = True) -> None:
    """
    Load model from SafeTensors format (sharded or single file).
    
    Args:
        model: PyTorch model to load weights into
        load_directory: Directory containing the saved model
        strict: Whether to strictly enforce that the keys match
    """
    if not is_safetensors_available():
        raise ImportError("safetensors is required. Install with: pip install safetensors")
    
    from safetensors.torch import load_file
    
    # Check for sharded model
    index_path = os.path.join(load_directory, "model.safetensors.index.json")
    if os.path.exists(index_path):
        # Load sharded model
        with open(index_path, 'r') as f:
            index = json.load(f)
        
        state_dict = {}
        weight_map = index["weight_map"]
        
        # Get unique shard files
        shard_files = list(set(weight_map.values()))
        
        for shard_file in shard_files:
            shard_path = os.path.join(load_directory, shard_file)
            shard_dict = load_file(shard_path)
            state_dict.update(shard_dict)
        
        print(f"Loaded sharded model from {len(shard_files)} files")
        
    else:
        # Load single file
        model_path = os.path.join(load_directory, "model.safetensors")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"No SafeTensors model found in {load_directory}")
        
        state_dict = load_file(model_path)
        print(f"Loaded model from single SafeTensors file")
    
    # Load state dict into model
    model.load_state_dict(state_dict, strict=strict)


def get_safetensors_metadata(file_path: str) -> Dict[str, Any]:
    """Get metadata from a SafeTensors file."""
    if not is_safetensors_available():
        raise ImportError("safetensors is required. Install with: pip install safetensors")
    
    from safetensors import safe_open
    
    metadata = {}
    with safe_open(file_path, framework="pt") as f:
        metadata = f.metadata()
    
    return metadata


def convert_pytorch_to_safetensors(pytorch_path: str, 
                                  safetensors_path: str,
                                  metadata: Optional[Dict[str, str]] = None) -> None:
    """Convert a PyTorch model file to SafeTensors format."""
    if not is_safetensors_available():
        raise ImportError("safetensors is required. Install with: pip install safetensors")
    
    from safetensors.torch import save_file
    
    # Load PyTorch state dict
    state_dict = torch.load(pytorch_path, map_location='cpu')
    
    # Prepare metadata
    if metadata is None:
        metadata = {}
    metadata.update({
        "format": "pt",
        "converted_from": "pytorch",
        "pytorch_version": torch.__version__,
    })
    
    # Save as SafeTensors
    save_file(state_dict, safetensors_path, metadata=metadata)
    print(f"Converted {pytorch_path} to {safetensors_path}")


def list_safetensors_tensors(file_path: str) -> List[str]:
    """List all tensor names in a SafeTensors file."""
    if not is_safetensors_available():
        raise ImportError("safetensors is required. Install with: pip install safetensors")
    
    from safetensors import safe_open
    
    tensor_names = []
    with safe_open(file_path, framework="pt") as f:
        tensor_names = list(f.keys())
    
    return tensor_names