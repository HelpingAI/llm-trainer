"""Training utilities."""

import os
import random
import logging
import torch
import numpy as np
from typing import Optional, Dict, Any
import json
from datetime import datetime


def set_seed(seed: int) -> None:
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Only set CUDA seed if CUDA is available
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # For deterministic behavior (may impact performance)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """Get the best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def setup_logging(log_level: str = "info", log_file: Optional[str] = None) -> None:
    """Setup logging configuration without duplicating handlers."""
    level_map = {
        "debug": logging.DEBUG,
        "info": logging.INFO,
        "warning": logging.WARNING,
        "error": logging.ERROR,
        "critical": logging.CRITICAL
    }

    level = level_map.get(log_level.lower(), logging.INFO)

    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # Remove existing llm-trainer handlers to avoid duplicate logs
    existing_handlers = list(root_logger.handlers)
    for handler in existing_handlers:
        if getattr(handler, "_llm_trainer_handler", False):
            root_logger.removeHandler(handler)
            handler.close()

    # Setup console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    console_handler._llm_trainer_handler = True  # type: ignore[attr-defined]
    root_logger.addHandler(console_handler)

    # Setup file handler if specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        file_handler._llm_trainer_handler = True  # type: ignore[attr-defined]
        root_logger.addHandler(file_handler)


def count_parameters(model: torch.nn.Module) -> Dict[str, int]:
    """Count model parameters."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return {
        "total": total_params,
        "trainable": trainable_params,
        "non_trainable": total_params - trainable_params
    }


def get_model_size_mb(model: torch.nn.Module) -> float:
    """Get model size in MB."""
    param_size = 0
    buffer_size = 0

    for param in model.parameters():
        param_size += param.nelement() * param.element_size()

    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_mb = (param_size + buffer_size) / 1024 / 1024
    return size_mb


def save_checkpoint(model: torch.nn.Module,
                   optimizer: torch.optim.Optimizer,
                   scheduler: Optional[Any],
                   epoch: int,
                   step: int,
                   loss: float,
                   save_path: str,
                   config: Optional[Dict[str, Any]] = None) -> None:
    """Save training checkpoint."""
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "step": step,
        "loss": loss,
        "timestamp": datetime.now().isoformat()
    }

    if scheduler is not None:
        checkpoint["scheduler_state_dict"] = scheduler.state_dict()

    if config is not None:
        checkpoint["config"] = config

    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Save checkpoint
    torch.save(checkpoint, save_path)


def load_checkpoint(checkpoint_path: str,
                   model: torch.nn.Module,
                   optimizer: Optional[torch.optim.Optimizer] = None,
                   scheduler: Optional[Any] = None,
                   device: Optional[torch.device] = None) -> Dict[str, Any]:
    """Load training checkpoint."""
    if device is None:
        device = get_device()

    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Load model state
    model.load_state_dict(checkpoint["model_state_dict"])

    # Load optimizer state
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    # Load scheduler state
    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    return {
        "epoch": checkpoint.get("epoch", 0),
        "step": checkpoint.get("step", 0),
        "loss": checkpoint.get("loss", float('inf')),
        "config": checkpoint.get("config", {})
    }


def cleanup_checkpoints(checkpoint_dir: str, keep_last_n: int = 3) -> None:
    """Clean up old checkpoints, keeping only the last N."""
    if not os.path.exists(checkpoint_dir):
        return

    # Get all checkpoint files
    checkpoint_files = []
    for file in os.listdir(checkpoint_dir):
        if file.startswith("checkpoint-") and file.endswith(".pt"):
            file_path = os.path.join(checkpoint_dir, file)
            checkpoint_files.append((file_path, os.path.getmtime(file_path)))

    # Sort by modification time (newest first)
    checkpoint_files.sort(key=lambda x: x[1], reverse=True)

    # Remove old checkpoints
    for file_path, _ in checkpoint_files[keep_last_n:]:
        try:
            os.remove(file_path)
            logging.info(f"Removed old checkpoint: {file_path}")
        except OSError as e:
            logging.warning(f"Failed to remove checkpoint {file_path}: {e}")


def format_time(seconds: float) -> str:
    """Format time in seconds to human readable format."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"


def calculate_eta(current_step: int, total_steps: int, elapsed_time: float) -> str:
    """Calculate estimated time of arrival."""
    if current_step == 0:
        return "Unknown"

    steps_per_second = current_step / elapsed_time
    remaining_steps = total_steps - current_step
    eta_seconds = remaining_steps / steps_per_second

    return format_time(eta_seconds)


def get_memory_usage() -> Dict[str, float]:
    """Get current memory usage."""
    memory_info = {}

    # CUDA memory tracking (only if CUDA is available and being used)
    if torch.cuda.is_available() and torch.cuda.device_count() > 0:
        try:
            memory_info["cuda_allocated"] = torch.cuda.memory_allocated() / 1024**3  # GB
            memory_info["cuda_reserved"] = torch.cuda.memory_reserved() / 1024**3  # GB
            memory_info["cuda_max_allocated"] = torch.cuda.max_memory_allocated() / 1024**3  # GB
        except RuntimeError:
            # Handle case where CUDA context is not initialized
            pass

    # System memory (always try to get this as fallback)
    try:
        import psutil
        process = psutil.Process()
        memory_info["system_memory"] = process.memory_info().rss / 1024**3  # GB
        # Add system-wide memory info for CPU training
        virtual_memory = psutil.virtual_memory()
        memory_info["system_memory_total"] = virtual_memory.total / 1024**3  # GB
        memory_info["system_memory_available"] = virtual_memory.available / 1024**3  # GB
        memory_info["system_memory_percent"] = virtual_memory.percent
    except ImportError:
        # Fallback to basic memory tracking without psutil
        try:
            import resource
            # Get memory usage in KB and convert to GB
            if hasattr(resource, 'getrusage') and hasattr(resource, 'RUSAGE_SELF'):
                memory_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                # On Linux, ru_maxrss is in KB; on macOS, it's in bytes
                import sys
                if sys.platform == 'darwin':  # macOS
                    memory_info["system_memory"] = memory_kb / 1024**3  # Convert from bytes to GB
                else:  # Linux and others
                    memory_info["system_memory"] = memory_kb / 1024**2  # Convert from KB to GB
            else:
                memory_info["system_memory"] = 0.0
        except (ImportError, Exception):
            # Ultimate fallback - just indicate we couldn't get memory info
            memory_info["system_memory"] = 0.0

    return memory_info


def log_model_info(model: torch.nn.Module, logger: Optional[logging.Logger] = None) -> None:
    """Log model information."""
    if logger is None:
        logger = logging.getLogger(__name__)

    param_counts = count_parameters(model)
    model_size = get_model_size_mb(model)

    logger.info("Model parameters:")
    logger.info(f"  Total: {param_counts['total']:,}")
    logger.info(f"  Trainable: {param_counts['trainable']:,}")
    logger.info(f"  Non-trainable: {param_counts['non_trainable']:,}")
    logger.info(f"Model size: {model_size:.2f} MB")


class MetricsTracker:
    """Track training metrics."""

    def __init__(self):
        self.metrics = {}
        self.step_counts = {}

    def update(self, metrics: Dict[str, float], step: int) -> None:
        """Update metrics."""
        for key, value in metrics.items():
            if key not in self.metrics:
                self.metrics[key] = []
                self.step_counts[key] = []

            self.metrics[key].append(value)
            self.step_counts[key].append(step)

    def get_average(self, key: str, last_n: Optional[int] = None) -> float:
        """Get average of a metric."""
        if key not in self.metrics:
            return 0.0

        values = self.metrics[key]
        if last_n is not None:
            values = values[-last_n:]

        return sum(values) / len(values) if values else 0.0

    def get_latest(self, key: str) -> float:
        """Get latest value of a metric."""
        if key not in self.metrics or not self.metrics[key]:
            return 0.0
        return self.metrics[key][-1]

    def reset(self) -> None:
        """Reset all metrics."""
        self.metrics.clear()
        self.step_counts.clear()

    def save(self, file_path: str) -> None:
        """Save metrics to file."""
        data = {
            "metrics": self.metrics,
            "step_counts": self.step_counts
        }

        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2)

    def load(self, file_path: str) -> None:
        """Load metrics from file."""
        with open(file_path, 'r') as f:
            data = json.load(f)

        self.metrics = data.get("metrics", {})
        self.step_counts = data.get("step_counts", {})
