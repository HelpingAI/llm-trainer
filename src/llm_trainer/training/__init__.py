"""Training infrastructure for LLM training."""

from importlib import import_module
from typing import Any, Dict, Tuple

_SYMBOL_MAP: Dict[str, Tuple[str, str]] = {
    "Trainer": ("llm_trainer.training.trainer", "Trainer"),
    "create_optimizer": ("llm_trainer.training.optimizer", "create_optimizer"),
    "create_scheduler": ("llm_trainer.training.scheduler", "create_scheduler"),
    "set_seed": ("llm_trainer.training.utils", "set_seed"),
    "get_device": ("llm_trainer.training.utils", "get_device"),
    "setup_logging": ("llm_trainer.training.utils", "setup_logging"),
}

__all__ = list(_SYMBOL_MAP.keys())


def __getattr__(name: str) -> Any:
    if name not in _SYMBOL_MAP:
        raise AttributeError(f"module 'llm_trainer.training' has no attribute '{name}'")

    module_name, attr_name = _SYMBOL_MAP[name]
    module = import_module(module_name)
    return getattr(module, attr_name)


def __dir__() -> Any:
    return sorted(list(globals().keys()) + __all__)
