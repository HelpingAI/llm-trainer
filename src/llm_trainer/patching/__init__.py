"""Patching system for integrating with Transformers and TRL."""

from .patch_transformers import patch_transformers
from .patch_trl import patch_trl

__all__ = [
    "patch_transformers",
    "patch_trl"
]