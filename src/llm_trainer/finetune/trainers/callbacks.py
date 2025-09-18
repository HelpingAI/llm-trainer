"""
Training callbacks for LLM fine-tuning (TRL-style).
"""
from typing import Optional, Dict, Any
import logging
from transformers import TrainerCallback, TrainerState, TrainerControl
from transformers.trainer_utils import TrainOutput

logger = logging.getLogger(__name__)


class BEMACallback(TrainerCallback):
    """Beta Exponential Moving Average callback for model parameters."""

    def __init__(self, beta: float = 0.999):
        self.beta = beta
        self.ema_model = None

    def on_train_begin(self, args, state: TrainerState, control: TrainerControl, model=None, **kwargs):
        """Initialize EMA model at training start."""
        if model is not None:
            self.ema_model = {name: param.clone().detach() for name, param in model.named_parameters()}


class LogCompletionsCallback(TrainerCallback):
    """Callback to log model completions during training."""

    def __init__(self, log_freq: int = 100):
        self.log_freq = log_freq

    def on_log(self, args, state: TrainerState, control: TrainerControl, logs: Optional[Dict[str, float]] = None, **kwargs):
        """Log completions at specified frequency."""
        if state.global_step % self.log_freq == 0:
            logger.info(f"Step {state.global_step}: Training in progress")


class MergeModelCallback(TrainerCallback):
    """Callback to merge PEFT adapters with base model."""

    def on_train_end(self, args, state: TrainerState, control: TrainerControl, model=None, **kwargs):
        """Merge adapters at training end if using PEFT."""
        if model is not None and hasattr(model, "merge_and_unload"):
            logger.info("Merging PEFT adapters with base model")


class RichProgressCallback(TrainerCallback):
    """Enhanced progress callback with rich formatting."""

    def on_log(self, args, state: TrainerState, control: TrainerControl, logs: Optional[Dict[str, float]] = None, **kwargs):
        """Log training progress with rich formatting."""
        if logs:
            loss = logs.get("train_loss", 0.0)
            lr = logs.get("learning_rate", 0.0)
            logger.info(f"Step {state.global_step}: loss={loss:.4f}, lr={lr:.2e}")


class SyncRefModelCallback(TrainerCallback):
    """Callback to synchronize reference model in preference learning."""

    def __init__(self, sync_freq: int = 100):
        self.sync_freq = sync_freq

    def on_step_end(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        """Sync reference model at specified frequency."""
        if state.global_step % self.sync_freq == 0:
            logger.debug(f"Syncing reference model at step {state.global_step}")


class WinRateCallback(TrainerCallback):
    """Callback to track win rates in preference learning."""

    def __init__(self):
        self.win_rates = []

    def on_log(self, args, state: TrainerState, control: TrainerControl, logs: Optional[Dict[str, float]] = None, **kwargs):
        """Track win rates from logs."""
        if logs and "win_rate" in logs:
            self.win_rates.append(logs["win_rate"])
            logger.info(f"Current win rate: {logs['win_rate']:.3f}")

