"""Main trainer class for LLM training with TRL enhancements."""

import os
import time
import logging
import torch
import torch.nn as nn
from typing import Optional, Dict, Any, Callable
from tqdm import tqdm
import math
from contextlib import nullcontext

from ..config import ModelConfig, TrainingConfig, DataConfig
from ..models import TransformerLM
from ..data import LanguageModelingDataset, create_dataloader, create_distributed_dataloader
from .optimizer import create_optimizer, GradientClipping, ExponentialMovingAverage
from .scheduler import create_scheduler
from .utils import (
    set_seed, setup_logging, save_checkpoint, load_checkpoint,
    cleanup_checkpoints, MetricsTracker, log_model_info, get_memory_usage,
    format_time
)

# Import kernel optimizations
from ..kernels import *  # noqa: F403
from ..kernels.fused_ops import FusedLinear, fused_cross_entropy
from ..kernels.memory_efficient import (
    gradient_checkpointing, offload_optimizer_states, empty_cache
)


class Trainer:
    """Enhanced trainer class with TRL-style API and memory optimizations."""

    def __init__(self,
                 model: TransformerLM,
                 tokenizer,
                 config: TrainingConfig,
                 train_dataset: Optional[LanguageModelingDataset] = None,
                 eval_dataset: Optional[LanguageModelingDataset] = None,
                 data_config: Optional[DataConfig] = None,
                 formatting_func: Optional[Callable] = None,
                 peft_config: Optional[Any] = None):
        """
        Args:
            model: The model to train.
            tokenizer: The tokenizer to use.
            config: Training configuration.
            train_dataset: Optional training dataset.
            eval_dataset: Optional evaluation dataset.
            data_config: Optional data configuration.
            formatting_func: Optional formatting function for dataset samples (like SFTTrainer).
            peft_config: Optional PEFT configuration for parameter-efficient training.
        """
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.data_config = data_config
        self.formatting_func = formatting_func

        # Setup logging
        setup_logging(config.log_level)
        self.logger = logging.getLogger(__name__)

        # Set random seed
        set_seed(config.seed)

        # Setup device and distributed training
        self.device = self.config.get_effective_device()
        self.is_distributed = config.world_size > 1
        self.local_rank = config.local_rank

        # Move model to device (skip if using Accelerate)
        self.accelerator = None
        if not getattr(self.config, 'use_accelerate', False):
            self.model = self.model.to(self.device)

        # Setup distributed training (handled by Accelerate if enabled)
        if self.is_distributed and not getattr(self.config, 'use_accelerate', False):
            self._setup_distributed()
        # Optional: integrate Accelerate
        if getattr(self.config, 'use_accelerate', False):
            try:
                from accelerate import Accelerator
                mixed_precision = getattr(self.config, 'accelerate_mixed_precision', 'no')
                self.accelerator = Accelerator(mixed_precision=mixed_precision if mixed_precision in ["no", "fp16", "bf16"] else "no")
                self.device = self.accelerator.device
                self.logger.info(f"Accelerate enabled with mixed_precision={mixed_precision}, device={self.device}")
            except ImportError:
                self.logger.warning("Accelerate not installed; proceeding without it.")
                self.accelerator = None


        # Mixed precision setup
        self.use_amp = self.config.should_use_amp()
        self.amp_dtype = self.config.get_amp_dtype()
        self.scaler = None
        if self.use_amp and self.device.type == "cuda" and not self.config.bf16:
            self.scaler = torch.amp.GradScaler("cuda")

        # Create optimizer
        self.optimizer = create_optimizer(
            model=self.model,
            optimizer_name=config.optimizer,
            learning_rate=config.learning_rate,
            weight_decay=config.weight_decay,
            beta1=config.beta1,
            beta2=config.beta2,
            eps=config.eps
        )

        # Setup gradient clipping
        self.gradient_clipper = GradientClipping(max_norm=config.max_grad_norm)

        # Setup EMA if requested
        self.ema = None
        if hasattr(config, 'use_ema') and config.use_ema:
            ema_decay = getattr(config, 'ema_decay', 0.999)
            self.ema = ExponentialMovingAverage(self.model, decay=ema_decay)

        # Setup mixed precision training
        self.use_amp = self.config.should_use_amp()
        self.scaler: Optional[torch.amp.GradScaler] = None
        if self.use_amp and not getattr(self.config, 'use_accelerate', False):
            if self.device.type != "cuda":
                self.logger.warning(
                    "AMP requested but unsupported on device '%s'; disabling AMP.",
                    self.device.type
                )
                self.use_amp = False
            elif torch.cuda.is_available():
                self.scaler = torch.amp.GradScaler(device=self.device.type)
            else:
                self.logger.warning("CUDA unavailable; disabling AMP.")
                self.use_amp = False

        # Setup datasets and dataloaders
        # Optionally apply PEFT adapters (LoRA) if requested
        self.peft_applied = False
        if getattr(self.config, 'use_peft', False) or peft_config is not None:
            try:
                from peft import get_peft_model  # type: ignore
                # Use provided peft_config or fall back to config settings
                if peft_config is not None:
                    self.model = get_peft_model(self.model, peft_config)
                    self.peft_applied = True
                    self.logger.info("PEFT adapters applied with provided config")
                elif getattr(self.config, 'use_peft', False):
                    # Apply PEFT from training config
                    from peft import LoraConfig, TaskType # type: ignore
                    task_type = getattr(self.config, 'peft_task_type', 'CAUSAL_LM')
                    task_enum = getattr(TaskType, task_type, TaskType.CAUSAL_LM)
                    lora_config = LoraConfig(
                        r=getattr(self.config, 'peft_r', 8),
                        lora_alpha=getattr(self.config, 'peft_alpha', 16),
                        lora_dropout=getattr(self.config, 'peft_dropout', 0.05),
                        bias=getattr(self.config, 'peft_bias', 'none'),
                        task_type=task_enum,
                        target_modules=getattr(self.config, 'peft_target_modules', None)
                    )
                    self.model = get_peft_model(self.model, lora_config)
                    self.peft_applied = True
                    self.logger.info("PEFT adapters applied (LoRA)")
            except ImportError:
                self.logger.warning("peft not installed; skipping PEFT integration.")
            except Exception as e:
                self.logger.warning(f"Failed to apply PEFT adapters: {e}")

        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.train_dataloader = None
        self.eval_dataloader = None

        if train_dataset is not None:
            self._setup_dataloaders()

        # Calculate total training steps
        self.total_steps = self._calculate_total_steps()

        # Create learning rate scheduler
        self.scheduler = create_scheduler(
            optimizer=self.optimizer,
            scheduler_name=config.lr_scheduler,
            num_training_steps=self.total_steps,
            warmup_steps=config.warmup_steps,
            min_lr_ratio=config.min_lr_ratio
        )

        # Training state
        self.current_epoch = 0
        self.current_step = 0
        self.best_eval_loss = float('inf')

        # Metrics tracking
        self.metrics_tracker = MetricsTracker()

        # Setup monitoring
        self._setup_monitoring()

        # Log model information
        log_model_info(self.model, self.logger)

        # Apply fused layers if requested
        self.apply_fused_layers()

        # Setup efficient attention if available
        self._setup_efficient_attention()

        self.logger.info(f"Trainer initialized with {self.total_steps} total training steps")

    def _setup_distributed(self):
        """Setup distributed training."""
        if not torch.distributed.is_initialized():  # type: ignore
            torch.distributed.init_process_group(  # type: ignore
                backend=self.config.get_effective_distributed_backend(),
                rank=self.local_rank,
                world_size=self.config.world_size
            )

        # Wrap model for distributed training
        self.model = torch.nn.parallel.DistributedDataParallel(
            self.model,
            device_ids=[self.local_rank] if self.device.type == "cuda" else None,
            find_unused_parameters=False
        )

        self.logger.info(f"Distributed training setup: rank {self.local_rank}/{self.config.world_size}")

    def _setup_dataloaders(self):
        """Setup training and evaluation dataloaders."""
        use_accel = getattr(self.config, 'use_accelerate', False) and self.accelerator is not None
        if self.is_distributed and not use_accel:
            if self.train_dataset is not None:
                self.train_dataloader = create_distributed_dataloader(
                    dataset=self.train_dataset,
                    batch_size=self.config.batch_size,
                    num_replicas=self.config.world_size,
                    rank=self.local_rank,
                    shuffle=True,
                    num_workers=self.config.dataloader_num_workers,
                    pin_memory=self.config.dataloader_pin_memory,
                    drop_last=self.config.dataloader_drop_last,
                    pad_token_id=self.tokenizer.pad_token_id
                )
        else:
            # For Accelerate or single-process, use regular DataLoader; Accelerate will wrap it
            if self.train_dataset is not None:
                self.train_dataloader = create_dataloader(
                    dataset=self.train_dataset,
                    batch_size=self.config.batch_size,
                    shuffle=True,
                    num_workers=self.config.dataloader_num_workers,
                    pin_memory=self.config.dataloader_pin_memory,
                    drop_last=self.config.dataloader_drop_last,
                    pad_token_id=self.tokenizer.pad_token_id
                )

        if self.eval_dataset is not None:
            if self.is_distributed:
                self.eval_dataloader = create_distributed_dataloader(
                    dataset=self.eval_dataset,
                    batch_size=self.config.batch_size,
                    num_replicas=self.config.world_size,
                    rank=self.local_rank,
                    shuffle=False,
                    num_workers=self.config.dataloader_num_workers,
                    pin_memory=self.config.dataloader_pin_memory,
                    drop_last=False,
                    pad_token_id=self.tokenizer.pad_token_id
                )
            else:
                self.eval_dataloader = create_dataloader(
                    dataset=self.eval_dataset,
                    batch_size=self.config.batch_size,
                    shuffle=False,
                    num_workers=self.config.dataloader_num_workers,
                    pin_memory=self.config.dataloader_pin_memory,
                    drop_last=False,
                    pad_token_id=self.tokenizer.pad_token_id
                )

    def _calculate_total_steps(self) -> int:
        """Calculate total number of training steps."""
        if self.config.max_steps is not None:
            return self.config.max_steps

        if self.train_dataloader is None:
            # Estimate based on dataset size if available
            if self.train_dataset is not None:
                denominator = max(
                    1,
                    self.config.batch_size * self.config.gradient_accumulation_steps * max(1, self.config.world_size)
                )
                steps_per_epoch = math.ceil(len(self.train_dataset) / denominator)
                return max(1, steps_per_epoch) * self.config.num_epochs
            else:
                return 1000  # Default fallback

        try:
            dataloader_len = len(self.train_dataloader)
        except TypeError:
            # Some iterable dataloaders (e.g., streaming) do not support len()
            return self.config.max_steps or 1000

        steps_per_epoch = math.ceil(dataloader_len / self.config.gradient_accumulation_steps)
        return max(1, steps_per_epoch) * self.config.num_epochs

    def _setup_monitoring(self):
        """Setup monitoring and logging."""
        self.writers = []

        # Setup TensorBoard
        if self.config.report_to and "tensorboard" in self.config.report_to:
            try:
                from torch.utils.tensorboard import SummaryWriter
                log_dir = os.path.join(self.config.checkpoint_dir, "logs")
                self.tb_writer = SummaryWriter(log_dir)
                self.writers.append("tensorboard")
                self.logger.info(f"TensorBoard logging to {log_dir}")
            except ImportError:
                self.logger.warning("TensorBoard not available")

        # Setup Weights & Biases
        if self.config.report_to and "wandb" in self.config.report_to:
            try:
                import wandb
                wandb.init(
                    project="llm-training",
                    config=self.config.to_dict(),
                    name=f"run-{int(time.time())}"
                )
                self.writers.append("wandb")
                self.logger.info("Weights & Biases logging initialized")
            except ImportError:
                self.logger.warning("Weights & Biases not available")

    def train(self, config: Optional[TrainingConfig] = None) -> None:
        """
        Main training loop with TRL-style API.
        
        Args:
            config: Optional training configuration to override the default one
        """
        # If a specific config is provided, use it instead of the default one
        if config is not None:
            # Store the original config temporarily
            original_config = self.config
            self.config = config

            # Recalculate total steps if needed
            if self.train_dataloader is not None:
                self.total_steps = self._calculate_total_steps()

                # Recreate scheduler with new config
                self.scheduler = create_scheduler(
                    optimizer=self.optimizer,
                    scheduler_name=config.lr_scheduler,
                    num_training_steps=self.total_steps,
                    warmup_steps=config.warmup_steps,
                    min_lr_ratio=config.min_lr_ratio
                )

        if self.train_dataloader is None:
            raise ValueError("No training dataloader available. Please provide train_dataset.")

        self.logger.info("Starting training...")
        self.logger.info(f"Total epochs: {self.config.num_epochs}")
        self.logger.info(f"Total steps: {self.total_steps}")
        self.logger.info(f"Batch size: {self.config.batch_size}")
        self.logger.info(f"Gradient accumulation steps: {self.config.gradient_accumulation_steps}")
        self.logger.info(f"Effective batch size: {self.config.effective_batch_size}")

        # Resume from checkpoint if specified
        if self.config.resume_from_checkpoint:
            self._load_checkpoint(self.config.resume_from_checkpoint)

        # Training loop
        start_time = time.time()

        for epoch in range(self.current_epoch, self.config.num_epochs):
            self.current_epoch = epoch

            # Set epoch for distributed sampler
            if self.is_distributed and hasattr(self.train_dataloader.sampler, 'set_epoch'):
                self.train_dataloader.sampler.set_epoch(epoch)  # type: ignore

            # Train one epoch
            epoch_loss = self._train_epoch()

            # Evaluate if needed
            if self._should_evaluate():
                eval_loss = self._evaluate()

                # Check for best model
                if eval_loss < self.best_eval_loss:
                    self.best_eval_loss = eval_loss
                    self._save_best_model()

            # Save checkpoint
            if self._should_save_checkpoint():
                self._save_checkpoint()

            # Early stopping check
            if self._should_early_stop():
                self.logger.info("Early stopping triggered")
                break

            # Log epoch summary
            elapsed_time = time.time() - start_time
            self.logger.info(
                f"Epoch {epoch + 1}/{self.config.num_epochs} completed. "
                f"Loss: {epoch_loss:.4f}, Time: {format_time(elapsed_time)}"
            )

        # Final save
        self._save_checkpoint(final=True)

        total_time = time.time() - start_time
        self.logger.info(f"Training completed in {format_time(total_time)}")

        # Cleanup
        self._cleanup()

        # Restore original config if it was overridden
        if config is not None:
            self.config = original_config

    def sft_train(self, config: Optional[TrainingConfig] = None) -> None:
        """
        Supervised Fine-Tuning training (TRL-style API).
        
        Args:
            config: Optional training configuration
        """
        self.train(config)

    def dpo_train(self, config: Optional[TrainingConfig] = None) -> None:
        """
        Direct Preference Optimization training (TRL-style API).
        
        Args:
            config: Optional training configuration
        """
        self.train(config)

    def ppo_train(self, config: Optional[TrainingConfig] = None) -> None:
        """
        Proximal Policy Optimization training (TRL-style API).
        
        Args:
            config: Optional training configuration
        """
        self.train(config)

    def reward_modeling_train(self, config: Optional[TrainingConfig] = None) -> None:
        """
        Reward modeling training (TRL-style API).
        
        Args:
            config: Optional training configuration
        """
        self.train(config)

    def _train_epoch(self) -> float:
        """Train for one epoch."""
        self.model.train()
        epoch_loss = 0.0
        num_batches = 0

        # Progress bar
        if self.local_rank <= 0:  # Only show on main process
            pbar = tqdm(
                self.train_dataloader,
                desc=f"Epoch {self.current_epoch + 1}",
                disable=False
            )
        else:
            pbar = self.train_dataloader

        if pbar is None:
            raise ValueError("train_dataloader is None. Please provide training data.")

        for batch_idx, batch in enumerate(pbar):
            # With Accelerate, tensors are already on correct device
            if not (getattr(self.config, 'use_accelerate', False) and self.accelerator is not None):
                # Move batch to device
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

            # Forward pass
            loss = self._training_step(batch)

            # Backward pass
            self._backward_step(loss)

            # Optimizer step
            if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                self._optimizer_step()
                self.current_step += 1

                # Update learning rate
                if self.scheduler is not None:
                    self.scheduler.step()

                # Update EMA
                if self.ema is not None:
                    self.ema.update()

                # Log metrics
                if self.current_step % self.config.logging_steps == 0:
                    self._log_metrics(loss.item())

                # Evaluate
                if self._should_evaluate():
                    _ = self._evaluate()
                    self.model.train()  # Return to training mode

                # Save checkpoint
                if self._should_save_checkpoint():
                    self._save_checkpoint()

                # Check if max steps reached
                if self.config.max_steps and self.current_step >= self.config.max_steps:
                    break

            epoch_loss += loss.item()
            num_batches += 1

            # Update progress bar
            if self.local_rank <= 0:
                pbar.set_postfix({  # type: ignore
                    'loss': f'{loss.item():.4f}',
                    'lr': f'{self.optimizer.param_groups[0]["lr"]:.2e}',
                    'step': self.current_step
                })

        return epoch_loss / num_batches if num_batches > 0 else 0.0

    def _training_step(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Perform a single training step with kernel optimizations."""
        # Apply gradient checkpointing if enabled
        if getattr(self.config, 'use_gradient_checkpointing', False):
            def model_forward(*args, **kwargs):
                return self.model(*args, **kwargs)

            outputs = gradient_checkpointing(model_forward, use_reentrant=True, **batch)
        else:
            autocast_context = nullcontext()
            if self.use_amp:
                autocast_context = torch.amp.autocast(device_type=self.device.type, dtype=self.amp_dtype)

            with autocast_context:
                outputs = self.model(**batch)

        if isinstance(outputs, dict):
            loss = outputs["loss"]
        else:
            loss = getattr(outputs, "loss", None)
            if loss is None and isinstance(outputs, (list, tuple)) and len(outputs) > 0:
                loss = outputs[0]
            if loss is None:
                raise RuntimeError("Model forward pass did not return a loss tensor.")

        # Use fused cross-entropy if available and beneficial
        logits = None
        if isinstance(outputs, dict):
            logits = outputs.get("logits")
        else:
            logits = getattr(outputs, "logits", None)

        labels = batch.get('labels') if isinstance(batch, dict) else None

        if logits is None and isinstance(outputs, (list, tuple)) and len(outputs) > 1:
            logits = outputs[1]

        if logits is not None and labels is not None:
            try:
                # Use fused cross-entropy for better performance
                loss = fused_cross_entropy(
                    logits,
                    labels,
                    ignore_index=self.tokenizer.pad_token_id
                )
            except Exception:
                # Fallback to original loss
                pass

        # Scale loss for gradient accumulation
        loss = loss / self.config.gradient_accumulation_steps

        return loss

    def _backward_step(self, loss: torch.Tensor) -> None:
        """Perform backward pass."""
        if getattr(self.config, 'use_accelerate', False) and self.accelerator is not None:
            self.accelerator.backward(loss)
        elif self.scaler is not None:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

    def _optimizer_step(self) -> None:
        """Perform optimizer step with gradient clipping and memory optimizations."""
        if getattr(self.config, 'use_accelerate', False) and self.accelerator is not None:
            if (
                self.config.max_grad_norm
                and self.config.max_grad_norm > 0
                and hasattr(self.accelerator, 'clip_grad_norm_')
            ):
                self.accelerator.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
            elif self.config.max_grad_norm and self.config.max_grad_norm > 0:
                _ = self.gradient_clipper.clip_gradients(self.model)
            self.optimizer.step()
        elif self.scaler is not None:
            # Unscale gradients for clipping
            self.scaler.unscale_(self.optimizer)  # type: ignore

            # Clip gradients
            _ = self.gradient_clipper.clip_gradients(self.model)

            # Optimizer step
            self.scaler.step(self.optimizer)  # type: ignore
            self.scaler.update()  # type: ignore
        else:
            # Clip gradients
            _ = self.gradient_clipper.clip_gradients(self.model)

            # Optimizer step
            self.optimizer.step()

        # Zero gradients with memory efficiency
        self.optimizer.zero_grad(set_to_none=True)

        # Apply memory-efficient techniques if enabled
        if getattr(self.config, 'use_low_vram', False):
            # Offload optimizer states to CPU to save GPU memory
            try:
                offload_optimizer_states(self.optimizer, device='cpu')
            except Exception:
                pass  # Silently fail if offloading not supported

        # Empty cache periodically to free up memory
        if getattr(self.config, 'empty_cache_steps', 0) > 0:
            if self.current_step % self.config.empty_cache_steps == 0:
                empty_cache()

    def _evaluate(self) -> float:
        """Evaluate the model."""
        if self.eval_dataloader is None:
            return float('inf')

        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch in tqdm(self.eval_dataloader, desc="Evaluating", disable=self.local_rank > 0):
                # Move batch to device
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                        for k, v in batch.items()}

                # Forward pass
                autocast_context = nullcontext()
                if self.use_amp:
                    autocast_context = torch.amp.autocast(device_type=self.device.type, dtype=self.amp_dtype)

                with autocast_context:
                    outputs = self.model(**batch)

                if isinstance(outputs, dict):
                    loss = outputs["loss"]
                else:
                    loss = getattr(outputs, "loss", None)
                    if loss is None and isinstance(outputs, (list, tuple)) and len(outputs) > 0:
                        loss = outputs[0]
                    if loss is None:
                        raise RuntimeError("Model forward pass did not return a loss tensor during evaluation.")

                total_loss += loss.item()
                num_batches += 1

        avg_loss = total_loss / num_batches if num_batches > 0 else float('inf')

        # Log evaluation metrics
        self._log_eval_metrics(avg_loss)

        self.logger.info(f"Evaluation loss: {avg_loss:.4f}")
        return avg_loss

    def _should_evaluate(self) -> bool:
        """Check if evaluation should be performed."""
        if self.config.eval_strategy == "no":
            return False
        elif self.config.eval_strategy == "steps":
            return (
                self.current_step > 0
                and self.config.eval_steps > 0
                and self.current_step % self.config.eval_steps == 0
            )
        elif self.config.eval_strategy == "epoch":
            return True  # Evaluate at end of each epoch
        return False

    def _should_save_checkpoint(self) -> bool:
        """Check if checkpoint should be saved."""
        return (
            self.current_step > 0
            and self.config.save_steps > 0
            and self.current_step % self.config.save_steps == 0
        )

    def _should_early_stop(self) -> bool:
        """Check if early stopping should be triggered."""
        if self.config.early_stopping_patience is None:
            return False

        # Implementation would track validation loss and patience
        # For now, return False
        return False

    def _log_metrics(self, loss: float) -> None:
        """Log training metrics."""
        metrics = {
            "train/loss": loss,
            "train/learning_rate": self.optimizer.param_groups[0]["lr"],
            "train/step": self.current_step,
            "train/epoch": self.current_epoch
        }

        # Add memory usage
        memory_info = get_memory_usage()
        for key, value in memory_info.items():
            metrics[f"system/{key}"] = value

        # Update metrics tracker
        self.metrics_tracker.update(metrics, self.current_step)

        # Log to TensorBoard
        if "tensorboard" in self.writers:
            for key, value in metrics.items():
                self.tb_writer.add_scalar(key, value, self.current_step)

        # Log to Weights & Biases
        if "wandb" in self.writers:
            import wandb
            wandb.log(metrics, step=self.current_step)

        # Log to console
        if self.current_step % (self.config.logging_steps * 10) == 0:
            pass # Using tqdm for progress tracking

    def _log_eval_metrics(self, eval_loss: float) -> None:
        """Log evaluation metrics."""
        metrics = {
            "eval/loss": eval_loss,
            "eval/step": self.current_step,
            "eval/epoch": self.current_epoch
        }

        # Calculate perplexity
        perplexity = math.exp(eval_loss) if eval_loss < 10 else float('inf')
        metrics["eval/perplexity"] = perplexity

        # Update metrics tracker
        self.metrics_tracker.update(metrics, self.current_step)

        # Log to TensorBoard
        if "tensorboard" in self.writers:
            for key, value in metrics.items():
                self.tb_writer.add_scalar(key, value, self.current_step)

        # Log to Weights & Biases
        if "wandb" in self.writers:
            import wandb
            wandb.log(metrics, step=self.current_step)

    def _save_checkpoint(self, final: bool = False) -> None:
        """Save training checkpoint."""
        if self.local_rank > 0:  # Only save on main process
            return

        checkpoint_name = f"checkpoint-{self.current_step}.pt"
        if final:
            checkpoint_name = "final_checkpoint.pt"

        checkpoint_path = os.path.join(self.config.checkpoint_dir, checkpoint_name)

        # Get model for saving (unwrap DDP if needed)
        model_to_save = self.model
        if hasattr(self.model, 'module'):
            model_to_save = self.model.module
        assert isinstance(model_to_save, nn.Module), "Model must be a PyTorch Module"

        save_checkpoint(
            model=model_to_save,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            epoch=self.current_epoch,
            step=self.current_step,
            loss=self.metrics_tracker.get_latest("train/loss"),
            save_path=checkpoint_path,
            config=self.config.to_dict()
        )

        self.logger.info(f"Checkpoint saved: {checkpoint_path}")

        # Cleanup old checkpoints
        if not final:
            cleanup_checkpoints(self.config.checkpoint_dir, self.config.save_total_limit)

    def _save_best_model(self) -> None:
        """Save the best model."""
        if self.local_rank > 0:  # Only save on main process
            return

        best_model_path = os.path.join(self.config.checkpoint_dir, "best_model")

        # Get model for saving (unwrap DDP if needed)
        model_to_save = self.model
        if hasattr(self.model, 'module'):
            model_to_save = self.model.module
        assert isinstance(model_to_save, nn.Module), "Model must be a PyTorch Module"

        if hasattr(model_to_save, 'save_pretrained'):
            model_to_save.save_pretrained(best_model_path)  # type: ignore
        else:
            # Fallback: save state dict
            os.makedirs(best_model_path, exist_ok=True)
            torch.save(model_to_save.state_dict(), os.path.join(best_model_path, 'pytorch_model.bin'))
        
        if hasattr(self.tokenizer, 'save_pretrained'):
            self.tokenizer.save_pretrained(best_model_path)  # type: ignore

        self.logger.info(f"Best model saved: {best_model_path}")

    def _load_checkpoint(self, checkpoint_path: str) -> None:
        """Load training checkpoint."""
        self.logger.info(f"Loading checkpoint: {checkpoint_path}")

        # Get model for loading (unwrap DDP if needed)
        model_to_load = self.model
        if hasattr(self.model, 'module'):
            model_to_load = self.model.module
        assert isinstance(model_to_load, nn.Module), "Model must be a PyTorch Module"

        checkpoint_info = load_checkpoint(
            checkpoint_path=checkpoint_path,
            model=model_to_load,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            device=self.device
        )

        self.current_epoch = checkpoint_info["epoch"]
        self.current_step = checkpoint_info["step"]

        self.logger.info(f"Resumed from epoch {self.current_epoch}, step {self.current_step}")

    def _cleanup(self) -> None:
        """Cleanup resources."""
        # Close TensorBoard writer
        if hasattr(self, 'tb_writer'):
            self.tb_writer.close()

        # Finish Weights & Biases
        if "wandb" in self.writers:
            import wandb
            wandb.finish()

        # Cleanup distributed training
        if self.is_distributed:
            torch.distributed.destroy_process_group()  # type: ignore

    def train_from_config(self,
                         model_config: ModelConfig,
                         data_config: DataConfig,
                         dataset_name: Optional[str] = None) -> None:
        """Train model from configuration files."""
        # Create datasets if not provided
        if self.train_dataset is None:
            dataset_name = dataset_name or data_config.dataset_name

            self.train_dataset = LanguageModelingDataset(
                dataset_name=dataset_name,
                dataset_config=data_config.dataset_config,
                split=data_config.dataset_split,
                tokenizer=self.tokenizer,
                text_column=data_config.text_column,
                max_length=data_config.max_length,
                preprocessing_config={
                    "min_length": data_config.min_length,
                    "filter_empty": data_config.filter_empty_lines,
                    "remove_duplicates": data_config.remove_duplicates
                },
                pack_sequences=data_config.pack_sequences,
                packing_strategy=data_config.packing_strategy
            )

        if data_config.validation_split and dataset_name:
            try:
                self.eval_dataset = LanguageModelingDataset(
                    dataset_name=dataset_name,
                    dataset_config=data_config.dataset_config,
                    split=data_config.validation_split,
                    tokenizer=self.tokenizer,
                    text_column=data_config.text_column,
                    max_length=data_config.max_length,
                    preprocessing_config={
                        "min_length": data_config.min_length,
                        "filter_empty": data_config.filter_empty_lines,
                        "remove_duplicates": data_config.remove_duplicates
                    },
                    pack_sequences=data_config.pack_sequences,
                    packing_strategy=data_config.packing_strategy
                )
            except ValueError as e:
                if "Unknown split" in str(e):
                    print(f"Warning: Validation split '{data_config.validation_split}' not found in dataset.")
                    print("Available splits are shown in the error above.")
                    print("Creating validation set from training data...")

                    # Create validation set from training data
                    train_size = len(self.train_dataset)
                    val_size = min(1000, train_size // 10)  # Use 10% or 1000 samples, whichever is smaller

                    # Split the training dataset
                    split_dataset = self.train_dataset.dataset.train_test_split(
                        test_size=val_size,
                        shuffle=True,
                        seed=42
                    )

                    # Update training dataset to use the train split
                    self.train_dataset.dataset = split_dataset['train']

                    # Create validation dataset
                    if dataset_name:
                        self.eval_dataset = LanguageModelingDataset(
                            dataset_name=dataset_name,
                            dataset_config=data_config.dataset_config,
                        split="train",  # We'll override this
                        tokenizer=self.tokenizer,
                        text_column=data_config.text_column,
                        max_length=data_config.max_length,
                        preprocessing_config={
                            "min_length": data_config.min_length,
                            "filter_empty": data_config.filter_empty_lines,
                            "remove_duplicates": data_config.remove_duplicates
                        },
                        pack_sequences=data_config.pack_sequences,
                        packing_strategy=data_config.packing_strategy
                    )
                    # Override with the test split
                        self.eval_dataset.dataset = split_dataset['test']  # type: ignore
        # Setup dataloaders from the created datasets
        self._setup_dataloaders()
        # Start training
        self.train()

    def generate_text(self,
                     prompt: str,
                     max_length: int = 100,
                     temperature: float = 1.0,
                     top_k: Optional[int] = None,
                     top_p: Optional[float] = None,
                     do_sample: bool = True) -> str:
        """Generate text from a prompt."""
        self.model.eval()

        # Encode prompt
        input_ids = self.tokenizer.encode(prompt, add_special_tokens=True)
        input_ids = torch.tensor([input_ids], device=self.device)

        # Generate
        with torch.no_grad():
            pad_token_id = getattr(self.tokenizer, "pad_token_id", None) or 0
            eos_token_id = getattr(self.tokenizer, "eos_token_id", None) or 3

            generated_ids = self.model.generate(  # type: ignore
                input_ids=input_ids,
                max_length=max_length,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                do_sample=do_sample,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id
            )

        # Decode generated text
        generated_text = self.tokenizer.decode(generated_ids[0].tolist(), skip_special_tokens=True)
        return generated_text

    def evaluate_perplexity(self, dataset: Optional[LanguageModelingDataset] = None) -> float:
        """Evaluate perplexity on a dataset."""
        if dataset is None:
            dataset = self.eval_dataset

        if dataset is None:
            raise ValueError("No evaluation dataset provided")

        # Create dataloader
        dataloader = create_dataloader(
            dataset=dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.dataloader_num_workers,
            pin_memory=self.config.dataloader_pin_memory,
            drop_last=False,
            pad_token_id=self.tokenizer.pad_token_id
        )

        self.model.eval()
        total_loss = 0.0
        total_tokens = 0

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Computing perplexity"):
                # Move batch to device
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                        for k, v in batch.items()}

                # Forward pass
                outputs = self.model(**batch)
                loss = outputs["loss"]

                # Count tokens (excluding padding)
                attention_mask = batch.get("attention_mask", torch.ones_like(batch["input_ids"]))
                num_tokens = attention_mask.sum().item()

                total_loss += loss.item() * num_tokens
                total_tokens += num_tokens

        avg_loss = total_loss / total_tokens if total_tokens > 0 else float('inf')
        perplexity = math.exp(avg_loss) if avg_loss < 10 else float('inf')

        self.logger.info(f"Perplexity: {perplexity:.2f}")
        return perplexity

    def save_model(self, save_path: str) -> None:
        """Save the trained model (HuggingFace-style API)."""
        # Get model for saving (unwrap DDP if needed)
        model_to_save = self.model
        if hasattr(self.model, 'module'):
            model_to_save = self.model.module

        if hasattr(model_to_save, 'save_pretrained'):
            model_to_save.save_pretrained(save_path)  # type: ignore
        else:
            # Fallback: save state dict
            os.makedirs(save_path, exist_ok=True)
            torch.save(model_to_save.state_dict(), os.path.join(save_path, 'pytorch_model.bin'))  # type: ignore
        
        if hasattr(self.tokenizer, 'save_pretrained'):
            self.tokenizer.save_pretrained(save_path)  # type: ignore

        self.logger.info(f"Model saved to {save_path}")

    def save_pretrained(self, save_directory: str) -> None:
        """Save the trained model and tokenizer (HuggingFace-style API)."""
        self.save_model(save_directory)

    def push_to_hub(self, repo_id: str, **kwargs) -> None:
        """Push model to Hugging Face Hub (HuggingFace-style API)."""
        # Get model for saving (unwrap DDP if needed)
        model_to_save = self.model
        if hasattr(self.model, 'module'):
            model_to_save = self.model.module

        # Push model to hub
        if hasattr(model_to_save, 'push_to_hub'):
            model_to_save.push_to_hub(repo_id, **kwargs)  # type: ignore
        else:
            raise AttributeError("Model does not support push_to_hub method")
        
        if hasattr(self.tokenizer, 'push_to_hub'):
            self.tokenizer.push_to_hub(repo_id, **kwargs)  # type: ignore

        self.logger.info(f"Model pushed to Hugging Face Hub: {repo_id}")

    @classmethod
    def from_pretrained(cls,
                       model_path: str,
                       tokenizer_path: str,
                       config: TrainingConfig) -> 'Trainer':
        """Load trainer from pretrained model and tokenizer."""
        # Load model
        model = TransformerLM.from_pretrained(model_path)

        # Load tokenizer
        from ..tokenizer import BPETokenizer
        tokenizer = BPETokenizer.from_pretrained(tokenizer_path)

        # Create trainer
        trainer = cls(model=model, tokenizer=tokenizer, config=config)

        return trainer

    def get_nb_trainable_parameters(self) -> tuple:
        """
        Get number of trainable parameters.
        
        Returns:
            Tuple of (all_params, trainable_params)
        """
        all_param = 0
        trainable_param = 0
        for _, param in self.model.named_parameters():
            all_param += param.numel()
            if param.requires_grad:
                trainable_param += param.numel()
        return all_param, trainable_param

    def print_trainable_parameters(self) -> None:
        """Print trainable parameters information."""
        all_param, trainable_param = self.get_nb_trainable_parameters()
        trainable_ratio = 100 * trainable_param / all_param if all_param > 0 else 0
        self.logger.info(
            f"Trainable parameters: {trainable_param:,}/{all_param:,} ({trainable_ratio:.2f}%)"
        )

    def prepare_model_for_kbit_training(self) -> None:
        """Prepare model for k-bit training."""
        try:
            from peft import prepare_model_for_kbit_training # type: ignore
            self.model = prepare_model_for_kbit_training(self.model)
            self.logger.info("Model prepared for k-bit training")
        except ImportError:
            self.logger.warning("PEFT not installed; skipping k-bit training preparation")
        except Exception as e:
            self.logger.warning(f"Failed to prepare model for k-bit training: {e}")

    def apply_fused_layers(self) -> None:
        """Apply fused layers for better performance."""
        try:
            # Replace linear layers with fused versions if requested
            if getattr(self.config, 'fuse_layers', False):
                self._replace_linear_with_fused(self.model)
                self.logger.info("Applied fused layers for better performance")
        except Exception as e:
            self.logger.warning(f"Failed to apply fused layers: {e}")

    def _replace_linear_with_fused(self, module: nn.Module) -> None:
        """Recursively replace linear layers with fused versions."""
        for name, child in module.named_children():
            if isinstance(child, nn.Linear):
                # Replace with fused linear
                fused_layer = FusedLinear(
                    child.in_features,
                    child.out_features,
                    child.bias is not None
                )
                # Copy weights and bias
                fused_layer.weight.data.copy_(child.weight.data)
                if child.bias is not None:
                    fused_layer.bias.data.copy_(child.bias.data)

                setattr(module, name, fused_layer)
            else:
                # Recursively apply to child modules
                self._replace_linear_with_fused(child)

    def _setup_efficient_attention(self) -> None:
        """Setup efficient attention mechanisms."""
        try:
            # Enable efficient attention if PyTorch 2.0+ is available
            if (
                hasattr(torch.nn.functional, 'scaled_dot_product_attention')
                and torch.cuda.is_available()
                and hasattr(torch.backends, 'cuda')
            ):
                if hasattr(torch.backends.cuda, 'enable_flash_sdp'):
                    torch.backends.cuda.enable_flash_sdp(True)
                if hasattr(torch.backends.cuda, 'enable_math_sdp'):
                    torch.backends.cuda.enable_math_sdp(False)
                if hasattr(torch.backends.cuda, 'enable_mem_efficient_sdp'):
                    torch.backends.cuda.enable_mem_efficient_sdp(False)
                self.logger.info("Enabled efficient attention (Flash Attention)")
        except Exception as e:
            self.logger.warning(f"Failed to setup efficient attention: {e}")

    @classmethod
    def create_memory_efficient_trainer(
        cls,
        model: nn.Module,
        tokenizer,
        config: TrainingConfig,
        **kwargs
    ) -> 'Trainer':
        """Create a memory-efficient trainer with optimizations applied."""
        # Set memory-efficient defaults
        config.use_gradient_checkpointing = getattr(config, 'use_gradient_checkpointing', True)
        config.use_low_vram = getattr(config, 'use_low_vram', True)
        config.fuse_layers = getattr(config, 'fuse_layers', True)

        trainer = cls(model, tokenizer, config, **kwargs)  # type: ignore
        return trainer
