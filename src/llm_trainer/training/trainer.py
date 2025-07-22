"""Main trainer class for LLM training."""

import os
import time
import logging
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Optional, Dict, Any, List, Union
from tqdm import tqdm
import math

from ..config import ModelConfig, TrainingConfig, DataConfig
from ..models import TransformerLM
from ..data import LanguageModelingDataset, create_dataloader, create_distributed_dataloader
from .optimizer import create_optimizer, GradientClipping, ExponentialMovingAverage
from .scheduler import create_scheduler
from .utils import (
    set_seed, get_device, setup_logging, save_checkpoint, load_checkpoint,
    cleanup_checkpoints, MetricsTracker, log_model_info, get_memory_usage,
    calculate_eta, format_time
)


class Trainer:
    """Main trainer class for language model training."""
    
    def __init__(self,
                 model: TransformerLM,
                 tokenizer,
                 config: TrainingConfig,
                 train_dataset: Optional[LanguageModelingDataset] = None,
                 eval_dataset: Optional[LanguageModelingDataset] = None,
                 data_config: Optional[DataConfig] = None):
        
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.data_config = data_config
        
        # Setup logging
        setup_logging(config.log_level)
        self.logger = logging.getLogger(__name__)
        
        # Set random seed
        set_seed(config.seed)
        
        # Setup device and distributed training
        self.device = get_device()
        self.is_distributed = config.world_size > 1
        self.local_rank = config.local_rank
        
        # Move model to device
        self.model = self.model.to(self.device)
        
        # Setup distributed training
        if self.is_distributed:
            self._setup_distributed()
        
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
        self.use_amp = config.use_amp
        self.scaler = None
        if self.use_amp:
            self.scaler = torch.cuda.amp.GradScaler()
        
        # Setup datasets and dataloaders
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
        
        self.logger.info(f"Trainer initialized with {self.total_steps} total training steps")
    
    def _setup_distributed(self):
        """Setup distributed training."""
        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(
                backend=self.config.distributed_backend,
                rank=self.local_rank,
                world_size=self.config.world_size
            )
        
        # Wrap model for distributed training
        self.model = torch.nn.parallel.DistributedDataParallel(
            self.model,
            device_ids=[self.local_rank] if torch.cuda.is_available() else None,
            find_unused_parameters=False
        )
        
        self.logger.info(f"Distributed training setup: rank {self.local_rank}/{self.config.world_size}")
    
    def _setup_dataloaders(self):
        """Setup training and evaluation dataloaders."""
        if self.is_distributed:
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
                steps_per_epoch = len(self.train_dataset) // (
                    self.config.batch_size * self.config.gradient_accumulation_steps * self.config.world_size
                )
                return steps_per_epoch * self.config.num_epochs
            else:
                return 1000  # Default fallback

        steps_per_epoch = len(self.train_dataloader) // self.config.gradient_accumulation_steps
        return steps_per_epoch * self.config.num_epochs

    def _setup_monitoring(self):
        """Setup monitoring and logging."""
        self.writers = []

        # Setup TensorBoard
        if "tensorboard" in self.config.report_to:
            try:
                from torch.utils.tensorboard import SummaryWriter
                log_dir = os.path.join(self.config.checkpoint_dir, "logs")
                self.tb_writer = SummaryWriter(log_dir)
                self.writers.append("tensorboard")
                self.logger.info(f"TensorBoard logging to {log_dir}")
            except ImportError:
                self.logger.warning("TensorBoard not available")

        # Setup Weights & Biases
        if "wandb" in self.config.report_to:
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

    def train(self) -> None:
        """Main training loop."""
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
                self.train_dataloader.sampler.set_epoch(epoch)

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

        for batch_idx, batch in enumerate(pbar):
            # Move batch to device
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()}

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
                    eval_loss = self._evaluate()
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
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'lr': f'{self.optimizer.param_groups[0]["lr"]:.2e}',
                    'step': self.current_step
                })

        return epoch_loss / num_batches if num_batches > 0 else 0.0

    def _training_step(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Perform a single training step."""
        if self.use_amp:
            with torch.cuda.amp.autocast():
                outputs = self.model(**batch)
                loss = outputs["loss"]
        else:
            outputs = self.model(**batch)
            loss = outputs["loss"]

        # Scale loss for gradient accumulation
        loss = loss / self.config.gradient_accumulation_steps

        return loss

    def _backward_step(self, loss: torch.Tensor) -> None:
        """Perform backward pass."""
        if self.use_amp:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

    def _optimizer_step(self) -> None:
        """Perform optimizer step with gradient clipping."""
        if self.use_amp:
            # Unscale gradients for clipping
            self.scaler.unscale_(self.optimizer)

            # Clip gradients
            grad_norm = self.gradient_clipper.clip_gradients(self.model)

            # Optimizer step
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            # Clip gradients
            grad_norm = self.gradient_clipper.clip_gradients(self.model)

            # Optimizer step
            self.optimizer.step()

        # Zero gradients
        self.optimizer.zero_grad(set_to_none=True)

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
                if self.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(**batch)
                        loss = outputs["loss"]
                else:
                    outputs = self.model(**batch)
                    loss = outputs["loss"]

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
            return self.current_step % self.config.eval_steps == 0
        elif self.config.eval_strategy == "epoch":
            return True  # Evaluate at end of each epoch
        return False

    def _should_save_checkpoint(self) -> bool:
        """Check if checkpoint should be saved."""
        return self.current_step % self.config.save_steps == 0

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
            self.logger.info(
                f"Step {self.current_step}: Loss={loss:.4f}, "
                f"LR={self.optimizer.param_groups[0]['lr']:.2e}"
            )

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

        model_to_save.save_pretrained(best_model_path)
        self.tokenizer.save_pretrained(best_model_path)

        self.logger.info(f"Best model saved: {best_model_path}")

    def _load_checkpoint(self, checkpoint_path: str) -> None:
        """Load training checkpoint."""
        self.logger.info(f"Loading checkpoint: {checkpoint_path}")

        # Get model for loading (unwrap DDP if needed)
        model_to_load = self.model
        if hasattr(self.model, 'module'):
            model_to_load = self.model.module

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
            torch.distributed.destroy_process_group()

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

        if data_config.validation_split:
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
                    self.eval_dataset.dataset = split_dataset['test']
                    
                    print(f"Created validation set with {val_size} samples from training data.")
                else:
                    raise e            # Setup dataloaders
            self._setup_dataloaders()

            # Recalculate total steps
            self.total_steps = self._calculate_total_steps()

            # Recreate scheduler with correct total steps
            self.scheduler = create_scheduler(
                optimizer=self.optimizer,
                scheduler_name=self.config.lr_scheduler,
                num_training_steps=self.total_steps,
                warmup_steps=self.config.warmup_steps,
                min_lr_ratio=self.config.min_lr_ratio
            )

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
            generated_ids = self.model.generate(
                input_ids=input_ids,
                max_length=max_length,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
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
        """Save the trained model."""
        # Get model for saving (unwrap DDP if needed)
        model_to_save = self.model
        if hasattr(self.model, 'module'):
            model_to_save = self.model.module

        model_to_save.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)

        self.logger.info(f"Model saved to {save_path}")

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
