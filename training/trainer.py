"""
AetherMind - Main Training Loop
==================================
The core training loop for Forge-1 with all memory optimizations.

Features:
  - Mixed precision training (FP16/BF16)
  - Gradient accumulation for effective larger batch sizes
  - Gradient checkpointing (toggle via config)
  - Checkpoint save/resume
  - Rich training dashboard integration
  - VRAM monitoring and OOM recovery hints

Dependencies: torch, training/optimizer.py, training/checkpointing.py, 
              training/losses.py, training/callbacks.py
Used by: scripts/train.py
"""

import torch
import time
import math
import logging
import json
from pathlib import Path
from typing import Dict, Any, Optional, List
from torch.utils.data import DataLoader
from contextlib import nullcontext

from training.optimizer import create_optimizer, create_scheduler
from training.checkpointing import CheckpointManager
from training.losses import compute_lm_loss, compute_perplexity
from training.callbacks import TrainingCallback, LoggingCallback, MetricsTracker

logger = logging.getLogger(__name__)


class Trainer:
    """
    Main trainer for Forge-1 language model.
    
    Handles the complete training loop with:
    - Mixed precision (FP16/BF16 via torch.amp)
    - Gradient accumulation
    - Gradient clipping
    - Checkpoint save/resume
    - Metrics tracking and logging
    
    Args:
        model: The Forge1Model instance
        config: Training configuration dict
        train_dataloader: Training data DataLoader
        val_dataloader: Optional validation DataLoader
        callbacks: List of TrainingCallback instances
    """
    
    def __init__(self, model, config: Dict[str, Any],
                 train_dataloader: DataLoader,
                 val_dataloader: Optional[DataLoader] = None,
                 callbacks: Optional[List[TrainingCallback]] = None):
        
        self.config = config
        self.callbacks = callbacks or [LoggingCallback(), MetricsTracker()]
        
        # Device setup
        self.device = self._setup_device(config.get("hardware", {}).get("device", "cuda"))
        self.model = model.to(self.device)
        
        # Mixed precision setup
        self.precision = config.get("hardware", {}).get("precision", "fp16")
        self.scaler = None
        self.amp_dtype = torch.float32
        self._setup_mixed_precision()
        
        # Gradient checkpointing
        if config.get("hardware", {}).get("gradient_checkpointing", False):
            self.model.enable_gradient_checkpointing()
        
        # Training hyperparameters
        train_cfg = config.get("training", {})
        self.batch_size = train_cfg.get("batch_size", 2)
        self.grad_accum_steps = train_cfg.get("gradient_accumulation_steps", 8)
        self.num_epochs = train_cfg.get("num_epochs", 3)
        self.max_grad_norm = train_cfg.get("max_grad_norm", 1.0)
        self.log_every = config.get("logging", {}).get("log_every_n_steps", 10)
        
        # Data
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        
        # Calculate total steps
        steps_per_epoch = len(train_dataloader) // self.grad_accum_steps
        self.total_steps = steps_per_epoch * self.num_epochs
        
        # Optimizer and scheduler
        self.optimizer = create_optimizer(
            model,
            lr=train_cfg.get("learning_rate", 3e-4),
            weight_decay=train_cfg.get("weight_decay", 0.1),
            beta1=train_cfg.get("adam_beta1", 0.9),
            beta2=train_cfg.get("adam_beta2", 0.95),
            eps=train_cfg.get("adam_epsilon", 1e-8),
            use_8bit=train_cfg.get("use_8bit_optimizer", False),
        )
        self.scheduler = create_scheduler(
            self.optimizer,
            num_training_steps=self.total_steps,
            warmup_ratio=train_cfg.get("warmup_ratio", 0.02),
            scheduler_type=train_cfg.get("lr_scheduler", "cosine"),
        )
        
        # Checkpoint manager
        ckpt_cfg = config.get("checkpointing", {})
        self.ckpt_manager = CheckpointManager(
            output_dir=ckpt_cfg.get("output_dir", "outputs/checkpoints"),
            keep_last_n=ckpt_cfg.get("keep_last_n", 3),
        )
        self.save_every = ckpt_cfg.get("save_every_n_steps", 500)
        
        # State
        self.global_step = 0
        self.current_epoch = 0
        self.loss_history: List[float] = []
        self.best_loss = float("inf")
        
        logger.info(f"Trainer initialized:")
        logger.info(f"  Device: {self.device}")
        logger.info(f"  Precision: {self.precision}")
        logger.info(f"  Batch size: {self.batch_size} x {self.grad_accum_steps} accum = "
                     f"{self.batch_size * self.grad_accum_steps} effective")
        logger.info(f"  Total steps: {self.total_steps}")
        logger.info(f"  Steps per epoch: {steps_per_epoch}")
    
    def _setup_device(self, device_str: str) -> torch.device:
        """Auto-detect and setup compute device."""
        if device_str == "cuda" and torch.cuda.is_available():
            device = torch.device("cuda")
            gpu_name = torch.cuda.get_device_name(0)
            vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            logger.info(f"Using GPU: {gpu_name} ({vram_gb:.1f} GB VRAM)")
            return device
        else:
            logger.warning("CUDA not available, using CPU (training will be very slow)")
            return torch.device("cpu")
    
    def _setup_mixed_precision(self):
        """Configure mixed precision training."""
        if self.device.type != "cuda":
            self.precision = "fp32"
            return
        
        if self.precision == "bf16":
            if torch.cuda.is_bf16_supported():
                self.amp_dtype = torch.bfloat16
                logger.info("Using BF16 mixed precision")
            else:
                logger.warning("BF16 not supported, falling back to FP16")
                self.precision = "fp16"
                self.amp_dtype = torch.float16
                self.scaler = torch.amp.GradScaler("cuda")
        elif self.precision == "fp16":
            self.amp_dtype = torch.float16
            self.scaler = torch.amp.GradScaler("cuda")
            logger.info("Using FP16 mixed precision with GradScaler")
        else:
            logger.info("Using FP32 (no mixed precision)")
    
    def _get_amp_context(self):
        """Get the appropriate autocast context manager."""
        if self.precision in ("fp16", "bf16"):
            return torch.amp.autocast("cuda", dtype=self.amp_dtype)
        return nullcontext()
    
    def train(self) -> Dict[str, Any]:
        """
        Run the full training loop.
        
        Returns:
            Dict with training results and metrics
        """
        state = {"model": self.model, "config": self.config}
        for cb in self.callbacks:
            cb.on_train_start(state)
        
        # Try to resume from checkpoint
        self._try_resume()
        
        logger.info("=" * 60)
        logger.info("Starting training")
        logger.info("=" * 60)
        
        try:
            for epoch in range(self.current_epoch, self.num_epochs):
                self.current_epoch = epoch
                for cb in self.callbacks:
                    cb.on_epoch_start(epoch, state)
                
                self._train_epoch(epoch)
                
                for cb in self.callbacks:
                    cb.on_epoch_end(epoch, state)
                
                # Validation at end of epoch
                if self.val_dataloader:
                    val_loss = self._validate()
                    logger.info(f"Epoch {epoch+1}/{self.num_epochs} | Val Loss: {val_loss:.4f}")
        
        except KeyboardInterrupt:
            logger.info("\nTraining interrupted by user. Saving checkpoint...")
            self._save_checkpoint()
            logger.info("Checkpoint saved. Resume training later to continue.")
        
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                logger.error("\n" + "=" * 60)
                logger.error("OUT OF MEMORY ERROR!")
                logger.error("=" * 60)
                logger.error("Try these fixes:")
                logger.error("  1. Reduce batch_size in config (current: {})".format(self.batch_size))
                logger.error("  2. Reduce context_length in config")
                logger.error("  3. Enable gradient_checkpointing: true")
                logger.error("  4. Use precision: fp16 or bf16")
                logger.error("  5. Increase gradient_accumulation_steps")
                if torch.cuda.is_available():
                    logger.error(f"  VRAM: {torch.cuda.memory_allocated()/1e9:.1f}GB / "
                                f"{torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB")
                raise
            raise
        
        # Save final checkpoint
        self._save_checkpoint()
        
        for cb in self.callbacks:
            cb.on_train_end(state)
        
        return {
            "final_loss": self.loss_history[-1] if self.loss_history else None,
            "total_steps": self.global_step,
            "epochs_completed": self.current_epoch + 1,
        }
    
    def _train_epoch(self, epoch: int):
        """Train for one epoch."""
        self.model.train()
        self.optimizer.zero_grad()
        
        epoch_loss = 0.0
        step_in_epoch = 0
        tokens_processed = 0
        epoch_start = time.time()
        step_start = time.time()
        
        for batch_idx, batch in enumerate(self.train_dataloader):
            # Move batch to device
            input_ids = batch["input_ids"].to(self.device)
            labels = batch["labels"].to(self.device)
            
            # Forward pass with mixed precision
            with self._get_amp_context():
                outputs = self.model(input_ids=input_ids, labels=labels)
                loss = outputs["loss"] / self.grad_accum_steps
            
            # Backward pass
            if self.scaler:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # Track tokens
            tokens_processed += input_ids.numel()
            epoch_loss += loss.item() * self.grad_accum_steps
            
            # Gradient accumulation step
            if (batch_idx + 1) % self.grad_accum_steps == 0:
                # Gradient clipping
                if self.scaler:
                    self.scaler.unscale_(self.optimizer)
                
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.max_grad_norm
                )
                
                # Optimizer step
                if self.scaler:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                
                self.scheduler.step()
                self.optimizer.zero_grad()
                self.global_step += 1
                step_in_epoch += 1
                
                # Track loss
                actual_loss = loss.item() * self.grad_accum_steps
                self.loss_history.append(actual_loss)
                
                # Calculate speed
                step_time = time.time() - step_start
                tokens_per_sec = tokens_processed / max(step_time, 0.001)
                
                # Callback
                state = {
                    "loss": actual_loss,
                    "lr": self.scheduler.get_last_lr()[0],
                    "epoch": epoch,
                    "step": self.global_step,
                    "tokens_per_sec": tokens_per_sec,
                }
                for cb in self.callbacks:
                    cb.on_step_end(self.global_step, state)
                
                # Reset for next step
                tokens_processed = 0
                step_start = time.time()
                
                # Save checkpoint
                if self.global_step % self.save_every == 0:
                    self._save_checkpoint()
        
        epoch_time = time.time() - epoch_start
        avg_loss = epoch_loss / max(step_in_epoch * self.grad_accum_steps, 1)
        logger.info(f"Epoch {epoch+1} complete | Avg Loss: {avg_loss:.4f} | "
                    f"Time: {epoch_time/60:.1f}min")
    
    @torch.no_grad()
    def _validate(self) -> float:
        """Run validation and return average loss."""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        for batch in self.val_dataloader:
            input_ids = batch["input_ids"].to(self.device)
            labels = batch["labels"].to(self.device)
            
            with self._get_amp_context():
                outputs = self.model(input_ids=input_ids, labels=labels)
            
            total_loss += outputs["loss"].item()
            num_batches += 1
        
        return total_loss / max(num_batches, 1)
    
    def _save_checkpoint(self):
        """Save current training state."""
        self.ckpt_manager.save(
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            step=self.global_step,
            epoch=self.current_epoch,
            loss_history=self.loss_history,
            config=self.config,
        )
    
    def _try_resume(self):
        """Try to resume from the latest checkpoint."""
        checkpoint = self.ckpt_manager.load_latest()
        if checkpoint:
            self.global_step, self.current_epoch, self.loss_history = \
                self.ckpt_manager.resume(
                    self.model, self.optimizer, self.scheduler, checkpoint
                )
            logger.info(f"Resumed training from step {self.global_step}")
