"""
AetherMind - Checkpoint Management
====================================
Save and load training checkpoints for resumable training.

A checkpoint includes: model weights, optimizer state, scheduler state,
training step, epoch, loss history, and config.

Dependencies: torch, safetensors
Used by: training/trainer.py, scripts/train.py
"""

import torch
import json
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List

logger = logging.getLogger(__name__)


class CheckpointManager:
    """
    Manages saving and loading of training checkpoints.
    
    Keeps only the last N checkpoints to save disk space.
    
    Args:
        output_dir: Directory to save checkpoints
        keep_last_n: Number of recent checkpoints to keep
    """
    
    def __init__(self, output_dir: str, keep_last_n: int = 3):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.keep_last_n = keep_last_n
    
    def save(self, model: torch.nn.Module, optimizer: torch.optim.Optimizer,
             scheduler, step: int, epoch: int,
             loss_history: List[float], config: Dict[str, Any]) -> str:
        """
        Save a training checkpoint.
        
        Returns:
            Path to the saved checkpoint file
        """
        ckpt_path = self.output_dir / f"step_{step}.pt"
        
        checkpoint = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
            "step": step,
            "epoch": epoch,
            "loss_history": loss_history[-1000:],  # Keep last 1000 loss values
            "config": config,
        }
        
        torch.save(checkpoint, ckpt_path)
        logger.info(f"Checkpoint saved: {ckpt_path}")
        
        # Clean up old checkpoints
        self._cleanup()
        
        return str(ckpt_path)
    
    def load_latest(self) -> Optional[Dict[str, Any]]:
        """Find and load the most recent checkpoint."""
        ckpts = sorted(self.output_dir.glob("step_*.pt"))
        if not ckpts:
            logger.info("No checkpoints found, starting from scratch")
            return None
        
        latest = ckpts[-1]
        logger.info(f"Loading checkpoint: {latest}")
        return torch.load(latest, map_location="cpu", weights_only=False)
    
    def load(self, path: str) -> Dict[str, Any]:
        """Load a specific checkpoint file."""
        logger.info(f"Loading checkpoint: {path}")
        return torch.load(path, map_location="cpu", weights_only=False)
    
    def resume(self, model, optimizer, scheduler, checkpoint):
        """Restore model/optimizer/scheduler state from checkpoint dict."""
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if scheduler and checkpoint.get("scheduler_state_dict"):
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        
        step = checkpoint["step"]
        epoch = checkpoint["epoch"]
        loss_history = checkpoint.get("loss_history", [])
        
        logger.info(f"Resumed from step {step}, epoch {epoch}")
        return step, epoch, loss_history
    
    def _cleanup(self):
        """Delete old checkpoints, keeping only the last N."""
        ckpts = sorted(self.output_dir.glob("step_*.pt"))
        if len(ckpts) > self.keep_last_n:
            for old in ckpts[:-self.keep_last_n]:
                old.unlink()
                logger.debug(f"Deleted old checkpoint: {old}")
