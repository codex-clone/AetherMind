"""
AetherMind - Training Callbacks
=================================
Event hooks for the training loop (logging, checkpointing, etc.)

Dependencies: logging
Used by: training/trainer.py
"""

import logging
import time
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


class TrainingCallback:
    """Base class for training callbacks."""
    
    def on_train_start(self, state: Dict[str, Any]) -> None:
        pass
    
    def on_train_end(self, state: Dict[str, Any]) -> None:
        pass
    
    def on_epoch_start(self, epoch: int, state: Dict[str, Any]) -> None:
        pass
    
    def on_epoch_end(self, epoch: int, state: Dict[str, Any]) -> None:
        pass
    
    def on_step_end(self, step: int, state: Dict[str, Any]) -> None:
        pass


class LoggingCallback(TrainingCallback):
    """Logs training metrics at regular intervals."""
    
    def __init__(self, log_every_n: int = 10):
        self.log_every_n = log_every_n
        self.step_start_time = None
    
    def on_step_end(self, step: int, state: Dict[str, Any]) -> None:
        if step % self.log_every_n == 0:
            loss = state.get("loss", 0)
            lr = state.get("lr", 0)
            tokens_per_sec = state.get("tokens_per_sec", 0)
            logger.info(
                f"Step {step} | Loss: {loss:.4f} | LR: {lr:.2e} | "
                f"Tok/s: {tokens_per_sec:.0f}"
            )


class MetricsTracker(TrainingCallback):
    """Tracks and stores training metrics over time."""
    
    def __init__(self):
        self.loss_history = []
        self.lr_history = []
        self.step_times = []
        self.start_time = None
    
    def on_train_start(self, state: Dict[str, Any]) -> None:
        self.start_time = time.time()
    
    def on_step_end(self, step: int, state: Dict[str, Any]) -> None:
        self.loss_history.append(state.get("loss", 0))
        self.lr_history.append(state.get("lr", 0))
    
    def get_avg_loss(self, last_n: int = 100) -> float:
        if not self.loss_history:
            return 0.0
        recent = self.loss_history[-last_n:]
        return sum(recent) / len(recent)
    
    def get_elapsed_time(self) -> float:
        if self.start_time is None:
            return 0.0
        return time.time() - self.start_time
