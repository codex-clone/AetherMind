"""
AetherMind - Optimizer & Scheduler Setup
=========================================
Creates the optimizer (AdamW or 8-bit AdamW) and learning rate scheduler
(warmup + cosine decay) for training Forge-1.

Dependencies: torch, (optional) bitsandbytes
Used by: training/trainer.py
"""

import torch
import math
import logging
from typing import Optional

logger = logging.getLogger(__name__)


def create_optimizer(model: torch.nn.Module, lr: float = 3e-4,
                     weight_decay: float = 0.1, beta1: float = 0.9,
                     beta2: float = 0.95, eps: float = 1e-8,
                     use_8bit: bool = False) -> torch.optim.Optimizer:
    """
    Create AdamW optimizer with proper weight decay handling.
    
    Weight decay is NOT applied to bias terms, norm weights, or embeddings
    (standard practice for transformer training).
    
    Args:
        model: The Forge-1 model
        lr: Learning rate
        weight_decay: Weight decay coefficient
        beta1: Adam first moment decay (0.9 standard)
        beta2: Adam second moment decay (0.95 for LLMs)
        eps: Epsilon for numerical stability
        use_8bit: Try to use bitsandbytes 8-bit AdamW (saves memory)
    
    Returns:
        Configured optimizer
    """
    # Separate parameters into decay and no-decay groups
    decay_params = []
    no_decay_params = []
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        # Don't decay bias, normalization weights, or embedding weights
        if "bias" in name or "norm" in name or "embedding" in name:
            no_decay_params.append(param)
        else:
            decay_params.append(param)
    
    param_groups = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]
    
    n_decay = sum(p.numel() for p in decay_params)
    n_no_decay = sum(p.numel() for p in no_decay_params)
    logger.info(f"Optimizer groups: {n_decay:,} decay params, {n_no_decay:,} no-decay params")
    
    # Try 8-bit optimizer if requested
    if use_8bit:
        try:
            import bitsandbytes as bnb
            optimizer = bnb.optim.AdamW8bit(
                param_groups, lr=lr, betas=(beta1, beta2), eps=eps,
            )
            logger.info("Using bitsandbytes 8-bit AdamW optimizer")
            return optimizer
        except ImportError:
            logger.warning("bitsandbytes not available, falling back to standard AdamW")
        except Exception as e:
            logger.warning(f"8-bit optimizer failed ({e}), falling back to standard AdamW")
    
    # Standard AdamW
    optimizer = torch.optim.AdamW(
        param_groups, lr=lr, betas=(beta1, beta2), eps=eps,
    )
    logger.info("Using standard AdamW optimizer")
    return optimizer


def create_scheduler(optimizer: torch.optim.Optimizer, num_training_steps: int,
                     warmup_ratio: float = 0.02,
                     scheduler_type: str = "cosine") -> torch.optim.lr_scheduler.LambdaLR:
    """
    Create learning rate scheduler with warmup + cosine decay.
    
    Args:
        optimizer: The optimizer
        num_training_steps: Total number of training steps
        warmup_ratio: Fraction of steps for linear warmup (0.02 = 2%)
        scheduler_type: "cosine" or "linear"
    
    Returns:
        LambdaLR scheduler
    """
    warmup_steps = int(num_training_steps * warmup_ratio)
    logger.info(f"LR schedule: {warmup_steps} warmup steps, "
                f"{num_training_steps - warmup_steps} decay steps ({scheduler_type})")
    
    def lr_lambda(current_step: int) -> float:
        # Warmup phase: linear increase from 0 to 1
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        
        # Decay phase
        progress = float(current_step - warmup_steps) / float(
            max(1, num_training_steps - warmup_steps)
        )
        
        if scheduler_type == "cosine":
            # Cosine decay from 1 to 0
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
        else:
            # Linear decay from 1 to 0
            return max(0.0, 1.0 - progress)
    
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
