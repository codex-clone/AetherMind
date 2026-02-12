"""
AetherMind - Loss Functions
=============================
Loss functions for training Forge-1.

Primary loss: Cross-entropy on next-token prediction.
Optional: Weighted loss that reduces weight on thinking tokens.

Dependencies: torch
Used by: training/trainer.py
"""

import torch
import torch.nn.functional as F
import logging

logger = logging.getLogger(__name__)


def compute_lm_loss(logits: torch.Tensor, labels: torch.Tensor,
                    ignore_index: int = -100) -> torch.Tensor:
    """
    Standard language modeling loss (cross-entropy on next-token prediction).
    
    Args:
        logits: Model output logits, shape (batch, seq_len, vocab_size)
        labels: Target token IDs, shape (batch, seq_len)
        ignore_index: Label value to ignore in loss (padding)
    
    Returns:
        Scalar loss tensor
    """
    # Shift: predict token i+1 from position i
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    
    loss = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=ignore_index,
    )
    return loss


def compute_perplexity(loss: torch.Tensor) -> float:
    """Compute perplexity from loss value."""
    return torch.exp(loss).item()
