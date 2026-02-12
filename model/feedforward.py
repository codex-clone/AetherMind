"""
AetherMind - Feed-Forward Network (SwiGLU)
==========================================
Implements the SwiGLU (Swish-Gated Linear Unit) feed-forward block used in
modern transformer architectures like LLaMA, Mistral, and Qwen.

Why SwiGLU over standard FFN (ReLU/GELU):
  - Empirically better performance on language modeling tasks
  - The gating mechanism allows the network to learn which features to amplify
  - Used in all state-of-the-art open-source LLMs (LLaMA 2/3, Mistral, etc.)
  - SwiGLU(x) = (Swish(xW_gate) * xW_up) @ W_down
  - Swish(x) = x * sigmoid(x), also known as SiLU in PyTorch

Architecture:
  Input (hidden_dim) -> gate_proj (intermediate_dim) -> SiLU activation
                     -> up_proj (intermediate_dim) -> element-wise multiply
                     -> down_proj (hidden_dim) -> Output

Note: SwiGLU uses 3 weight matrices instead of 2, so for the same parameter
count as a standard FFN with 4x expansion, SwiGLU uses ~2.67x expansion.

Dependencies:
  - torch

Used by:
  - model/architecture.py (inside each transformer layer)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SwiGLUFeedForward(nn.Module):
    """
    SwiGLU Feed-Forward Network block.

    Computes: output = down_proj(SiLU(gate_proj(x)) * up_proj(x))

    This is the standard FFN used in LLaMA-style architectures. The gating
    mechanism (SiLU activation on the gate path, multiplied with the up path)
    allows the network to learn adaptive feature selection.

    Args:
        hidden_dim: Input and output dimension (model's hidden size).
        intermediate_dim: Inner dimension of the FFN. Typically ~2.67x hidden_dim
                          for SwiGLU (vs 4x for standard ReLU FFN) to maintain
                          similar parameter count.
        dropout: Dropout rate applied after the down projection. Default: 0.0
    """

    def __init__(
        self,
        hidden_dim: int,
        intermediate_dim: int,
        dropout: float = 0.0,
    ):
        super().__init__()

        # Gate projection: projects input to intermediate dim, then applies SiLU
        # This acts as a learned "gate" that controls information flow
        self.gate_proj = nn.Linear(hidden_dim, intermediate_dim, bias=False)

        # Up projection: projects input to intermediate dim (no activation)
        # This is the "value" path that gets gated
        self.up_proj = nn.Linear(hidden_dim, intermediate_dim, bias=False)

        # Down projection: projects back from intermediate to hidden dim
        # Reduces dimensionality back to the model's hidden size
        self.down_proj = nn.Linear(intermediate_dim, hidden_dim, bias=False)

        # Dropout for regularization (usually 0.0 during pretraining)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the SwiGLU FFN.

        Args:
            x: Input tensor of shape (batch_size, seq_len, hidden_dim)

        Returns:
            Output tensor of shape (batch_size, seq_len, hidden_dim)
        """
        # Gate path: apply SiLU (Swish) activation
        # SiLU(x) = x * sigmoid(x) — smooth, non-monotonic activation
        gate = F.silu(self.gate_proj(x))

        # Up path: linear projection without activation
        up = self.up_proj(x)

        # Element-wise gating: gate controls which features from 'up' pass through
        # This is the key innovation of GLU-style architectures
        gated = gate * up

        # Down projection back to hidden_dim
        output = self.down_proj(gated)

        # Apply dropout (identity if dropout=0.0)
        output = self.dropout(output)

        return output
