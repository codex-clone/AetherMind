"""
AetherMind - Rotary Position Embeddings (RoPE)
==============================================
Implements Rotary Position Embeddings as described in the RoFormer paper
(Su et al., 2021). RoPE encodes position information by rotating query and
key vectors in the complex plane, which naturally captures relative positions
through the dot product.

Why RoPE over learned/sinusoidal embeddings:
  - Encodes RELATIVE position (not absolute) — better for length generalization
  - No extra parameters to learn — position info comes from rotation
  - Compatible with linear attention and KV caching
  - Used by LLaMA, Mistral, Qwen, and most modern LLMs

Dependencies:
  - torch
  - einops (optional, for clarity)

Used by:
  - model/attention.py (applied to Q and K before attention computation)
"""

import torch
import torch.nn as nn
from typing import Tuple


class RotaryPositionalEmbedding(nn.Module):
    """
    Rotary Position Embedding (RoPE) module.

    Precomputes sin/cos frequency tables for a given dimension and max sequence
    length. These are applied to query and key tensors to inject position info.

    Args:
        dim: Head dimension (hidden_dim // num_heads). Must be even.
        max_seq_len: Maximum sequence length to precompute frequencies for.
        theta: Base frequency for the geometric progression. Default 10000.0
               (standard value from the original paper).
    """

    def __init__(
        self,
        dim: int,
        max_seq_len: int = 2048,
        theta: float = 10000.0,
    ):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.theta = theta

        # Validate that dimension is even (required for pairing)
        assert dim % 2 == 0, f"RoPE dimension must be even, got {dim}"

        # Precompute the frequency table
        # Shape: (dim // 2,) — one frequency per pair of dimensions
        # Formula: freq_i = 1 / (theta^(2i/dim)) for i in [0, dim//2)
        inv_freq = 1.0 / (
            theta ** (torch.arange(0, dim, 2).float() / dim)
        )
        # Register as buffer so it moves with the model to GPU but isn't a parameter
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Precompute cos/sin for all positions up to max_seq_len
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int) -> None:
        """
        Build the cosine and sine cache for positions [0, seq_len).

        Args:
            seq_len: Number of positions to cache.
        """
        # Position indices: [0, 1, 2, ..., seq_len-1]
        positions = torch.arange(seq_len, dtype=self.inv_freq.dtype)

        # Outer product: (seq_len,) x (dim//2,) -> (seq_len, dim//2)
        # Each row has the angle for each frequency at that position
        angles = torch.outer(positions, self.inv_freq)

        # Duplicate angles for the full dimension: (seq_len, dim)
        # We need cos and sin for pairs, so repeat each angle
        emb = torch.cat([angles, angles], dim=-1)

        # Register cos/sin as buffers
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        position_ids: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply rotary embeddings to query and key tensors.

        Args:
            q: Query tensor of shape (batch, num_heads, seq_len, head_dim)
            k: Key tensor of shape (batch, num_kv_heads, seq_len, head_dim)
            position_ids: Optional position indices. If None, uses [0..seq_len).
                          Shape: (batch, seq_len) or (seq_len,)

        Returns:
            Tuple of (rotated_q, rotated_k) with same shapes as input.
        """
        seq_len = q.shape[2]

        # Extend cache if needed (handles sequences longer than initial max)
        if seq_len > self.max_seq_len:
            self._build_cache(seq_len)
            self.max_seq_len = seq_len

        if position_ids is not None:
            # Use custom position indices
            cos = self.cos_cached[position_ids].unsqueeze(1)  # (batch, 1, seq_len, dim)
            sin = self.sin_cached[position_ids].unsqueeze(1)
        else:
            # Use sequential positions [0, 1, ..., seq_len-1]
            cos = self.cos_cached[:seq_len].unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, dim)
            sin = self.sin_cached[:seq_len].unsqueeze(0).unsqueeze(0)

        # Apply rotation to Q and K
        q_rotated = self._apply_rotation(q, cos, sin)
        k_rotated = self._apply_rotation(k, cos, sin)

        return q_rotated, k_rotated

    @staticmethod
    def _apply_rotation(
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply the rotary transformation to a tensor.

        The rotation works by splitting the last dimension into pairs,
        then rotating each pair by the corresponding angle:
          [x0, x1] -> [x0*cos - x1*sin, x0*sin + x1*cos]

        This is equivalent to complex number multiplication:
          (x0 + ix1) * (cos + i*sin)

        Args:
            x: Input tensor of shape (..., dim) where dim is even.
            cos: Cosine values, broadcastable to x's shape.
            sin: Sine values, broadcastable to x's shape.

        Returns:
            Rotated tensor of the same shape as x.
        """
        # Split into first half and second half of dimensions
        # For dim=64: x1 = x[..., :32], x2 = x[..., 32:]
        half_dim = x.shape[-1] // 2
        x1 = x[..., :half_dim]
        x2 = x[..., half_dim:]

        # Create the "rotated" version: [-x2, x1]
        x_rotated = torch.cat([-x2, x1], dim=-1)

        # Apply rotation: x * cos + x_rotated * sin
        return x * cos + x_rotated * sin
