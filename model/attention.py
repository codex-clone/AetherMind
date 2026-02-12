"""
AetherMind - Grouped Query Attention (GQA) with RoPE
=====================================================
Implements Grouped Query Attention, a memory-efficient variant of Multi-Head
Attention where multiple query heads share the same key/value heads.

Why GQA over standard MHA:
  - Reduces KV cache memory by num_heads/num_kv_heads factor
  - Minimal quality loss compared to full MHA
  - Used in LLaMA 2 70B, LLaMA 3, Mistral, and most modern LLMs
  - When num_kv_heads == num_heads: equivalent to standard MHA
  - When num_kv_heads == 1: equivalent to Multi-Query Attention (MQA)

This module also integrates:
  - RoPE (Rotary Position Embeddings) for position encoding
  - PyTorch's scaled_dot_product_attention (SDPA) for efficient computation
  - Optional Flash Attention 2 backend (auto-detected by SDPA)

Dependencies:
  - torch
  - model/positional_encoding.py (RotaryPositionalEmbedding)

Used by:
  - model/architecture.py (inside each transformer layer)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import logging
from typing import Optional, Tuple

from model.positional_encoding import RotaryPositionalEmbedding

logger = logging.getLogger(__name__)


class GroupedQueryAttention(nn.Module):
    """
    Grouped Query Attention (GQA) with Rotary Position Embeddings.

    In GQA, query heads are divided into groups, and each group shares
    a single key/value head. This reduces memory usage for the KV cache
    during inference while maintaining most of the quality of full MHA.

    Args:
        hidden_dim: Model hidden dimension (embedding size).
        num_heads: Number of query attention heads.
        num_kv_heads: Number of key/value heads (must divide num_heads evenly).
        max_seq_len: Maximum sequence length for RoPE precomputation.
        rope_theta: Base frequency for RoPE. Default: 10000.0
        dropout: Attention dropout rate. Default: 0.0
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        num_kv_heads: int,
        max_seq_len: int = 2048,
        rope_theta: float = 10000.0,
        dropout: float = 0.0,
    ):
        super().__init__()

        # Validate configuration
        assert hidden_dim % num_heads == 0, (
            f"hidden_dim ({hidden_dim}) must be divisible by num_heads ({num_heads})"
        )
        assert num_heads % num_kv_heads == 0, (
            f"num_heads ({num_heads}) must be divisible by num_kv_heads ({num_kv_heads})"
        )

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = hidden_dim // num_heads  # Dimension per head
        self.num_groups = num_heads // num_kv_heads  # Q heads per KV head
        self.dropout = dropout

        # Query projection: projects to all query heads
        # Shape: (hidden_dim) -> (num_heads * head_dim)
        self.q_proj = nn.Linear(hidden_dim, num_heads * self.head_dim, bias=False)

        # Key projection: projects to KV heads only (fewer than Q heads)
        # Shape: (hidden_dim) -> (num_kv_heads * head_dim)
        self.k_proj = nn.Linear(hidden_dim, num_kv_heads * self.head_dim, bias=False)

        # Value projection: same size as key projection
        self.v_proj = nn.Linear(hidden_dim, num_kv_heads * self.head_dim, bias=False)

        # Output projection: combines all heads back to hidden_dim
        self.o_proj = nn.Linear(num_heads * self.head_dim, hidden_dim, bias=False)

        # Rotary Position Embeddings
        self.rope = RotaryPositionalEmbedding(
            dim=self.head_dim,
            max_seq_len=max_seq_len,
            theta=rope_theta,
        )

        # Check if SDPA Flash Attention is available
        self._check_sdpa_backends()

    def _check_sdpa_backends(self) -> None:
        """Log which SDPA backends are available for this configuration."""
        try:
            # Flash Attention requires head_dim to be a multiple of 8
            # and specific GPU architecture (Ampere+)
            if self.head_dim % 8 == 0:
                logger.debug("Head dim is compatible with Flash Attention backend")
            else:
                logger.debug(
                    f"Head dim {self.head_dim} not divisible by 8; "
                    "Flash Attention backend may not be used"
                )
        except Exception:
            pass  # Non-critical — SDPA will fall back automatically

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        is_causal: bool = True,
    ) -> torch.Tensor:
        """
        Forward pass through Grouped Query Attention.

        Args:
            x: Input tensor of shape (batch_size, seq_len, hidden_dim)
            attention_mask: Optional mask tensor. For causal/autoregressive
                            models, set is_causal=True instead of passing a mask.
                            Shape: (batch_size, 1, seq_len, seq_len) or broadcastable.
            position_ids: Optional position indices for RoPE.
                          Shape: (batch_size, seq_len)
            is_causal: If True, applies causal (autoregressive) masking.
                       This is more efficient than passing an explicit mask.

        Returns:
            Output tensor of shape (batch_size, seq_len, hidden_dim)
        """
        batch_size, seq_len, _ = x.shape

        # === Project to Q, K, V ===
        q = self.q_proj(x)  # (batch, seq_len, num_heads * head_dim)
        k = self.k_proj(x)  # (batch, seq_len, num_kv_heads * head_dim)
        v = self.v_proj(x)  # (batch, seq_len, num_kv_heads * head_dim)

        # === Reshape to (batch, num_heads, seq_len, head_dim) ===
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # === Apply Rotary Position Embeddings to Q and K ===
        # RoPE only applies to Q and K (not V) — it encodes position through
        # the rotation of query/key vectors in the complex plane
        q, k = self.rope(q, k, position_ids)

        # === Expand KV heads to match Q heads (GQA grouping) ===
        # If num_kv_heads < num_heads, we need to repeat KV heads
        # so each group of Q heads has matching K/V to attend to
        if self.num_groups > 1:
            # Repeat each KV head 'num_groups' times along the head dimension
            # (batch, num_kv_heads, seq, head_dim) -> (batch, num_heads, seq, head_dim)
            k = k.repeat_interleave(self.num_groups, dim=1)
            v = v.repeat_interleave(self.num_groups, dim=1)

        # === Compute Attention using PyTorch SDPA ===
        # SDPA automatically selects the best backend:
        # 1. Flash Attention 2 (fastest, if available)
        # 2. Memory-efficient attention (xformers-style)
        # 3. Standard scaled dot-product attention (fallback)
        dropout_p = self.dropout if self.training else 0.0

        attn_output = F.scaled_dot_product_attention(
            query=q,
            key=k,
            value=v,
            attn_mask=attention_mask,
            dropout_p=dropout_p,
            is_causal=is_causal if attention_mask is None else False,
        )
        # attn_output shape: (batch, num_heads, seq_len, head_dim)

        # === Reshape and project output ===
        # Transpose back: (batch, num_heads, seq_len, head_dim) -> (batch, seq_len, num_heads, head_dim)
        attn_output = attn_output.transpose(1, 2).contiguous()
        # Merge heads: (batch, seq_len, num_heads * head_dim)
        attn_output = attn_output.view(batch_size, seq_len, self.hidden_dim)
        # Final linear projection
        output = self.o_proj(attn_output)

        return output
