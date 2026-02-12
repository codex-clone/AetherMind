"""
AetherMind - Forge-1 Model Architecture
========================================
Decoder-only transformer in the LLaMA/Mistral style.

Architecture features:
  - RMSNorm (pre-normalization, more efficient than LayerNorm)
  - Grouped Query Attention with RoPE
  - SwiGLU Feed-Forward Networks
  - Optional gradient checkpointing for memory savings
  - Weight tying between input embeddings and output head

Dependencies:
  - torch, model/attention.py, model/feedforward.py
  
Used by: training/trainer.py, inference/generate.py, scripts/train.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import math
from dataclasses import dataclass, field
from typing import Optional, Tuple

from model.attention import GroupedQueryAttention
from model.feedforward import SwiGLUFeedForward

logger = logging.getLogger(__name__)


@dataclass
class Forge1Config:
    """Configuration for the Forge-1 model architecture.
    
    Args:
        num_layers: Number of transformer decoder layers
        hidden_dim: Model hidden dimension (embedding size)
        num_heads: Number of query attention heads
        num_kv_heads: Number of KV heads for GQA
        intermediate_dim: FFN intermediate dimension
        context_length: Maximum sequence length
        vocab_size: Tokenizer vocabulary size
        dropout: Dropout rate (0.0 for pretraining)
        rope_theta: RoPE base frequency
        norm_eps: RMSNorm epsilon
        tie_word_embeddings: Whether to tie input/output embeddings
        gradient_checkpointing: Whether to use activation checkpointing
    """
    num_layers: int = 12
    hidden_dim: int = 768
    num_heads: int = 12
    num_kv_heads: int = 4
    intermediate_dim: int = 2048
    context_length: int = 1024
    vocab_size: int = 50304
    dropout: float = 0.0
    rope_theta: float = 10000.0
    norm_eps: float = 1e-5
    tie_word_embeddings: bool = True
    gradient_checkpointing: bool = False
    
    @classmethod
    def nano(cls) -> "Forge1Config":
        """125M parameter Nano variant for 4GB VRAM."""
        return cls(
            num_layers=12, hidden_dim=768, num_heads=12, num_kv_heads=4,
            intermediate_dim=2048, context_length=1024, vocab_size=50304,
        )
    
    @classmethod
    def mini(cls) -> "Forge1Config":
        """350M parameter Mini variant for 15GB VRAM."""
        return cls(
            num_layers=24, hidden_dim=1024, num_heads=16, num_kv_heads=4,
            intermediate_dim=2816, context_length=1024, vocab_size=50304,
        )
    
    @classmethod
    def from_variant(cls, variant: str) -> "Forge1Config":
        """Create config from variant name string."""
        variants = {"nano": cls.nano, "mini": cls.mini}
        if variant not in variants:
            raise ValueError(f"Unknown variant {variant}. Choose from: {list(variants.keys())}")
        return variants[variant]()
    
    def count_parameters_estimate(self) -> int:
        """Rough estimate of total parameters."""
        embed = self.vocab_size * self.hidden_dim
        attn_per_layer = (
            self.hidden_dim * self.num_heads * (self.hidden_dim // self.num_heads) +  # Q
            self.hidden_dim * self.num_kv_heads * (self.hidden_dim // self.num_heads) * 2 +  # K, V
            self.hidden_dim * self.hidden_dim  # O
        )
        ffn_per_layer = self.hidden_dim * self.intermediate_dim * 3  # gate, up, down
        norm_per_layer = self.hidden_dim * 2  # 2 RMSNorms per layer
        total = embed + self.num_layers * (attn_per_layer + ffn_per_layer + norm_per_layer)
        if not self.tie_word_embeddings:
            total += embed  # output projection
        return total


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization.
    
    More efficient than LayerNorm - skips mean centering, only does
    RMS re-scaling. Used in LLaMA, Mistral, and most modern LLMs.
    
    Args:
        dim: Dimension to normalize over (hidden_dim)
        eps: Small epsilon for numerical stability
    """
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))  # Learnable scale
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Compute RMS: sqrt(mean(x^2) + eps)
        rms = torch.sqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        # Normalize and scale
        return (x / rms) * self.weight


class TransformerBlock(nn.Module):
    """Single transformer decoder block with pre-normalization.
    
    Architecture (pre-norm style, same as LLaMA):
        x -> RMSNorm -> GQA -> residual_add -> RMSNorm -> SwiGLU_FFN -> residual_add
    
    Args:
        config: Forge1Config with model hyperparameters
    """
    def __init__(self, config: Forge1Config):
        super().__init__()
        self.attn_norm = RMSNorm(config.hidden_dim, config.norm_eps)
        self.attention = GroupedQueryAttention(
            hidden_dim=config.hidden_dim,
            num_heads=config.num_heads,
            num_kv_heads=config.num_kv_heads,
            max_seq_len=config.context_length,
            rope_theta=config.rope_theta,
            dropout=config.dropout,
        )
        self.ffn_norm = RMSNorm(config.hidden_dim, config.norm_eps)
        self.feed_forward = SwiGLUFeedForward(
            hidden_dim=config.hidden_dim,
            intermediate_dim=config.intermediate_dim,
            dropout=config.dropout,
        )
    
    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None,
                position_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Pre-norm attention with residual connection
        h = x + self.attention(self.attn_norm(x), attention_mask, position_ids)
        # Pre-norm FFN with residual connection
        out = h + self.feed_forward(self.ffn_norm(h))
        return out


class Forge1Model(nn.Module):
    """
    Forge-1: A decoder-only transformer language model.
    
    Designed for chain-of-thought reasoning with special thinking tokens.
    Follows the LLaMA architecture: RMSNorm + GQA + SwiGLU + RoPE.
    
    Args:
        config: Forge1Config defining the model architecture
    """
    def __init__(self, config: Forge1Config):
        super().__init__()
        self.config = config
        
        # Token embedding layer
        self.token_embedding = nn.Embedding(config.vocab_size, config.hidden_dim)
        
        # Transformer decoder layers
        self.layers = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.num_layers)
        ])
        
        # Final normalization before output projection
        self.final_norm = RMSNorm(config.hidden_dim, config.norm_eps)
        
        # Output projection (language model head)
        self.lm_head = nn.Linear(config.hidden_dim, config.vocab_size, bias=False)
        
        # Optionally tie input/output embeddings (saves parameters)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.token_embedding.weight
        
        # Gradient checkpointing flag
        self.gradient_checkpointing = config.gradient_checkpointing
        
        # Initialize weights
        self._init_weights()
        
        # Log model info
        n_params = sum(p.numel() for p in self.parameters())
        logger.info(f"Forge-1 initialized: {n_params/1e6:.1f}M parameters, "
                    f"{config.num_layers} layers, {config.hidden_dim} hidden_dim")
    
    def _init_weights(self):
        """Initialize weights using standard transformer initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # Normal init with std scaled by depth
                std = 0.02
                nn.init.normal_(module.weight, mean=0.0, std=std)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def enable_gradient_checkpointing(self):
        """Enable gradient checkpointing for memory savings."""
        self.gradient_checkpointing = True
        logger.info("Gradient checkpointing enabled")
    
    def disable_gradient_checkpointing(self):
        """Disable gradient checkpointing."""
        self.gradient_checkpointing = False
    
    def forward(self, input_ids: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                position_ids: Optional[torch.Tensor] = None,
                labels: Optional[torch.Tensor] = None,
                ) -> dict:
        """
        Forward pass through the model.
        
        Args:
            input_ids: Token IDs, shape (batch_size, seq_len)
            attention_mask: Optional attention mask
            position_ids: Optional position IDs for RoPE
            labels: Optional labels for loss computation (shifted internally)
        
        Returns:
            Dict with keys: "logits", optionally "loss"
        """
        # Get token embeddings
        x = self.token_embedding(input_ids)  # (batch, seq_len, hidden_dim)
        
        # Pass through transformer layers
        for layer in self.layers:
            if self.gradient_checkpointing and self.training:
                # Use gradient checkpointing to save memory
                x = torch.utils.checkpoint.checkpoint(
                    layer, x, attention_mask, position_ids,
                    use_reentrant=False,
                )
            else:
                x = layer(x, attention_mask, position_ids)
        
        # Final normalization
        x = self.final_norm(x)
        
        # Project to vocabulary logits
        logits = self.lm_head(x)  # (batch, seq_len, vocab_size)
        
        result = {"logits": logits}
        
        # Compute loss if labels provided
        if labels is not None:
            # Shift logits and labels for next-token prediction
            # logits: predict token at position i+1 using position i
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            loss = F.cross_entropy(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1),
                ignore_index=-100,  # Standard ignore index for padding
            )
            result["loss"] = loss
        
        return result
    
    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def estimate_memory_mb(self, batch_size: int = 1, seq_len: int = 1024,
                           precision: str = "fp16") -> dict:
        """Estimate memory usage in MB."""
        n_params = self.count_parameters()
        bytes_per_param = 2 if precision in ("fp16", "bf16") else 4
        
        model_mb = (n_params * bytes_per_param) / (1024**2)
        # Rough activation estimate
        act_mb = (batch_size * seq_len * self.config.hidden_dim * 
                  self.config.num_layers * bytes_per_param * 2) / (1024**2)
        # Optimizer states (AdamW needs 2x model size for moments)
        opt_mb = (n_params * 4 * 2) / (1024**2)  # Always fp32 for optimizer
        
        return {
            "model_mb": round(model_mb, 1),
            "activations_mb": round(act_mb, 1),
            "optimizer_mb": round(opt_mb, 1),
            "total_mb": round(model_mb + act_mb + opt_mb, 1),
        }
