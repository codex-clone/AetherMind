"""
AetherMind - Text Generation
==============================
Autoregressive text generation with various sampling strategies.

Supports: greedy, top-k, top-p (nucleus), temperature scaling.
Token streaming for real-time output.

Dependencies: torch, model/architecture.py, model/tokenizer.py
Used by: inference/thinking_pipeline.py, scripts/chat.py
"""

import torch
import torch.nn.functional as F
import logging
from typing import Optional, List, Generator

logger = logging.getLogger(__name__)


@torch.no_grad()
def generate(model, tokenizer, prompt: str, max_new_tokens: int = 512,
             temperature: float = 0.7, top_k: int = 50, top_p: float = 0.9,
             do_sample: bool = True, device: str = "cuda") -> str:
    """
    Generate text autoregressively from a prompt.
    
    Args:
        model: Forge1Model instance
        tokenizer: Forge1Tokenizer instance
        prompt: Input text prompt
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature (higher = more random)
        top_k: Keep only top-k logits before sampling
        top_p: Nucleus sampling threshold
        do_sample: If False, use greedy decoding
        device: Device to run on
    
    Returns:
        Generated text string (including prompt)
    """
    model.eval()
    input_ids = tokenizer.encode(prompt, add_special_tokens=False)
    input_ids = torch.tensor([input_ids], dtype=torch.long, device=device)
    
    eos_id = tokenizer.eos_token_id
    generated = list(input_ids[0].cpu().numpy())
    
    for _ in range(max_new_tokens):
        # Truncate to context length
        ctx_len = model.config.context_length
        if input_ids.size(1) > ctx_len:
            input_ids = input_ids[:, -ctx_len:]
        
        # Forward pass
        outputs = model(input_ids=input_ids)
        logits = outputs["logits"][:, -1, :]  # Last token logits
        
        # Sample next token
        next_token = _sample_token(logits, temperature, top_k, top_p, do_sample)
        
        # Check for EOS
        if next_token == eos_id:
            break
        
        # Append and continue
        generated.append(next_token)
        input_ids = torch.cat([
            input_ids,
            torch.tensor([[next_token]], dtype=torch.long, device=device)
        ], dim=1)
    
    return tokenizer.decode(generated)


@torch.no_grad()
def generate_stream(model, tokenizer, prompt: str, max_new_tokens: int = 512,
                    temperature: float = 0.7, top_k: int = 50, top_p: float = 0.9,
                    device: str = "cuda") -> Generator[str, None, None]:
    """
    Stream tokens one at a time (generator). Yields each new token as text.
    """
    model.eval()
    input_ids = tokenizer.encode(prompt, add_special_tokens=False)
    input_ids = torch.tensor([input_ids], dtype=torch.long, device=device)
    
    eos_id = tokenizer.eos_token_id
    
    for _ in range(max_new_tokens):
        ctx_len = model.config.context_length
        if input_ids.size(1) > ctx_len:
            input_ids = input_ids[:, -ctx_len:]
        
        outputs = model(input_ids=input_ids)
        logits = outputs["logits"][:, -1, :]
        
        next_token = _sample_token(logits, temperature, top_k, top_p, True)
        
        if next_token == eos_id:
            break
        
        token_text = tokenizer.decode([next_token])
        yield token_text
        
        input_ids = torch.cat([
            input_ids,
            torch.tensor([[next_token]], dtype=torch.long, device=device)
        ], dim=1)


def _sample_token(logits: torch.Tensor, temperature: float = 1.0,
                  top_k: int = 0, top_p: float = 1.0,
                  do_sample: bool = True) -> int:
    """Sample a single token from logits with temp/top-k/top-p."""
    if not do_sample:
        return logits.argmax(dim=-1).item()
    
    # Temperature scaling
    if temperature != 1.0:
        logits = logits / temperature
    
    # Top-k filtering
    if top_k > 0:
        top_k = min(top_k, logits.size(-1))
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = float("-inf")
    
    # Top-p (nucleus) filtering
    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        
        # Remove tokens with cumulative probability above threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        
        indices_to_remove = sorted_indices_to_remove.scatter(
            -1, sorted_indices, sorted_indices_to_remove
        )
        logits[indices_to_remove] = float("-inf")
    
    # Sample from distribution
    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1).item()
