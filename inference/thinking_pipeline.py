"""
AetherMind - Thinking Pipeline
================================
Wraps the generation module to extract and separate the thinking
(chain-of-thought) block from the final answer.

The model generates text in the format:
  <|think|>...reasoning...<|/think|>
  <|assistant|>...final answer...<|/assistant|>

This module parses the output and returns structured results.

Dependencies: inference/generate.py, model/tokenizer.py
Used by: scripts/chat.py, ui/chat_ui.py
"""

import time
import logging
from typing import Optional, Dict, Generator
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ThinkingResult:
    """Structured output from the thinking pipeline."""
    thinking: str          # Chain-of-thought reasoning
    answer: str            # Final answer text
    full_text: str         # Complete raw generated text
    num_tokens: int        # Total tokens generated
    generation_time: float # Time taken in seconds
    tokens_per_sec: float  # Generation speed
    
    @property
    def has_thinking(self) -> bool:
        return bool(self.thinking.strip())


def extract_thinking(text: str) -> tuple:
    """
    Parse generated text to extract thinking and answer blocks.
    
    Returns:
        (thinking_text, answer_text)
    """
    thinking = ""
    answer = text  # Default: entire text is the answer
    
    think_start = "<|think|>"
    think_end = "<|/think|>"
    asst_start = "<|assistant|>"
    asst_end = "<|/assistant|>"
    
    # Extract thinking block
    if think_start in text and think_end in text:
        ts_idx = text.index(think_start) + len(think_start)
        te_idx = text.index(think_end)
        thinking = text[ts_idx:te_idx].strip()
    
    # Extract assistant answer
    if asst_start in text:
        as_idx = text.index(asst_start) + len(asst_start)
        if asst_end in text:
            ae_idx = text.index(asst_end)
            answer = text[as_idx:ae_idx].strip()
        else:
            answer = text[as_idx:].strip()
    elif think_end in text:
        # Answer comes after thinking block
        te_idx = text.index(think_end) + len(think_end)
        answer = text[te_idx:].strip()
    
    return thinking, answer


def run_thinking_pipeline(model, tokenizer, user_message: str,
                          system_prompt: Optional[str] = None,
                          max_new_tokens: int = 512,
                          temperature: float = 0.7,
                          device: str = "cuda") -> ThinkingResult:
    """
    Run the full thinking pipeline: format prompt, generate, parse output.
    
    Args:
        model: Forge1Model
        tokenizer: Forge1Tokenizer
        user_message: User input text
        system_prompt: Optional system prompt override
        max_new_tokens: Max tokens to generate
        temperature: Sampling temperature
        device: Compute device
    
    Returns:
        ThinkingResult with separated thinking and answer
    """
    from inference.generate import generate
    
    if system_prompt is None:
        system_prompt = "You are Forge-1, a helpful thinking AI. Reason step-by-step."
    
    # Format the prompt
    prompt = tokenizer.format_conversation(
        system_prompt=system_prompt,
        user_message=user_message,
    )
    # Add thinking start to prompt the model to think
    prompt += "\n<|think|>\n"
    
    # Generate
    start_time = time.time()
    full_output = generate(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        device=device,
    )
    gen_time = time.time() - start_time
    
    # Extract the generated part (after the prompt)
    generated_text = full_output[len(prompt):]
    
    # Parse thinking and answer
    # Reconstruct with the think_start we added
    parse_text = "<|think|>\n" + generated_text
    thinking, answer = extract_thinking(parse_text)
    
    # Count tokens
    num_tokens = len(tokenizer.encode(generated_text, add_special_tokens=False))
    tps = num_tokens / max(gen_time, 0.001)
    
    return ThinkingResult(
        thinking=thinking,
        answer=answer,
        full_text=full_output,
        num_tokens=num_tokens,
        generation_time=gen_time,
        tokens_per_sec=tps,
    )
