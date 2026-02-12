"""
AetherMind - Forge-1 Tokenizer
===============================
Sets up the tokenizer for Forge-1, reusing GPT-2 tokenizer from HuggingFace
and adding custom special tokens for thinking/reasoning conversation format.

Why GPT-2 tokenizer:
  - Well-tested BPE tokenizer with ~50k vocabulary
  - Native HuggingFace integration, fast Rust backend
  - Good coverage of English, code, and math notation
  - Easy to extend with custom special tokens

Dependencies: transformers (AutoTokenizer)
Used by: data/preprocess.py, data/dataset_loader.py, inference/generate.py
"""

import logging
from pathlib import Path
from typing import List, Dict, Optional
from transformers import AutoTokenizer, PreTrainedTokenizerFast

logger = logging.getLogger(__name__)

# Special tokens for Forge-1 conversation format
FORGE1_SPECIAL_TOKENS: List[str] = [
    "<|pad|>",
    "<|bos|>",
    "<|eos|>",
    "<|system|>",
    "<|/system|>",
    "<|user|>",
    "<|/user|>",
    "<|assistant|>",
    "<|/assistant|>",
    "<|think|>",
    "<|/think|>",
    "<|no_think|>"
]

# Semantic role mappings for convenience
TOKEN_ROLES = {
    "pad": "<|pad|>",
    "bos": "<|bos|>",
    "eos": "<|eos|>",
    "system_start": "<|system|>",
    "system_end": "<|/system|>",
    "user_start": "<|user|>",
    "user_end": "<|/user|>",
    "assistant_start": "<|assistant|>",
    "assistant_end": "<|/assistant|>",
    "think_start": "<|think|>",
    "think_end": "<|/think|>",
    "no_think": "<|no_think|>",
}


class Forge1Tokenizer:
    """
    Wrapper around a HuggingFace tokenizer with Forge-1 special tokens.
    
    Loads GPT-2 tokenizer, adds special tokens, and provides helper methods
    for encoding/decoding in the Forge-1 conversation format.
    
    Args:
        model_name: Base HuggingFace tokenizer to load. Default: "gpt2"
        cache_dir: Directory to cache the downloaded tokenizer files.
    """
    
    BASE_MODEL = "gpt2"
    
    def __init__(self, model_name: str = "gpt2", cache_dir: Optional[str] = None):
        self.model_name = model_name
        
        logger.info(f"Loading base tokenizer: {model_name}")
        self.tokenizer: PreTrainedTokenizerFast = AutoTokenizer.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            use_fast=True,  # Use Rust-based fast tokenizer
        )
        
        # Add Forge-1 special tokens
        num_added = self.tokenizer.add_special_tokens({
            "additional_special_tokens": FORGE1_SPECIAL_TOKENS,
            "pad_token": TOKEN_ROLES["pad"],
            "bos_token": TOKEN_ROLES["bos"],
            "eos_token": TOKEN_ROLES["eos"],
        })
        logger.info(f"Added {num_added} special tokens to tokenizer")
        
        # Store vocab size (needed for model embedding layer)
        self._vocab_size = len(self.tokenizer)
        
        # Build token ID lookup for fast access
        self.special_token_ids: Dict[str, int] = {}
        for role, token_str in TOKEN_ROLES.items():
            tid = self.tokenizer.convert_tokens_to_ids(token_str)
            self.special_token_ids[role] = tid
        
        logger.info(f"Tokenizer ready: vocab_size={self._vocab_size}")
    
    @property
    def vocab_size(self) -> int:
        """Total vocabulary size including special tokens."""
        return self._vocab_size

    @property
    def pad_token_id(self) -> int:
        return self.special_token_ids["pad"]
    
    @property
    def bos_token_id(self) -> int:
        return self.special_token_ids["bos"]
    
    @property
    def eos_token_id(self) -> int:
        return self.special_token_ids["eos"]
    
    @property
    def think_start_id(self) -> int:
        return self.special_token_ids["think_start"]
    
    @property
    def think_end_id(self) -> int:
        return self.special_token_ids["think_end"]
    
    def encode(self, text: str, add_special_tokens: bool = True,
               max_length: Optional[int] = None) -> List[int]:
        """Encode text to token IDs."""
        kwargs = {"add_special_tokens": add_special_tokens}
        if max_length:
            kwargs["max_length"] = max_length
            kwargs["truncation"] = True
        return self.tokenizer.encode(text, **kwargs)
    
    def decode(self, token_ids: List[int], skip_special_tokens: bool = False) -> str:
        """Decode token IDs back to text."""
        return self.tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)
    
    def format_conversation(self, system_prompt: str, user_message: str,
                           thinking: Optional[str] = None,
                           assistant_response: Optional[str] = None) -> str:
        """
        Format a conversation turn in Forge-1 format.
        
        Args:
            system_prompt: System instruction text
            user_message: User's input message
            thinking: Optional chain-of-thought reasoning
            assistant_response: Optional final response
        
        Returns:
            Formatted conversation string
        """
        parts = []
        parts.append(f'{TOKEN_ROLES["system_start"]}{system_prompt}{TOKEN_ROLES["system_end"]}')
        parts.append(f'{TOKEN_ROLES["user_start"]}{user_message}{TOKEN_ROLES["user_end"]}')
        
        if thinking:
            parts.append(f'{TOKEN_ROLES["think_start"]}\n{thinking}\n{TOKEN_ROLES["think_end"]}')
        elif thinking is None and assistant_response:
            # No thinking block provided - use no_think marker
            parts.append(TOKEN_ROLES["no_think"])
        
        if assistant_response:
            parts.append(f'{TOKEN_ROLES["assistant_start"]}{assistant_response}{TOKEN_ROLES["assistant_end"]}')
        
        return "\n".join(parts)
    
    def save(self, save_dir: str) -> None:
        """Save tokenizer to directory."""
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        self.tokenizer.save_pretrained(save_dir)
        logger.info(f"Tokenizer saved to {save_dir}")
    
    @classmethod
    def load(cls, load_dir: str) -> "Forge1Tokenizer":
        """Load a saved Forge1Tokenizer from directory."""
        instance = cls.__new__(cls)
        instance.tokenizer = AutoTokenizer.from_pretrained(load_dir, use_fast=True)
        instance._vocab_size = len(instance.tokenizer)
        instance.special_token_ids = {}
        for role, token_str in TOKEN_ROLES.items():
            tid = instance.tokenizer.convert_tokens_to_ids(token_str)
            instance.special_token_ids[role] = tid
        logger.info(f"Tokenizer loaded from {load_dir}: vocab_size={instance._vocab_size}")
        return instance
    
    def get_vocab_size_padded(self, multiple_of: int = 64) -> int:
        """Get vocab size padded to a multiple (for efficient GPU computation)."""
        v = self._vocab_size
        return ((v + multiple_of - 1) // multiple_of) * multiple_of
