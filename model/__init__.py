"""
AetherMind Model Package
========================
Contains the Forge-1 transformer model architecture and all sub-components.

Modules:
    - architecture.py: Main Forge1Model class (full transformer)
    - attention.py: Multi-Head / Grouped Query Attention with RoPE
    - feedforward.py: SwiGLU Feed-Forward Network blocks
    - positional_encoding.py: Rotary Position Embeddings (RoPE)
    - tokenizer.py: Tokenizer setup with special tokens for thinking

Note: Imports are lazy to avoid requiring torch at import time
for lightweight scripts that only need the tokenizer.
"""

def __getattr__(name):
    if name == "Forge1Model":
        from model.architecture import Forge1Model
        return Forge1Model
    elif name == "Forge1Config":
        from model.architecture import Forge1Config
        return Forge1Config
    elif name == "Forge1Tokenizer":
        from model.tokenizer import Forge1Tokenizer
        return Forge1Tokenizer
    raise AttributeError(f"module 'model' has no attribute {name}")

__all__ = ["Forge1Model", "Forge1Config", "Forge1Tokenizer"]
