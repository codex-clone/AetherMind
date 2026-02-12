"""
AetherMind - Dataset Loader
=============================
PyTorch Dataset and DataLoader classes for training Forge-1.

Handles tokenization, padding, truncation, and batching.

Dependencies: torch, model/tokenizer.py, data/preprocess.py
Used by: training/trainer.py, scripts/train.py
"""

import torch
from torch.utils.data import Dataset, DataLoader
import logging
import json
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)


class Forge1Dataset(Dataset):
    """
    PyTorch Dataset for Forge-1 training data.
    
    Loads preprocessed text, tokenizes on-the-fly, and returns
    input_ids and labels tensors for next-token prediction.
    
    Args:
        texts: List of formatted training text strings
        tokenizer: Forge1Tokenizer instance
        max_length: Maximum sequence length (truncate longer sequences)
    """
    
    def __init__(self, texts: List[str], tokenizer, max_length: int = 1024):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
        logger.info(f"Created dataset with {len(texts)} examples, max_length={max_length}")
    
    def __len__(self) -> int:
        return len(self.texts)
    
    def __getitem__(self, idx: int) -> dict:
        text = self.texts[idx]
        
        # Tokenize with truncation
        token_ids = self.tokenizer.encode(text, add_special_tokens=False,
                                          max_length=self.max_length)
        
        # Truncate if needed
        if len(token_ids) > self.max_length:
            token_ids = token_ids[:self.max_length]
        
        # For causal LM: input_ids = labels (shifted inside the model)
        input_ids = torch.tensor(token_ids, dtype=torch.long)
        labels = input_ids.clone()
        
        return {"input_ids": input_ids, "labels": labels}
    
    @classmethod
    def from_jsonl(cls, path: str, tokenizer, max_length: int = 1024,
                   max_samples: Optional[int] = None) -> "Forge1Dataset":
        """Create dataset from a JSONL file."""
        texts = []
        with open(path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if max_samples and i >= max_samples:
                    break
                if line.strip():
                    data = json.loads(line)
                    texts.append(data["text"])
        return cls(texts, tokenizer, max_length)


def collate_fn(batch: List[dict], pad_token_id: int = 0) -> dict:
    """
    Collate function for DataLoader - pads sequences to same length.
    
    Args:
        batch: List of dicts with "input_ids" and "labels" tensors
        pad_token_id: Token ID to use for padding
    
    Returns:
        Dict with padded "input_ids", "labels", and "attention_mask"
    """
    input_ids = [item["input_ids"] for item in batch]
    labels = [item["labels"] for item in batch]
    
    # Find max length in this batch
    max_len = max(ids.size(0) for ids in input_ids)
    
    # Pad all sequences to max_len
    padded_input_ids = []
    padded_labels = []
    attention_masks = []
    
    for ids, lbl in zip(input_ids, labels):
        pad_len = max_len - ids.size(0)
        
        # Pad input_ids with pad_token_id
        padded_input_ids.append(
            torch.cat([ids, torch.full((pad_len,), pad_token_id, dtype=torch.long)])
        )
        # Pad labels with -100 (ignore index for cross-entropy)
        padded_labels.append(
            torch.cat([lbl, torch.full((pad_len,), -100, dtype=torch.long)])
        )
        # Attention mask: 1 for real tokens, 0 for padding
        attention_masks.append(
            torch.cat([torch.ones(ids.size(0)), torch.zeros(pad_len)])
        )
    
    return {
        "input_ids": torch.stack(padded_input_ids),
        "labels": torch.stack(padded_labels),
        "attention_mask": torch.stack(attention_masks),
    }


def create_dataloader(dataset: Forge1Dataset, batch_size: int,
                      pad_token_id: int = 0, shuffle: bool = True,
                      num_workers: int = 0, pin_memory: bool = False) -> DataLoader:
    """Create a DataLoader with proper collation."""
    from functools import partial
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=partial(collate_fn, pad_token_id=pad_token_id),
        drop_last=True,  # Drop incomplete last batch for stability
    )
