"""
AetherMind - Dataset Downloader
================================
Downloads and caches training datasets from Hugging Face Hub.

Datasets used:
  1. OpenHermes-2.5 (teknium) - 1M instruction-following examples
  2. Open-Platypus (garage-bAInd) - STEM reasoning with CoT
  3. OpenThoughts-114k (open-thoughts) - Synthetic CoT reasoning

Dependencies: datasets (HuggingFace), huggingface_hub
Used by: data/preprocess.py, scripts/train.py
"""

import logging
from pathlib import Path
from typing import Optional
from datasets import load_dataset, Dataset, DatasetDict

logger = logging.getLogger(__name__)

# Dataset definitions
DATASET_REGISTRY = {
    "openhermes": {
        "repo": "teknium/OpenHermes-2.5",
        "subset": None,
        "split": "train",
        "description": "1M high-quality instruction-following examples",
    },
    "platypus": {
        "repo": "garage-bAInd/Open-Platypus",
        "subset": None,
        "split": "train",
        "description": "25K STEM reasoning questions with CoT",
    },
    "openthoughts": {
        "repo": "open-thoughts/OpenThoughts-114k",
        "subset": None,
        "split": "train",
        "description": "114K synthetic reasoning with detailed CoT traces",
    },
}


def download_dataset(name: str, cache_dir: Optional[str] = None,
                     max_samples: Optional[int] = None) -> Dataset:
    """
    Download a single dataset by name.
    
    Args:
        name: Dataset name from DATASET_REGISTRY
        cache_dir: Where to cache downloaded data
        max_samples: Limit number of samples (for testing)
    
    Returns:
        HuggingFace Dataset object
    """
    if name not in DATASET_REGISTRY:
        raise ValueError(f"Unknown dataset: {name}. Available: {list(DATASET_REGISTRY.keys())}")
    
    info = DATASET_REGISTRY[name]
    logger.info(f"Downloading {name}: {info['description']}")
    
    try:
        ds = load_dataset(
            info["repo"],
            info.get("subset"),
            split=info["split"],
            cache_dir=cache_dir,
        )
        
        if max_samples and len(ds) > max_samples:
            ds = ds.select(range(max_samples))
            logger.info(f"  Truncated to {max_samples} samples")
        
        logger.info(f"  Downloaded {len(ds)} samples from {name}")
        return ds
        
    except Exception as e:
        logger.error(f"Failed to download {name}: {e}")
        logger.error(f"Manual download: https://huggingface.co/datasets/{info['repo']}")
        raise


def download_all(cache_dir: Optional[str] = None,
                 max_samples: Optional[int] = None) -> dict:
    """Download all datasets. Returns dict of name -> Dataset."""
    results = {}
    for name in DATASET_REGISTRY:
        try:
            results[name] = download_dataset(name, cache_dir, max_samples)
        except Exception as e:
            logger.warning(f"Skipping {name}: {e}")
    return results


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    max_s = int(sys.argv[1]) if len(sys.argv) > 1 else None
    datasets = download_all(cache_dir="data/cache", max_samples=max_s)
    print(f"\nDownloaded {len(datasets)} datasets successfully!")
    for name, ds in datasets.items():
        print(f"  {name}: {len(ds)} samples")
