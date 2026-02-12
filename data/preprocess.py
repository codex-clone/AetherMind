"""
AetherMind - Data Preprocessing
================================
Converts raw datasets into the unified Forge-1 conversation format,
then tokenizes them for training.

Conversation format:
  <|bos|>
  <|system|>system_prompt<|/system|>
  <|user|>user_message<|/user|>
  <|think|>reasoning<|/think|>
  <|assistant|>answer<|/assistant|>
  <|eos|>

Usage:
    python data/preprocess.py                    # Preprocess all downloaded datasets
    python data/preprocess.py --max-samples 500  # Limit samples per dataset
    python data/preprocess.py --output data/train.jsonl

Dependencies: data/download_datasets.py
Used by: data/dataset_loader.py, scripts/train.py
"""

import logging
import json
import argparse
from pathlib import Path
from typing import List, Dict, Optional, Any
from datasets import Dataset

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = "You are Forge-1, a helpful thinking AI assistant. Always reason step-by-step before answering."


def extract_conversations_openhermes(example: Dict[str, Any]) -> Dict[str, str]:
    """
    Extract user/assistant from OpenHermes-2.5 ShareGPT format.
    
    Schema: {"conversations": [{"from": "human"/"gpt"/"system", "value": "..."}], ...}
    """
    convs = example.get("conversations", [])
    user_msg = ""
    assistant_msg = ""
    system_msg = example.get("system_prompt") or SYSTEM_PROMPT
    
    for turn in convs:
        role = turn.get("from", "")
        value = turn.get("value", "")
        if role == "system":
            system_msg = value
        elif role == "human":
            user_msg = value
        elif role == "gpt":
            assistant_msg = value
    
    return {"system": system_msg, "user": user_msg, "assistant": assistant_msg, "thinking": ""}


def extract_conversations_platypus(example: Dict[str, Any]) -> Dict[str, str]:
    """
    Extract from Open-Platypus format.
    
    Schema: {"instruction": "...", "input": "...", "output": "...", "data_source": "..."}
    """
    instruction = example.get("instruction", "")
    inp = example.get("input", "")
    # Combine instruction and input if both present
    if inp:
        user_msg = f"{instruction}\n\n{inp}"
    else:
        user_msg = instruction
    output = example.get("output", "")
    
    return {"system": SYSTEM_PROMPT, "user": user_msg, "assistant": output, "thinking": ""}


def extract_conversations_openthoughts(example: Dict[str, Any]) -> Dict[str, str]:
    """
    Extract from OpenThoughts-114k format with CoT reasoning.
    
    Schema: {"system": "...", "conversations": [{"from": "user", "value": "..."}, {"from": "assistant", "value": "..."}]}
    
    The assistant response typically contains a <think>...</think> block
    followed by the actual answer. We parse this out.
    """
    system_msg = example.get("system", SYSTEM_PROMPT)
    convs = example.get("conversations", [])
    
    user_msg = ""
    full_response = ""
    
    for turn in convs:
        role = turn.get("from", "")
        value = turn.get("value", "")
        if role == "user":
            user_msg = value
        elif role == "assistant":
            full_response = value
    
    # Parse thinking from response
    # OpenThoughts format often has: <think>reasoning</think>\nfinal answer
    thinking = ""
    answer = full_response
    
    think_start_tag = "<|think|>"
    think_end_tag = "<|/think|>"
    
    # Also check for HTML-style <think> tags used by some entries
    for ts, te in [(think_start_tag, think_end_tag), ("<think>", "</think>")]:
        if ts in full_response and te in full_response:
            ts_idx = full_response.index(ts) + len(ts)
            te_idx = full_response.index(te)
            thinking = full_response[ts_idx:te_idx].strip()
            answer = full_response[te_idx + len(te):].strip()
            break
    
    return {"system": system_msg, "user": user_msg, "assistant": answer, "thinking": thinking}


EXTRACTORS = {
    "openhermes": extract_conversations_openhermes,
    "platypus": extract_conversations_platypus,
    "openthoughts": extract_conversations_openthoughts,
}


def format_for_training(item: Dict[str, str]) -> str:
    """Convert extracted conversation to Forge-1 training format string."""
    parts = []
    parts.append("<|bos|>")
    parts.append(f"<|system|>{item['system']}<|/system|>")
    parts.append(f"<|user|>{item['user']}<|/user|>")
    
    if item.get("thinking"):
        parts.append(f"<|think|>\n{item['thinking']}\n<|/think|>")
    else:
        parts.append("<|no_think|>")
    
    parts.append(f"<|assistant|>{item['assistant']}<|/assistant|>")
    parts.append("<|eos|>")
    
    return "\n".join(parts)


def preprocess_dataset(dataset: Dataset, dataset_name: str,
                       max_samples: Optional[int] = None) -> List[str]:
    """
    Preprocess a dataset into formatted training strings.
    
    Args:
        dataset: HuggingFace Dataset object
        dataset_name: Name key for selecting the right extractor
        max_samples: Optional limit on number of samples
    
    Returns:
        List of formatted training strings
    """
    extractor = EXTRACTORS.get(dataset_name)
    if not extractor:
        logger.warning(f"No extractor for {dataset_name}, using platypus format")
        extractor = extract_conversations_platypus
    
    formatted = []
    limit = max_samples or len(dataset)
    skipped = 0
    
    for i, example in enumerate(dataset):
        if i >= limit:
            break
        try:
            item = extractor(example)
            if item["user"] and item["assistant"]:
                text = format_for_training(item)
                formatted.append(text)
            else:
                skipped += 1
        except Exception as e:
            logger.debug(f"Skipping example {i}: {e}")
            skipped += 1
            continue
    
    logger.info(f"Preprocessed {len(formatted)}/{limit} examples from {dataset_name} ({skipped} skipped)")
    return formatted


def save_preprocessed(texts: List[str], output_path: str) -> None:
    """Save preprocessed texts to a JSONL file."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for text in texts:
            json.dump({"text": text}, f, ensure_ascii=False)
            f.write("\n")
    logger.info(f"Saved {len(texts)} examples to {output_path}")


def load_preprocessed(input_path: str) -> List[str]:
    """Load preprocessed texts from a JSONL file."""
    texts = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                texts.append(data["text"])
    logger.info(f"Loaded {len(texts)} examples from {input_path}")
    return texts


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess datasets for Forge-1 training")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Max samples per dataset (None = use all)")
    parser.add_argument("--output", type=str, default="data/train.jsonl",
                        help="Output JSONL file path")
    parser.add_argument("--cache-dir", type=str, default="data/cache",
                        help="HuggingFace cache directory")
    args = parser.parse_args()
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from data.download_datasets import download_all
    
    # Step 1: Download datasets (cached if already downloaded)
    logger.info("Step 1: Downloading/loading datasets...")
    datasets = download_all(cache_dir=args.cache_dir, max_samples=args.max_samples)
    
    if not datasets:
        logger.error("No datasets downloaded! Check your internet connection.")
        exit(1)
    
    # Step 2: Preprocess each dataset
    logger.info("\nStep 2: Preprocessing datasets...")
    all_texts = []
    stats = {}
    
    for name, ds in datasets.items():
        logger.info(f"Processing {name} ({len(ds)} samples)...")
        texts = preprocess_dataset(ds, name, max_samples=args.max_samples)
        all_texts.extend(texts)
        stats[name] = len(texts)
    
    # Step 3: Save to JSONL
    logger.info(f"\nStep 3: Saving {len(all_texts)} total examples to {args.output}...")
    save_preprocessed(all_texts, args.output)
    
    # Summary
    print("\n" + "=" * 60)
    print("Preprocessing Complete!")
    print("=" * 60)
    for name, count in stats.items():
        print(f"  {name}: {count} examples")
    print(f"  TOTAL: {len(all_texts)} examples")
    print(f"  Output: {args.output}")
    
    # Show a sample
    if all_texts:
        print(f"\nSample (first 300 chars):")
        print(all_texts[0][:300])
    print("=" * 60)
