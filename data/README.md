# AetherMind Datasets

## Datasets Used

### 1. OpenHermes-2.5 (`teknium/OpenHermes-2.5`)
- **Size:** ~1M examples, ~1.9 GB
- **Format:** ShareGPT (multi-turn conversations)
- **License:** Apache 2.0
- **Why:** High-quality, diverse instruction-following data compiled from multiple sources

### 2. Open-Platypus (`garage-bAInd/Open-Platypus`)
- **Size:** ~25K examples
- **Format:** Instruction/Output pairs
- **License:** CC-BY-4.0  
- **Why:** STEM and reasoning-focused with step-by-step solutions

### 3. OpenThoughts-114k (`open-thoughts/OpenThoughts-114k`)
- **Size:** ~114K examples
- **Format:** Problem/Thought/Solution with detailed CoT
- **License:** Apache 2.0
- **Why:** Explicit chain-of-thought reasoning traces - critical for teaching the model to "think"

## Usage

```bash
# Download all datasets
python data/download_datasets.py

# Download with sample limit (for testing)
python data/download_datasets.py 1000
```
