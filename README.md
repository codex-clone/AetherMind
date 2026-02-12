# AetherMind

> **Train a small reasoning language model from scratch.**

AetherMind is a project to build **Forge-1**, a compact decoder-only transformer that can reason step-by-step using chain-of-thought (CoT) thinking tokens. Two variants are available:

| Variant | Parameters | VRAM Required | Target Hardware |
|---------|-----------|---------------|-----------------|
| **Forge-1-Nano** | ~125M | 4 GB | RTX 3050 Ti, GTX 1650 |
| **Forge-1-Mini** | ~350M | 15 GB | Colab T4, RTX 3060+ |

## Architecture

Forge-1 follows the modern LLaMA/Mistral architecture:

- **RMSNorm** — Pre-normalization (more efficient than LayerNorm)
- **Grouped Query Attention (GQA)** — Reduces KV-cache memory by sharing KV heads
- **Rotary Position Embeddings (RoPE)** — Relative position encoding via rotation
- **SwiGLU Feed-Forward** — Gated activation function for better feature selection
- **Weight Tying** — Input embeddings shared with output projection head

## Quick Start

### 1. Setup Environment

```bash
# Clone the repo
git clone https://github.com/yourusername/AetherMind.git
cd AetherMind

# Create virtual environment
python -m venv venv
venv\Scripts\activate       # Windows
# source venv/bin/activate  # Linux/macOS

# Install dependencies
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

### 2. Download & Preprocess Data

```bash
# Download datasets (use 1000 for a quick test)
python data/download_datasets.py 1000

# Full download (takes longer)
python data/download_datasets.py
```

### 3. Train

```bash
# Local training (RTX 3050 Ti / 4GB VRAM)
python scripts/train.py --config configs/local_config.yaml --variant nano

# Or on Colab (T4 / 15GB VRAM)
python scripts/train.py --config configs/colab_config.yaml --variant mini

# Resume interrupted training
python scripts/train.py --config configs/local_config.yaml --variant nano --resume
```

### 4. Chat

```bash
python scripts/chat.py --model outputs/final_model/forge1_nano_final.pt --variant nano
```

### 5. Evaluate

```bash
python scripts/evaluate.py --model outputs/final_model/forge1_nano_final.pt --variant nano
```

### 6. Export

```bash
python scripts/export.py --model outputs/final_model/forge1_nano_final.pt --variant nano
```

## Google Colab

```python
!git clone https://github.com/yourusername/AetherMind.git
%cd AetherMind
!python scripts/colab_setup.py
!python scripts/train.py --config configs/colab_config.yaml --variant mini
```

## Training Datasets

| Dataset | Size | Purpose |
|---------|------|---------|
| [OpenHermes-2.5](https://huggingface.co/datasets/teknium/OpenHermes-2.5) | 1M examples | Diverse instruction following |
| [Open-Platypus](https://huggingface.co/datasets/garage-bAInd/Open-Platypus) | 25K examples | STEM reasoning with CoT |
| [OpenThoughts-114k](https://huggingface.co/datasets/open-thoughts/OpenThoughts-114k) | 114K examples | Synthetic CoT reasoning traces |

## Memory Optimization

For low-VRAM GPUs (4GB), AetherMind uses:

- **FP16 mixed precision** — Halves memory for activations
- **Gradient checkpointing** — Trades compute for memory
- **Gradient accumulation** — Small physical batch, large effective batch
- **Weight tying** — Shares embedding weights between input and output layers
- **GQA** — Fewer KV heads reduces KV-cache memory

## License

MIT License — see [LICENSE](LICENSE) for details.