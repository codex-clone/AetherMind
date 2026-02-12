# AetherMind — Complete Training Guide

> How to train Forge-1 from scratch, test it, and chat with it.

---

## Table of Contents

1. [Model Variants](#1-model-variants)
2. [Environment Setup](#2-environment-setup)
3. [Data Pipeline](#3-data-pipeline)
4. [Training](#4-training)
5. [Testing & Evaluation](#5-testing--evaluation)
6. [Chat Interface](#6-chat-interface)
7. [Exporting the Model](#7-exporting-the-model)
8. [Google Colab Guide](#8-google-colab-guide)
9. [Troubleshooting](#9-troubleshooting)
10. [Architecture Deep Dive](#10-architecture-deep-dive)

---

## 1. Model Variants

### Forge-1-Nano (~125M Parameters)

| Setting | Value |
|---------|-------|
| **Layers** | 12 |
| **Hidden Dim** | 768 |
| **Attention Heads** | 12 |
| **KV Heads (GQA)** | 4 |
| **FFN Hidden** | 2048 |
| **Context Length** | 1024 tokens |
| **Min VRAM** | 4 GB |
| **Target GPU** | RTX 3050 Ti, GTX 1650, RTX 2060 |
| **Training Time** | ~6-12 hrs (1M samples) |

Best for: Learning, experimentation, low-end GPUs.

### Forge-1-Mini (~350M Parameters)

| Setting | Value |
|---------|-------|
| **Layers** | 24 |
| **Hidden Dim** | 1024 |
| **Attention Heads** | 16 |
| **KV Heads (GQA)** | 4 |
| **FFN Hidden** | 2816 |
| **Context Length** | 2048 tokens |
| **Min VRAM** | 12-15 GB |
| **Target GPU** | Colab T4, RTX 3060+, RTX 4070 |
| **Training Time** | ~12-24 hrs (1M samples) |

Best for: Better reasoning quality, Colab T4, or 8GB+ GPUs.

### Architecture Choices

Both variants use the **LLaMA/Mistral architecture pattern**:

- **RMSNorm** instead of LayerNorm (more efficient, fewer ops)
- **Grouped Query Attention (GQA)** — 4 KV heads shared across all query heads (saves 60-70% KV-cache memory)
- **Rotary Position Embeddings (RoPE)** — relative position encoding via rotation matrices, no absolute position embeddings
- **SwiGLU Feed-Forward** — gated activation: `SiLU(x*W_gate) * (x*W_up)`, then `W_down`
- **Weight Tying** — embedding and output projection share the same weight matrix (saves parameters)
- **No bias terms** in any linear layer (matches modern practice)
- **Pre-normalization** — RMSNorm applied before attention/FFN, not after

---

## 2. Environment Setup

### Local Setup (Windows)

```powershell
# Clone the repository
git clone https://github.com/Mr-Dark-debug/AetherMind.git
cd AetherMind

# Create virtual environment
python -m venv venv
.\venv\Scripts\activate

# Install PyTorch (CUDA 12.1)
pip install torch --index-url https://download.pytorch.org/whl/cu121

# If the above fails (Python 3.13+), try default PyPI:
pip install torch

# Install all other dependencies
pip install -r requirements.txt
```

### Verify Installation

```powershell
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
```

Expected output:
```
PyTorch 2.x.x, CUDA: True
```

---

## 3. Data Pipeline

### Step 1: Download Datasets

```powershell
# Quick test (1000 samples per dataset = 3000 total, ~2 min)
python data/download_datasets.py 1000

# Medium run (10K samples per dataset = 30K total, ~5 min)
python data/download_datasets.py 10000

# Full dataset (all available samples, ~10-15 min)
python data/download_datasets.py
```

**Datasets downloaded:**

| Dataset | Samples | Size | What It Teaches |
|---------|---------|------|-----------------|
| [OpenHermes-2.5](https://huggingface.co/datasets/teknium/OpenHermes-2.5) | 1,001,551 | ~1.9 GB | General instruction following |
| [Open-Platypus](https://huggingface.co/datasets/garage-bAInd/Open-Platypus) | 24,926 | ~16 MB | STEM reasoning, math, science |
| [OpenThoughts-114k](https://huggingface.co/datasets/open-thoughts/OpenThoughts-114k) | 113,957 | ~1.1 GB | Chain-of-thought reasoning traces |

### Step 2: Preprocess into Training Format

```powershell
# Preprocess downloaded data with limit (uses cached downloads)
python data/preprocess.py --max-samples 1000

# For a bigger training run:
python data/preprocess.py --max-samples 10000

# Full preprocessing (all data):
python data/preprocess.py

# Custom output path:
python data/preprocess.py --output data/my_train.jsonl --max-samples 5000
```

This creates `data/train.jsonl` — each line is a JSON object with a `"text"` field containing the Forge-1 conversation format:

```
<|bos|>
<|system|>You are Forge-1, a helpful thinking AI assistant...<|/system|>
<|user|>What is 2+2?<|/user|>
<|think|>
Let me calculate... 2+2 equals 4.
<|/think|>
<|assistant|>The answer is 4.<|/assistant|>
<|eos|>
```

### Special Tokens

| Token | Purpose |
|-------|---------|
| `<|bos|>` | Beginning of sequence |
| `<|eos|>` | End of sequence |
| `<|pad|>` | Padding token |
| `<|system|>` / `<|/system|>` | System prompt boundaries |
| `<|user|>` / `<|/user|>` | User message boundaries |
| `<|assistant|>` / `<|/assistant|>` | Assistant response boundaries |
| `<|think|>` / `<|/think|>` | Chain-of-thought reasoning block |
| `<|no_think|>` | Placeholder when no thinking is needed |

---

## 4. Training

### Quick Test Training (CPU, 5 minutes)

```powershell
# Download small dataset
python data/download_datasets.py 100

# Preprocess
python data/preprocess.py --max-samples 100

# Train with minimal settings (edit local_config.yaml: device: "cpu")
python scripts/train.py --config configs/local_config.yaml --variant nano
```

### Local GPU Training (RTX 3050 Ti, 4 GB VRAM)

```powershell
# 1. Download data (1000 samples for testing, increase for real training)
python data/download_datasets.py 1000

# 2. Preprocess
python data/preprocess.py --max-samples 1000

# 3. Train Nano on local GPU
python scripts/train.py --config configs/local_config.yaml --variant nano
```

**What the config does** (`configs/local_config.yaml`):

| Setting | Value | Why |
|---------|-------|-----|
| `precision: fp16` | Half precision | Halves memory for activations |
| `gradient_checkpointing: true` | Recompute activations | Saves ~40% memory, costs ~20% speed |
| `batch_size: 2` | Tiny batch | Fits in 4GB VRAM |
| `gradient_accumulation_steps: 8` | Accumulate gradients | Effective batch = 2*8 = 16 |
| `context_length: 1024` | Short context | Fits in 4GB VRAM |
| `max_grad_norm: 1.0` | Gradient clipping | Prevents exploding gradients |
| `learning_rate: 3e-4` | Peak LR | Standard for small LLMs |
| `warmup_ratio: 0.02` | LR warmup | 2% of steps for stable startup |
| `lr_scheduler: cosine` | Cosine decay | Smooth LR decay to near zero |

### Google Colab Training (T4, 15 GB VRAM)

```python
# In Colab:
!git clone https://github.com/Mr-Dark-debug/AetherMind.git
%cd AetherMind
!python scripts/colab_setup.py

# Download & preprocess data
!python data/download_datasets.py 10000
!python data/preprocess.py --max-samples 10000

# Train Mini model
!python scripts/train.py --config configs/colab_config.yaml --variant mini
```

### Resume Interrupted Training

```powershell
# Just add --resume flag — it picks up from the latest checkpoint
python scripts/train.py --config configs/local_config.yaml --variant nano --resume
```

### Training Dashboard

The training script displays a live Rich dashboard showing:
- **Progress bar** — Current step / total steps with ETA
- **Loss curve** — Real-time training loss
- **Learning rate** — Current LR value
- **Speed** — Tokens/second and steps/second
- **Memory** — GPU VRAM usage
- **Epoch progress** — Current epoch / total epochs

### Recommended Training Recipes

#### Recipe 1: Quick Sanity Check (10 min)
```powershell
python data/download_datasets.py 500
python data/preprocess.py --max-samples 500
python scripts/train.py --config configs/local_config.yaml --variant nano
```
Goal: Verify everything works. Loss should decrease from ~10.5 to ~7-8.

#### Recipe 2: Small but Meaningful (2-4 hours)
```powershell
python data/download_datasets.py 5000
python data/preprocess.py --max-samples 5000
python scripts/train.py --config configs/local_config.yaml --variant nano
```
Goal: See basic instruction following emerge. Loss should reach ~4-5.

#### Recipe 3: Full Training (8-24 hours)
```powershell
python data/download_datasets.py
python data/preprocess.py
python scripts/train.py --config configs/local_config.yaml --variant nano
```
Goal: Best achievable quality for Nano. Loss should reach ~2.5-3.5.

#### Recipe 4: Colab Mini (6-12 hours)
```python
!python data/download_datasets.py
!python data/preprocess.py
!python scripts/train.py --config configs/colab_config.yaml --variant mini
```
Goal: Larger model, better reasoning quality.

---

## 5. Testing & Evaluation

### Run Evaluation Suite

```powershell
python scripts/evaluate.py --model outputs/final_model/forge1_nano_final.pt --variant nano
```

This runs 8 test prompts covering:
- Arithmetic ("What is 15 * 23?")
- Word problems ("If a train travels at 60mph for 2.5 hours...")
- Creative writing ("Write a haiku about programming")
- Factual recall ("What is the capital of France?")
- Sorting ("Sort these numbers: 5, 2, 8, 1, 9")

For each prompt, it shows:
- Whether the model used chain-of-thought thinking
- The thinking process (if any)
- The final answer
- Token count and generation time

### Manual Testing

```python
import sys; sys.path.insert(0, '.')
import torch
from model.architecture import Forge1Model, Forge1Config
from model.tokenizer import Forge1Tokenizer
from inference.thinking_pipeline import run_thinking_pipeline

# Load model
config = Forge1Config.nano()
tokenizer = Forge1Tokenizer()
config.vocab_size = tokenizer.get_vocab_size_padded()
model = Forge1Model(config)

ckpt = torch.load("outputs/final_model/forge1_nano_final.pt", map_location="cpu")
model.load_state_dict(ckpt)
model.eval()

# Generate
result = run_thinking_pipeline(
    model=model,
    tokenizer=tokenizer,
    user_message="What is the meaning of life?",
    max_new_tokens=256,
    temperature=0.7,
    device="cuda",
)

print(f"Thinking: {result.thinking}")
print(f"Answer: {result.answer}")
```

---

## 6. Chat Interface

### Start Interactive Chat

```powershell
python scripts/chat.py --model outputs/final_model/forge1_nano_final.pt --variant nano
```

### Chat Commands

| Command | Description |
|---------|-------------|
| `/think` | Toggle showing chain-of-thought blocks on/off |
| `/save` | Save conversation history to file |
| `/clear` | Clear conversation history |
| `/help` | Show all available commands |
| `/quit` | Exit the chat |

### Chat Options

```powershell
# Show thinking by default
python scripts/chat.py --model path/to/model.pt --variant nano --show-thinking

# Use CPU if no GPU
python scripts/chat.py --model path/to/model.pt --variant nano --device cpu

# Use a specific tokenizer directory
python scripts/chat.py --model path/to/model.pt --variant nano --tokenizer outputs/final_model/tokenizer
```

---

## 7. Exporting the Model

### Export to Both Formats

```powershell
python scripts/export.py --model outputs/final_model/forge1_nano_final.pt --variant nano
```

This creates:
- `outputs/exports/forge1_nano.pt` — PyTorch state dict
- `outputs/exports/forge1_nano.safetensors` — SafeTensors format (requires `pip install safetensors`)
- `outputs/exports/tokenizer/` — Saved tokenizer

### Export Only One Format

```powershell
# PyTorch only
python scripts/export.py --model path/to/model.pt --variant nano --format pytorch

# SafeTensors only
python scripts/export.py --model path/to/model.pt --variant nano --format safetensors
```

---

## 8. Google Colab Guide

### Complete Colab Notebook

```python
# Cell 1: Setup
!git clone https://github.com/Mr-Dark-debug/AetherMind.git
%cd AetherMind
!python scripts/colab_setup.py

# Cell 2: Download data
!python data/download_datasets.py 10000

# Cell 3: Preprocess
!python data/preprocess.py --max-samples 10000

# Cell 4: Train (this takes a while)
!python scripts/train.py --config configs/colab_config.yaml --variant mini

# Cell 5: Evaluate
!python scripts/evaluate.py --model outputs/final_model/forge1_mini_final.pt --variant mini

# Cell 6: Chat
!python scripts/chat.py --model outputs/final_model/forge1_mini_final.pt --variant mini
```

### Tips for Colab
- Checkpoints save to Google Drive automatically (mounted in setup)
- If runtime disconnects, reconnect and run with `--resume` to continue training
- T4 GPU has 15GB VRAM — enough for Mini variant
- A100 GPU (Colab Pro) can train Mini much faster with larger batch sizes

---

## 9. Troubleshooting

### Common Issues

#### "CUDA out of memory"
```
Try these fixes in order:
1. Reduce batch_size to 1 in local_config.yaml
2. Reduce context_length to 512
3. Ensure gradient_checkpointing: true
4. Ensure precision: fp16
5. Increase gradient_accumulation_steps to 16 or 32
6. Switch to the nano variant
```

#### "No module named 'torch'"
```powershell
pip install torch
```

#### "trust_remote_code is not supported"
This is a harmless warning from HuggingFace. It has been fixed in the latest download_datasets.py.

#### Training loss is not decreasing
- Check that your data was preprocessed correctly
- Increase `max_samples` — more data usually helps
- Make sure learning rate is 3e-4 (not too high, not too low)
- Verify the model loaded correctly (check parameter count)

#### Slow training speed
- Ensure CUDA is being used (check for "Using GPU:" in logs)
- Enable mixed precision (`precision: fp16` or `bf16`)
- Increase `num_workers` to 2-4 in config
- Enable `pin_memory: true`

---

## 10. Architecture Deep Dive

### The Transformer Block

Each Forge-1 transformer layer consists of:

```
Input
  │
  ├── RMSNorm ──→ Grouped Query Attention ──→ Residual Add
  │                      (with RoPE)
  │
  ├── RMSNorm ──→ SwiGLU Feed-Forward ──→ Residual Add
  │
Output
```

### Grouped Query Attention (GQA)

Standard multi-head attention uses separate K,V projections per head. GQA shares K,V heads across groups of query heads:

```
Nano: 12 query heads, 4 KV heads → each KV head serves 3 query heads
Mini: 16 query heads, 4 KV heads → each KV head serves 4 query heads
```

This reduces KV-cache memory by 3-4x with minimal quality loss.

### RoPE (Rotary Position Embeddings)

Instead of adding position embeddings to tokens, RoPE rotates the query and key vectors:

```
q_rotated = q * cos(theta) + rotate_half(q) * sin(theta)
k_rotated = k * cos(theta) + rotate_half(k) * sin(theta)
```

Benefits:
- Relative position awareness (attention decays naturally with distance)
- No extra parameters needed
- Supports extrapolation to longer sequences

### SwiGLU Feed-Forward

The gated activation function used in modern LLMs:

```
FFN(x) = W_down * (SiLU(W_gate * x) ⊙ W_up * x)
```

Where ⊙ is element-wise multiplication. The gate controls information flow.

### Training Loop

```
For each epoch:
  For each batch:
    1. Forward pass (with AMP autocast)
    2. Compute loss (cross-entropy, ignoring padding)
    3. Scale loss by gradient_accumulation_steps
    4. Backward pass (with GradScaler if FP16)
    5. If accumulated enough steps:
       a. Unscale gradients
       b. Clip gradients (max_norm=1.0)
       c. Optimizer step
       d. Scheduler step
       e. Zero gradients
       f. Log metrics
    6. Save checkpoint every N steps
```

### Memory Budget (Nano on 4GB VRAM)

| Component | Memory |
|-----------|--------|
| Model weights (FP16) | ~228 MB |
| Activations (1 batch, 1024 seq) | ~500 MB |
| Gradient checkpointing savings | -200 MB |
| Optimizer states (AdamW) | ~456 MB |
| Gradient buffers | ~228 MB |
| CUDA overhead | ~200 MB |
| **Total** | **~1.4 GB** ✅ |

---

## Complete Command Reference

```powershell
# ============== SETUP ==============
python -m venv venv
.\venv\Scripts\activate
pip install torch
pip install -r requirements.txt

# ============== DATA ==============
python data/download_datasets.py 1000        # Download (1K per dataset)
python data/download_datasets.py 10000       # Download (10K per dataset)
python data/download_datasets.py             # Download (full datasets)

python data/preprocess.py --max-samples 1000 # Preprocess (1K samples each)
python data/preprocess.py --max-samples 10000
python data/preprocess.py                    # Preprocess all

# ============== TRAIN ==============
python scripts/train.py --config configs/local_config.yaml --variant nano
python scripts/train.py --config configs/colab_config.yaml --variant mini
python scripts/train.py --config configs/local_config.yaml --variant nano --resume

# ============== TEST ==============
python scripts/evaluate.py --model outputs/final_model/forge1_nano_final.pt --variant nano
python scripts/evaluate.py --model outputs/final_model/forge1_mini_final.pt --variant mini

# ============== CHAT ==============
python scripts/chat.py --model outputs/final_model/forge1_nano_final.pt --variant nano
python scripts/chat.py --model outputs/final_model/forge1_mini_final.pt --variant mini --show-thinking

# ============== EXPORT ==============
python scripts/export.py --model outputs/final_model/forge1_nano_final.pt --variant nano
python scripts/export.py --model outputs/final_model/forge1_mini_final.pt --variant mini --format safetensors
```

---

*Built with \u2764\ufe0f by AetherMind Team*
