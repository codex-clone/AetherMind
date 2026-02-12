"""
AetherMind - Model Export
==========================
Export trained model to various formats for deployment.

Supports: PyTorch state dict, SafeTensors, ONNX

Usage:
    python scripts/export.py --model outputs/final_model/forge1_nano_final.pt --variant nano
"""

import sys
import torch
import argparse
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from model.architecture import Forge1Model, Forge1Config
from model.tokenizer import Forge1Tokenizer


def export_safetensors(model, output_path: str):
    """Export model weights in SafeTensors format."""
    try:
        from safetensors.torch import save_model
        save_model(model, output_path)
        print(f"Saved SafeTensors: {output_path}")
    except ImportError:
        print("safetensors not installed. Install with: pip install safetensors")


def export_pytorch(model, output_path: str):
    """Export as standard PyTorch state dict."""
    torch.save(model.state_dict(), output_path)
    print(f"Saved PyTorch weights: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Export Forge-1 model")
    parser.add_argument("--model", type=str, required=True, help="Model checkpoint path")
    parser.add_argument("--variant", type=str, default="nano")
    parser.add_argument("--output-dir", type=str, default="outputs/exports")
    parser.add_argument("--format", type=str, default="all",
                        choices=["pytorch", "safetensors", "all"])
    args = parser.parse_args()
    
    # Load model
    config = Forge1Config.from_variant(args.variant)
    tokenizer = Forge1Tokenizer()
    config.vocab_size = tokenizer.get_vocab_size_padded()
    
    model = Forge1Model(config)
    ckpt = torch.load(args.model, map_location="cpu", weights_only=False)
    if "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        model.load_state_dict(ckpt)
    
    # Export
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if args.format in ("pytorch", "all"):
        export_pytorch(model, str(output_dir / f"forge1_{args.variant}.pt"))
    
    if args.format in ("safetensors", "all"):
        export_safetensors(model, str(output_dir / f"forge1_{args.variant}.safetensors"))
    
    tokenizer.save(str(output_dir / "tokenizer"))
    print("Export complete!")


if __name__ == "__main__":
    main()
