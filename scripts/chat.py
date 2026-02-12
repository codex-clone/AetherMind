"""
AetherMind - Chat Entry Point
================================
Interactive chat with a trained Forge-1 model.

Usage:
    python scripts/chat.py --model outputs/final_model/forge1_nano_final.pt --variant nano
    python scripts/chat.py --model outputs/checkpoints/step_5000.pt --variant nano

Dependencies: All project modules
"""

import sys
import torch
import argparse
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from model.architecture import Forge1Model, Forge1Config
from model.tokenizer import Forge1Tokenizer
from ui.chat_ui import ChatInterface


def main():
    parser = argparse.ArgumentParser(description="Chat with Forge-1")
    parser.add_argument("--model", type=str, required=True,
                        help="Path to model checkpoint (.pt file)")
    parser.add_argument("--variant", type=str, default="nano",
                        choices=["nano", "mini"], help="Model variant")
    parser.add_argument("--tokenizer", type=str, default=None,
                        help="Path to saved tokenizer directory")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device (cuda or cpu)")
    parser.add_argument("--show-thinking", action="store_true",
                        help="Show thinking blocks by default")
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.WARNING)
    logger = logging.getLogger("chat")
    
    # Setup device
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        args.device = "cpu"
    
    device = torch.device(args.device)
    
    # Load model
    print(f"Loading Forge-1-{args.variant.title()}...")
    config = Forge1Config.from_variant(args.variant)
    
    # Load tokenizer
    if args.tokenizer and Path(args.tokenizer).exists():
        tokenizer = Forge1Tokenizer.load(args.tokenizer)
    else:
        tokenizer = Forge1Tokenizer()
    
    config.vocab_size = tokenizer.get_vocab_size_padded()
    model = Forge1Model(config)
    
    # Load weights
    checkpoint = torch.load(args.model, map_location="cpu", weights_only=False)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    model.eval()
    
    n_params = model.count_parameters()
    print(f"Model loaded: {n_params/1e6:.1f}M parameters on {device}")
    
    # Start chat
    chat = ChatInterface(
        model=model,
        tokenizer=tokenizer,
        device=str(device),
        show_thinking=args.show_thinking,
    )
    chat.run()


if __name__ == "__main__":
    main()
