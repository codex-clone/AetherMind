"""
AetherMind - Model Evaluation
================================
Evaluate a trained Forge-1 model on test prompts.

Usage:
    python scripts/evaluate.py --model outputs/final_model/forge1_nano_final.pt --variant nano
"""

import sys
import torch
import argparse
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from model.architecture import Forge1Model, Forge1Config
from model.tokenizer import Forge1Tokenizer
from inference.thinking_pipeline import run_thinking_pipeline


# Test prompts to evaluate reasoning quality
TEST_PROMPTS = [
    "What is 15 * 23?",
    "If a train travels at 60mph for 2.5 hours, how far does it go?",
    "Write a haiku about programming.",
    "What are the first 5 prime numbers?",
    "Explain why the sky is blue in one sentence.",
    "If I have 3 apples and give away 1, how many do I have?",
    "What is the capital of France?",
    "Sort these numbers: 5, 2, 8, 1, 9",
]


def main():
    parser = argparse.ArgumentParser(description="Evaluate Forge-1")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--variant", type=str, default="nano")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.WARNING)
    
    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"
    
    # Load model
    config = Forge1Config.from_variant(args.variant)
    tokenizer = Forge1Tokenizer()
    config.vocab_size = tokenizer.get_vocab_size_padded()
    
    model = Forge1Model(config)
    ckpt = torch.load(args.model, map_location="cpu", weights_only=False)
    state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt

    # Check for embedding size mismatch
    if "token_embedding.weight" in state_dict:
        ckpt_vocab_size = state_dict["token_embedding.weight"].shape[0]
        if ckpt_vocab_size != config.vocab_size:
            print(f"Resizing model embeddings from {config.vocab_size} to {ckpt_vocab_size} to match checkpoint")
            config.vocab_size = ckpt_vocab_size
            model = Forge1Model(config)
            
    model.load_state_dict(state_dict)
    
    model = model.to(args.device)
    model.eval()
    
    print(f"Forge-1-{args.variant.title()} loaded ({model.count_parameters()/1e6:.1f}M params)")
    print("=" * 60)
    
    for i, prompt in enumerate(TEST_PROMPTS):
        print(f"\n[{i+1}/{len(TEST_PROMPTS)}] {prompt}")
        print("-" * 40)
        
        result = run_thinking_pipeline(
            model=model,
            tokenizer=tokenizer,
            user_message=prompt,
            max_new_tokens=256,
            temperature=0.7,
            device=args.device,
        )
        
        if result.has_thinking:
            print(f"Thinking: {result.thinking[:200]}...")
        print(f"Answer: {result.answer}")
        print(f"({result.num_tokens} tok, {result.generation_time:.1f}s)")
    
    print("\n" + "=" * 60)
    print("Evaluation complete!")


if __name__ == "__main__":
    main()
