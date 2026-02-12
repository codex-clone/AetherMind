"""
AetherMind - Google Colab Setup Script
========================================
Run this first in a Colab notebook to set up the environment.

Usage (in Colab):
    !git clone https://github.com/Mr-Dark-debug/AetherMind.git
    %cd AetherMind
    !python scripts/colab_setup.py
"""

import subprocess
import sys
import os


def run_cmd(cmd: str):
    """Run a shell command and print output."""
    print(f"$ {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.stdout:
        print(result.stdout)
    if result.returncode != 0:
        print(f"WARNING: {result.stderr}")
    return result.returncode


def main():
    print("=" * 60)
    print("AetherMind Colab Setup")
    print("=" * 60)
    
    # Check GPU
    print("\n[1/4] Checking GPU...")
    import torch
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"  GPU: {gpu_name} ({vram:.1f} GB)")
        
        if vram >= 14:
            print("  Recommended variant: mini (~350M params)")
        else:
            print("  Recommended variant: nano (~125M params)")
    else:
        print("  No GPU detected! Training will be very slow.")
    
    # Install dependencies
    print("\n[2/4] Installing dependencies...")
    run_cmd(f"{sys.executable} -m pip install -q -r requirements.txt")
    
    # Mount Google Drive (for persistent checkpoints)
    print("\n[3/4] Mounting Google Drive...")
    try:
        from google.colab import drive
        drive.mount("/content/drive")
        
        # Create checkpoint directory on Drive
        os.makedirs("/content/drive/MyDrive/AetherMind/checkpoints", exist_ok=True)
        os.makedirs("/content/drive/MyDrive/AetherMind/logs", exist_ok=True)
        print("  Google Drive mounted successfully")
    except ImportError:
        print("  Not running in Colab, skipping Drive mount")
    except Exception as e:
        print(f"  Drive mount failed: {e}")
    
    # Verify setup
    print("\n[4/4] Verifying setup...")
    try:
        from model.architecture import Forge1Model, Forge1Config
        from model.tokenizer import Forge1Tokenizer
        
        config = Forge1Config.nano()
        print(f"  Model config loaded: {config.count_parameters_estimate()/1e6:.0f}M params (nano)")
        
        tokenizer = Forge1Tokenizer()
        print(f"  Tokenizer ready: {tokenizer.vocab_size} tokens")
        
        print("\n" + "=" * 60)
        print("Setup complete! Start training with:")
        print("  python scripts/train.py --config configs/colab_config.yaml --variant mini")
        print("=" * 60)
    except Exception as e:
        print(f"  Verification failed: {e}")
        print("  Please check installation and try again.")


if __name__ == "__main__":
    main()
