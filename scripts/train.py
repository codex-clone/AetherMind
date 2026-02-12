"""
AetherMind - Training Entry Point
====================================
Main script to train Forge-1 from scratch or resume from checkpoint.

Usage:
    python scripts/train.py --config configs/local_config.yaml --variant nano
    python scripts/train.py --config configs/colab_config.yaml --variant mini
    python scripts/train.py --config configs/local_config.yaml --variant nano --resume

Dependencies: All project modules
"""

import sys
import os
import yaml
import torch
import logging
import argparse
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from model.architecture import Forge1Model, Forge1Config
from model.tokenizer import Forge1Tokenizer
from data.dataset_loader import Forge1Dataset, create_dataloader
from data.preprocess import load_preprocessed
from training.trainer import Trainer
from training.callbacks import LoggingCallback, MetricsTracker
from ui.training_ui import TrainingDashboard


def load_config(config_path: str) -> dict:
    """Load YAML configuration file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def setup_logging(config: dict):
    """Configure logging based on config."""
    log_level = config.get("logging", {}).get("log_level", "INFO")
    log_dir = config.get("logging", {}).get("log_dir", "outputs/logs")
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    
    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(f"{log_dir}/training.log"),
        ],
    )


def main():
    parser = argparse.ArgumentParser(description="Train Forge-1 model")
    parser.add_argument("--config", type=str, default="configs/local_config.yaml",
                        help="Path to training config YAML")
    parser.add_argument("--variant", type=str, default="nano",
                        choices=["nano", "mini"], help="Model variant")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from latest checkpoint")
    parser.add_argument("--data-path", type=str, default=None,
                        help="Path to preprocessed JSONL data file")
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    setup_logging(config)
    logger = logging.getLogger("train")
    
    logger.info("=" * 60)
    logger.info(f"AetherMind - Training Forge-1-{args.variant.title()}")
    logger.info("=" * 60)
    
    # Create model config
    model_config = Forge1Config.from_variant(args.variant)
    if config.get("hardware", {}).get("gradient_checkpointing", False):
        model_config.gradient_checkpointing = True
    
    logger.info(f"Model config: {model_config}")
    logger.info(f"Estimated parameters: {model_config.count_parameters_estimate()/1e6:.1f}M")
    
    # Setup tokenizer
    logger.info("Setting up tokenizer...")
    tokenizer = Forge1Tokenizer()
    actual_vocab_size = tokenizer.get_vocab_size_padded()
    model_config.vocab_size = actual_vocab_size
    logger.info(f"Vocab size (padded): {actual_vocab_size}")
    
    # Create model
    logger.info("Creating model...")
    model = Forge1Model(model_config)
    n_params = model.count_parameters()
    logger.info(f"Model created: {n_params/1e6:.1f}M trainable parameters")
    
    # Resize embeddings to match tokenizer
    if tokenizer.vocab_size != model_config.vocab_size:
        logger.warning(f"Adjusting vocab_size from {model_config.vocab_size} to {tokenizer.vocab_size}")
        model_config.vocab_size = tokenizer.vocab_size
        model = Forge1Model(model_config)
    
    # Load training data
    data_path = args.data_path or config.get("data", {}).get("preprocessed_path", "data/train.jsonl")
    if not Path(data_path).exists():
        logger.error(f"Training data not found: {data_path}")
        logger.error("Run data preprocessing first:")
        logger.error("  python data/download_datasets.py")
        logger.error("  python data/preprocess.py")
        sys.exit(1)
    
    logger.info(f"Loading training data from {data_path}...")
    max_samples = config.get("data", {}).get("max_samples", None)
    dataset = Forge1Dataset.from_jsonl(
        path=data_path,
        tokenizer=tokenizer,
        max_length=model_config.context_length,
        max_samples=max_samples,
    )
    
    train_dataloader = create_dataloader(
        dataset=dataset,
        batch_size=config.get("training", {}).get("batch_size", 2),
        pad_token_id=tokenizer.pad_token_id,
        shuffle=True,
        num_workers=config.get("data", {}).get("num_workers", 0),
        pin_memory=config.get("data", {}).get("pin_memory", False),
    )
    
    logger.info(f"Dataset: {len(dataset)} samples")
    logger.info(f"Batches per epoch: {len(train_dataloader)}")
    
    # Training callbacks
    grad_accum = config.get("training", {}).get("gradient_accumulation_steps", 8)
    steps_per_epoch = len(train_dataloader) // grad_accum
    total_epochs = config.get("training", {}).get("num_epochs", 3)
    total_steps = steps_per_epoch * total_epochs
    
    callbacks = [
        LoggingCallback(log_every_n=config.get("logging", {}).get("log_every_n_steps", 10)),
        MetricsTracker(),
    ]
    
    # Add Rich dashboard if enabled
    if config.get("logging", {}).get("use_rich_ui", True):
        dashboard = TrainingDashboard(
            total_steps=total_steps,
            total_epochs=total_epochs,
            steps_per_epoch=steps_per_epoch,
            model_name=f"Forge-1-{args.variant.title()}",
        )
        callbacks.append(dashboard)
    
    # Create trainer
    trainer = Trainer(
        model=model,
        config=config,
        train_dataloader=train_dataloader,
        callbacks=callbacks,
    )
    
    # Train!
    results = trainer.train()
    
    # Save final model
    final_dir = config.get("checkpointing", {}).get("final_model_dir", "outputs/final_model")
    Path(final_dir).mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), f"{final_dir}/forge1_{args.variant}_final.pt")
    tokenizer.save(f"{final_dir}/tokenizer")
    
    logger.info(f"Training complete! Final model saved to {final_dir}")
    logger.info(f"Results: {results}")


if __name__ == "__main__":
    main()
