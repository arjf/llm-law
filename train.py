#!/usr/bin/env python3
"""
Hierarchical InLegalBERT - Multi GPU Training Script
Main entry point for training using the restructured modular codebase.

Expected training time: 3-4 hours for 5 epochs on 2x RTX 4090

Usage:
    # With torchrun
    torchrun --nproc_per_node=2 train.py

    # With accelerate
    accelerate launch --num_processes=2 train.py

    # With custom arguments
    torchrun --nproc_per_node=2 train.py --epochs 10 --batch_size 16 --lr 3e-5

    # Quick test mode
    torchrun --nproc_per_node=2 train.py --quick_test

Author: Your Name
Date: October 2025
"""

import argparse
import warnings

from src.config import Config
from src.utils import setup_distributed, cleanup_distributed, is_main_process
from src.training import train

warnings.filterwarnings("ignore")


def main():
    """Main execution"""

    parser = argparse.ArgumentParser(
        description="Train Hierarchical InLegalBERT on ILDC dataset"
    )
    parser.add_argument(
        "--epochs", type=int, default=5, help="Number of training epochs"
    )
    parser.add_argument("--batch_size", type=int, default=12, help="Batch size per GPU")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate")
    parser.add_argument(
        "--quick_test",
        action="store_true",
        help="Run in quick test mode with reduced dataset",
    )
    args = parser.parse_args()

    # Update config
    Config.num_epochs = args.epochs
    Config.batch_size = args.batch_size
    Config.learning_rate = args.lr

    if args.quick_test:
        Config.max_train_samples = 5000
        Config.max_eval_samples = 1000
        print("\n🚀 QUICK TEST MODE")

    # Setup distributed
    rank, world_size, local_rank, is_distributed = setup_distributed()

    if is_main_process(rank):
        print(f"\n🚀 Training on {world_size} GPU(s)")

    try:
        # Train
        metrics = train(Config, rank, world_size, local_rank, is_distributed)

        if is_main_process(rank):
            print("\n" + "=" * 60)
            print("TRAINING COMPLETE!")
            print("=" * 60)
            print(f"\nBest Test F1: {metrics['f1']:.4f}")
            print(f"Model saved to: {Config.output_dir}")

    finally:
        # Cleanup
        if is_distributed:
            cleanup_distributed()


if __name__ == "__main__":
    main()
