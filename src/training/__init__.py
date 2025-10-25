"""
Training module for Hierarchical InLegalBERT.
Contains training loop, trainer functions, and evaluation utilities.
"""

from .trainer import train_epoch, evaluate
from .train_loop import train

__all__ = ["train_epoch", "evaluate", "train"]
