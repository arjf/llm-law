"""
Hierarchical InLegalBERT - Multi-GPU Training Framework
A modular implementation for training hierarchical document classification models
on Indian legal case data.

Modules:
    - config: Configuration settings
    - models: Model architectures
    - data: Dataset loading and processing
    - training: Training and evaluation loops
    - utils: Utilities for distributed training, logging, and helpers
"""

from .config import Config
from .models import HierarchicalInLegalBERT
from .data import ILDCDataset, load_ildc_dataset
from .training import train, train_epoch, evaluate
from .utils import (
    setup_distributed,
    cleanup_distributed,
    is_main_process,
    set_seed,
    setup_wandb,
)

__version__ = "1.0.0"

__all__ = [
    # Config
    "Config",
    # Models
    "HierarchicalInLegalBERT",
    # Data
    "ILDCDataset",
    "load_ildc_dataset",
    # Training
    "train",
    "train_epoch",
    "evaluate",
    # Utils
    "setup_distributed",
    "cleanup_distributed",
    "is_main_process",
    "set_seed",
    "setup_wandb",
]
