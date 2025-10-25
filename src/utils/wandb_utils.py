"""
Weights & Biases (wandb) integration utilities.
"""

import torch
import wandb
from .distributed import is_main_process


def setup_wandb(rank, config):
    """Setup W&B only on main process"""

    if not is_main_process(rank) or not config.use_wandb:
        return False

    try:
        wandb.init(
            project=config.wandb_project,
            name=config.wandb_run_name,
            config={
                "model": config.base_model,
                "gpus": torch.cuda.device_count(),
                "chunk_size": config.chunk_size,
                "max_chunks": config.max_chunks,
                "batch_size_per_gpu": config.batch_size,
                "total_batch_size": config.batch_size * torch.cuda.device_count(),
                "learning_rate": config.learning_rate,
                "epochs": config.num_epochs,
                "bf16": config.bf16,
            },
        )
        return True
    except Exception as e:
        print(f" W&B initialization failed: {e}")
        return False
