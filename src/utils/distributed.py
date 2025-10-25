"""
Distributed training utilities for multi-GPU setup.
"""

import os
import torch
import torch.distributed as dist


def setup_distributed():
    """Initialize distributed training"""

    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        # torchrun sets these automatically
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])
    elif "LOCAL_RANK" in os.environ:
        # Accelerate sets LOCAL_RANK
        local_rank = int(os.environ["LOCAL_RANK"])
        rank = local_rank
        world_size = torch.cuda.device_count()
    else:
        # Single GPU fallback
        print(" No distributed environment detected - using single GPU")
        return 0, 1, 0, False

    torch.cuda.set_device(local_rank)
    from ..config import Config

    dist.init_process_group(backend=Config.backend, init_method="env://")

    return rank, world_size, local_rank, True


def cleanup_distributed():
    """Cleanup distributed training"""
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_process(rank):
    """Check if current process is main"""
    return rank == 0
