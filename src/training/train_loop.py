"""
Main training loop for Hierarchical InLegalBERT.
Handles the complete training pipeline including data loading, model initialization,
training epochs, validation, and model checkpointing.
"""

import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from torch.optim import AdamW
from sklearn.metrics import classification_report
import wandb

from ..config import Config
from ..models import HierarchicalInLegalBERT
from ..data import ILDCDataset, load_ildc_dataset
from ..utils import setup_wandb, set_seed, is_main_process
from .trainer import train_epoch, evaluate


def train(config, rank, world_size, local_rank, is_distributed):
    """Main training function"""

    if is_main_process(rank):
        print("=" * 60)
        print("HIERARCHICAL INLEGALBERT - MULTI-GPU TRAINING")
        print("=" * 60)
        print(f"GPUs: {world_size}")
        print(f"Batch size per GPU: {config.batch_size}")
        print(f"Total batch size: {config.batch_size * world_size}")
        print(f"Mixed precision (BF16): {config.bf16}")

    set_seed(config.seed)

    # Setup W&B
    wandb_enabled = setup_wandb(rank, config)

    # Load dataset
    dataset = load_ildc_dataset(rank)

    # Load tokenizer
    if is_main_process(rank):
        print(f"\nLoading tokenizer: {config.base_model}")

    tokenizer = AutoTokenizer.from_pretrained(
        config.base_model, cache_dir=config.cache_dir
    )

    # Create datasets
    train_dataset = ILDCDataset(
        dataset["train"], tokenizer, config, config.max_train_samples
    )
    val_dataset = ILDCDataset(
        dataset["validation"], tokenizer, config, config.max_eval_samples
    )
    test_dataset = ILDCDataset(dataset["test"], tokenizer, config)

    if is_main_process(rank):
        print(f"\nDataset sizes:")
        print(f"  Train: {len(train_dataset)}")
        print(f"  Validation: {len(val_dataset)}")
        print(f"  Test: {len(test_dataset)}")

    # Create distributed samplers
    if is_distributed:
        train_sampler = DistributedSampler(
            train_dataset, num_replicas=world_size, rank=rank, shuffle=True
        )
        val_sampler = DistributedSampler(
            val_dataset, num_replicas=world_size, rank=rank, shuffle=False
        )
        test_sampler = DistributedSampler(
            test_dataset, num_replicas=world_size, rank=rank, shuffle=False
        )
    else:
        train_sampler = None
        val_sampler = None
        test_sampler = None

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        prefetch_factor=config.prefetch_factor,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        sampler=val_sampler,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        sampler=test_sampler,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
    )

    # Initialize model
    if is_main_process(rank):
        print(f"\nInitializing model...")

    model = HierarchicalInLegalBERT(config).to(local_rank)

    if is_distributed:
        model = DDP(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=config.find_unused_parameters,
        )

    if is_main_process(rank):
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")

    # Optimizer and scheduler
    optimizer = AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )

    total_steps = len(train_loader) * config.num_epochs
    warmup_steps = int(total_steps * config.warmup_ratio)

    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    scaler = torch.amp.GradScaler("cuda") if config.bf16 else None

    # Training loop
    best_f1 = 0

    if is_main_process(rank):
        print(f"\nStarting training for {config.num_epochs} epochs...")
        print(f"Warmup steps: {warmup_steps}")
        print(f"Total steps: {total_steps}")
        print("=" * 60)

    for epoch in range(config.num_epochs):
        if is_main_process(rank):
            print(f"\nEpoch {epoch + 1}/{config.num_epochs}")
            print("-" * 60)

        if is_distributed and train_sampler is not None:
            train_sampler.set_epoch(epoch)

        # Train
        train_metrics = train_epoch(
            model, train_loader, optimizer, scheduler, local_rank, rank, config, scaler
        )

        if is_main_process(rank):
            print(f"Train Loss: {train_metrics['loss']:.4f}")
            print(f"Train Accuracy: {train_metrics['accuracy']:.4f}")
            print(f"Train F1: {train_metrics['f1']:.4f}")

        # Evaluate
        val_metrics = evaluate(model, val_loader, local_rank, rank, config)

        if is_main_process(rank):
            print(f"\nVal Loss: {val_metrics['loss']:.4f}")
            print(f"Val Accuracy: {val_metrics['accuracy']:.4f}")
            print(f"Val F1: {val_metrics['f1']:.4f}")
            print(f"Val Precision: {val_metrics['precision']:.4f}")
            print(f"Val Recall: {val_metrics['recall']:.4f}")

        # Log to W&B
        if wandb_enabled and is_main_process(rank):
            wandb.log(
                {
                    "epoch": epoch + 1,
                    "train/loss": train_metrics["loss"],
                    "train/accuracy": train_metrics["accuracy"],
                    "train/f1": train_metrics["f1"],
                    "val/loss": val_metrics["loss"],
                    "val/accuracy": val_metrics["accuracy"],
                    "val/f1": val_metrics["f1"],
                    "val/precision": val_metrics["precision"],
                    "val/recall": val_metrics["recall"],
                    "learning_rate": scheduler.get_last_lr()[0],
                }
            )

        # Save best model (only main process)
        if is_main_process(rank) and val_metrics["f1"] > best_f1:
            best_f1 = val_metrics["f1"]
            print(f"\n✓ New best F1: {best_f1:.4f} - Saving model...")

            os.makedirs(config.output_dir, exist_ok=True)

            # Save model (unwrap DDP if needed)
            model_to_save = model.module if is_distributed else model

            # Extract config as dict
            config_dict = (
                {k: v for k, v in vars(config).items() if not k.startswith("_")}
                if hasattr(config, "__dict__")
                else {
                    k: getattr(config, k)
                    for k in dir(config)
                    if not k.startswith("_") and not callable(getattr(config, k))
                }
            )

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model_to_save.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_f1": best_f1,
                    "config": config_dict,
                },
                os.path.join(config.output_dir, "best_model.pt"),
            )

            tokenizer.save_pretrained(config.output_dir)

    # Test evaluation (only main process)
    if is_main_process(rank):
        print("\n" + "=" * 60)
        print("FINAL TEST EVALUATION")
        print("=" * 60)

        # Load best model
        checkpoint = torch.load(os.path.join(config.output_dir, "best_model.pt"))
        if is_distributed:
            model.module.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint["model_state_dict"])

    # Synchronize before test
    if is_distributed:
        dist.barrier()

    test_metrics = evaluate(model, test_loader, local_rank, rank, config)

    if is_main_process(rank):
        print(f"\nTest Loss: {test_metrics['loss']:.4f}")
        print(f"Test Accuracy: {test_metrics['accuracy']:.4f}")
        print(f"Test F1: {test_metrics['f1']:.4f}")
        print(f"Test Precision: {test_metrics['precision']:.4f}")
        print(f"Test Recall: {test_metrics['recall']:.4f}")

        print("\nClassification Report:")
        print(
            classification_report(
                test_metrics["true_labels"],
                test_metrics["predictions"],
                target_names=["Reject", "Accept"],
            )
        )

        if wandb_enabled:
            wandb.log(
                {
                    "test/loss": test_metrics["loss"],
                    "test/accuracy": test_metrics["accuracy"],
                    "test/f1": test_metrics["f1"],
                    "test/precision": test_metrics["precision"],
                    "test/recall": test_metrics["recall"],
                }
            )
            wandb.finish()

    return test_metrics
