#!/usr/bin/env python3
"""
Hierarchical InLegalBERT - Multi GPU
HF Accel

Expected training time was 3-4 hours for 5 epochs

#Usage
    accelerate launch --num_processes 2 train_hierarchical_multi_gpu.py

Author: Your Name
Date: October 2025
"""

import os
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers import (
    AutoTokenizer,
    AutoModel,
    get_linear_schedule_with_warmup,
)
from torch.optim import AdamW
from datasets import load_dataset
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    classification_report,
)
from tqdm import tqdm
import wandb
import argparse
import numpy as np
import warnings

warnings.filterwarnings("ignore")


# ==================== CONFIGURATION ====================


class Config:
    # Model architecture
    base_model = "law-ai/InLegalBERT"
    chunk_size = 512  # Increased from 384 (more VRAM available)
    chunk_overlap = 100
    max_chunks = 16
    hidden_size = 768
    gru_hidden_size = 256
    gru_num_layers = 2
    num_classes = 2
    dropout = 0.3

    # Training
    batch_size = 12  # Per GPU
    gradient_accumulation_steps = 1  # No need with larger batch
    num_epochs = 5
    learning_rate = 2e-5
    weight_decay = 0.01
    warmup_ratio = 0.1
    max_grad_norm = 1.0

    # Precision
    fp16 = False  # 4090s support BF16 better
    bf16 = True  # Better numerical stability than FP16

    # Quick test mode
    max_train_samples = 5000  # Set to 5000 for quick test
    max_eval_samples = 1000  # Set to 1000 for quick test

    # Paths
    cache_dir = "./cache"
    output_dir = "./models/hierarchical_inlegalbert"

    # W&B
    wandb_project = "indian-legal-llm"
    wandb_run_name = "hierarchical-inlegalbert-4090x2"
    use_wandb = True

    # HuggingFace Hub
    push_to_hub = False  # Set to True to push model to HF Hub
    hub_model_id = None  # e.g., "your-username/hierarchical-inlegalbert"
    hub_private_repo = False  # Set to True for private repository

    # DDP settings
    backend = "nccl"  # Best for NVIDIA GPUs
    find_unused_parameters = False

    # Device
    num_workers = 8  # More workers for faster data loading
    pin_memory = True
    prefetch_factor = 2
    seed = 42

    # Logging
    logging_steps = 50
    eval_steps = 500
    save_steps = 500


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
    dist.init_process_group(backend=Config.backend, init_method="env://")

    return rank, world_size, local_rank, True


def cleanup_distributed():
    """Cleanup distributed training"""
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_process(rank):
    """Check if current process is main"""
    return rank == 0


# ==================== W&B SETUP ====================


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


# ==================== DATA LOADING ====================


def load_ildc_dataset(rank):
    """Load ILDC dataset from HuggingFace"""

    if is_main_process(rank):
        print("=" * 60)
        print("LOADING ILDC DATASET")
        print("=" * 60)

    try:
        dataset = load_dataset(
            "Exploration-Lab/IL-TUR", "cjpe", cache_dir=Config.cache_dir
        )

        if is_main_process(rank):
            print(f"✓ Dataset loaded:")
            print(f"  Available splits: {list(dataset.keys())}")
            print(f"  Multi-train: {len(dataset['multi_train'])}")
            print(f"  Multi-dev: {len(dataset['multi_dev'])}")
            print(f"  Test: {len(dataset['test'])}")

        # Rename splits to match expected names
        from datasets import DatasetDict

        dataset = DatasetDict(
            {
                "train": dataset["multi_train"],
                "validation": dataset["multi_dev"],
                "test": dataset["test"],
            }
        )

        return dataset

    except Exception as e:
        if is_main_process(rank):
            print(f"✗ Error loading dataset: {e}")
        raise


class ILDCDataset(Dataset):
    """ILDC Dataset with hierarchical chunking"""

    def __init__(self, dataset, tokenizer, config, max_samples=None):
        if max_samples:
            indices = list(range(min(max_samples, len(dataset))))
            self.data = dataset.select(indices)
        else:
            self.data = dataset

        self.tokenizer = tokenizer
        self.config = config

    def __len__(self):
        return len(self.data)

    def chunk_text(self, text):
        """Split text into overlapping chunks"""

        tokens = self.tokenizer.tokenize(text)
        chunks = []
        stride = self.config.chunk_size - self.config.chunk_overlap

        for i in range(0, len(tokens), stride):
            chunk_tokens = tokens[i : i + self.config.chunk_size]
            if len(chunk_tokens) < 30:
                break
            chunk_text = self.tokenizer.convert_tokens_to_string(chunk_tokens)
            chunks.append(chunk_text)
            if len(chunks) >= self.config.max_chunks:
                break

        if not chunks:
            chunks = [text[: self.config.chunk_size * 3]]

        return chunks

    def __getitem__(self, idx):
        sample = self.data[idx]

        text = sample.get("text", sample.get("facts", sample.get("case_text", "")))
        label = sample.get("label", sample.get("decision", sample.get("judgment", 0)))

        if isinstance(label, str):
            label = 1 if label.lower() in ["accept", "accepted", "1", "yes"] else 0

        chunks = self.chunk_text(text)
        encodings = self.tokenizer(
            chunks,
            padding="max_length",
            truncation=True,
            max_length=self.config.chunk_size,
            return_tensors="pt",
        )

        num_chunks = encodings["input_ids"].size(0)
        if num_chunks < self.config.max_chunks:
            pad_size = self.config.max_chunks - num_chunks
            input_ids = torch.cat(
                [
                    encodings["input_ids"],
                    torch.zeros(pad_size, self.config.chunk_size, dtype=torch.long),
                ]
            )
            attention_mask = torch.cat(
                [
                    encodings["attention_mask"],
                    torch.zeros(pad_size, self.config.chunk_size, dtype=torch.long),
                ]
            )
        else:
            input_ids = encodings["input_ids"][: self.config.max_chunks]
            attention_mask = encodings["attention_mask"][: self.config.max_chunks]

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": torch.tensor(label, dtype=torch.long),
            "num_chunks": torch.tensor(num_chunks, dtype=torch.long),
        }


# ==================== MODEL ====================


class HierarchicalInLegalBERT(nn.Module):
    """Hierarchical BERT: InLegalBERT + BiGRU + Attention"""

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.bert = AutoModel.from_pretrained(
            config.base_model, cache_dir=config.cache_dir
        )

        self.bigru = nn.GRU(
            input_size=config.hidden_size,
            hidden_size=config.gru_hidden_size,
            num_layers=config.gru_num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=config.dropout if config.gru_num_layers > 1 else 0,
        )

        gru_output_size = config.gru_hidden_size * 2
        self.attention = nn.Sequential(
            nn.Linear(gru_output_size, 128), nn.Tanh(), nn.Linear(128, 1)
        )

        self.classifier = nn.Sequential(
            nn.Dropout(config.dropout),
            nn.Linear(gru_output_size, 256),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(256, config.num_classes),
        )

    def forward(self, input_ids, attention_mask, num_chunks):
        batch_size, max_chunks, seq_len = input_ids.size()

        input_ids_flat = input_ids.view(-1, seq_len)
        attention_mask_flat = attention_mask.view(-1, seq_len)

        bert_output = self.bert(
            input_ids=input_ids_flat, attention_mask=attention_mask_flat
        )

        cls_embeddings = bert_output.last_hidden_state[:, 0, :]
        chunk_embeddings = cls_embeddings.view(batch_size, max_chunks, -1)

        gru_output, _ = self.bigru(chunk_embeddings)

        attention_scores = self.attention(gru_output).squeeze(-1)
        chunk_mask = torch.arange(max_chunks).unsqueeze(0).to(num_chunks.device)
        chunk_mask = (chunk_mask < num_chunks.unsqueeze(1)).float()
        attention_scores = attention_scores.masked_fill(chunk_mask == 0, -1e9)
        attention_weights = torch.softmax(attention_scores, dim=1)

        document_embedding = torch.bmm(
            attention_weights.unsqueeze(1), gru_output
        ).squeeze(1)

        logits = self.classifier(document_embedding)

        return logits, attention_weights


# ==================== TRAINING ====================


def set_seed(seed=42):
    """Set random seeds for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    import random

    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_epoch(
    model, dataloader, optimizer, scheduler, local_rank, rank, config, scaler=None
):
    """Train for one epoch"""

    model.train()
    total_loss = 0
    predictions = []
    true_labels = []

    if is_main_process(rank):
        pbar = tqdm(dataloader, desc="Training")
    else:
        pbar = dataloader

    for step, batch in enumerate(pbar):
        input_ids = batch["input_ids"].to(local_rank)
        attention_mask = batch["attention_mask"].to(local_rank)
        labels = batch["labels"].to(local_rank)
        num_chunks = batch["num_chunks"].to(local_rank)

        if config.bf16 and scaler is not None:
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                logits, _ = model(input_ids, attention_mask, num_chunks)
                loss = nn.CrossEntropyLoss()(logits, labels)
        else:
            logits, _ = model(input_ids, attention_mask, num_chunks)
            loss = nn.CrossEntropyLoss()(logits, labels)

        if config.bf16 and scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            optimizer.step()

        scheduler.step()
        optimizer.zero_grad()

        total_loss += loss.item()
        preds = torch.argmax(logits, dim=1).cpu().numpy()
        predictions.extend(preds)
        true_labels.extend(labels.cpu().numpy())

        if is_main_process(rank) and isinstance(pbar, tqdm):
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(true_labels, predictions)
    f1 = f1_score(true_labels, predictions, average="macro")

    return {"loss": avg_loss, "accuracy": accuracy, "f1": f1}


def evaluate(model, dataloader, local_rank, rank, config):
    """Evaluate model"""

    model.eval()
    total_loss = 0
    predictions = []
    true_labels = []

    if is_main_process(rank):
        pbar = tqdm(dataloader, desc="Evaluating")
    else:
        pbar = dataloader

    with torch.no_grad():
        for batch in pbar:
            input_ids = batch["input_ids"].to(local_rank)
            attention_mask = batch["attention_mask"].to(local_rank)
            labels = batch["labels"].to(local_rank)
            num_chunks = batch["num_chunks"].to(local_rank)

            if config.bf16:
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    logits, _ = model(input_ids, attention_mask, num_chunks)
                    loss = nn.CrossEntropyLoss()(logits, labels)
            else:
                logits, _ = model(input_ids, attention_mask, num_chunks)
                loss = nn.CrossEntropyLoss()(logits, labels)

            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            predictions.extend(preds)
            true_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(true_labels, predictions)
    f1 = f1_score(true_labels, predictions, average="macro")
    precision = precision_score(true_labels, predictions, average="macro")
    recall = recall_score(true_labels, predictions, average="macro")

    return {
        "loss": avg_loss,
        "accuracy": accuracy,
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "predictions": predictions,
        "true_labels": true_labels,
    }


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

            # Push to HuggingFace Hub if enabled
            if config.push_to_hub and config.hub_model_id:
                try:
                    print(
                        f"\n📤 Pushing model to HuggingFace Hub: {config.hub_model_id}"
                    )

                    # Save model in HuggingFace format
                    model_to_save.save_pretrained(
                        config.output_dir,
                        safe_serialization=True,
                    )

                    # Push to hub
                    from huggingface_hub import HfApi

                    api = HfApi()

                    api.create_repo(
                        repo_id=config.hub_model_id,
                        private=config.hub_private_repo,
                        exist_ok=True,
                    )

                    api.upload_folder(
                        folder_path=config.output_dir,
                        repo_id=config.hub_model_id,
                        commit_message=f"Upload model - Epoch {epoch + 1} - F1: {best_f1:.4f}",
                    )

                    print(
                        f"✓ Model pushed to https://huggingface.co/{config.hub_model_id}"
                    )

                except Exception as e:
                    print(f"⚠ Failed to push to Hub: {e}")
                    print("  Model saved locally. You can push manually later.")

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


# ==================== MAIN ====================


def main():
    """Main execution"""

    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=12)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--quick_test", action="store_true")
    parser.add_argument(
        "--push_to_hub", action="store_true", help="Push model to HuggingFace Hub"
    )
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="HuggingFace Hub model ID (e.g., username/model-name)",
    )
    parser.add_argument(
        "--hub_private",
        action="store_true",
        help="Make HuggingFace Hub repository private",
    )
    args = parser.parse_args()

    # Update config
    Config.num_epochs = args.epochs
    Config.batch_size = args.batch_size
    Config.learning_rate = args.lr

    # HuggingFace Hub settings
    if args.push_to_hub:
        Config.push_to_hub = True
        Config.hub_model_id = args.hub_model_id
        Config.hub_private_repo = args.hub_private

        if not args.hub_model_id:
            print("\n⚠ Warning: --push_to_hub requires --hub_model_id")
            print("  Example: --hub_model_id username/hierarchical-inlegalbert")
            Config.push_to_hub = False

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
