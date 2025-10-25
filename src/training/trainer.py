"""
Training and evaluation functions for Hierarchical InLegalBERT.
"""

import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from tqdm import tqdm
from ..utils.distributed import is_main_process


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
