"""
ILDC Dataset implementation with hierarchical chunking support.
"""

import torch
from torch.utils.data import Dataset
from datasets import load_dataset


def load_ildc_dataset(rank):
    """Load ILDC dataset from HuggingFace"""
    from ..utils.distributed import is_main_process
    from ..config import Config

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

        # Suppress warnings for long sequences - we chunk them anyway
        import warnings

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Token indices sequence length")
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
