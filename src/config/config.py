"""
Configuration settings for Hierarchical InLegalBERT training.
"""


class Config:
    """Configuration class for model training and evaluation."""

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
