#!/usr/bin/env python3
"""
Multi-task Fine-tuning: PCR + LSI + SUMM on Gemma 3n-E4B with RAG Integration
Uses Zilliz Cloud (Milvus) for vector retrieval + InLegalBERT embeddings
Optimized with Unsloth

Environment Variables:
  MILVUS_URI (e.g., https://your-cluster.zillizcloud.com)
  MILVUS_TOKEN (API key or username:password)
  MILVUS_COLLECTION

Usage:
  python iltur_multitask_finetune.py --build-rag-index   # Build vector DB first
  python iltur_multitask_finetune.py --train             # Fine-tune model
  python iltur_multitask_finetune.py --inference         # Test trained model
"""

import os
import json
import torch
import numpy as np
from typing import List, Dict, Tuple
from tqdm import tqdm
from datasets import load_dataset, concatenate_datasets, Dataset
from transformers import AutoTokenizer, AutoModel, TextStreamer

from pymilvus import (
    MilvusClient,
    connections,
    FieldSchema,
    CollectionSchema,
    DataType,
    Collection,
    utility,
)

# ==================== CONFIG ====================


class Config:
    # Models
    base_model = "unsloth/gemma-3n-E4B-it-unsloth-bnb-4bit"
    retrieval_model = "law-ai/InLegalBERT"

    # Training
    output_dir = "gemma-3n-iltur-multitask"
    max_seq_length = 2048  # Reduce to 1024 if OOM
    per_device_train_batch_size = 1
    gradient_accumulation_steps = 8
    num_train_epochs = 3
    learning_rate = 1e-4
    warmup_steps = 20
    logging_steps = 10
    save_steps = 500

    # LoRA
    lora_r = 16
    lora_alpha = 16
    lora_dropout = 0

    # RAG
    embedding_dim = 768
    top_k_retrieve = 5
    chunk_size = 512
    chunk_overlap = 100

    from dotenv import load_dotenv

    load_dotenv()

    milvus_uri = os.environ.get("MILVUS_URI")
    milvus_token = os.environ.get("MILVUS_TOKEN")  # API key or username:password
    milvus_port = os.environ.get("MILVUS_PORT", default=443)
    print(f"URI={milvus_uri} \t TOKEN={milvus_token} \t PORT={milvus_port}")

    # Legacy approach: Host + Port + User/Password
    milvus_collection = os.environ.get("MILVUS_COLLECTION", "iltur_legal_kb")

    # Dataset ratios for multi-task (PCR: 40%, LSI: 35%, SUMM: 25%)
    task_ratios = {"pcr": 0.40, "lsi": 0.35, "summ": 0.25}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ==================== RAG INDEXER ====================


class ZillizRAGIndexer:
    """Build and query Zilliz Cloud vector DB for legal documents"""

    def __init__(self, config: Config):
        self.config = config
        print(f"[RAG] Loading retrieval model: {config.retrieval_model}")
        self.tokenizer = AutoTokenizer.from_pretrained(config.retrieval_model)
        self.model = AutoModel.from_pretrained(config.retrieval_model).to(config.device)
        self.model.eval()

        self._connect_milvus()
        self.collection = None

    def _connect_milvus(self):
        if self.config.milvus_uri and self.config.milvus_token:
            print(f"[RAG] Connecting to Zilliz Cloud: {self.config.milvus_uri}")
            connections.connect(
                alias="default",
                uri=self.config.milvus_uri,
                token=self.config.milvus_token,
            )
        else:
            raise ValueError("Mising URI and/or TOKEN")

    def _create_collection(self):
        """Create collection with same schema as legal_rag_qa.py"""
        if utility.has_collection(self.config.milvus_collection):
            self.collection = Collection(self.config.milvus_collection)
            print(f"[RAG] Using existing collection: {self.config.milvus_collection}")
            return

        fields = [
            FieldSchema(
                name="primary_key", dtype=DataType.INT64, is_primary=True, auto_id=True
            ),
            FieldSchema(
                name="embedding",
                dtype=DataType.FLOAT_VECTOR,
                dim=self.config.embedding_dim,
            ),
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=8192),
            FieldSchema(name="label", dtype=DataType.VARCHAR, max_length=256),
            FieldSchema(name="case_id", dtype=DataType.VARCHAR, max_length=256),
        ]
        schema = CollectionSchema(fields, description="IL-TUR Legal Knowledge Base")
        self.collection = Collection(name=self.config.milvus_collection, schema=schema)
        print(f"[RAG] Created collection: {self.config.milvus_collection}")

    def _encode_batch(self, texts: List[str]) -> np.ndarray:
        """Encode texts to embeddings (CLS token)"""
        enc = self.tokenizer(
            texts, max_length=512, truncation=True, padding=True, return_tensors="pt"
        ).to(self.config.device)
        with torch.no_grad():
            outs = self.model(**enc)
            embs = outs.last_hidden_state[:, 0, :].cpu().numpy().astype("float32")
        return embs

    def _chunk_text(self, text: str) -> List[str]:
        """Chunk text with overlap"""
        # Truncate text to avoid tokenizer warnings (approx 4 chars per token)
        max_chars = self.config.chunk_size * 16 * 4  # 16 max chunks * 512 tokens * ~4 chars
        if len(text) > max_chars:
            text = text[:max_chars]

        tokens = self.tokenizer.tokenize(text)
        stride = self.config.chunk_size - self.config.chunk_overlap
        chunks = []
        for i in range(0, len(tokens), stride):
            chunk_tokens = tokens[i : i + self.config.chunk_size]
            if len(chunk_tokens) < 30:
                break
            chunks.append(self.tokenizer.convert_tokens_to_string(chunk_tokens))
            if len(chunks) >= 16:
                break
        return chunks if chunks else [text[:1500]]

    def build_index(self, dataset_name="Exploration-Lab/IL-TUR", all_tasks=True):
        """
        Build index from IL-TUR datasets
        Args:
            all_tasks: If True, index ALL 8 IL-TUR tasks for comprehensive KB
                      If False, only index the 3 training tasks (PCR, LSI, SUMM)
        """
        self._create_collection()

        # All 8 IL-TUR tasks or just training tasks
        # Map subset names to their correct config and split names
        if all_tasks:
            # Some datasets have multiple splits we should load all of them
            subset_configs = {
                "lner": ["fold_1", "fold_2", "fold_3"],  # Legal NER - all folds
                "lsi": ["train"],   # Legal Statute Identification
                "pcr": ["train_queries"],  # Prior Case Retrieval (use queries)
                "summ": ["train"],  # Summarization
                "cjpe": ["single_train"],  # Court Judgment Prediction
                "bail": ["train_all"],  # Bail Prediction
                "rr": ["CL_train"],  # Rhetorical Role (CL_train or IT_train)
                "lmt": ["acts", "cci_faq", "ip"],  # Legal Machine Translation - all splits
            }
            print("[RAG] Building COMPREHENSIVE knowledge base from all 8 IL-TUR tasks")
        else:
            subset_configs = {
                "pcr": ["train_queries"],
                "lsi": ["train"],
                "summ": ["train"],
            }
            print("[RAG] Building knowledge base from 3 training tasks only")

        all_chunks, all_labels, all_case_ids = [], [], []

        for subset, split_names in subset_configs.items():
            for split_name in split_names:
                try:
                    print(f"[RAG] Loading {subset} dataset (split: {split_name})...")
                    ds = load_dataset(dataset_name, subset, split=split_name)

                    for idx, item in enumerate(tqdm(ds, desc=f"Chunking {subset}_{split_name}")):
                        # Extract text based on task type
                        if subset == "pcr":
                            text = item.get("query", item.get("text", ""))
                        elif subset == "lsi":
                            text = item.get("facts", item.get("text", ""))
                        elif subset == "summ":
                            text = item.get("text", item.get("document", ""))
                        elif subset == "cjpe":
                            text = item.get("facts", item.get("text", ""))
                        elif subset == "lner":
                            text = item.get("text", "")
                        elif subset == "rr":
                            text = item.get("text", "")
                        elif subset == "bail":
                            text = item.get("text", item.get("facts", ""))
                        elif subset == "lmt":
                            text = item.get("source", item.get("text", ""))
                        else:
                            text = item.get("text", "")

                        # Skip empty texts
                        if not isinstance(text, str) or len(text) < 50:
                            continue

                        # Chunk and add to lists
                        chunks = self._chunk_text(text)
                        for c_id, chunk in enumerate(chunks):
                            all_chunks.append(chunk)
                            label = f"{subset}_{split_name}_doc{idx}_chunk{c_id}"
                            case_id = item.get("case_id", item.get("id", f"{subset}_{split_name}_{idx}"))
                            if not isinstance(case_id, str):
                                case_id = str(case_id)
                            all_labels.append(label)
                            all_case_ids.append(case_id)

                except Exception as e:
                    print(f"[RAG] Warning: Failed to load {subset} (split: {split_name}): {e}")
                    continue

        print(f"[RAG] Total chunks to index: {len(all_chunks)}")

        if len(all_chunks) == 0:
            print("[RAG] No chunks to insert. Skipping.")
            return

        # Pre-encode all embeddings (faster than encoding during insertion)
        print("[RAG] Encoding embeddings...")
        all_embeddings = []
        encoding_batch_size = 128  # Larger batch for encoding
        for i in tqdm(range(0, len(all_chunks), encoding_batch_size), desc="Encoding"):
            batch_texts = all_chunks[i : i + encoding_batch_size]
            embeddings = self._encode_batch(batch_texts)
            all_embeddings.append(embeddings)
            torch.cuda.empty_cache()  # Clear GPU memory

        # Concatenate all embeddings
        import numpy as np
        all_embeddings = np.vstack(all_embeddings)
        print(f"[RAG] Encoded {len(all_embeddings)} embeddings")

        # Batch insert (larger batches for faster network transfer)
        batch_size = 256  # Increased from 64
        print(f"[RAG] Inserting into Zilliz with batch_size={batch_size}...")
        for i in tqdm(
            range(0, len(all_chunks), batch_size), desc="Inserting to Zilliz"
        ):
            batch_embeddings = all_embeddings[i : i + batch_size]
            batch_texts = all_chunks[i : i + batch_size]
            batch_labels = all_labels[i : i + batch_size]
            batch_case_ids = all_case_ids[i : i + batch_size]

            entities = [batch_embeddings.tolist(), batch_texts, batch_labels, batch_case_ids]
            self.collection.insert(entities)

            # Flush less frequently (every 20 batches instead of 10)
            if (i // batch_size) % 20 == 0:
                self.collection.flush()

        # Final flush
        self.collection.flush()

        # Create index (skip if already exists)
        print("[RAG] Creating HNSW index...")
        index_params = {
            "index_type": "HNSW",
            "metric_type": "COSINE",
            "params": {"M": 16, "efConstruction": 200},
        }
        try:
            self.collection.create_index(field_name="embedding", index_params=index_params)
            print("[RAG] Index created successfully")
        except Exception as e:
            if "already exists" in str(e).lower() or "distinct index" in str(e).lower():
                print("[RAG] Index already exists, skipping creation")
            else:
                raise

        self.collection.load()
        print(f"[RAG] Index built! Total entities: {self.collection.num_entities}")

    def load_index(self):
        """Load existing index"""
        if not utility.has_collection(self.config.milvus_collection):
            raise RuntimeError(f"Collection {self.config.milvus_collection} not found!")
        self.collection = Collection(self.config.milvus_collection)
        self.collection.load()
        print(f"[RAG] Loaded collection: {self.collection.num_entities} entities")

    def retrieve(self, query: str, top_k: int = 5) -> List[Dict]:
        """Retrieve top-k relevant chunks"""
        q_emb = self._encode_batch([query]).astype("float32").tolist()
        search_params = {"metric_type": "COSINE", "params": {"ef": 50}}

        results = self.collection.search(
            q_emb,
            "embedding",
            param=search_params,
            limit=top_k,
            output_fields=["text", "label", "case_id"],
        )

        retrieved = []
        for hits in results:
            for hit in hits:
                ent = hit.entity
                retrieved.append(
                    {
                        "text": ent.get("text", ""),
                        "score": float(getattr(hit, "score", 0.0)),
                        "label": ent.get("label", ""),
                        "case_id": ent.get("case_id", ""),
                    }
                )
        return retrieved


# ==================== DATA PREPARATION ====================


class ILTURDataPrep:
    """Prepare multi-task training data for PCR, LSI, SUMM"""

    def __init__(self, config: Config, rag_indexer: ZillizRAGIndexer = None):
        self.config = config
        self.rag = rag_indexer

    def _format_pcr_with_rag(self, item: Dict) -> Dict:
        """Prior Case Retrieval: Retrieve candidates, model ranks them"""
        query_case = item.get("query", item.get("text", ""))[:1000]

        # Use RAG to get candidate cases
        if self.rag:
            candidates = self.rag.retrieve(query_case, top_k=10)
            candidate_text = "\n\n".join(
                [
                    f"Candidate {i + 1}: {c['text'][:300]}..."
                    for i, c in enumerate(candidates[:5])
                ]
            )
        else:
            candidate_text = "[Candidates would be retrieved via RAG]"

        # Ground truth (assuming dataset has "relevant_cases" field)
        relevant = item.get("relevant_cases", item.get("label", "Case 1"))

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"[TASK: Prior Case Retrieval]\n\nQuery Case:\n{query_case}\n\nCandidate Cases:\n{candidate_text}\n\nRank these cases by legal relevance and explain why.",
                    }
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "text",
                        "text": f"Most Relevant: {relevant}\n\nReasoning: These cases share similar legal principles and factual patterns...",
                    }
                ],
            },
        ]
        return {"conversations": messages}

    def _format_lsi(self, item: Dict) -> Dict:
        """Legal Statute Identification: Facts -> Applicable statutes"""
        facts = item.get("facts", item.get("text", ""))[:1000]
        statutes = item.get("statutes", item.get("label", "Section 302 IPC"))

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"[TASK: Legal Statute Identification]\n\nCase Facts:\n{facts}\n\nIdentify all applicable legal statutes and explain their relevance.",
                    }
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "text",
                        "text": f"Applicable Statutes:\n{statutes}\n\nReasoning: Based on the facts presented, these provisions are relevant because...",
                    }
                ],
            },
        ]
        return {"conversations": messages}

    def _format_summ(self, item: Dict) -> Dict:
        """Summarization: Long document -> Concise summary"""
        document = item.get("text", item.get("document", ""))[:2000]
        summary = item.get("summary", "This case involves...")

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"[TASK: Legal Summarization]\n\nDocument:\n{document}\n\nProvide a concise summary capturing key facts, arguments, and rulings.",
                    }
                ],
            },
            {"role": "assistant", "content": [{"type": "text", "text": summary}]},
        ]
        return {"conversations": messages}

    def prepare_multitask_dataset(self) -> Dataset:
        """Load and combine all three tasks with specified ratios"""
        print("[DATA] Loading IL-TUR datasets...")

        # Load datasets
        pcr_ds = load_dataset("Exploration-Lab/IL-TUR", "pcr", split="train")
        lsi_ds = load_dataset("Exploration-Lab/IL-TUR", "lsi", split="train")
        summ_ds = load_dataset("Exploration-Lab/IL-TUR", "summ", split="train")

        # Apply formatters
        print("[DATA] Formatting PCR...")
        pcr_formatted = pcr_ds.map(
            self._format_pcr_with_rag, remove_columns=pcr_ds.column_names
        )
        print("[DATA] Formatting LSI...")
        lsi_formatted = lsi_ds.map(self._format_lsi, remove_columns=lsi_ds.column_names)
        print("[DATA] Formatting SUMM...")
        summ_formatted = summ_ds.map(
            self._format_summ, remove_columns=summ_ds.column_names
        )

        # Sample according to ratios (for efficient training)
        target_samples = 5000  # Adjust based on your needs
        pcr_n = int(target_samples * self.config.task_ratios["pcr"])
        lsi_n = int(target_samples * self.config.task_ratios["lsi"])
        summ_n = int(target_samples * self.config.task_ratios["summ"])

        pcr_sampled = pcr_formatted.shuffle(seed=42).select(
            range(min(pcr_n, len(pcr_formatted)))
        )
        lsi_sampled = lsi_formatted.shuffle(seed=42).select(
            range(min(lsi_n, len(lsi_formatted)))
        )
        summ_sampled = summ_formatted.shuffle(seed=42).select(
            range(min(summ_n, len(summ_formatted)))
        )

        # Combine
        combined = concatenate_datasets([pcr_sampled, lsi_sampled, summ_sampled])
        combined = combined.shuffle(seed=42)

        print(f"[DATA] Multi-task dataset ready: {len(combined)} samples")
        print(
            f"  PCR: {len(pcr_sampled)}, LSI: {len(lsi_sampled)}, SUMM: {len(summ_sampled)}"
        )

        return combined


# ==================== TRAINING ====================


def train_multitask_model(config: Config, dataset: Dataset):
    """Fine-tune Gemma 3n-E4B on multi-task dataset"""
    from unsloth import FastModel
    from unsloth.chat_templates import get_chat_template, train_on_responses_only
    from trl import SFTTrainer, SFTConfig

    print("[TRAIN] Loading Gemma 3n-E4B...")
    model, tokenizer = FastModel.from_pretrained(
        model_name=config.base_model,
        max_seq_length=config.max_seq_length,
        dtype=None,
        load_in_4bit=True,
    )

    # Add LoRA adapters
    print("[TRAIN] Adding LoRA adapters...")
    model = FastModel.get_peft_model(
        model,
        finetune_vision_layers=False,  # Text-only for legal tasks
        finetune_language_layers=True,
        finetune_attention_modules=True,
        finetune_mlp_modules=True,
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        bias="none",
        random_state=3407,
    )

    # Setup chat template
    tokenizer = get_chat_template(tokenizer, chat_template="gemma-3")

    # Format dataset for training
    def formatting_prompts_func(examples):
        convos = examples["conversations"]
        texts = [
            tokenizer.apply_chat_template(
                convo, tokenize=False, add_generation_prompt=False
            ).removeprefix("<bos>")
            for convo in convos
        ]
        return {"text": texts}

    dataset = dataset.map(formatting_prompts_func, batched=True)

    # Training config
    print("[TRAIN] Setting up trainer...")
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        args=SFTConfig(
            dataset_text_field="text",
            per_device_train_batch_size=config.per_device_train_batch_size,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            warmup_steps=config.warmup_steps,
            num_train_epochs=config.num_train_epochs,
            learning_rate=config.learning_rate,
            logging_steps=config.logging_steps,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="cosine",
            seed=3407,
            output_dir=config.output_dir,
            save_steps=config.save_steps,
            save_total_limit=3,
            report_to="tensorboard",
        ),
    )

    # Train only on assistant responses
    trainer = train_on_responses_only(
        trainer,
        instruction_part="<start_of_turn>user\n",
        response_part="<start_of_turn>model\n",
    )

    # Show memory stats
    gpu_stats = torch.cuda.get_device_properties(0)
    start_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    print(f"[TRAIN] GPU: {gpu_stats.name}, Memory: {start_memory} GB reserved")

    # Train!
    print("[TRAIN] Starting training...")
    trainer_stats = trainer.train()

    # Memory stats
    used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    print(f"[TRAIN] Training complete!")
    print(f"  Time: {round(trainer_stats.metrics['train_runtime'] / 60, 2)} minutes")
    print(f"  Peak memory: {used_memory} GB")

    # Save model
    print(f"[TRAIN] Saving model to {config.output_dir}...")
    model.save_pretrained(config.output_dir)
    tokenizer.save_pretrained(config.output_dir)

    return model, tokenizer


# ==================== INFERENCE ====================


def test_multitask_inference(config: Config, rag_indexer: ZillizRAGIndexer):
    """Test trained model on all three tasks with RAG"""
    from unsloth import FastModel
    from unsloth.chat_templates import get_chat_template

    print("[INFERENCE] Loading fine-tuned model...")
    model, tokenizer = FastModel.from_pretrained(
        model_name=config.output_dir,
        max_seq_length=config.max_seq_length,
        load_in_4bit=True,
    )
    tokenizer = get_chat_template(tokenizer, chat_template="gemma-3")

    # Test cases
    test_cases = [
        {
            "task": "PCR",
            "query": "A case involving murder under Section 302 IPC with circumstantial evidence",
        },
        {
            "task": "LSI",
            "query": "Facts: The accused trespassed into private property at night and stole valuables worth 50,000 rupees. Identify applicable statutes.",
        },
        {
            "task": "SUMM",
            "query": "Summarize: The Supreme Court held that the right to privacy is a fundamental right under Article 21. The judgment examined historical precedents and constitutional provisions...",
        },
    ]

    for test in test_cases:
        print(f"\n{'=' * 60}")
        print(f"Testing: {test['task']}")
        print(f"{'=' * 60}")

        # For PCR, use RAG to get candidates
        if test["task"] == "PCR" and rag_indexer:
            candidates = rag_indexer.retrieve(test["query"], top_k=5)
            candidate_text = "\n".join(
                [
                    f"Case {i + 1}: {c['text'][:200]}..."
                    for i, c in enumerate(candidates)
                ]
            )
            full_query = f"[TASK: Prior Case Retrieval]\nQuery: {test['query']}\n\nCandidates:\n{candidate_text}\n\nRank by relevance:"
        else:
            full_query = f"[TASK: {test['task']}]\n{test['query']}"

        messages = [{"role": "user", "content": [{"type": "text", "text": full_query}]}]
        inputs = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
            tokenize=True,
            return_dict=True,
        ).to("cuda")

        print(f"\nQuery: {test['query'][:200]}...")
        print("\nAnswer:")
        _ = model.generate(
            **inputs,
            max_new_tokens=256,
            temperature=1.0,
            top_p=0.95,
            top_k=64,
            streamer=TextStreamer(tokenizer, skip_prompt=True),
        )


# ==================== MAIN ====================


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="IL-TUR Multi-task Fine-tuning with RAG"
    )
    parser.add_argument(
        "--build-rag-index",
        action="store_true",
        help="Build Zilliz vector DB from ALL 8 IL-TUR tasks",
    )
    parser.add_argument(
        "--train",
        action="store_true",
        help="Fine-tune model on 3 tasks (PCR, LSI, SUMM)",
    )
    parser.add_argument(
        "--inference",
        action="store_true",
        help="Test trained model with RAG retrieval from all 8 tasks",
    )
    parser.add_argument(
        "--rag-all-tasks",
        action="store_true",
        default=True,
        help="Index all 8 IL-TUR tasks (default: True)",
    )
    args = parser.parse_args()

    config = Config()

    # Initialize RAG indexer
    rag_indexer = ZillizRAGIndexer(config)

    if args.build_rag_index:
        print("\n" + "=" * 70)
        print("[STEP 1] Building RAG Index in Zilliz Cloud")
        print("=" * 70)
        print("Strategy: Index ALL 8 IL-TUR tasks for comprehensive retrieval")
        print("Training: Fine-tune on 3 tasks (PCR, LSI, SUMM)")
        print("Benefit: Model can leverage entire legal knowledge base via RAG")
        print("=" * 70 + "\n")

        # Build comprehensive index from all 8 tasks
        rag_indexer.build_index(all_tasks=True)
        print("\n[DONE] RAG index built with ALL 8 IL-TUR tasks!")
        print("Your model will now be able to retrieve from:")
        print("  ✓ NER, RR, CJPE, BAIL (indexed but not trained)")
        print("  ✓ LSI, PCR, SUMM (indexed AND will be trained)")
        print("  ✓ MT (Machine Translation)")
        return

    if args.train:
        print("\n" + "=" * 70)
        print("[STEP 2] Multi-task Fine-tuning")
        print("=" * 70)
        print("Training on 3 tasks: PCR (40%), LSI (35%), SUMM (25%)")
        print("Model: unsloth/gemma-3n-E4B-it-unsloth-bnb-4bit")
        print("=" * 70 + "\n")

        # Load existing index for RAG-augmented data prep
        try:
            rag_indexer.load_index()
            print("[INFO] RAG index loaded - will augment training with retrieval")
        except Exception as e:
            print(f"[WARNING] Could not load RAG index: {e}")
            print("[INFO] Training will proceed without RAG augmentation")
            rag_indexer = None

        data_prep = ILTURDataPrep(config, rag_indexer)
        dataset = data_prep.prepare_multitask_dataset()

        print("\n[STEP 3] Starting Fine-tuning...")
        model, tokenizer = train_multitask_model(config, dataset)
        print(f"\n[DONE] Model saved to {config.output_dir}")
        print("\nYour fine-tuned model can now:")
        print("  ✓ Rank prior cases (PCR)")
        print("  ✓ Identify applicable statutes (LSI)")
        print("  ✓ Summarize legal documents (SUMM)")
        print("  ✓ Use RAG to retrieve from all 8 IL-TUR task datasets!")
        return

    if args.inference:
        print("\n" + "=" * 70)
        print("[STEP 4] Testing Fine-tuned Model with RAG")
        print("=" * 70)
        print(
            "Model retrieves from ALL 8 IL-TUR tasks, generates with 3 trained skills"
        )
        print("=" * 70 + "\n")

        try:
            rag_indexer.load_index()
            print("[INFO] RAG index loaded - using all 8 tasks for retrieval")
        except Exception as e:
            print(f"[WARNING] RAG index not available: {e}")
            rag_indexer = None

        test_multitask_inference(config, rag_indexer)
        print("\n[DONE] Inference complete!")
        return

    # Show helpful usage
    print("=" * 70)
    print("IL-TUR Multi-task Fine-tuning with RAG")
    print("=" * 70)
    print("\nUsage:")
    print("  1. Build RAG index (ALL 8 tasks):")
    print("     python iltur_multitask_finetune.py --build-rag-index")
    print("\n  2. Fine-tune model (3 tasks: PCR, LSI, SUMM):")
    print("     python iltur_multitask_finetune.py --train")
    print("\n  3. Test with RAG retrieval:")
    print("     python iltur_multitask_finetune.py --inference")
    print("\n" + "=" * 70)
    print("\nStrategy:")
    print("  • RAG Index: ALL 8 IL-TUR tasks → Comprehensive knowledge base")
    print("  • Training: 3 generative tasks → Efficient fine-tuning")
    print("  • Inference: Model retrieves from 8, generates with 3 trained skills")
    print("=" * 70)


if __name__ == "__main__":
    main()
