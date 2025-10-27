#!/usr/bin/env python3
"""
LegalAI Backend Server with Gemma-3n-IL-TUR-multitask Model
FastAPI server that uses the fine-tuned Gemma model with RAG retrieval
using Zilliz Cloud (Milvus) vector database.

Model: arjf/gemma-3n-IL-TUR-multitask
RAG: Zilliz Cloud with InLegalBERT embeddings
"""

import os
import sys
import json
import asyncio
import time
import logging
from typing import List, Optional, AsyncGenerator, Dict, Any
from pathlib import Path

import torch
import numpy as np
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModel,
    TextStreamer,
)
from peft import PeftModel, PeftConfig
from datasets import load_dataset
from dotenv import load_dotenv

from pymilvus import (
    connections,
    FieldSchema,
    CollectionSchema,
    DataType,
    Collection,
    utility,
)

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# ==================== CONFIGURATION ====================


class Config:
    """Configuration matching rag.py settings"""

    # Models
    MODEL_NAME = "arjf/gemma-3n-IL-TUR-multitask"
    BASE_MODEL = "unsloth/gemma-3n-E4B-it-unsloth-bnb-4bit"
    RETRIEVAL_MODEL = "law-ai/InLegalBERT"

    # RAG Configuration (matching rag.py)
    EMBEDDING_DIM = 768
    TOP_K_RETRIEVE = 5
    CHUNK_SIZE = 512
    CHUNK_OVERLAP = 100

    # Milvus/Zilliz Configuration
    MILVUS_URI = os.environ.get("MILVUS_URI")
    MILVUS_TOKEN = os.environ.get("MILVUS_TOKEN")
    MILVUS_COLLECTION = os.environ.get("MILVUS_COLLECTION", "iltur_legal_kb")

    # Generation settings
    MAX_NEW_TOKENS = 512
    TEMPERATURE = 0.7
    TOP_P = 0.9
    DO_SAMPLE = True

    # Server settings
    HOST = "0.0.0.0"
    PORT = 8000
    CACHE_DIR = "./cache"

    # Device
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @classmethod
    def validate(cls):
        """Validate required environment variables"""
        if not cls.MILVUS_URI or not cls.MILVUS_TOKEN:
            raise ValueError(
                "MILVUS_URI and MILVUS_TOKEN must be set in environment variables"
            )


# ==================== RAG INDEXER ====================


class ZillizRAGIndexer:
    """RAG system using Zilliz Cloud (matching rag.py implementation)"""

    def __init__(self, config: Config):
        self.config = config
        logger.info(f"[RAG] Loading retrieval model: {config.RETRIEVAL_MODEL}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.RETRIEVAL_MODEL, cache_dir=config.CACHE_DIR
        )
        self.model = AutoModel.from_pretrained(
            config.RETRIEVAL_MODEL, cache_dir=config.CACHE_DIR
        ).to(config.DEVICE)
        self.model.eval()

        self._connect_milvus()
        self.collection = None

    def _connect_milvus(self):
        """Connect to Zilliz Cloud"""
        if self.config.MILVUS_URI and self.config.MILVUS_TOKEN:
            logger.info(f"[RAG] Connecting to Zilliz Cloud: {self.config.MILVUS_URI}")
            connections.connect(
                alias="default",
                uri=self.config.MILVUS_URI,
                token=self.config.MILVUS_TOKEN,
            )
        else:
            raise ValueError("Missing MILVUS_URI and/or MILVUS_TOKEN")

    def load_index(self):
        """Load existing index from Zilliz Cloud"""
        if not utility.has_collection(self.config.MILVUS_COLLECTION):
            raise RuntimeError(
                f"Collection {self.config.MILVUS_COLLECTION} not found! "
                f"Please run 'python train/rag.py --build-rag-index' first."
            )
        self.collection = Collection(self.config.MILVUS_COLLECTION)
        self.collection.load()
        logger.info(f"[RAG] Loaded collection: {self.collection.num_entities} entities")

    def _encode_batch(self, texts: List[str]) -> np.ndarray:
        """Encode texts to embeddings (CLS token)"""
        enc = self.tokenizer(
            texts, max_length=512, truncation=True, padding=True, return_tensors="pt"
        ).to(self.config.DEVICE)
        with torch.no_grad():
            outs = self.model(**enc)
            embs = outs.last_hidden_state[:, 0, :].cpu().numpy().astype("float32")
        return embs

    def retrieve(self, query: str, top_k: int = 5) -> List[Dict]:
        """Retrieve top-k relevant chunks from Zilliz Cloud"""
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


# ==================== GEMMA MODEL HANDLER ====================


class GemmaLegalAssistant:
    """Handles the fine-tuned Gemma model for legal tasks"""

    def __init__(self, config: Config, rag_indexer: ZillizRAGIndexer):
        self.config = config
        self.rag = rag_indexer
        self.tokenizer = None
        self.model = None

    def load_model(self):
        """Load the fine-tuned Gemma model from HuggingFace"""
        logger.info(f"Loading model: {self.config.MODEL_NAME}")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.MODEL_NAME,
            cache_dir=self.config.CACHE_DIR,
            trust_remote_code=True,
        )

        logger.info("Loading base model + adapter separately...")

        # 1. Load the 4-bit base model (forced to GPU)
        base_model = AutoModelForCausalLM.from_pretrained(
            self.config.BASE_MODEL,
            device_map={"": 0},
            cache_dir=self.config.CACHE_DIR,
            trust_remote_code=True,
            dtype=torch.float16,
        )

        # 2. Load the PEFT adapter on top
        self.model = PeftModel.from_pretrained(
            base_model,
            self.config.MODEL_NAME,
            cache_dir=self.config.CACHE_DIR,
        )
        logger.info("Model loaded successfully with PEFT adapter")
        # --- END REVISED LOGIC ---

        self.model.eval()
        logger.info("Model ready for inference")

    def format_prompt_with_rag(self, query: str, task: str = "general") -> str:
        """Format prompt with RAG context (matching rag.py formatting)"""

        # Retrieve relevant context
        retrieved = self.rag.retrieve(query, top_k=self.config.TOP_K_RETRIEVE)

        # Build context from retrieved documents
        context_parts = []
        for i, doc in enumerate(retrieved[:3], 1):  # Use top 3 for context
            snippet = (
                doc["text"][:300] + "..." if len(doc["text"]) > 300 else doc["text"]
            )
            context_parts.append(f"[{i}] {snippet}")

        context = (
            "\n".join(context_parts)
            if context_parts
            else "No relevant precedents found."
        )

        # Format prompt based on task
        if task == "pcr":
            # Prior Case Retrieval
            prompt = f"""Based on the following legal precedents:

{context}

Query: {query}

Provide the most relevant case law and explain why it applies."""

        elif task == "lsi":
            # Legal Statute Identification
            prompt = f"""Given the following legal context:

{context}

Query: {query}

Identify the applicable Indian legal statutes and provisions."""

        elif task == "summ":
            # Summarization
            prompt = f"""Summarize the following legal matter:

{query}

Key Points:"""

        else:
            # General legal query with RAG
            prompt = f"""You are a knowledgeable Indian legal assistant. Use the following precedents to answer the query.

Legal Precedents:
{context}

Query: {query}

Answer:"""

        return prompt

    async def generate_streaming(
        self, query: str, task: str = "general"
    ) -> AsyncGenerator[str, None]:
        """Generate streaming response"""

        try:
            # Format prompt with RAG context
            prompt = self.format_prompt_with_rag(query, task)

            # Tokenize
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

            # Generate with streaming
            generation_kwargs = {
                "input_ids": inputs["input_ids"],
                "attention_mask": inputs["attention_mask"],
                "max_new_tokens": self.config.MAX_NEW_TOKENS,
                "temperature": self.config.TEMPERATURE,
                "top_p": self.config.TOP_P,
                "do_sample": self.config.DO_SAMPLE,
                "pad_token_id": self.tokenizer.pad_token_id
                or self.tokenizer.eos_token_id,
            }

            # Generate tokens
            with torch.no_grad():
                generated_ids = self.model.generate(**generation_kwargs)

            # Decode only the new tokens (excluding input prompt)
            input_length = inputs["input_ids"].shape[1]
            generated_text = self.tokenizer.decode(
                generated_ids[0][input_length:], skip_special_tokens=True
            )

            # Stream the response word by word
            words = generated_text.split()
            for i, word in enumerate(words):
                chunk = word if i == 0 else f" {word}"
                yield chunk
                await asyncio.sleep(0.02)  # Simulate streaming delay

        except Exception as e:
            logger.error(f"Error during generation: {e}", exc_info=True)
            yield f"\n\nError: {str(e)}"


# ==================== FASTAPI APP ====================

app = FastAPI(
    title="LegalAI Backend - Gemma-3n-IL-TUR",
    description="Indian Legal Assistant with RAG-enhanced Gemma model",
    version="3.0.0",
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        # Add production URLs here
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global system
legal_assistant: Optional[GemmaLegalAssistant] = None
rag_indexer: Optional[ZillizRAGIndexer] = None


# ==================== REQUEST/RESPONSE MODELS ====================


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=10000)
    task: Optional[str] = Field(
        default="general", description="Task type: pcr, lsi, summ, or general"
    )
    stream: Optional[bool] = Field(
        default=True, description="Enable streaming response"
    )


class HealthResponse(BaseModel):
    status: str
    version: str
    model: str
    rag_collection: str
    rag_entities: int
    device: str


# ==================== STARTUP ====================


@app.on_event("startup")
async def startup_event():
    """Initialize system on startup"""
    global legal_assistant, rag_indexer

    logger.info("=" * 70)
    logger.info("LegalAI Backend - Gemma-3n-IL-TUR-multitask with RAG")
    logger.info("=" * 70)

    try:
        # Validate configuration
        Config.validate()

        # Initialize RAG indexer
        logger.info("Initializing RAG indexer...")
        config = Config()
        rag_indexer = ZillizRAGIndexer(config)
        rag_indexer.load_index()

        # Initialize Gemma model
        logger.info("Loading Gemma model...")
        legal_assistant = GemmaLegalAssistant(config, rag_indexer)
        legal_assistant.load_model()

        logger.info("✓ System ready!")
        logger.info(f"Device: {config.DEVICE}")
        logger.info(f"Model: {config.MODEL_NAME}")
        logger.info(f"RAG Collection: {config.MILVUS_COLLECTION}")
        logger.info(f"RAG Entities: {rag_indexer.collection.num_entities}")
        logger.info("=" * 70)

    except Exception as e:
        logger.error(f"Failed to initialize system: {e}", exc_info=True)
        raise


# ==================== API ENDPOINTS ====================


@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "name": "LegalAI Backend",
        "version": "3.0.0",
        "model": Config.MODEL_NAME,
        "description": "Indian Legal Assistant with RAG-enhanced Gemma model",
        "endpoints": {
            "POST /query": "Submit a legal query",
            "POST /query/stream": "Submit a legal query with streaming response",
            "GET /health": "Health check",
            "GET /metrics": "System metrics",
        },
    }


@app.post("/query")
async def query_legal(request: QueryRequest):
    """Non-streaming query endpoint"""
    if not legal_assistant:
        raise HTTPException(status_code=503, detail="System not initialized")

    try:
        start_time = time.time()

        # Collect full response
        full_response = ""
        async for chunk in legal_assistant.generate_streaming(
            request.query, request.task
        ):
            full_response += chunk

        # Get RAG context for metadata
        retrieved_docs = rag_indexer.retrieve(
            request.query, top_k=Config.TOP_K_RETRIEVE
        )

        return {
            "response": full_response.strip(),
            "retrieved_docs": [
                {
                    "text": doc["text"][:200] + "..."
                    if len(doc["text"]) > 200
                    else doc["text"],
                    "score": doc["score"],
                    "label": doc["label"],
                    "case_id": doc["case_id"],
                }
                for doc in retrieved_docs
            ],
            "metadata": {
                "task": request.task,
                "model": Config.MODEL_NAME,
                "num_docs_retrieved": len(retrieved_docs),
                "processing_time_ms": int((time.time() - start_time) * 1000),
            },
        }

    except Exception as e:
        logger.error(f"Error processing query: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/query/stream")
async def query_stream(request: QueryRequest):
    """Streaming query endpoint"""
    if not legal_assistant:
        raise HTTPException(status_code=503, detail="System not initialized")

    async def generate_wrapper():
        try:
            start_time = time.time()

            # Stream the main response
            async for chunk in legal_assistant.generate_streaming(
                request.query, request.task
            ):
                yield chunk

            # Send metadata as final chunk
            retrieved_docs = rag_indexer.retrieve(
                request.query, top_k=Config.TOP_K_RETRIEVE
            )
            metadata = {
                "retrieved_docs": [
                    {
                        "text": doc["text"][:200] + "..."
                        if len(doc["text"]) > 200
                        else doc["text"],
                        "score": round(doc["score"], 3),
                        "label": doc["label"],
                        "case_id": doc["case_id"],
                    }
                    for doc in retrieved_docs
                ],
                "metadata": {
                    "task": request.task,
                    "model": Config.MODEL_NAME,
                    "num_docs_retrieved": len(retrieved_docs),
                    "processing_time_ms": int((time.time() - start_time) * 1000),
                    "timestamp": time.time(),
                },
            }

            yield "\n\n__METADATA__\n" + json.dumps(metadata)

        except Exception as e:
            logger.error(f"Error in streaming: {e}", exc_info=True)
            yield f"\n\nError: {str(e)}"

    return StreamingResponse(generate_wrapper(), media_type="text/plain")


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    if not legal_assistant or not rag_indexer:
        raise HTTPException(status_code=503, detail="System not initialized")

    return HealthResponse(
        status="healthy",
        version="3.0.0",
        model=Config.MODEL_NAME,
        rag_collection=Config.MILVUS_COLLECTION,
        rag_entities=rag_indexer.collection.num_entities,
        device=str(Config.DEVICE),
    )


@app.get("/metrics")
async def get_metrics():
    """System metrics endpoint"""
    if not legal_assistant or not rag_indexer:
        raise HTTPException(status_code=503, detail="System not initialized")

    return {
        "system": {
            "status": "running",
            "device": str(Config.DEVICE),
            "cuda_available": torch.cuda.is_available(),
        },
        "model": {
            "name": Config.MODEL_NAME,
            "base_model": Config.BASE_MODEL,
            "retrieval_model": Config.RETRIEVAL_MODEL,
        },
        "rag": {
            "collection": Config.MILVUS_COLLECTION,
            "num_entities": rag_indexer.collection.num_entities,
            "top_k": Config.TOP_K_RETRIEVE,
            "embedding_dim": Config.EMBEDDING_DIM,
        },
    }


# ==================== MAIN ====================

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host=Config.HOST,
        port=Config.PORT,
        reload=False,
        log_level="info",
    )
