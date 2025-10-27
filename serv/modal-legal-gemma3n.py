#!/usr/bin/env python3
"""
LegalAI Backend Server - Modal Deployment
FastAPI server that uses the fine-tuned Gemma model with RAG retrieval
using Zilliz Cloud (Milvus) vector database.

For deploying on Modal

Model: arjf/gemma-3n-IL-TUR-multitask
RAG: Zilliz Cloud with InLegalBERT embeddings
"""

import json
import time
import logging
from typing import List, Optional, AsyncGenerator, Dict

import torch
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel
from peft import PeftModel
import modal
from threading import Thread
from transformers import TextIteratorStreamer

from pymilvus import connections, Collection, utility

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# ==================== MODAL CONFIGURATION ====================

app = modal.App(name="legal-ai-Gemma-3n-IL-TUR-multitask")


legal_gemma3n_image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "timm>=1.0.21",
    "pillow>=11.3.0",
    "torch>=2.0.0",
    "transformers>=4.35.0",
    "peft>=0.7.0",
    "accelerate>=0.24.0",
    "bitsandbytes>=0.41.0",
    "pymilvus>=2.3.0",
    "numpy>=1.24.0",
    "fastapi>=0.104.1",
    "uvicorn[standard]>=0.24.0",
    "pydantic>=2.5.0",
    "python-dotenv>=1.0.0",
)

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

    # Milvus/Zilliz Configuration (loaded from Modal secrets)
    MILVUS_URI = None
    MILVUS_TOKEN = None
    MILVUS_COLLECTION = "iltur_legal_kb"

    # Generation settings
    MAX_NEW_TOKENS = 512
    TEMPERATURE = 0.7
    TOP_P = 0.9
    DO_SAMPLE = True

    # Device
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @classmethod
    def load_from_env(cls):
        """Load configuration from environment variables (Modal secrets)"""
        import os

        cls.MILVUS_URI = os.environ.get("MILVUS_URI")
        cls.MILVUS_TOKEN = os.environ.get("MILVUS_TOKEN")
        cls.MILVUS_COLLECTION = os.environ.get("MILVUS_COLLECTION", "iltur_legal_kb")

    @classmethod
    def validate(cls):
        """Validate required environment variables"""
        if not cls.MILVUS_URI or not cls.MILVUS_TOKEN:
            raise ValueError("MILVUS_URI and MILVUS_TOKEN must be set in Modal secrets")


# ==================== RAG INDEXER ====================


class ZillizRAGIndexer:
    """RAG system"""

    def __init__(self, config: Config):
        self.config = config
        logger.info(f"[RAG] Loading retrieval model: {config.RETRIEVAL_MODEL}")
        self.tokenizer = AutoTokenizer.from_pretrained(config.RETRIEVAL_MODEL)
        self.model = AutoModel.from_pretrained(config.RETRIEVAL_MODEL).to(config.DEVICE)
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
                f"Please build the index."
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


# ==================== GEMMA MODEL HANDLER ====================


class GemmaLegalAssistant:
    """Handles the fine-tuned model"""

    def __init__(self, config: Config, rag_indexer: ZillizRAGIndexer):
        self.config = config
        self.rag = rag_indexer
        self.tokenizer = None
        self.model = None

    def load_model(self):
        """Load the fine-tuned Gemma model"""
        logger.info(f"Loading model: {self.config.MODEL_NAME}")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.MODEL_NAME,
            trust_remote_code=True,
        )

        logger.info("Loading base model...")
        base_model = AutoModelForCausalLM.from_pretrained(
            self.config.BASE_MODEL,
            device_map={"": 0},
            trust_remote_code=True,
            dtype=torch.float16,
            load_in_4bit=True,
        )

        # Load PEFT adapter
        logger.info("Loading PEFT adapter...")
        try:
            self.model = PeftModel.from_pretrained(base_model, self.config.MODEL_NAME)
            logger.info("PEFT adapter loaded successfully.")
            self.model = self.model.merge_and_unload()
            logger.info("Model reloaded after merging with PEFT adapters.")
        except Exception as e:
            logger.warning(f"Could not load PEFT adapter: {e}")
            logger.info("Using base model without adapter")
            self.model = base_model

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
            prompt = f"""Based on the following legal precedents:

{context}

Query: {query}

Provide the most relevant case law and explain why it applies."""

        elif task == "lsi":
            prompt = f"""Given the following legal context:

{context}

Query: {query}

Identify the applicable Indian legal statutes and provisions."""

        elif task == "summ":
            prompt = f"""Summarize the following legal matter:

{query}

Key Points:"""

        else:
            prompt = f"""You are a knowledgeable Indian legal assistant. Use the following precedents to answer the query.

Legal Precedents:
{context}

Query: {query}

Answer:"""

        return prompt

    async def generate_streaming(
        self, query: str, task: str = "general"
    ) -> AsyncGenerator[str, None]:
        """Generate streaming response using TextIteratorStreamer"""

        try:
            # Format prompt with RAG context
            prompt = self.format_prompt_with_rag(query, task)

            # Tokenize
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

            # --- TRUE STREAMING IMPLEMENTATION ---

            # 1. Create the streamer
            streamer = TextIteratorStreamer(
                self.tokenizer, skip_prompt=True, skip_special_tokens=True
            )

            # 2. Define generation kwargs
            generation_kwargs = {
                "input_ids": inputs["input_ids"],
                "attention_mask": inputs["attention_mask"],
                "streamer": streamer,  # Pass the streamer
                "max_new_tokens": self.config.MAX_NEW_TOKENS,
                "temperature": self.config.TEMPERATURE,
                "top_p": self.config.TOP_P,
                "do_sample": self.config.DO_SAMPLE,
                "pad_token_id": self.tokenizer.pad_token_id
                or self.tokenizer.eos_token_id,
            }

            # 3. Define the generation function to run in a thread
            def generate_in_thread():
                with torch.no_grad():
                    self.model.generate(**generation_kwargs)

            # 4. Start the thread
            thread = Thread(target=generate_in_thread)
            thread.start()

            # 5. Yield tokens as they become available
            logger.info("Streaming response...")
            for new_token in streamer:
                yield new_token

            # 6. Wait for the thread to finish
            thread.join()
            logger.info("Streaming complete.")

            # --- END TRUE STREAMING ---

        except Exception as e:
            logger.error(f"Error during generation: {e}", exc_info=True)
            yield f"\n\nError: {str(e)}"


# ==================== FASTAPI APP ====================

web_app = FastAPI(
    title="LegalAI Backend - Gemma-3n-IL-TUR (Modal)",
    description="Indian Legal Assistant with RAG-enhanced Gemma model",
    version="0.0.1-modal",
)

# CORS configuration
web_app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "*",  # Allow all for Modal
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global system (will be initialized in Modal container)
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


# ==================== API ENDPOINTS ====================


@web_app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "name": "LegalAI Backend (Modal)",
        "version": "3.0.0-modal",
        "model": Config.MODEL_NAME,
        "description": "Indian Legal Assistant with RAG-enhanced Gemma model",
        "endpoints": {
            "POST /query": "Submit a legal query",
            "POST /query/stream": "Submit a legal query with streaming response",
            "GET /health": "Health check",
            "GET /metrics": "System metrics",
        },
    }


@web_app.post("/query")
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


@web_app.post("/query/stream")
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


@web_app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    if not legal_assistant or not rag_indexer:
        raise HTTPException(status_code=503, detail="System not initialized")

    return HealthResponse(
        status="healthy",
        version="3.0.0-modal",
        model=Config.MODEL_NAME,
        rag_collection=Config.MILVUS_COLLECTION,
        rag_entities=rag_indexer.collection.num_entities,
        device=str(Config.DEVICE),
    )


@web_app.get("/metrics")
async def get_metrics():
    """System metrics endpoint"""
    if not legal_assistant or not rag_indexer:
        raise HTTPException(status_code=503, detail="System not initialized")

    return {
        "system": {
            "status": "running",
            "device": str(Config.DEVICE),
            "cuda_available": torch.cuda.is_available(),
            "platform": "Modal",
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


# ==================== MODAL DEPLOYMENT ====================


@app.function(
    image=legal_gemma3n_image,
    gpu="A10G",  # Request NV A10G
    secrets=[modal.Secret.from_dotenv()],  # Load .env file as secrets
    timeout=900,  # 15 minutes timeout for model loading
    scaledown_window=300,  # Keep container alive for 5 minutes
)
@modal.asgi_app()
def fastapi_app():
    """
    Modal entrypoint that initializes the system and returns the FastAPI app.
    This runs once when the container starts.
    """
    global legal_assistant, rag_indexer

    logger.info("=" * 70)
    logger.info("LegalAI Backend - Modal Deployment")
    logger.info("=" * 70)

    try:
        # Load configuration from Modal secrets
        Config.load_from_env()
        Config.validate()

        # Initialize RAG indexer
        logger.info("Initializing RAG indexer...")
        config = Config()
        rag_indexer = ZillizRAGIndexer(config)
        rag_indexer.load_index()

        # Initialize model
        logger.info("Loading model...")
        legal_assistant = GemmaLegalAssistant(config, rag_indexer)
        legal_assistant.load_model()

        logger.info("Model loaded and ready.")
        logger.info(f"Device: {config.DEVICE}")
        logger.info(f"Model: {config.MODEL_NAME}")
        logger.info(f"RAG Collection: {config.MILVUS_COLLECTION}")
        logger.info(f"RAG Entities: {rag_indexer.collection.num_entities}")
        logger.info("=" * 70)

    except Exception as e:
        logger.error(f"Failed to initialize system: {e}", exc_info=True)
        raise

    return web_app


# ==================== LOCAL TESTING ====================

if __name__ == "__main__":
    # This is for local testing only
    # For Modal deployment, use: modal deploy modal-legal-gemma3n.py
    import uvicorn
    from dotenv import load_dotenv

    load_dotenv()

    # Initialize system locally
    Config.load_from_env()
    Config.validate()

    config = Config()
    rag_indexer = ZillizRAGIndexer(config)
    rag_indexer.load_index()

    legal_assistant = GemmaLegalAssistant(config, rag_indexer)
    legal_assistant.load_model()

    uvicorn.run(web_app, host="0.0.0.0", port=8000)
