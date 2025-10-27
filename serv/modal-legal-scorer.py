#!/usr/bin/env python3
"""
LegalAI CJPE Scorer
FastAPI server that uses the hierarchical InLegalBERT model for
Court Judgment Prediction and Explanation (CJPE).

For deployment on Modal

Model: arjf/InLegalBERT-HiGRU-CJPE-Scorer
"""

import time
import logging
from typing import List, Optional, Dict, Any

import torch
import torch.nn as nn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from transformers import AutoTokenizer, AutoModel
import modal

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# ==================== MODAL CONFIGURATION ====================

app = modal.App(name="legal-ai-InLegalBERT-HiGRU-CJPE-Scorer")

legal_scorer_image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "torch>=2.0.0",
    "transformers>=4.35.0",
    "fastapi>=0.104.1",
    "uvicorn[standard]>=0.24.0",
    "pydantic>=2.5.0",
    "huggingface-hub>=0.19.0",
)

# ==================== CONFIGURATION ====================


class Config:
    """Configuration for hierarchical InLegalBERT with CJPE"""

    # Model configuration
    MODEL_NAME = "arjf/InLegalBERT-HiGRU-CJPE-Scorer"
    BASE_MODEL = "law-ai/InLegalBERT"

    # Model architecture parameters
    HIDDEN_SIZE = 768
    GRU_HIDDEN_SIZE = 256
    GRU_NUM_LAYERS = 2
    NUM_CLASSES = 2  # Binary classification for CJPE
    DROPOUT = 0.3
    MAX_CHUNKS = 16
    CHUNK_SIZE = 512

    # Device
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ==================== HIERARCHICAL MODEL ====================


class HierarchicalInLegalBERT(nn.Module):
    """Hierarchical BERT: InLegalBERT + BiGRU + Attention for CJPE"""

    def __init__(self, config):
        super().__init__()
        self.config = config

        # Load InLegalBERT base model
        self.bert = AutoModel.from_pretrained(config.BASE_MODEL)

        # Bidirectional GRU for document-level encoding
        self.bigru = nn.GRU(
            input_size=config.HIDDEN_SIZE,
            hidden_size=config.GRU_HIDDEN_SIZE,
            num_layers=config.GRU_NUM_LAYERS,
            batch_first=True,
            bidirectional=True,
            dropout=config.DROPOUT if config.GRU_NUM_LAYERS > 1 else 0,
        )

        # Attention mechanism
        gru_output_size = config.GRU_HIDDEN_SIZE * 2
        self.attention = nn.Sequential(
            nn.Linear(gru_output_size, 128), nn.Tanh(), nn.Linear(128, 1)
        )

        # Classifier for CJPE
        self.classifier = nn.Sequential(
            nn.Dropout(config.DROPOUT),
            nn.Linear(gru_output_size, 256),
            nn.ReLU(),
            nn.Dropout(config.DROPOUT),
            nn.Linear(256, config.NUM_CLASSES),
        )

    def forward(self, input_ids, attention_mask, num_chunks):
        batch_size, max_chunks, seq_len = input_ids.size()

        # Flatten for BERT processing
        input_ids_flat = input_ids.view(-1, seq_len)
        attention_mask_flat = attention_mask.view(-1, seq_len)

        # Get BERT embeddings
        bert_output = self.bert(
            input_ids=input_ids_flat, attention_mask=attention_mask_flat
        )

        # Get CLS embeddings and reshape
        cls_embeddings = bert_output.last_hidden_state[:, 0, :]
        chunk_embeddings = cls_embeddings.view(batch_size, max_chunks, -1)

        # Process through BiGRU
        gru_output, _ = self.bigru(chunk_embeddings)

        # Attention mechanism
        attention_scores = self.attention(gru_output).squeeze(-1)
        chunk_mask = torch.arange(max_chunks).unsqueeze(0).to(num_chunks.device)
        chunk_mask = (chunk_mask < num_chunks.unsqueeze(1)).float()
        attention_scores = attention_scores.masked_fill(chunk_mask == 0, -1e9)
        attention_weights = torch.softmax(attention_scores, dim=1)

        # Weighted document embedding
        document_embedding = torch.bmm(
            attention_weights.unsqueeze(1), gru_output
        ).squeeze(1)

        # Classification logits
        logits = self.classifier(document_embedding)

        return logits, attention_weights


# ==================== CJPE SCORER ====================


class LegalCJPEScorer:
    """Handles the hierarchical InLegalBERT model for CJPE"""

    def __init__(self, config: Config):
        self.config = config
        self.tokenizer = None
        self.model = None

    def load_model(self):
        """Load the hierarchical InLegalBERT model"""
        logger.info(f"Loading tokenizer from {self.config.BASE_MODEL}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.BASE_MODEL)

        logger.info(f"Loading hierarchical model from {self.config.MODEL_NAME}")
        self.model = HierarchicalInLegalBERT(self.config)

        try:
            # Load from HuggingFace
            from huggingface_hub import hf_hub_download

            model_path = hf_hub_download(
                repo_id=self.config.MODEL_NAME,
                filename="best_model.pt",
            )
            state_dict = torch.load(model_path, map_location=self.config.DEVICE)

            # Handle different state dict formats
            if "model_state_dict" in state_dict:
                self.model.load_state_dict(state_dict["model_state_dict"])
            else:
                self.model.load_state_dict(state_dict)

            logger.info("Loaded model from HuggingFace Hub")
        except Exception as e:
            logger.warning(f"Could not load from HuggingFace: {e}")
            logger.info("Initializing model with random weights (fine-tune first!)")

        self.model.to(self.config.DEVICE)
        self.model.eval()
        logger.info("Model ready for inference")

    def _chunk_text(self, text: str) -> List[str]:
        """Split text into chunks"""
        tokens = self.tokenizer.tokenize(text)
        chunks = []

        for i in range(0, len(tokens), self.config.CHUNK_SIZE - 100):
            chunk_tokens = tokens[i : i + self.config.CHUNK_SIZE]
            if len(chunk_tokens) < 50:
                break
            chunk_text = self.tokenizer.convert_tokens_to_string(chunk_tokens)
            chunks.append(chunk_text)
            if len(chunks) >= self.config.MAX_CHUNKS:
                break

        if not chunks:
            chunks = [text[: self.config.CHUNK_SIZE * 3]]

        return chunks

    def predict(self, text: str) -> Dict[str, Any]:
        """
        Predict Court Judgment (CJPE)

        Returns:
            Dictionary with prediction results
        """
        start_time = time.time()

        # Chunk the text
        chunks = self._chunk_text(text)

        # Encode chunks
        with torch.no_grad():
            encoded = self.tokenizer(
                chunks,
                padding="max_length",
                truncation=True,
                max_length=512,
                return_tensors="pt",
            )

            batch_size = len(chunks)
            max_chunks = self.config.MAX_CHUNKS

            # Prepare inputs
            input_ids = torch.zeros(1, max_chunks, 512, dtype=torch.long)
            attention_mask = torch.zeros(1, max_chunks, 512, dtype=torch.long)

            for i in range(min(batch_size, max_chunks)):
                input_ids[0, i] = encoded["input_ids"][i]
                attention_mask[0, i] = encoded["attention_mask"][i]

            num_chunks = torch.tensor([min(batch_size, max_chunks)])

            # Move to device
            input_ids = input_ids.to(self.config.DEVICE)
            attention_mask = attention_mask.to(self.config.DEVICE)
            num_chunks = num_chunks.to(self.config.DEVICE)

            # Get prediction
            logits, attention_weights = self.model(
                input_ids, attention_mask, num_chunks
            )
            probabilities = torch.softmax(logits, dim=-1)
            predicted_class = torch.argmax(probabilities, dim=-1).item()

        prediction_time = int((time.time() - start_time) * 1000)

        # Get top attention chunks for explainability
        attn_weights_list = (
            attention_weights[0].cpu().tolist()[: min(batch_size, max_chunks)]
        )
        top_chunk_indices = sorted(
            range(len(attn_weights_list)),
            key=lambda i: attn_weights_list[i],
            reverse=True,
        )[:3]

        return {
            "prediction": "favorable" if predicted_class == 1 else "unfavorable",
            "predicted_class": predicted_class,
            "confidence": round(float(probabilities[0, predicted_class].item()), 4),
            "probabilities": {
                "unfavorable": round(float(probabilities[0, 0].item()), 4),
                "favorable": round(float(probabilities[0, 1].item()), 4),
            },
            "num_chunks": min(batch_size, max_chunks),
            "prediction_time_ms": prediction_time,
            "attention_weights": [round(w, 4) for w in attn_weights_list],
            "top_chunks": [
                {
                    "index": idx,
                    "attention_weight": round(attn_weights_list[idx], 4),
                    "text": chunks[idx][:200] + "..."
                    if len(chunks[idx]) > 200
                    else chunks[idx],
                }
                for idx in top_chunk_indices
            ],
        }


# ==================== FASTAPI APP ====================

web_app = FastAPI(
    title="LegalAI CJPE Scorer (Modal)",
    description="Court Judgment Prediction and Explanation using Hierarchical InLegalBERT",
    version="2.0.0-modal",
)

# CORS configuration
web_app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global scorer
legal_scorer: Optional[LegalCJPEScorer] = None


# ==================== REQUEST/RESPONSE MODELS ====================


class PredictRequest(BaseModel):
    text: str = Field(
        ..., min_length=1, max_length=50000, description="Legal text to analyze"
    )


class PredictionResponse(BaseModel):
    prediction: str
    predicted_class: int
    confidence: float
    probabilities: Dict[str, float]
    num_chunks: int
    prediction_time_ms: int
    attention_weights: List[float]
    top_chunks: List[Dict[str, Any]]


class HealthResponse(BaseModel):
    status: str
    version: str
    model: str
    device: str


# ==================== API ENDPOINTS ====================


@web_app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "name": "LegalAI CJPE Scorer",
        "version": "2.0.0-modal",
        "model": Config.MODEL_NAME,
        "description": "Court Judgment Prediction and Explanation using Hierarchical InLegalBERT",
        "architecture": {
            "encoder": "InLegalBERT",
            "aggregator": "Bidirectional GRU",
            "attention": "Hierarchical Attention",
            "classifier": "Binary (favorable/unfavorable)",
        },
        "endpoints": {
            "POST /predict": "Get CJPE prediction for legal text",
            "GET /health": "Health check",
            "GET /metrics": "System metrics",
        },
    }


@web_app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictRequest):
    """
    Predict court judgment outcome (CJPE)

    Returns binary classification: favorable or unfavorable
    """
    if not legal_scorer:
        raise HTTPException(status_code=503, detail="System not initialized")

    try:
        result = legal_scorer.predict(request.text)
        return result

    except Exception as e:
        logger.error(f"Error in CJPE prediction: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@web_app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    if not legal_scorer:
        raise HTTPException(status_code=503, detail="System not initialized")

    return HealthResponse(
        status="healthy",
        version="2.0.0-modal",
        model=Config.MODEL_NAME,
        device=str(Config.DEVICE),
    )


@web_app.get("/metrics")
async def get_metrics():
    """System metrics endpoint"""
    if not legal_scorer:
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
            "type": "Hierarchical InLegalBERT",
            "task": "Court Judgment Prediction and Explanation (CJPE)",
        },
        "architecture": {
            "encoder": "InLegalBERT",
            "aggregator": "Bidirectional GRU",
            "attention": "Hierarchical Attention",
            "classifier": "Binary (favorable/unfavorable)",
            "parameters": {
                "hidden_size": Config.HIDDEN_SIZE,
                "gru_hidden_size": Config.GRU_HIDDEN_SIZE,
                "gru_layers": Config.GRU_NUM_LAYERS,
                "max_chunks": Config.MAX_CHUNKS,
                "chunk_size": Config.CHUNK_SIZE,
            },
        },
    }


# ==================== MODAL DEPLOYMENT ====================


@app.function(
    image=legal_scorer_image,
    gpu="T4",  # Req NVIDIA T4 GPU
    timeout=900,  # 15 minutes timeout for model loading
    scaledown_window=300,  # Keep container alive for 5 minutes
)
@modal.asgi_app()
def fastapi_app():
    """
    Modal entrypoint that initializes the CJPE scorer and returns the FastAPI app.
    This runs once when the container starts.
    """
    global legal_scorer

    logger.info("=" * 70)
    logger.info("LegalAI CJPE Scorer - Modal Deployment")
    logger.info("=" * 70)

    try:
        # Initialize CJPE scorer
        logger.info("Loading CJPE scorer model...")
        config = Config()
        legal_scorer = LegalCJPEScorer(config)
        legal_scorer.load_model()

        logger.info("CJPE Scorer loaded and ready.")
        logger.info(f"Device: {config.DEVICE}")
        logger.info(f"Model: {config.MODEL_NAME}")
        logger.info("=" * 70)

    except Exception as e:
        logger.error(f"Failed to initialize system: {e}", exc_info=True)
        raise

    return web_app
