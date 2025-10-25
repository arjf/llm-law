#!/usr/bin/env python3
"""
Example Streaming Backend for LegalAI
FastAPI implementation with SSE streaming support

This is a reference implementation showing how to integrate
the RAG system with the Next.js frontend using streaming responses.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import json
import asyncio
from typing import List, Optional, AsyncGenerator
import time

app = FastAPI(
    title="LegalAI Backend",
    description="Streaming API for Indian Legal Assistant",
    version="1.0.0",
)

# CORS configuration for Next.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        # Add production URLs here
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request/Response Models
class QueryRequest(BaseModel):
    query: str
    files: Optional[List[str]] = None


class HealthResponse(BaseModel):
    status: str
    version: str
    models: dict
    index_loaded: bool


# Mock data for demonstration
MOCK_LEGAL_RESPONSES = {
    "termination": "Based on Indian employment law, wrongful termination refers to the dismissal of an employee without just cause or in violation of contractual terms. According to Section 2(kkk) of the Industrial Disputes Act, 1947, an employer must provide valid grounds for termination. Key considerations include: (1) Notice period compliance as per employment contract or statutory requirements, (2) Just cause requirement under labor laws, (3) Procedural fairness including opportunity to respond. The terminated employee may seek remedies including reinstatement, back wages, or compensation through labor courts.",
    "notice": "Under Indian labor law, notice period requirements vary based on employment terms and applicable legislation. The Industrial Employment (Standing Orders) Act, 1946 mandates minimum notice periods. For workmen, typically 30 days notice or wages in lieu is required. For employees covered under the Shops and Establishments Act, notice periods range from 30-90 days depending on the state and tenure. Employment contracts may specify longer notice periods. Failure to serve proper notice may result in recovery of notice period wages. Payment in lieu of notice must be at the employee's last drawn salary rate.",
    "default": "Based on the relevant provisions of Indian employment and workplace law, including the Industrial Disputes Act 1947, the Payment of Wages Act 1936, and applicable state legislation, this matter requires careful consideration of statutory requirements, contractual obligations, and established precedents. Key factors to evaluate include notice periods, just cause requirements, procedural fairness, and available remedies. For specific guidance on your situation, consultation with a qualified legal professional is recommended.",
}


async def generate_legal_response(
    query: str, files: Optional[List[str]] = None
) -> AsyncGenerator[str, None]:
    """
    Generate streaming legal response

    This is a mock implementation. Replace with actual RAG system:
    - Use legal_rag_qa.LegalRAGSystem for document retrieval
    - Stream tokens from Mistral-7B as they're generated
    - Calculate CJPE score from retrieved documents
    """

    # Simulate retrieval phase
    await asyncio.sleep(0.1)

    # Select appropriate response based on query keywords
    response_text = MOCK_LEGAL_RESPONSES["default"]
    if "termination" in query.lower() or "dismiss" in query.lower():
        response_text = MOCK_LEGAL_RESPONSES["termination"]
    elif "notice" in query.lower():
        response_text = MOCK_LEGAL_RESPONSES["notice"]

    # Stream response word by word (simulating token generation)
    words = response_text.split()
    for i, word in enumerate(words):
        # Add space before word (except first word)
        chunk = word if i == 0 else f" {word}"
        yield chunk

        # Simulate generation delay
        await asyncio.sleep(0.03)

    # Calculate mock CJPE score (replace with actual calculation)
    cjpe_score = 0.75 + (hash(query) % 20) / 100  # Deterministic but varied

    # Mock retrieved cases
    retrieved_cases = [
        {
            "case_id": "CJPE-2019-SC-1234",
            "title": "State Bank of India vs. Employee Union",
            "relevance_score": 0.92,
            "excerpt": "Pertaining to employment termination procedures...",
        },
        {
            "case_id": "CJPE-2020-HC-5678",
            "title": "Tech Corp India vs. Software Engineer",
            "relevance_score": 0.87,
            "excerpt": "Regarding notice period requirements...",
        },
        {
            "case_id": "CJPE-2021-LC-9012",
            "title": "Manufacturing Ltd vs. Workers Association",
            "relevance_score": 0.81,
            "excerpt": "Concerning procedural fairness in dismissal...",
        },
    ]

    # Send metadata as final chunk
    metadata = {
        "cjpe_score": round(cjpe_score, 3),
        "retrieved_cases": retrieved_cases,
        "metadata": {
            "retrieval_time_ms": 150,
            "generation_time_ms": len(words) * 30,
            "model_version": "mistral-7b-instruct-v0.3",
            "retrieval_model": "InLegalBERT",
            "timestamp": time.time(),
        },
    }

    yield "\n\n__METADATA__\n" + json.dumps(metadata)


@app.post("/query/stream")
async def query_stream(request: QueryRequest):
    """
    Streaming endpoint for legal queries

    Returns Server-Sent Events (SSE) stream with:
    1. Incremental text response
    2. Metadata including CJPE score and retrieved cases

    Example usage:
        POST /query/stream
        {
            "query": "What are grounds for termination?",
            "files": ["contract.pdf"]
        }
    """
    try:
        if not request.query or len(request.query.strip()) == 0:
            raise HTTPException(status_code=400, detail="Query cannot be empty")

        if len(request.query) > 2000:
            raise HTTPException(
                status_code=400, detail="Query too long (max 2000 chars)"
            )

        return StreamingResponse(
            generate_legal_response(request.query, request.files),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",  # Disable nginx buffering
            },
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")


@app.get("/health")
async def health_check():
    """
    Health check endpoint

    Returns system status and model information
    """
    return {
        "status": "healthy",
        "version": "1.0.0",
        "models": {
            "retrieval": "law-ai/InLegalBERT",
            "generation": "mistralai/Mistral-7B-Instruct-v0.3",
        },
        "index_loaded": True,
        "features": {"streaming": True, "file_upload": True, "document_analysis": True},
    }


@app.get("/metrics")
async def get_metrics():
    """
    System metrics endpoint (optional)

    Returns performance and usage statistics
    """
    return {
        "total_queries": 1234,
        "avg_cjpe_score": 0.82,
        "avg_response_time_ms": 2450,
        "avg_tokens_per_second": 15.3,
        "index_size": 15000,
        "uptime_seconds": 86400,
    }


@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "service": "LegalAI Backend API",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "POST /query/stream": "Submit legal query (streaming)",
            "GET /health": "Health check",
            "GET /metrics": "System metrics",
        },
        "documentation": "/docs",
    }


# ==================== INTEGRATION WITH ACTUAL RAG SYSTEM ====================
"""
To integrate with the actual RAG system (legal_rag_qa.py):

1. Import the RAG system:
   from legal_rag_qa import LegalRAGSystem, RAGConfig

2. Initialize on startup:
   @app.on_event("startup")
   async def startup_event():
       global rag_system
       config = RAGConfig()
       rag_system = LegalRAGSystem(config)
       rag_system.build_knowledge_base()
       print("RAG system initialized")

3. Replace generate_legal_response with actual generation:

   async def generate_legal_response(query: str, files: Optional[List[str]] = None):
       # Retrieve relevant documents
       retrieved_docs = rag_system.indexer.search(query, k=5)

       # Calculate CJPE score
       cjpe_score = calculate_cjpe_score(query, retrieved_docs)

       # Stream generated answer
       for token in rag_system.generator.generate_streaming(query, retrieved_docs):
           yield token

       # Send metadata
       metadata = {
           "cjpe_score": cjpe_score,
           "retrieved_cases": format_retrieved_cases(retrieved_docs)
       }
       yield "\n\n__METADATA__\n" + json.dumps(metadata)

4. Add file upload handling:
   from fastapi import File, UploadFile

   @app.post("/upload")
   async def upload_document(file: UploadFile = File(...)):
       # Process uploaded document
       # Add to index or use for context
       pass
"""


if __name__ == "__main__":
    import uvicorn

    print("=" * 60)
    print("LegalAI Streaming Backend - Example Implementation")
    print("=" * 60)
    print("\nStarting server on http://localhost:8000")
    print("\nAvailable endpoints:")
    print("  POST   /query/stream  - Submit legal query (streaming)")
    print("  GET    /health        - Health check")
    print("  GET    /metrics       - System metrics")
    print("  GET    /docs          - API documentation")
    print("\nPress Ctrl+C to stop")
    print("=" * 60 + "\n")

    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info", access_log=True)
