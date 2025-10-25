# Backend Integration Guide

This document provides detailed instructions for connecting the Next.js frontend to the Python RAG backend with streaming support.

## Architecture Overview

```
┌─────────────────┐   Streaming SSE    ┌──────────────────┐
│                 │  ◄─────────────────► │                  │
│  Next.js UI     │                      │  Python Backend  │
│  (Port 3000)    │                      │  (Port 8000)     │
│                 │                      │                  │
└─────────────────┘                      └──────────────────┘
        │                                         │
        │                                         │
        ▼                                         ▼
  React Components                    InLegalBERT + Mistral-7B
  - ChatInterface (Streaming)         - Document Retrieval
  - CJPEScore (Real-time)             - Answer Generation
  - Document Upload                   - FAISS Index
```

## Implementation Options

### Option 1: Streaming via Next.js API Route (Recommended)

Frontend receives real-time streaming responses through Next.js API proxy.

**Advantages:**
- Real-time response generation
- Better user experience with streaming
- CORS handled automatically
- Can add authentication/rate limiting
- Hides backend URL from client

**Implementation:**

1. Update `app/api/query/route.ts`:

```typescript
import { NextRequest } from "next/server";

const BACKEND_URL = process.env.BACKEND_API_URL || "http://localhost:8000";

export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    
    const response = await fetch(`${BACKEND_URL}/query/stream`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });

    if (!response.ok) {
      throw new Error(`Backend error: ${response.status}`);
    }

    // Stream the response back to client
    return new Response(response.body, {
      headers: {
        "Content-Type": "text/event-stream",
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
      },
    });
  } catch (error) {
    return new Response(
      JSON.stringify({ error: "Backend connection failed" }),
      {
        status: 500,
        headers: { "Content-Type": "application/json" },
      }
    );
  }
}
```

2. The frontend (`components/ChatInterface.tsx`) already implements streaming:

```typescript
const response = await fetch("/api/query", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({
    query: userMessage.content,
    files: uploadedFiles.map((f) => f.name),
  }),
});

const reader = response.body?.getReader();
const decoder = new TextDecoder();

let fullText = "";
while (true) {
  const { done, value } = await reader.read();
  if (done) break;

  const chunk = decoder.decode(value, { stream: true });
  
  // Handle metadata separately
  if (chunk.includes("__METADATA__")) {
    const [text, metadata] = chunk.split("__METADATA__");
    fullText += text;
    // Process metadata (CJPE score, etc.)
  } else {
    fullText += chunk;
  }
  
  // Update message in real-time
  setMessages((prev) =>
    prev.map((msg) =>
      msg.id === assistantMessageId
        ? { ...msg, content: fullText }
        : msg
    )
  );
}
```

### Option 2: Direct Backend Streaming

Frontend calls Python backend directly for streaming responses.

**Advantages:**
- Simpler setup
- Lower latency
- Direct connection

**Disadvantages:**
- Must handle CORS
- Backend URL exposed to client
- No middleware layer

**Implementation:**

1. Enable CORS in Python backend:

```python
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

2. Call backend directly from frontend:

```typescript
const backendUrl = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
const response = await fetch(`${backendUrl}/query/stream`, {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({ query, files }),
});

// Then use the same streaming logic as Option 1
```

## Python Backend Requirements

The backend must expose the following streaming endpoint:

### POST /query/stream (Primary - Streaming)

**Request:**
```json
{
  "query": "What are the grounds for termination?",
  "files": ["document1.pdf", "document2.pdf"]
}
```

**Response:** Server-Sent Events (SSE) stream

The response should stream text chunks followed by metadata:

```
Based on Indian employment law, wrongful termination refers to...

[More text chunks as they are generated]

__METADATA__
{
  "cjpe_score": 0.87,
  "retrieved_cases": [
    {
      "case_id": "CJPE-123",
      "title": "Case Title",
      "relevance_score": 0.92,
      "excerpt": "Relevant excerpt from the case..."
    }
  ],
  "metadata": {
    "retrieval_time_ms": 150,
    "generation_time_ms": 2300,
    "model_version": "v1.0"
  }
}
```

**Headers:**
```
Content-Type: text/event-stream
Cache-Control: no-cache
Connection: keep-alive
```

### GET /health

**Response:**
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "models": {
    "retrieval": "InLegalBERT",
    "generation": "Mistral-7B-Instruct-v0.3"
  },
  "index_loaded": true
}
```

### GET /metrics (Optional)

**Response:**
```json
{
  "total_queries": 1234,
  "avg_cjpe_score": 0.82,
  "avg_response_time_ms": 2450,
  "index_size": 15000
}
```

## Sample Python Backend Implementation

Create `backend/api.py` with streaming support:

```python
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from legal_rag_qa import LegalRAGSystem, RAGConfig
import json
import time
from typing import List, Optional

app = FastAPI()

# Initialize RAG system
config = RAGConfig()
rag_system = LegalRAGSystem(config)
rag_system.build_knowledge_base()

class QueryRequest(BaseModel):
    query: str
    files: Optional[List[str]] = None

async def generate_stream(query: str, files: Optional[List[str]] = None):
    """Generate streaming response for legal query"""
    try:
        start_time = time.time()
        
        # Retrieve relevant documents
        retrieved_docs = rag_system.indexer.search(query, k=5)
        retrieval_time = int((time.time() - start_time) * 1000)
        
        # Generate answer with streaming
        gen_start = time.time()
        for chunk in rag_system.generator.generate_streaming(query, retrieved_docs):
            yield chunk
        
        generation_time = int((time.time() - gen_start) * 1000)
        
        # Calculate CJPE score
        cjpe_score = rag_system.calculate_cjpe_score(query, retrieved_docs)
        
        # Send metadata as final chunk
        metadata = {
            "cjpe_score": cjpe_score,
            "retrieved_cases": [
                {
                    "case_id": doc["id"],
                    "title": doc["title"],
                    "relevance_score": doc["score"],
                }
                for doc in retrieved_docs
            ],
            "metadata": {
                "retrieval_time_ms": retrieval_time,
                "generation_time_ms": generation_time,
                "model_version": "1.0.0"
            }
        }
        
        yield "\n\n__METADATA__\n" + json.dumps(metadata)
        
    except Exception as e:
        yield f"\n\nError: {str(e)}"

@app.post("/query/stream")
async def query_stream_endpoint(request: QueryRequest):
    """Streaming endpoint for legal queries"""
    return StreamingResponse(
        generate_stream(request.query, request.files),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "version": "1.0.0",
        "models": {
            "retrieval": config.retrieval_model,
            "generation": config.generation_model
        },
        "index_loaded": rag_system.indexer.index is not None
    }
```

**Add streaming support to your generator class:**

```python
def generate_streaming(self, query: str, context_docs: List[Dict]):
    """Generate answer with streaming support"""
    prompt = self._create_prompt(query, context_docs)
    
    # Use text generation with streaming
    streamer = TextIteratorStreamer(self.tokenizer, skip_special_tokens=True)
    generation_kwargs = {
        "inputs": self.tokenizer(prompt, return_tensors="pt").to(self.device),
        "max_new_tokens": self.config.max_answer_length,
        "temperature": self.config.temperature,
        "top_p": self.config.top_p,
        "streamer": streamer,
    }
    
    # Start generation in separate thread
    thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
    thread.start()
    
    # Yield chunks as they're generated
    for text in streamer:
        yield text
    
    thread.join()
```

Run the backend:

```bash
uvicorn backend.api:app --host 0.0.0.0 --port 8000 --reload
```

## Environment Configuration

### Frontend (.env.local)

```bash
# Backend API URL
NEXT_PUBLIC_API_URL=http://localhost:8000

# For production
# NEXT_PUBLIC_API_URL=https://api.yourdomain.com

# Optional: API authentication
# NEXT_PUBLIC_API_KEY=your_secret_key
```

### Backend (.env)

```bash
# Model cache directory
CACHE_DIR=./cache

# Index storage
INDEX_DIR=./rag_index

# Device
CUDA_VISIBLE_DEVICES=0

# API configuration
HOST=0.0.0.0
PORT=8000
```

## Testing the Integration

### 1. Start Backend

```bash
cd ../
python -m uvicorn backend.api:app --host 0.0.0.0 --port 8000 --reload
```

### 2. Start Frontend

```bash
cd web
npm run dev
```

### 3. Test Backend Health

Open browser console and run:

```javascript
fetch('http://localhost:8000/health')
  .then(r => r.json())
  .then(console.log)
```

### 4. Test Streaming

Test the streaming endpoint directly:

```javascript
fetch('http://localhost:8000/query/stream', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ query: 'Test query' })
})
.then(response => {
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  
  function read() {
    reader.read().then(({ done, value }) => {
      if (done) return;
      console.log(decoder.decode(value));
      read();
    });
  }
  read();
});
```

### 5. Test Full Integration

Use the chat interface to send a query. Verify:
- Message appears in chat immediately
- Response streams in real-time
- CJPE score updates after response completes
- File upload works (if implemented)
- No console errors
- Theme toggle works

## Troubleshooting

### CORS Errors

**Symptom:** Browser console shows CORS policy errors

**Solution:**
- Add CORS middleware to Python backend
- Use Next.js API route proxy instead of direct calls

### Connection Refused

**Symptom:** `ERR_CONNECTION_REFUSED`

**Solution:**
- Verify backend is running: `curl http://localhost:8000/health`
- Check firewall settings
- Verify port is not in use: `lsof -i :8000`
- Check CORS configuration

### Streaming Not Working

**Symptom:** Response doesn't stream, arrives all at once

**Solution:**
- Verify `Content-Type: text/event-stream` header
- Check that backend yields chunks properly
- Ensure no response buffering in middleware
- Test with curl: `curl -N http://localhost:8000/query/stream`

### Slow Responses

**Symptom:** Requests timeout or take too long

**Solution:**
- Optimize retrieval (reduce top_k)
- Use GPU acceleration
- Cache embeddings
- Implement request queuing
- Consider smaller model for faster inference

### Type Errors

**Symptom:** TypeScript errors about response types

**Solution:**
- Ensure backend response matches `QueryResponse` type in `types/index.ts`
- Add response validation in API route

## Production Deployment

### Backend

1. Use production ASGI server with streaming support:
   ```bash
   gunicorn backend.api:app -w 4 -k uvicorn.workers.UvicornWorker \
     --timeout 120 --keep-alive 30
   ```

2. Set up reverse proxy (Nginx) with streaming:
   ```nginx
   location /api {
       proxy_pass http://localhost:8000;
       proxy_http_version 1.1;
       proxy_set_header Host $host;
       proxy_set_header Connection "";
       proxy_buffering off;
       proxy_cache off;
       chunked_transfer_encoding on;
   }
   ```

### Frontend

1. Build optimized bundle:
   ```bash
   npm run build
   ```

2. Deploy to Vercel (automatic streaming support):
   ```bash
   vercel deploy
   ```

3. Or self-host with PM2:
   ```bash
   pm2 start npm --name "legal-ai" -- start
   ```

4. Set environment variables:
   ```bash
   NEXT_PUBLIC_API_URL=https://api.yourdomain.com
   ```

## Security Considerations

1. **API Authentication:** Add authentication middleware
2. **Rate Limiting:** Prevent abuse with rate limits
3. **Input Validation:** Sanitize user queries
4. **HTTPS:** Use SSL certificates in production
5. **Secrets Management:** Use environment variables, never commit secrets

## Monitoring

Add logging to track:
- Request latency
- Streaming performance
- Error rates
- CJPE score distribution
- Popular queries
- Token generation speed

Example using Python logging:

```python
import logging
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@app.post("/query/stream")
async def query_stream_endpoint(request: QueryRequest):
    start_time = time.time()
    logger.info(f"Streaming query received: {request.query[:50]}...")
    
    async def logged_stream():
        chunk_count = 0
        async for chunk in generate_stream(request.query, request.files):
            chunk_count += 1
            yield chunk
        
        elapsed = time.time() - start_time
        logger.info(f"Stream completed: {chunk_count} chunks in {elapsed:.2f}s")
    
    return StreamingResponse(logged_stream(), media_type="text/event-stream")
```

### Performance Metrics

Track these metrics for optimization:
- Time to first byte (TTFB)
- Tokens per second
- Average CJPE score
- User engagement (messages per session)

## Next Steps

1. Implement authentication if needed
2. Add request caching for common queries
3. Set up monitoring and analytics
4. Implement feedback collection
5. Add A/B testing for model comparisons