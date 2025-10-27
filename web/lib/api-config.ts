/**
 * API Configuration for LegalAI Backends
 *
 * This module provides configuration and utilities for communicating with:
 * 1. Gemma-3n Legal Assistant (RAG-enhanced streaming responses)
 * 2. CJPE Scorer (Court Judgment Prediction and Explanation)
 */

// API Endpoints
export const API_ENDPOINTS = {
  // Gemma-3n Legal Assistant with RAG
  gemma: {
    base: process.env.NEXT_PUBLIC_GEMMA_API_URL || process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000',
    query: '/query',
    queryStream: '/query/stream',
    health: '/health',
    metrics: '/metrics',
  },

  // CJPE Scorer (Court Judgment Prediction)
  cjpe: {
    base: process.env.NEXT_PUBLIC_CJPE_API_URL || 'http://localhost:8001',
    predict: '/predict',
    health: '/health',
    metrics: '/metrics',
  },
} as const;

// API Configuration
export const API_CONFIG = {
  timeout: parseInt(process.env.NEXT_PUBLIC_API_TIMEOUT || '30000'),
  debug: process.env.NEXT_PUBLIC_DEBUG === 'true',
  apiKey: process.env.NEXT_PUBLIC_API_KEY,
} as const;

// Types
export interface GemmaQueryRequest {
  query: string;
  task?: 'pcr' | 'lsi' | 'summ' | 'general';
  files?: string[];
  conversation_history?: Array<{
    role: 'user' | 'assistant';
    content: string;
    documentIds?: string[];
  }>;
  documents?: Array<{
    id: string;
    name: string;
    size: number;
    type: string;
  }>;
}

export interface GemmaQueryResponse {
  response: string;
  retrieved_docs?: Array<{
    text: string;
    score: number;
    label: string;
    case_id: string;
  }>;
  metadata: {
    task: string;
    model: string;
    num_docs_retrieved: number;
    processing_time_ms: number;
  };
}

export interface GemmaStreamMetadata {
  retrieved_docs: Array<{
    text: string;
    score: number;
    label: string;
    case_id: string;
  }>;
  metadata: {
    task: string;
    model: string;
    num_docs_retrieved: number;
    processing_time_ms: number;
    timestamp: number;
  };
}

export interface CJPEPredictRequest {
  text: string;
}

export interface CJPEPredictionResponse {
  prediction: 'favorable' | 'unfavorable';
  predicted_class: number;
  confidence: number;
  probabilities: {
    unfavorable: number;
    favorable: number;
  };
  num_chunks: number;
  prediction_time_ms: number;
  attention_weights: number[];
  top_chunks: Array<{
    index: number;
    attention_weight: number;
    text: string;
  }>;
}

export interface HealthResponse {
  status: string;
  version: string;
  model: string;
  device: string;
  [key: string]: any;
}

// Utility Functions
export const getGemmaUrl = (endpoint: keyof typeof API_ENDPOINTS.gemma): string => {
  const base = API_ENDPOINTS.gemma.base;
  const path = API_ENDPOINTS.gemma[endpoint];
  return `${base}${path}`;
};

export const getCJPEUrl = (endpoint: keyof typeof API_ENDPOINTS.cjpe): string => {
  const base = API_ENDPOINTS.cjpe.base;
  const path = API_ENDPOINTS.cjpe[endpoint];
  return `${base}${path}`;
};

export const getHeaders = (): HeadersInit => {
  const headers: HeadersInit = {
    'Content-Type': 'application/json',
  };

  if (API_CONFIG.apiKey) {
    headers['Authorization'] = `Bearer ${API_CONFIG.apiKey}`;
  }

  return headers;
};

// API Client Functions
export async function queryGemma(request: GemmaQueryRequest): Promise<GemmaQueryResponse> {
  const url = getGemmaUrl('query');

  const response = await fetch(url, {
    method: 'POST',
    headers: getHeaders(),
    body: JSON.stringify(request),
    signal: AbortSignal.timeout(API_CONFIG.timeout),
  });

  if (!response.ok) {
    throw new Error(`Gemma API error: ${response.status} ${response.statusText}`);
  }

  return response.json();
}

export async function predictCJPE(request: CJPEPredictRequest): Promise<CJPEPredictionResponse> {
  const url = getCJPEUrl('predict');

  const response = await fetch(url, {
    method: 'POST',
    headers: getHeaders(),
    body: JSON.stringify(request),
    signal: AbortSignal.timeout(API_CONFIG.timeout),
  });

  if (!response.ok) {
    throw new Error(`CJPE API error: ${response.status} ${response.statusText}`);
  }

  return response.json();
}

export async function checkGemmaHealth(): Promise<HealthResponse> {
  const url = getGemmaUrl('health');
  const response = await fetch(url);

  if (!response.ok) {
    throw new Error(`Health check failed: ${response.status}`);
  }

  return response.json();
}

export async function checkCJPEHealth(): Promise<HealthResponse> {
  const url = getCJPEUrl('health');
  const response = await fetch(url);

  if (!response.ok) {
    throw new Error(`Health check failed: ${response.status}`);
  }

  return response.json();
}

// Debug logging
export function debugLog(message: string, data?: any) {
  if (API_CONFIG.debug) {
    console.log(`[API] ${message}`, data || '');
  }
}

// Error handling
export class APIError extends Error {
  constructor(
    message: string,
    public statusCode?: number,
    public endpoint?: string,
  ) {
    super(message);
    this.name = 'APIError';
  }
}

export function handleAPIError(error: unknown, endpoint?: string): APIError {
  if (error instanceof APIError) {
    return error;
  }

  if (error instanceof Error) {
    return new APIError(error.message, undefined, endpoint);
  }

  return new APIError('Unknown API error', undefined, endpoint);
}
