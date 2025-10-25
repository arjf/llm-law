/**
 * Type definitions for Legal RAG Web UI
 */

// Chat message types
export interface Message {
  id: string;
  role: "user" | "assistant";
  content: string;
  timestamp: Date;
}

// API request/response types
export interface QueryRequest {
  query: string;
}

export interface RetrievedCase {
  case_id: string;
  title: string;
  relevance_score: number;
  excerpt?: string;
}

export interface QueryMetadata {
  retrieval_time_ms: number;
  generation_time_ms: number;
  model_version?: string;
}

export interface QueryResponse {
  answer: string;
  cjpe_score: number;
  retrieved_cases: RetrievedCase[];
  metadata: QueryMetadata;
  error?: string;
}

// CJPE Score types
export interface CJPEMetrics {
  score: number;
  confidence: "high" | "medium" | "low";
  timestamp: Date;
}

// System status types
export interface SystemStatus {
  backend_connected: boolean;
  retrieval_model: string;
  generation_model: string;
  dataset: string;
  version: string;
}

// Component prop types
export interface CJPEScoreProps {
  score?: number;
  loading?: boolean;
}

export interface ChatInterfaceProps {
  onScoreUpdate?: (score: number) => void;
}
