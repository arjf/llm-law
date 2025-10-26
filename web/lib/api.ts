/**
 * API Client for Legal RAG Backend
 *
 * Handles communication between the Next.js frontend and the Python RAG backend.
 * Replace the placeholder implementation with actual backend calls.
 */

import { QueryRequest, QueryResponse } from "@/types";

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

/**
 * Query the RAG system with a legal question
 *
 * @param query - User's legal question
 * @returns Promise with answer, CJPE score, and retrieved cases
 */
export async function queryRAGSystem(query: string): Promise<QueryResponse> {
  try {
    // Use Next.js API route as proxy to Python backend
    const response = await fetch("/api/query", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ query } as QueryRequest),
    });

    if (!response.ok) {
      throw new Error(`API error: ${response.status} ${response.statusText}`);
    }

    const data: QueryResponse = await response.json();
    return data;
  } catch (error) {
    console.error("Error querying RAG system:", error);
    throw error;
  }
}

/**
 * Direct call to Python backend (alternative to Next.js API route)
 * Use this if you want to bypass the Next.js API layer
 *
 * @param query - User's legal question
 * @returns Promise with answer and metadata
 */
export async function queryBackendDirect(
  query: string,
): Promise<QueryResponse> {
  try {
    const response = await fetch(`${API_BASE_URL}/query`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ query }),
    });

    if (!response.ok) {
      throw new Error(`Backend error: ${response.status}`);
    }

    return await response.json();
  } catch (error) {
    console.error("Error calling backend directly:", error);
    throw error;
  }
}

/**
 * Check backend health status
 *
 * @returns Promise with backend status information
 */
export async function checkBackendStatus(): Promise<{
  status: string;
  version: string;
  models: {
    retrieval: string;
    generation: string;
  };
}> {
  try {
    const response = await fetch(`${API_BASE_URL}/health`, {
      method: "GET",
    });

    if (!response.ok) {
      throw new Error("Backend not available");
    }

    return await response.json();
  } catch (error) {
    return {
      status: "disconnected",
      version: "unknown",
      models: {
        retrieval: "InLegalBERT",
        generation: "Mistral-7B",
      },
    };
  }
}

/**
 * Fetch system statistics and metrics
 *
 * @returns Promise with system metrics
 */
export async function getSystemMetrics(): Promise<{
  total_queries: number;
  avg_cjpe_score: number;
  avg_response_time_ms: number;
  index_size: number;
}> {
  try {
    const response = await fetch(`${API_BASE_URL}/metrics`, {
      method: "GET",
    });

    if (!response.ok) {
      throw new Error("Failed to fetch metrics");
    }

    return await response.json();
  } catch (error) {
    console.error("Error fetching metrics:", error);
    return {
      total_queries: 0,
      avg_cjpe_score: 0,
      avg_response_time_ms: 0,
      index_size: 0,
    };
  }
}
