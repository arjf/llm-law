import { NextRequest } from "next/server";

interface CJPEPredictRequest {
  text: string;
}

interface CJPEPredictionResponse {
  prediction: "favorable" | "unfavorable";
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

// Get CJPE backend URL from environment
const CJPE_API_URL =
  process.env.NEXT_PUBLIC_CJPE_API_URL || "http://localhost:8001";

export async function POST(request: NextRequest) {
  try {
    const body: CJPEPredictRequest = await request.json();
    const { text } = body;

    if (!text || typeof text !== "string") {
      return new Response(
        JSON.stringify({ error: "Invalid text parameter" }),
        {
          status: 400,
          headers: { "Content-Type": "application/json" },
        },
      );
    }

    // Call CJPE backend
    const response = await fetch(`${CJPE_API_URL}/predict`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text }),
    });

    if (!response.ok) {
      throw new Error(`CJPE Backend error: ${response.status}`);
    }

    const data: CJPEPredictionResponse = await response.json();

    return new Response(JSON.stringify(data), {
      headers: { "Content-Type": "application/json" },
    });
  } catch (error) {
    console.error("Error calling CJPE backend:", error);
    return new Response(
      JSON.stringify({
        error:
          "Failed to connect to CJPE backend. Make sure the CJPE server is running.",
        details: error instanceof Error ? error.message : "Unknown error",
      }),
      {
        status: 500,
        headers: { "Content-Type": "application/json" },
      },
    );
  }
}

export async function GET() {
  return new Response(
    JSON.stringify({
      message: "CJPE Prediction API",
      status: "Ready",
      version: "2.0.0",
      backend: {
        url: CJPE_API_URL,
        model: "InLegalBERT-HiGRU-CJPE-Scorer",
      },
      usage: {
        method: "POST",
        body: {
          text: "Legal text to analyze for court judgment prediction",
        },
        response: {
          prediction: "favorable | unfavorable",
          confidence: "0.0 to 1.0",
          probabilities: "Class probabilities",
          top_chunks: "Most important text segments with attention weights",
        },
      },
    }),
    {
      status: 200,
      headers: { "Content-Type": "application/json" },
    },
  );
}
