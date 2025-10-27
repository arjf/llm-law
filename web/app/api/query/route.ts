import { NextRequest } from "next/server";

interface ConversationMessage {
  role: "user" | "assistant";
  content: string;
  documentIds?: string[];
}

interface DocumentMetadata {
  id: string;
  name: string;
  size: number;
  type: string;
}

interface QueryRequest {
  query: string;
  task?: "pcr" | "lsi" | "summ" | "general";
  files?: string[];
  conversation_history?: ConversationMessage[];
  documents?: DocumentMetadata[];
  useCJPE?: boolean; // Flag to use CJPE prediction
}

// Get backend URLs from environment
const GEMMA_API_URL =
  process.env.NEXT_PUBLIC_GEMMA_API_URL ||
  process.env.NEXT_PUBLIC_API_URL ||
  "http://localhost:8000";

const CJPE_API_URL =
  process.env.NEXT_PUBLIC_CJPE_API_URL || "http://localhost:8001";

export async function POST(request: NextRequest) {
  try {
    const body: QueryRequest = await request.json();
    const {
      query,
      task = "general",
      files,
      conversation_history,
      documents,
      useCJPE = false,
    } = body;

    if (!query || typeof query !== "string") {
      return new Response(
        JSON.stringify({ error: "Invalid query parameter" }),
        {
          status: 400,
          headers: { "Content-Type": "application/json" },
        },
      );
    }

    // If CJPE prediction is requested, call CJPE backend
    if (useCJPE) {
      try {
        const cjpeResponse = await fetch(`${CJPE_API_URL}/predict`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ text: query }),
        });

        if (!cjpeResponse.ok) {
          throw new Error(`CJPE Backend error: ${cjpeResponse.status}`);
        }

        const cjpeData = await cjpeResponse.json();

        return new Response(JSON.stringify(cjpeData), {
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

    // Default: Call Gemma backend with streaming
    try {
      const response = await fetch(`${GEMMA_API_URL}/query/stream`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          query,
          task,
          files,
          conversation_history,
          documents,
        }),
      });

      if (!response.ok) {
        throw new Error(`Gemma Backend error: ${response.status}`);
      }

      // Stream the response back to client
      return new Response(response.body, {
        headers: {
          "Content-Type": "text/plain",
          "Cache-Control": "no-cache",
          Connection: "keep-alive",
        },
      });
    } catch (error) {
      console.error("Error calling Gemma backend:", error);
      return new Response(
        JSON.stringify({
          error:
            "Failed to connect to Gemma backend. Make sure the backend server is running.",
          details: error instanceof Error ? error.message : "Unknown error",
        }),
        {
          status: 500,
          headers: { "Content-Type": "application/json" },
        },
      );
    }
  } catch (error) {
    console.error("Error processing query:", error);
    return new Response(
      JSON.stringify({
        error: "Internal server error",
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
      message: "LegalAI Query API",
      status: "Ready",
      version: "2.0.0",
      backends: {
        gemma: {
          url: GEMMA_API_URL,
          features: ["streaming", "rag", "multitask"],
        },
        cjpe: {
          url: CJPE_API_URL,
          features: ["judgment_prediction", "explainability"],
        },
      },
      endpoints: {
        "POST /api/query": "Query Gemma backend (streaming)",
        "POST /api/query?useCJPE=true": "Get CJPE prediction",
        "GET /api/cjpe/predict": "Direct CJPE prediction endpoint",
        "GET /api/health": "Health check for all backends",
      },
    }),
    {
      status: 200,
      headers: { "Content-Type": "application/json" },
    },
  );
}
