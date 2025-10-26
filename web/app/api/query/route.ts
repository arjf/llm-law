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
  files?: string[];
  conversation_history?: ConversationMessage[];
  documents?: DocumentMetadata[];
}

export async function POST(request: NextRequest) {
  try {
    const body: QueryRequest = await request.json();
    const { query, files, conversation_history, documents } = body;

    if (!query || typeof query !== "string") {
      return new Response(
        JSON.stringify({ error: "Invalid query parameter" }),
        {
          status: 400,
          headers: { "Content-Type": "application/json" },
        },
      );
    }

    // Call the Python backend
    const backendUrl = process.env.BACKEND_API_URL || "http://localhost:8000";
    const response = await fetch(`${backendUrl}/query/stream`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        query,
        files,
        conversation_history,
        documents,
      }),
    });

    if (!response.ok) {
      throw new Error(`Backend error: ${response.status}`);
    }

    // Stream the response back to client
    return new Response(response.body, {
      headers: {
        "Content-Type": "text/event-stream",
        "Cache-Control": "no-cache",
        Connection: "keep-alive",
      },
    });
  } catch (error) {
    console.error("Error processing query:", error);
    return new Response(
      JSON.stringify({
        error:
          "Failed to connect to backend. Make sure the backend server is running on port 8000.",
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
      message: "RAG Query API with Streaming Support",
      status: "Ready",
      version: "1.0.0",
    }),
    {
      status: 200,
      headers: { "Content-Type": "application/json" },
    },
  );
}
