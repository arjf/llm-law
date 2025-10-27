import { NextRequest } from "next/server";

// Get backend URLs from environment
const GEMMA_API_URL =
  process.env.NEXT_PUBLIC_GEMMA_API_URL ||
  process.env.NEXT_PUBLIC_API_URL ||
  "http://localhost:8000";

const CJPE_API_URL =
  process.env.NEXT_PUBLIC_CJPE_API_URL || "http://localhost:8001";

interface BackendHealth {
  status: "healthy" | "unhealthy" | "unknown";
  version?: string;
  model?: string;
  error?: string;
  responseTime?: number;
}

async function checkBackendHealth(url: string): Promise<BackendHealth> {
  const startTime = Date.now();
  try {
    const response = await fetch(`${url}/health`, {
      method: "GET",
      headers: { "Content-Type": "application/json" },
      signal: AbortSignal.timeout(5000), // 5 second timeout
    });

    const responseTime = Date.now() - startTime;

    if (!response.ok) {
      return {
        status: "unhealthy",
        error: `HTTP ${response.status}`,
        responseTime,
      };
    }

    const data = await response.json();

    return {
      status: data.status === "healthy" ? "healthy" : "unhealthy",
      version: data.version,
      model: data.model,
      responseTime,
    };
  } catch (error) {
    return {
      status: "unhealthy",
      error: error instanceof Error ? error.message : "Unknown error",
      responseTime: Date.now() - startTime,
    };
  }
}

export async function GET(request: NextRequest) {
  const startTime = Date.now();

  // Check both backends in parallel
  const [gemmaHealth, cjpeHealth] = await Promise.all([
    checkBackendHealth(GEMMA_API_URL),
    checkBackendHealth(CJPE_API_URL),
  ]);

  const totalTime = Date.now() - startTime;

  // Determine overall status
  const allHealthy =
    gemmaHealth.status === "healthy" && cjpeHealth.status === "healthy";
  const anyHealthy =
    gemmaHealth.status === "healthy" || cjpeHealth.status === "healthy";

  const overallStatus = allHealthy
    ? "healthy"
    : anyHealthy
      ? "degraded"
      : "unhealthy";

  const response = {
    status: overallStatus,
    timestamp: new Date().toISOString(),
    backends: {
      gemma: {
        url: GEMMA_API_URL,
        ...gemmaHealth,
      },
      cjpe: {
        url: CJPE_API_URL,
        ...cjpeHealth,
      },
    },
    metadata: {
      totalCheckTime: totalTime,
      environment: process.env.NODE_ENV || "development",
    },
  };

  // Return appropriate status code
  const statusCode = overallStatus === "healthy" ? 200 : 503;

  return new Response(JSON.stringify(response), {
    status: statusCode,
    headers: { "Content-Type": "application/json" },
  });
}
