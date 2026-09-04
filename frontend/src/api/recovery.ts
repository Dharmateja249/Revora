import {
  AuthTokenResponse,
  HealthCheckResponse,
  RecoveryDecisionRequest,
  RecoveryDecisionResponse,
} from "../types/recovery";

// In development with Vite, we can use empty string if proxied, or explicit VITE_API_BASE_URL
const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || "";

// Standard demo customer UUID recognized by backend auth checks
const DEFAULT_AUTH_CUSTOMER_ID =
  import.meta.env.VITE_DEMO_CUSTOMER_ID || "e9cd4c97-979b-4753-9925-640623f74eee";

interface CachedToken {
  token: string;
  expiresAt: number; // Timestamp in milliseconds
}

// In-memory token cache (customerId -> CachedToken) strictly isolated from local/sessionStorage
const tokenCache = new Map<string, CachedToken>();

export class ApiError extends Error {
  status: number;
  detail: string;

  constructor(status: number, detail: string) {
    super(`API Error ${status}: ${detail}`);
    this.status = status;
    this.detail = detail;
    this.name = "ApiError";
  }
}

/**
 * Clears the in-memory demo authentication token cache.
 */
export function clearAuthTokenCache(): void {
  tokenCache.clear();
}

/**
 * Requests a cryptographically verifiable demo authentication token for a customer UUID.
 * Tokens are cached in-memory until near expiration to avoid redundant requests.
 *
 * @param customerId Customer UUID to bind to the authentication token
 * @returns Cryptographically verifiable access token string
 */
export async function fetchCustomerAuthToken(customerId: string): Promise<string> {
  const normalizedId = (customerId || "").trim();
  if (!normalizedId) {
    throw new ApiError(400, "A valid customer UUID is required to obtain an authentication token.");
  }

  // Check in-memory cache (with a 60-second safety window before expiration)
  const cached = tokenCache.get(normalizedId);
  if (cached && cached.expiresAt > Date.now() + 60_000) {
    return cached.token;
  }

  const url = `${API_BASE_URL}/api/auth/token`;
  try {
    const res = await fetch(url, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Accept: "application/json",
      },
      body: JSON.stringify({ customer_id: normalizedId }),
    });

    if (!res.ok) {
      let detail = `Failed to obtain authentication token (HTTP ${res.status})`;
      try {
        const errorData = await res.json();
        if (errorData && typeof errorData === "object") {
          if (typeof errorData.detail === "string") {
            detail = errorData.detail;
          } else if (Array.isArray(errorData.detail)) {
            detail = errorData.detail
              .map((d: { msg?: string; loc?: string[] }) => d.msg || JSON.stringify(d))
              .join("; ");
          }
        }
      } catch {
        detail = res.statusText || detail;
      }
      throw new ApiError(res.status, detail);
    }

    const data: AuthTokenResponse = await res.json();
    if (!data || !data.access_token) {
      throw new ApiError(500, "Malformed authentication response: missing access_token.");
    }

    // Cache token in memory; expires_in is in seconds
    const expiresInMs = (data.expires_in || 86400) * 1000;
    tokenCache.set(normalizedId, {
      token: data.access_token,
      expiresAt: Date.now() + expiresInMs,
    });

    return data.access_token;
  } catch (err: unknown) {
    if (err instanceof ApiError) throw err;
    throw new ApiError(
      0,
      err instanceof Error
        ? err.message
        : "Failed to connect to authentication service. Please verify backend status."
    );
  }
}

/**
 * Health check to verify backend connectivity.
 */
export async function checkBackendHealth(): Promise<HealthCheckResponse> {
  const url = `${API_BASE_URL}/health`;
  try {
    const res = await fetch(url, {
      method: "GET",
      headers: {
        Accept: "application/json",
      },
    });

    if (!res.ok) {
      throw new ApiError(res.status, `Backend health check failed (${res.statusText})`);
    }

    return await res.json();
  } catch (err: unknown) {
    if (err instanceof ApiError) throw err;
    throw new ApiError(0, "Unable to reach backend service. Is the server running on port 8000?");
  }
}

/**
 * Evaluates recovery decision and optionally triggers external gateway action.
 *
 * @param request Complete recovery decision payload matching backend DTO
 * @returns Evaluated decision response including explainability and optional execution telemetry
 */
export async function evaluateRecoveryDecision(
  request: RecoveryDecisionRequest
): Promise<RecoveryDecisionResponse> {
  const url = `${API_BASE_URL}/api/recovery/decision`;

  // Ensure customer_id matches authenticated principal or is passed cleanly
  const customerId = request.customer?.customer_id || DEFAULT_AUTH_CUSTOMER_ID;
  const payload: RecoveryDecisionRequest = {
    ...request,
    customer: {
      total_payments: 0,
      successful_payments: 0,
      failed_payments: 0,
      historical_success_rate: 0.0,
      ...request.customer,
      customer_id: customerId,
    },
    // Ensure execute_action is explicitly boolean
    execute_action: Boolean(request.execute_action),
  };

  // Obtain server-issued verifiable token (never send raw customer UUID)
  const authToken = await fetchCustomerAuthToken(customerId);

  try {
    const res = await fetch(url, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Accept: "application/json",
        Authorization: `Bearer ${authToken}`,
      },
      body: JSON.stringify(payload),
    });

    if (!res.ok) {
      let detail = `Request failed with HTTP status ${res.status}`;
      try {
        const errorData = await res.json();
        if (errorData && typeof errorData === "object") {
          if (typeof errorData.detail === "string") {
            detail = errorData.detail;
          } else if (Array.isArray(errorData.detail)) {
            detail = errorData.detail
              .map((d: { msg?: string; loc?: string[] }) => d.msg || JSON.stringify(d))
              .join("; ");
          }
        }
      } catch {
        // Fallback to generic status text
        detail = res.statusText || detail;
      }
      throw new ApiError(res.status, detail);
    }

    return await res.json();
  } catch (err: unknown) {
    if (err instanceof ApiError) throw err;
    throw new ApiError(
      0,
      err instanceof Error
        ? err.message
        : "Failed to connect to Revora backend. Please verify network and server state."
    );
  }
}
