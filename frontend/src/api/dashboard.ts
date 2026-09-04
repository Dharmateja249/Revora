import { DashboardMetrics } from "../types/recovery";
import { ApiError, fetchCustomerAuthToken } from "./recovery";

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || "";
const DEFAULT_AUTH_CUSTOMER_ID =
  import.meta.env.VITE_DEMO_CUSTOMER_ID || "e9cd4c97-979b-4753-9925-640623f74eee";

/**
 * Fetches real-time recovery, financial, execution, and AI telemetry metrics
 * calculated from the authenticated customer's backend database records and runtime VectorIndex.
 *
 * @param customerId Optional customer UUID (defaults to configured demo tenant)
 * @returns DashboardMetrics object
 */
export async function fetchDashboardMetrics(
  customerId: string = DEFAULT_AUTH_CUSTOMER_ID
): Promise<DashboardMetrics> {
  const token = await fetchCustomerAuthToken(customerId);

  const url = `${API_BASE_URL}/api/dashboard/metrics`;
  try {
    const res = await fetch(url, {
      method: "GET",
      headers: {
        Accept: "application/json",
        Authorization: `Bearer ${token}`,
      },
    });

    if (!res.ok) {
      let detail = `Failed to fetch dashboard metrics (HTTP ${res.status})`;
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

    const data: DashboardMetrics = await res.json();
    return data;
  } catch (err: unknown) {
    if (err instanceof ApiError) throw err;
    throw new ApiError(
      0,
      err instanceof Error
        ? err.message
        : "Network error while loading dashboard metrics."
    );
  }
}
