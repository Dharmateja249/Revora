/**
 * Revora Recovery Decision API Data Transfer Models.
 * Strictly mirrors the backend Pydantic models in backend/app/schemas/decision.py.
 */

export type RecoveryAction =
  | "retry_payment"
  | "wait_and_retry"
  | "payment_link"
  | "change_payment_method"
  | "customer_outreach"
  | "escalate_support"
  | "no_action";

export interface CustomerProfileDTO {
  customer_id?: string;
  total_payments: number;
  successful_payments: number;
  failed_payments: number;
  historical_success_rate: number;
}

export interface RecoveryAttemptDTO {
  action: string;
  status: string;
  amount_recovered?: number;
  error_code?: string | null;
}

export interface RecoveryDecisionRequest {
  amount: number;
  currency?: string;
  payment_method: string;
  failure_reason: string;
  payment_status?: string;
  payment_id?: string | null;
  customer?: CustomerProfileDTO;
  previous_attempts?: RecoveryAttemptDTO[];
  opportunity_status?: string;
  revenue_at_risk?: number | null;
  max_attempts?: number;
  execute_action?: boolean;
}

export interface ActionExecutionResultDTO {
  action: RecoveryAction;
  attempted: boolean;
  status: "success" | "simulated" | "failed" | "prohibited" | "skipped" | "unsupported" | string;
  success: boolean;
  reference_id?: string | null;
  resource_url?: string | null;
  message: string;
  error?: string | null;
}

export interface RecoveryDecisionResponse {
  recommended_action: RecoveryAction;
  confidence: number;
  reasoning: string;
  key_factors: string[];
  referenced_case_ids: string[];
  agent_used: boolean;
  policy_overridden: boolean;
  is_fallback: boolean;
  fallback_reason?: string | null;
  execution?: ActionExecutionResultDTO | null;
}

export interface HealthCheckResponse {
  status: string;
  app: string;
  version: string;
}

/**
 * Frontend Demo Case Schema for representative scenarios.
 */
export interface DemoPaymentCase {
  id: string;
  customerName: string;
  customerEmail: string;
  customerTier: "Enterprise" | "Growth" | "Starter";
  requestPayload: RecoveryDecisionRequest;
  timestamp: string;
  displayStatus: "Failed" | "Recovered" | "In Recovery";
  description: string;
  badgeColor?: string;
}
