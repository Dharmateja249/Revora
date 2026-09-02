"""
Revora Recovery Decision API Request and Response DTO Schemas.

Defines client-facing data transfer contracts for POST /api/recovery/decision,
reusing domain representations where appropriate and preventing internal leakages.
"""

from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from app.decision_engine import RecoveryAction


class CustomerProfileDTO(BaseModel):
    """
    Customer profile and baseline recovery statistics.
    """

    model_config = ConfigDict(extra="ignore")

    customer_id: UUID | None = Field(
        default=None,
        description="Optional unique identifier for the customer.",
    )
    total_payments: int = Field(
        default=0,
        ge=0,
        description="Total historical payment transactions.",
    )
    successful_payments: int = Field(
        default=0,
        ge=0,
        description="Total successful historical payment transactions.",
    )
    failed_payments: int = Field(
        default=0,
        ge=0,
        description="Total failed historical payment transactions.",
    )
    historical_success_rate: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Historical payment success rate bounded between 0.0 and 1.0.",
    )


class RecoveryAttemptDTO(BaseModel):
    """
    Execution details and outcome of a previous recovery attempt.
    """

    model_config = ConfigDict(extra="ignore")

    action: str = Field(
        ...,
        min_length=1,
        description="Recovery action executed in this attempt.",
    )
    status: str = Field(
        ...,
        min_length=1,
        description="Outcome status of the attempt (e.g., 'failed', 'succeeded').",
    )
    amount_recovered: float = Field(
        default=0.0,
        ge=0.0,
        description="Amount recovered during this attempt.",
    )
    error_code: str | None = Field(
        default=None,
        description="Gateway or technical error code, if attempt failed.",
    )


class RecoveryDecisionRequest(BaseModel):
    """
    Request DTO for evaluating a failed payment recovery decision via the API.
    """

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
    )

    amount: float = Field(
        ...,
        gt=0.0,
        description="Transaction amount of the failed payment.",
    )
    currency: str = Field(
        default="INR",
        min_length=3,
        max_length=3,
        description="Three-letter currency code (e.g., 'INR', 'USD').",
    )
    payment_method: str = Field(
        ...,
        min_length=1,
        description="Payment method used (e.g., 'upi', 'card', 'netbanking').",
    )
    failure_reason: str = Field(
        ...,
        min_length=1,
        description="Payment failure reason code or gateway message.",
    )
    payment_status: str = Field(
        default="failed",
        description="Current payment status.",
    )
    payment_id: UUID | None = Field(
        default=None,
        description="Optional identifier for the payment transaction.",
    )
    customer: CustomerProfileDTO = Field(
        default_factory=CustomerProfileDTO,
        description="Customer profile and transaction history summary.",
    )
    previous_attempts: list[RecoveryAttemptDTO] = Field(
        default_factory=list,
        description="Chronological history of prior attempts on this payment.",
    )
    opportunity_status: str = Field(
        default="open",
        description="Current recovery opportunity lifecycle status.",
    )
    revenue_at_risk: float | None = Field(
        default=None,
        ge=0.0,
        description="Total revenue at risk; defaults to payment amount if omitted.",
    )
    max_attempts: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Maximum allowed recovery attempts for this opportunity.",
    )
    execute_action: bool = Field(
        default=False,
        description="Whether to execute the policy-approved recovery action via gateway adapter. Defaults to False (opt-in).",
    )


class ActionExecutionResultDTO(BaseModel):
    """
    Client-facing DTO reporting the outcome of recovery action execution.
    Excludes credentials, raw payment tokens, and internal secrets.
    """

    model_config = ConfigDict(
        frozen=True,
        from_attributes=True,
    )

    action: RecoveryAction = Field(
        ...,
        description="Action evaluated or executed.",
    )
    attempted: bool = Field(
        ...,
        description="Whether execution was attempted.",
    )
    status: str = Field(
        ...,
        description="Execution status: 'success', 'simulated', 'failed', 'prohibited', 'skipped', 'unsupported'.",
    )
    success: bool = Field(
        ...,
        description="True if execution succeeded or was successfully simulated.",
    )
    reference_id: str | None = Field(
        default=None,
        description="Gateway reference identifier (e.g., Razorpay 'plink_...' ID).",
    )
    resource_url: str | None = Field(
        default=None,
        description="Public URL for recovery action (e.g. short payment link URL).",
    )
    message: str = Field(
        ...,
        description="Human-readable outcome summary.",
    )
    error: str | None = Field(
        default=None,
        description="Safe diagnostic error if execution failed.",
    )


class RecoveryDecisionResponse(BaseModel):
    """
    Response DTO exposing the evaluated recovery decision, explainability, and telemetry.
    Strictly excludes internal credentials, API keys, and internal database IDs.
    """

    model_config = ConfigDict(
        frozen=True,
        from_attributes=True,
    )

    recommended_action: RecoveryAction = Field(
        ...,
        description="Effective, policy-validated recovery action.",
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Decision confidence score bounded between 0.0 and 1.0.",
    )
    reasoning: str = Field(
        ...,
        description="Human-readable explanation of why this action was selected.",
    )
    key_factors: list[str] = Field(
        default_factory=list,
        description="Domain and contextual signals influencing the recommendation.",
    )
    referenced_case_ids: list[str] = Field(
        default_factory=list,
        description="Historical reference case identifiers supporting the decision.",
    )
    agent_used: bool = Field(
        ...,
        description="True if recommendation was generated by LLM agent; False if deterministic fallback.",
    )
    policy_overridden: bool = Field(
        default=False,
        description="True if PolicyValidator modified the LLM's candidate recommendation.",
    )
    is_fallback: bool = Field(
        default=False,
        description="True if deterministic fallback was triggered due to provider failure or error.",
    )
    fallback_reason: str | None = Field(
        default=None,
        description="Diagnostic explanation for fallback activation, if applicable.",
    )
    execution: ActionExecutionResultDTO | None = Field(
        default=None,
        description="Structured outcome of recovery action execution, if attempted.",
    )
