"""
Revora Dashboard Metrics Schema.

Defines client-facing data transfer objects for aggregated recovery and AI performance metrics.
"""

from pydantic import BaseModel, ConfigDict, Field


class DashboardMetricsResponse(BaseModel):
    """
    Response DTO containing real-time recovery, execution, and AI telemetry metrics
    calculated directly from tenant database records and active runtime state.
    """

    model_config = ConfigDict(
        frozen=True,
        from_attributes=True,
    )

    recovery_rate: float = Field(
        ...,
        description="Percentage of failed payment recovery cases successfully recovered (0.0 to 100.0).",
    )
    amount_recovered: float = Field(
        ...,
        description="Total monetary value successfully recovered across all attempts.",
    )
    total_cases: int = Field(
        ...,
        description="Total number of recovery opportunity cases tracked for the tenant.",
    )
    recovered_cases: int = Field(
        ...,
        description="Count of successfully recovered cases.",
    )
    failed_cases: int = Field(
        ...,
        description="Count of failed or abandoned recovery cases.",
    )
    pending_cases: int = Field(
        default=0,
        description="Count of active/in-progress recovery cases.",
    )
    revenue_at_risk: float = Field(
        default=0.0,
        description="Total monetary amount currently at risk in active failed payment cases.",
    )
    execution_success_rate: float = Field(
        ...,
        description="Percentage of external gateway execution attempts that succeeded (0.0 to 100.0).",
    )
    total_executions: int = Field(
        default=0,
        description="Total number of recovery execution attempts dispatched.",
    )
    successful_executions: int = Field(
        ...,
        description="Count of successful execution attempts.",
    )
    failed_executions: int = Field(
        ...,
        description="Count of failed execution attempts.",
    )
    pending_executions: int = Field(
        ...,
        description="Count of in-progress or pending execution attempts.",
    )
    policy_overrides: int = Field(
        ...,
        description="Count of candidate recommendations overridden by PolicyValidator.",
    )
    rag_precedents: int = Field(
        ...,
        description="Current number of precedent cases stored in the runtime RAG vector index.",
    )
    average_confidence: float = Field(
        default=0.0,
        description="Average confidence score of recovery recommendations (0.0 to 1.0).",
    )
    fallback_decisions: int = Field(
        default=0,
        description="Count of recovery decisions generated via deterministic fallback.",
    )
