"""
Revora Recovery Evaluation API Request and Response DTO Schemas.

Defines client-facing data transfer contracts for evaluating payment recovery decisions,
ensuring clean separation from internal ORM entities and protecting customer PII.
"""

from datetime import datetime, timezone
from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.decision_engine import RecoveryAction


def utc_now() -> datetime:
    """Return current UTC datetime with timezone awareness."""
    return datetime.now(timezone.utc)


class RecoveryEvaluationRequest(BaseModel):
    """
    Request DTO for evaluating a failed payment recovery decision.
    """

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        str_strip_whitespace=True,
    )

    customer_id: UUID = Field(
        ...,
        description="Internal UUID identifier for the customer.",
    )
    payment_id: UUID = Field(
        ...,
        description="Internal UUID identifier for the failed payment.",
    )
    use_rag: bool = Field(
        default=True,
        description="Whether to retrieve and incorporate empirical historical RAG evidence.",
    )
    use_agent: bool | None = Field(
        default=None,
        description="Explicitly enable/disable agent decision engine. If None, uses service default.",
    )


class RecoveryEvaluationResponse(BaseModel):
    """
    Response DTO returning the recovery decision and explainability metadata.
    Excludes internal ORM entities and customer PII.
    """

    model_config = ConfigDict(
        frozen=True,
        from_attributes=True,
    )

    payment_id: UUID = Field(
        ...,
        description="UUID of the evaluated payment.",
    )
    customer_id: UUID = Field(
        ...,
        description="UUID of the owning customer.",
    )
    opportunity_id: UUID | None = Field(
        default=None,
        description="UUID of the associated recovery opportunity, if present.",
    )
    recommended_action: RecoveryAction = Field(
        ...,
        description="Recommended recovery action.",
    )
    reason: str = Field(
        ...,
        description="Human-readable decision explanation.",
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Decision confidence score bounded between 0.0 and 1.0.",
    )
    decision_basis: dict[str, Any] = Field(
        default_factory=dict,
        description="Detailed explainability signals, rules matched, and telemetry.",
    )

    @field_validator("decision_basis", mode="before")
    @classmethod
    def _unfreeze_decision_basis(cls, v: Any) -> Any:
        def _unfreeze(item: Any) -> Any:
            if hasattr(item, "items"):
                return {k: _unfreeze(val) for k, val in item.items()}
            if isinstance(item, (list, tuple, set)):
                return [_unfreeze(val) for val in item]
            return item

        if v is None:
            return {}
        return _unfreeze(v)

    historical_rag_used: bool = Field(
        default=False,
        description="Flag indicating if historical RAG evidence was utilized.",
    )
    retrieved_evidence_count: int = Field(
        default=0,
        ge=0,
        description="Number of historical recovery cases retrieved and evaluated.",
    )
    provider: str | None = Field(
        default=None,
        description="Payment provider context identifier (e.g., 'razorpay').",
    )
    policy_version: str | None = Field(
        default=None,
        description="Version identifier of the applied recovery policy rules.",
    )
    applied_policy_ids: list[str] = Field(
        default_factory=list,
        description="Identifiers of policy rules applied during decision evaluation.",
    )
    policy_overridden: bool = Field(
        default=False,
        description="Flag indicating if the candidate action was overridden by policy validation.",
    )
    agent_used: bool = Field(
        default=False,
        description="Flag indicating if the agent decision pipeline was utilized.",
    )
    is_fallback: bool = Field(
        default=False,
        description="Flag indicating if deterministic fallback was applied.",
    )
    fallback_reason: str | None = Field(
        default=None,
        description="Sanitized reason for applying fallback, if applicable.",
    )
    evaluated_at: datetime = Field(
        default_factory=utc_now,
        description="UTC timestamp when the recovery decision was computed.",
    )
