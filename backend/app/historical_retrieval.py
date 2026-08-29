"""
Revora Historical Retrieval Domain Contracts and Schemas.

Defines the clean, immutable, typed Pydantic v2 schema representing a single
historical recovery case with relevance scoring, designed as the canonical contract
returned by any retrieval backend (deterministic or vector-based).
"""

from datetime import datetime, timezone
import types
from typing import Any, Dict, Mapping, Optional, Set
from uuid import UUID

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_serializer,
    field_validator,
    model_validator,
)


def utc_now() -> datetime:
    """Return current UTC timestamp with timezone awareness."""
    return datetime.now(timezone.utc)


def _freeze_nested(value: Any) -> Any:
    """Recursively freeze dictionaries to MappingProxyType and sequences to tuples."""
    if isinstance(value, (dict, types.MappingProxyType)):
        return types.MappingProxyType({k: _freeze_nested(v) for k, v in value.items()})
    if isinstance(value, (list, set)):
        return tuple(_freeze_nested(v) for v in value)
    if isinstance(value, tuple):
        return tuple(_freeze_nested(v) for v in value)
    return value


def _unfreeze_for_serialization(val: Any) -> Any:
    """Recursively convert MappingProxyType and tuples to dicts and lists for clean serialization."""
    if isinstance(val, (dict, types.MappingProxyType)):
        return {k: _unfreeze_for_serialization(v) for k, v in val.items()}
    if isinstance(val, (list, tuple, set)):
        return [_unfreeze_for_serialization(v) for v in val]
    return val


SUPPORTED_RECOVERY_STATUSES: Set[str] = {
    "open",
    "in_progress",
    "recovered",
    "failed",
    "abandoned",
    "succeeded",
    "pending",
}


class HistoricalCase(BaseModel):
    """
    Immutable representation of a historical recovery experience returned by retrieval backends.

    Contains the essential historical evidence and relevance scoring required by
    downstream retrieval ranking and the deterministic decision engine.
    """

    model_config = ConfigDict(
        frozen=True,
        from_attributes=True,
        validate_default=True,
        arbitrary_types_allowed=True,
    )

    # Identifiers (No PII)
    payment_id: UUID
    customer_id: UUID
    external_payment_id: Optional[str] = None
    external_customer_id: Optional[str] = None

    # Payment details
    amount: float = Field(ge=0.0)
    currency: str = Field(default="INR")
    payment_method: str
    failure_reason: Optional[str] = None

    # Recovery actions & outcomes
    recovery_action: Optional[str] = None
    recovery_status: str
    amount_recovered: float = Field(default=0.0, ge=0.0)
    was_recovered: bool = False

    # Retrieval relevance scoring [0.0, 1.0]
    relevance_score: Optional[float] = Field(default=None, ge=0.0, le=1.0)

    # Temporal & retrieval metadata
    created_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    metadata: Mapping[str, Any] = Field(default_factory=dict)

    @field_validator("recovery_status", mode="before")
    @classmethod
    def _validate_recovery_status(cls, v: Any) -> str:
        if not isinstance(v, str) or not v.strip():
            raise ValueError(f"Recovery status must be a non-empty string, got: {v!r}")
        normalized = v.strip().lower()
        if normalized not in SUPPORTED_RECOVERY_STATUSES:
            raise ValueError(
                f"Invalid or unsupported recovery status '{v}'. "
                f"Must be one of {sorted(list(SUPPORTED_RECOVERY_STATUSES))}."
            )
        return normalized

    @field_validator("payment_method", mode="before")
    @classmethod
    def _validate_payment_method(cls, v: Any) -> str:
        if not isinstance(v, str) or not v.strip():
            raise ValueError(f"Payment method must be a non-empty string, got: {v!r}")
        return v.strip()

    @field_validator("metadata", mode="before")
    @classmethod
    def _normalize_none_metadata(cls, v: Any) -> Any:
        if v is None:
            return {}
        return v

    @field_validator("metadata", mode="after")
    @classmethod
    def _freeze_metadata(cls, v: Any) -> Mapping[str, Any]:
        if v is None:
            return types.MappingProxyType({})
        return _freeze_nested(v)

    @field_serializer("metadata")
    def _serialize_metadata(self, v: Mapping[str, Any], _info: Any) -> Dict[str, Any]:
        return _unfreeze_for_serialization(v)

    @model_validator(mode="after")
    def _validate_amounts_and_consistency(self) -> "HistoricalCase":
        if self.amount_recovered > self.amount + 1e-5:
            raise ValueError(
                f"Amount recovered ({self.amount_recovered}) cannot exceed payment amount ({self.amount})."
            )
        return self
