"""
Revora Historical Retrieval Document Representation and Canonical Text Construction.

Defines the immutable RetrievalDocument contract and deterministic conversion from
HistoricalCase into canonical normalized text ready for downstream embedding models.
"""

import types
from collections.abc import Mapping
from typing import Any
from uuid import UUID

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_serializer,
    field_validator,
)

from app.historical_retrieval import HistoricalCase


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


class RetrievalDocument(BaseModel):
    """
    Canonical, immutable document representation of a historical recovery experience.

    Serves as the deterministic text input contract for downstream embedding models
    and vector database indexing without coupling to any specific embedding provider.
    """

    model_config = ConfigDict(
        frozen=True,
        from_attributes=True,
        validate_default=True,
        arbitrary_types_allowed=True,
    )

    case_id: UUID
    text: str
    metadata: Mapping[str, Any] = Field(default_factory=dict)

    @field_validator("text", mode="before")
    @classmethod
    def _validate_text(cls, v: Any) -> str:
        if not isinstance(v, str) or not v.strip():
            raise ValueError("Document text must be a non-empty string.")
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
    def _serialize_metadata(self, v: Mapping[str, Any], _info: Any) -> dict[str, Any]:
        return _unfreeze_for_serialization(v)


def construct_canonical_case_text(case: HistoricalCase) -> str:
    """
    Construct a deterministic, normalized text representation of a HistoricalCase.

    Normalization standards:
    - Consistent field ordering
    - Predictable lowercasing and whitespace normalization
    - Deterministic missing value handling ('none')
    - Uniform floating point formatting (2 decimal places)
    - Lowercase boolean representation ('true' / 'false')
    - Strict exclusion of customer PII
    """
    failure_reason_norm = (
        case.failure_reason.strip().lower() if case.failure_reason else "none"
    )
    payment_method_norm = (
        case.payment_method.strip().lower() if case.payment_method else "unknown"
    )
    amount_norm = f"{case.amount:.2f}"
    currency_norm = case.currency.strip().upper() if case.currency else "INR"
    recovery_action_norm = (
        case.recovery_action.strip().lower() if case.recovery_action else "none"
    )
    recovery_status_norm = (
        case.recovery_status.strip().lower() if case.recovery_status else "unknown"
    )
    was_recovered_norm = "true" if case.was_recovered else "false"
    amount_recovered_norm = f"{case.amount_recovered:.2f}"

    lines = [
        f"failure_reason: {failure_reason_norm}",
        f"payment_method: {payment_method_norm}",
        f"amount: {amount_norm}",
        f"currency: {currency_norm}",
        f"recovery_action: {recovery_action_norm}",
        f"recovery_status: {recovery_status_norm}",
        f"was_recovered: {was_recovered_norm}",
        f"amount_recovered: {amount_recovered_norm}",
    ]

    return "\n".join(lines)


PII_KEY_TOKENS = ("name", "email", "phone", "address")


def historical_case_to_document(case: HistoricalCase) -> RetrievalDocument:
    """
    Convert an immutable HistoricalCase into a canonical RetrievalDocument.

    Extracts clean non-PII metadata and produces deterministic canonical text.
    """
    text = construct_canonical_case_text(case)

    # Build safe, non-PII retrieval metadata
    doc_metadata: dict[str, Any] = {
        "payment_id": str(case.payment_id),
        "customer_id": str(case.customer_id),
        "amount": case.amount,
        "currency": case.currency.strip().upper() if case.currency else "INR",
        "payment_method": case.payment_method.strip().lower()
        if case.payment_method
        else "unknown",
        "failure_reason": case.failure_reason.strip().lower()
        if case.failure_reason
        else None,
        "recovery_status": case.recovery_status.strip().lower()
        if case.recovery_status
        else "unknown",
        "recovery_action": case.recovery_action.strip().lower()
        if case.recovery_action
        else None,
        "amount_recovered": case.amount_recovered,
        "was_recovered": case.was_recovered,
    }

    if case.external_payment_id:
        doc_metadata["external_payment_id"] = case.external_payment_id
    if case.external_customer_id:
        doc_metadata["external_customer_id"] = case.external_customer_id
    if case.created_at:
        doc_metadata["created_at"] = case.created_at.isoformat()
    if case.completed_at:
        doc_metadata["completed_at"] = case.completed_at.isoformat()
    if case.relevance_score is not None:
        doc_metadata["relevance_score"] = case.relevance_score

    # Include existing case metadata if present (excluding any PII keys and protecting canonical keys)
    if case.metadata:
        unfrozen = _unfreeze_for_serialization(case.metadata)
        for k, v in unfrozen.items():
            key_norm = str(k).strip().lower()
            if any(token in key_norm for token in PII_KEY_TOKENS):
                continue
            if key_norm in doc_metadata:
                continue
            doc_metadata[key_norm] = v

    return RetrievalDocument(
        case_id=case.payment_id,
        text=text,
        metadata=doc_metadata,
    )
