"""
Revora Historical Recovery Case Representation and Mappers.

Defines the clean, immutable, typed domain contract representing a single
historical payment recovery experience (HistoricalRecoveryCase and HistoricalAttempt),
along with deterministic, side-effect-free mappers from domain and context contracts.
"""

from datetime import datetime, timezone
import types
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple, Union
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_serializer, field_validator, model_validator

from app.context import (
    CustomerContext,
    CustomerRecoveryContext,
    HistoricalPaymentContext,
    PaymentContext,
    RecoveryAttemptContext,
    RecoveryOpportunityContext,
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


# Supported domain status sets for recovery outcomes and attempt executions
SUPPORTED_RECOVERY_STATUSES: Set[str] = {
    "open",
    "in_progress",
    "recovered",
    "failed",
    "abandoned",
    "succeeded",
    "pending",
}

SUPPORTED_ATTEMPT_STATUSES: Set[str] = {
    "pending",
    "succeeded",
    "failed",
}


class HistoricalAttempt(BaseModel):
    """
    Immutable representation of an individual recovery attempt within a historical case.
    """

    model_config = ConfigDict(
        frozen=True,
        from_attributes=True,
        validate_default=True,
        arbitrary_types_allowed=False,
    )

    attempt_id: Optional[UUID] = None
    action: str
    status: str
    amount_recovered: float = Field(default=0.0, ge=0.0)
    error_code: Optional[str] = None
    external_reference: Optional[str] = None
    created_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    @field_validator("status", mode="before")
    @classmethod
    def _validate_attempt_status(cls, v: Any) -> str:
        if not isinstance(v, str) or not v.strip():
            raise ValueError(f"Attempt status must be a non-empty string, got: {v!r}")
        normalized = v.strip().lower()
        if normalized not in SUPPORTED_ATTEMPT_STATUSES:
            raise ValueError(
                f"Invalid or unsupported attempt status '{v}'. "
                f"Must be one of {sorted(list(SUPPORTED_ATTEMPT_STATUSES))}."
            )
        return normalized

    @field_validator("action", mode="before")
    @classmethod
    def _validate_action(cls, v: Any) -> str:
        if not isinstance(v, str) or not v.strip():
            raise ValueError(f"Attempt action must be a non-empty string, got: {v!r}")
        return v.strip()


class HistoricalRecoveryCase(BaseModel):
    """
    Canonical, immutable, typed representation of a single historical payment recovery experience.

    Serves as the input contract for deterministic historical retrieval and downstream
    vector embedding retrieval without coupling to any database or vector store.
    """

    model_config = ConfigDict(
        frozen=True,
        from_attributes=True,
        validate_default=True,
        arbitrary_types_allowed=True,
    )

    # Identifiers and traceability (No PII)
    payment_id: UUID
    external_payment_id: Optional[str] = None
    opportunity_id: Optional[UUID] = None
    customer_id: Optional[UUID] = None
    external_customer_id: Optional[str] = None

    # Payment details
    amount: float = Field(ge=0.0)
    currency: str = Field(default="INR")
    payment_method: str
    failure_reason: Optional[str] = None

    # Recovery outcome and summary
    recovery_status: str
    amount_recovered: float = Field(default=0.0, ge=0.0)
    successful_action: Optional[str] = None

    # Chronologically ordered execution attempts
    attempts: Tuple[HistoricalAttempt, ...] = Field(default_factory=tuple)

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

    @field_validator("attempts", mode="before")
    @classmethod
    def _normalize_attempts(cls, v: Any) -> Tuple[Any, ...]:
        if v is None:
            return ()
        if isinstance(v, (list, tuple, set)):
            # Convert incoming collection to a defensive tuple copy
            converted = []
            for item in v:
                if isinstance(item, HistoricalAttempt):
                    converted.append(item)
                elif isinstance(item, RecoveryAttemptContext):
                    converted.append(
                        HistoricalAttempt(
                            attempt_id=item.attempt_id,
                            action=item.action,
                            status=item.status,
                            amount_recovered=item.amount_recovered,
                            error_code=item.error_code,
                            external_reference=item.external_reference,
                            created_at=item.created_at,
                            completed_at=item.completed_at,
                        )
                    )
                elif isinstance(item, dict):
                    converted.append(HistoricalAttempt(**item))
                else:
                    converted.append(item)
            return tuple(converted)
        return (v,)

    @field_validator("attempts", mode="after")
    @classmethod
    def _sort_attempts(
        cls, attempts: Tuple[HistoricalAttempt, ...]
    ) -> Tuple[HistoricalAttempt, ...]:
        if not attempts:
            return ()
        # Deterministically sort attempts in chronological order
        sorted_list = sorted(
            attempts,
            key=lambda a: (
                a.created_at.timestamp() if a.created_at else 0.0,
                str(a.attempt_id or ""),
            ),
        )
        return tuple(sorted_list)

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
    def _validate_amounts_and_consistency(self) -> "HistoricalRecoveryCase":
        # Ensure amount_recovered does not exceed payment amount
        if self.amount_recovered > self.amount + 1e-5:
            raise ValueError(
                f"Amount recovered ({self.amount_recovered}) cannot exceed payment amount ({self.amount})."
            )
        return self


# ============================================================================
# Deterministic Mappers
# ============================================================================


def map_context_to_historical_case(
    payment: PaymentContext,
    opportunity: Optional[RecoveryOpportunityContext] = None,
    attempts: Optional[Sequence[Union[RecoveryAttemptContext, HistoricalAttempt, Dict[str, Any]]]] = None,
    customer: Optional[CustomerContext] = None,
    customer_id: Optional[UUID] = None,
    external_customer_id: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> HistoricalRecoveryCase:
    """
    Deterministically map PaymentContext and related domain contexts into an immutable HistoricalRecoveryCase.

    Omits all customer PII (e.g. name, email) preserving only opaque identifiers and recovery signals.
    """
    # Resolve identifiers
    resolved_customer_id = customer_id or (customer.customer_id if customer else None)
    resolved_external_cust_id = external_customer_id or (
        customer.external_customer_id if customer else None
    )
    opportunity_id = opportunity.opportunity_id if opportunity else None

    # Determine recovery status
    if opportunity is not None:
        recovery_status = opportunity.status
    elif payment.status == "succeeded":
        recovery_status = "recovered"
    elif payment.status == "failed":
        recovery_status = "failed"
    else:
        recovery_status = payment.status

    # Normalize and sort attempts
    raw_attempts = list(attempts) if attempts else []
    converted_attempts: List[HistoricalAttempt] = []
    for att in raw_attempts:
        if isinstance(att, HistoricalAttempt):
            converted_attempts.append(att)
        elif isinstance(att, RecoveryAttemptContext):
            converted_attempts.append(
                HistoricalAttempt(
                    attempt_id=att.attempt_id,
                    action=att.action,
                    status=att.status,
                    amount_recovered=att.amount_recovered,
                    error_code=att.error_code,
                    external_reference=att.external_reference,
                    created_at=att.created_at,
                    completed_at=att.completed_at,
                )
            )
        elif isinstance(att, dict):
            converted_attempts.append(HistoricalAttempt(**att))

    # Calculate recovered amount and successful action
    amount_recovered = 0.0
    successful_action: Optional[str] = None
    latest_completed_at: Optional[datetime] = None

    for att in converted_attempts:
        if att.status == "succeeded":
            amount_recovered += att.amount_recovered
            if successful_action is None:
                successful_action = att.action
        if att.completed_at and (
            latest_completed_at is None or att.completed_at > latest_completed_at
        ):
            latest_completed_at = att.completed_at

    # If no attempts provided or attempts didn't yield amount, check opportunity
    if amount_recovered == 0.0 and opportunity is not None:
        if opportunity.status == "recovered":
            amount_recovered = (
                opportunity.expected_recovery
                if opportunity.expected_recovery > 0.0
                else payment.amount
            )
            if successful_action is None:
                successful_action = opportunity.recommended_action
    elif amount_recovered == 0.0 and payment.status == "succeeded":
        amount_recovered = payment.amount

    # Ensure amount_recovered does not exceed payment amount
    amount_recovered = min(amount_recovered, payment.amount)

    case_metadata: Dict[str, Any] = {}
    if metadata:
        case_metadata.update(metadata)
    if opportunity is not None:
        if opportunity.confidence is not None:
            case_metadata["confidence"] = opportunity.confidence
        if opportunity.recommended_action:
            case_metadata["opportunity_recommended_action"] = opportunity.recommended_action

    return HistoricalRecoveryCase(
        payment_id=payment.payment_id,
        external_payment_id=payment.external_payment_id,
        opportunity_id=opportunity_id,
        customer_id=resolved_customer_id,
        external_customer_id=resolved_external_cust_id,
        amount=payment.amount,
        currency=payment.currency,
        payment_method=payment.payment_method,
        failure_reason=payment.failure_reason,
        recovery_status=recovery_status,
        amount_recovered=amount_recovered,
        successful_action=successful_action,
        attempts=tuple(converted_attempts),
        created_at=payment.created_at,
        completed_at=latest_completed_at,
        metadata=case_metadata,
    )


def map_historical_payment_to_case(
    historical_payment: HistoricalPaymentContext,
    customer_id: Optional[UUID] = None,
    external_customer_id: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> HistoricalRecoveryCase:
    """
    Deterministically map a HistoricalPaymentContext into a HistoricalRecoveryCase.
    """
    recovery_status = (
        "recovered"
        if historical_payment.was_recovered
        else ("failed" if historical_payment.status == "failed" else historical_payment.status)
    )
    amount_recovered = (
        historical_payment.amount if historical_payment.was_recovered else 0.0
    )
    successful_action = (
        historical_payment.recovery_action if historical_payment.was_recovered else None
    )

    case_metadata: Dict[str, Any] = {
        "recovery_attempts_count": historical_payment.recovery_attempts_count,
        "was_recovered": historical_payment.was_recovered,
    }
    if metadata:
        case_metadata.update(metadata)

    return HistoricalRecoveryCase(
        payment_id=historical_payment.payment_id,
        external_payment_id=historical_payment.external_payment_id,
        customer_id=customer_id,
        external_customer_id=external_customer_id,
        amount=historical_payment.amount,
        currency=historical_payment.currency,
        payment_method=historical_payment.payment_method,
        failure_reason=historical_payment.failure_reason,
        recovery_status=recovery_status,
        amount_recovered=amount_recovered,
        successful_action=successful_action,
        attempts=(),
        created_at=historical_payment.created_at,
        completed_at=None,
        metadata=case_metadata,
    )


def map_customer_recovery_context_to_cases(
    context: CustomerRecoveryContext,
) -> List[HistoricalRecoveryCase]:
    """
    Extract all historical recovery cases from a CustomerRecoveryContext.

    Includes the current payment (if resolved) and all prior historical payments.
    """
    cases: List[HistoricalRecoveryCase] = []
    customer_id = context.customer.customer_id
    external_customer_id = context.customer.external_customer_id

    # 1. Map current payment experience if present
    if context.current_payment is not None:
        current_case = map_context_to_historical_case(
            payment=context.current_payment,
            opportunity=context.current_opportunity,
            attempts=context.current_payment_attempts,
            customer_id=customer_id,
            external_customer_id=external_customer_id,
        )
        cases.append(current_case)

    # 2. Map historical payments
    for hist_payment in context.historical_payments:
        cases.append(
            map_historical_payment_to_case(
                historical_payment=hist_payment,
                customer_id=customer_id,
                external_customer_id=external_customer_id,
            )
        )

    return cases
