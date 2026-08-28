"""
Revora Context Schemas and Domain Exceptions.

Defines the typed, immutable schemas representing the deterministic context
retrieved at decision time for failed payment recovery, along with domain-specific
retrieval exceptions.
"""

from datetime import datetime, timezone
from typing import List, Optional
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


def utc_now() -> datetime:
    """Return current UTC timestamp with timezone awareness."""
    return datetime.now(timezone.utc)


# ============================================================================
# Domain Exceptions
# ============================================================================


class ContextRetrievalError(Exception):
    """Base exception for all context retrieval errors."""

    pass


class CustomerNotFoundError(ContextRetrievalError):
    """Raised when a requested customer is not found in the database."""

    def __init__(self, identifier: str | UUID):
        self.identifier = str(identifier)
        super().__init__(f"Customer with identifier '{self.identifier}' was not found.")


class PaymentNotFoundError(ContextRetrievalError):
    """Raised when a requested payment is not found in the database."""

    def __init__(self, identifier: str | UUID):
        self.identifier = str(identifier)
        super().__init__(f"Payment with identifier '{self.identifier}' was not found.")


class PaymentCustomerMismatchError(ContextRetrievalError):
    """Raised when a payment exists but does not belong to the specified customer."""

    def __init__(self, payment_id: str | UUID, customer_id: str | UUID, actual_customer_id: str | UUID):
        self.payment_id = str(payment_id)
        self.customer_id = str(customer_id)
        self.actual_customer_id = str(actual_customer_id)
        super().__init__(
            f"Payment '{self.payment_id}' belongs to customer '{self.actual_customer_id}', "
            f"not expected customer '{self.customer_id}'."
        )


class RecoveryOpportunityNotFoundError(ContextRetrievalError):
    """Raised when a recovery opportunity associated with a payment is not found."""

    def __init__(self, identifier: str | UUID):
        self.identifier = str(identifier)
        super().__init__(
            f"Recovery opportunity for identifier '{self.identifier}' was not found."
        )


# ============================================================================
# Immutable Context Schemas (Read-Oriented Data Contracts)
# ============================================================================


class CustomerContext(BaseModel):
    """
    Deterministic customer profile and baseline transaction stats.
    """

    model_config = ConfigDict(frozen=True, from_attributes=True)

    customer_id: UUID
    external_customer_id: Optional[str] = None
    name: Optional[str] = None
    email: Optional[str] = None
    total_payments: int = Field(default=0, ge=0)
    successful_payments: int = Field(default=0, ge=0)
    failed_payments: int = Field(default=0, ge=0)
    historical_success_rate: float = Field(default=0.0, ge=0.0, le=1.0)
    created_at: Optional[datetime] = None


class PaymentContext(BaseModel):
    """
    Context for the current failed payment triggering the recovery evaluation.
    """

    model_config = ConfigDict(frozen=True, from_attributes=True)

    payment_id: UUID
    external_payment_id: Optional[str] = None
    amount: float = Field(ge=0.0)
    currency: str = Field(default="INR")
    payment_method: str
    status: str
    failure_reason: Optional[str] = None
    created_at: Optional[datetime] = None


class RecoveryOpportunityContext(BaseModel):
    """
    Active recovery opportunity tracking revenue at risk and recovery lifecycle state.
    """

    model_config = ConfigDict(frozen=True, from_attributes=True)

    opportunity_id: UUID
    status: str
    revenue_at_risk: float = Field(ge=0.0)
    expected_recovery: float = Field(default=0.0, ge=0.0)
    recommended_action: Optional[str] = None
    confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    created_at: Optional[datetime] = None


class RecoveryAttemptContext(BaseModel):
    """
    Execution details and outcome of a specific recovery attempt.
    """

    model_config = ConfigDict(frozen=True, from_attributes=True)

    attempt_id: Optional[UUID] = None
    action: str
    status: str
    amount_recovered: float = Field(default=0.0, ge=0.0)
    error_code: Optional[str] = None
    external_reference: Optional[str] = None
    created_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None


class HistoricalPaymentContext(BaseModel):
    """
    Summary of a prior payment belonging to the same customer.
    """

    model_config = ConfigDict(frozen=True, from_attributes=True)

    payment_id: UUID
    external_payment_id: Optional[str] = None
    amount: float = Field(ge=0.0)
    currency: str = Field(default="INR")
    payment_method: str
    status: str
    failure_reason: Optional[str] = None
    created_at: Optional[datetime] = None
    was_recovered: bool = False
    recovery_action: Optional[str] = None
    recovery_attempts_count: int = Field(default=0, ge=0)


class CustomerRecoveryStatsContext(BaseModel):
    """
    Deterministic historical recovery aggregates calculated from database records.
    """

    model_config = ConfigDict(frozen=True, from_attributes=True)

    total_recovery_opportunities: int = Field(default=0, ge=0)
    recovered_opportunities: int = Field(default=0, ge=0)
    failed_opportunities: int = Field(default=0, ge=0)
    recovery_rate: float = Field(default=0.0, ge=0.0, le=1.0)
    previously_successful_actions: List[str] = Field(default_factory=list)
    previously_failed_actions: List[str] = Field(default_factory=list)
    total_amount_recovered: float = Field(default=0.0, ge=0.0)


class CustomerRecoveryContext(BaseModel):
    """
    Top-level immutable context contract containing all deterministic information
    required by the decision engine for failed payment recovery.
    """

    model_config = ConfigDict(frozen=True, from_attributes=True)

    customer: CustomerContext
    current_payment: Optional[PaymentContext] = None
    current_opportunity: Optional[RecoveryOpportunityContext] = None
    current_payment_attempts: List[RecoveryAttemptContext] = Field(default_factory=list)
    historical_payments: List[HistoricalPaymentContext] = Field(default_factory=list)
    recovery_statistics: CustomerRecoveryStatsContext = Field(
        default_factory=CustomerRecoveryStatsContext
    )
    retrieved_at: datetime = Field(default_factory=utc_now)
