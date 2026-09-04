import uuid
from datetime import datetime, timezone
from typing import Any, Optional

from sqlalchemy import (
    JSON,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    Uuid,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base


def utc_now() -> datetime:
    """Return current UTC timestamp with timezone awareness."""
    return datetime.now(timezone.utc)


class Customer(Base):
    """
    Customer entity representing end-users whose payments are tracked.
    """

    __tablename__ = "customers"

    id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    external_customer_id: Mapped[str | None] = mapped_column(
        String(255), unique=True, nullable=True, index=True
    )
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    email: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    total_payments: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    successful_payments: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    failed_payments: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, nullable=False
    )

    # Relationships
    payments: Mapped[list["Payment"]] = relationship(
        "Payment", back_populates="customer", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return f"<Customer id={self.id} email='{self.email}'>"


class Payment(Base):
    """
    Payment transaction record, storing payment method, status, and failure details.
    """

    __tablename__ = "payments"

    id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    external_payment_id: Mapped[str | None] = mapped_column(
        String(255), unique=True, nullable=True, index=True
    )
    customer_id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey("customers.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    amount: Mapped[float] = mapped_column(Float, nullable=False)
    currency: Mapped[str] = mapped_column(String(10), default="INR", nullable=False)
    payment_method: Mapped[str] = mapped_column(String(50), nullable=False)
    status: Mapped[str] = mapped_column(
        String(50), nullable=False, index=True
    )  # e.g., 'failed', 'succeeded', 'pending'
    failure_reason: Mapped[str | None] = mapped_column(String(500), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, onupdate=utc_now, nullable=False
    )

    # Relationships
    customer: Mapped["Customer"] = relationship("Customer", back_populates="payments")
    recovery_opportunity: Mapped[Optional["RecoveryOpportunity"]] = relationship(
        "RecoveryOpportunity",
        back_populates="payment",
        uselist=False,
        cascade="all, delete-orphan",
    )

    def __repr__(self) -> str:
        return f"<Payment id={self.id} amount={self.amount} status='{self.status}'>"


class RecoveryOpportunity(Base):
    """
    Recovery opportunity identified when a payment fails, estimating revenue at risk
    and driving policy-bounded recovery actions.
    """

    __tablename__ = "recovery_opportunities"

    id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    payment_id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey("payments.id", ondelete="CASCADE"),
        unique=True,
        nullable=False,
        index=True,
    )
    status: Mapped[str] = mapped_column(
        String(50), nullable=False, index=True
    )  # e.g., 'open', 'in_progress', 'recovered', 'failed', 'abandoned'
    revenue_at_risk: Mapped[float] = mapped_column(Float, nullable=False)
    expected_recovery: Mapped[float] = mapped_column(Float, nullable=False)
    recommended_action: Mapped[str | None] = mapped_column(String(100), nullable=True)
    confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, onupdate=utc_now, nullable=False
    )

    # Relationships
    payment: Mapped["Payment"] = relationship(
        "Payment", back_populates="recovery_opportunity"
    )
    attempts: Mapped[list["RecoveryAttempt"]] = relationship(
        "RecoveryAttempt", back_populates="opportunity", cascade="all, delete-orphan"
    )
    audit_events: Mapped[list["AuditEvent"]] = relationship(
        "AuditEvent", back_populates="opportunity", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return f"<RecoveryOpportunity id={self.id} status='{self.status}' risk={self.revenue_at_risk}>"


class RecoveryAttempt(Base):
    """
    An execution attempt to recover a failed payment (e.g., smart retry, customer prompt).
    """

    __tablename__ = "recovery_attempts"

    id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    opportunity_id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey("recovery_opportunities.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    action: Mapped[str] = mapped_column(String(100), nullable=False)
    status: Mapped[str] = mapped_column(
        String(50), nullable=False, index=True
    )  # e.g., 'pending', 'succeeded', 'failed'
    amount_recovered: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    external_reference: Mapped[str | None] = mapped_column(
        String(255), unique=True, nullable=True, index=True
    )
    idempotency_key: Mapped[str | None] = mapped_column(
        String(255), unique=True, nullable=True, index=True
    )
    error_code: Mapped[str | None] = mapped_column(String(100), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, nullable=False
    )
    completed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    # Relationships
    opportunity: Mapped["RecoveryOpportunity"] = relationship(
        "RecoveryOpportunity", back_populates="attempts"
    )

    def __repr__(self) -> str:
        return f"<RecoveryAttempt id={self.id} action='{self.action}' status='{self.status}'>"


class AuditEvent(Base):
    """
    Immutable audit log for decision tracking and explainability across recovery actions.
    """

    __tablename__ = "audit_events"

    id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    opportunity_id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey("recovery_opportunities.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    event_type: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    description: Mapped[str] = mapped_column(Text, nullable=False)
    # Map to DB column 'metadata', using attribute name 'metadata_payload' to avoid conflict with Base.metadata
    metadata_payload: Mapped[dict[str, Any]] = mapped_column(
        "metadata", JSON, default=dict, nullable=False
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, nullable=False, index=True
    )

    # Relationships
    opportunity: Mapped["RecoveryOpportunity"] = relationship(
        "RecoveryOpportunity", back_populates="audit_events"
    )

    def __repr__(self) -> str:
        return f"<AuditEvent id={self.id} event_type='{self.event_type}'>"
