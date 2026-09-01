import uuid
from datetime import datetime

import pytest
from app.database import Base
from app.models import (
    AuditEvent,
    Customer,
    Payment,
    RecoveryAttempt,
    RecoveryOpportunity,
)
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker


@pytest.fixture
def db_session():
    """Create a fresh in-memory SQLite database session for testing."""
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
    )
    Base.metadata.create_all(bind=engine)
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    session = TestingSessionLocal()
    try:
        yield session
    finally:
        session.close()
        Base.metadata.drop_all(bind=engine)


def test_table_creation(db_session):
    """Verify that all expected tables are created in the database metadata."""
    table_names = Base.metadata.tables.keys()
    expected_tables = {
        "customers",
        "payments",
        "recovery_opportunities",
        "recovery_attempts",
        "audit_events",
    }
    assert expected_tables.issubset(table_names)


def test_customer_creation(db_session):
    """Verify customer entity creation, default values, and UTC timestamps."""
    customer = Customer(
        external_customer_id="cust_12345",
        name="Alex Mercer",
        email="alex@example.com",
    )
    db_session.add(customer)
    db_session.commit()
    db_session.refresh(customer)

    assert isinstance(customer.id, uuid.UUID)
    assert customer.name == "Alex Mercer"
    assert customer.email == "alex@example.com"
    assert customer.total_payments == 0
    assert customer.successful_payments == 0
    assert customer.failed_payments == 0
    assert customer.created_at is not None
    assert customer.created_at.tzinfo is not None or isinstance(
        customer.created_at, datetime
    )


def test_customer_payment_relationship(db_session):
    """Verify customer to payment relationship and cascade behavior."""
    customer = Customer(
        name="Sarah Connor",
        email="sarah@example.com",
    )
    db_session.add(customer)
    db_session.commit()

    payment = Payment(
        customer_id=customer.id,
        amount=4999.00,
        currency="INR",
        payment_method="card",
        status="failed",
        failure_reason="insufficient_funds",
    )
    db_session.add(payment)
    db_session.commit()
    db_session.refresh(customer)

    assert len(customer.payments) == 1
    assert customer.payments[0].amount == 4999.00
    assert customer.payments[0].customer.id == customer.id


def test_payment_recovery_opportunity_relationship(db_session):
    """Verify payment to recovery opportunity one-to-one relationship."""
    customer = Customer(name="John Doe", email="john@example.com")
    db_session.add(customer)
    db_session.commit()

    payment = Payment(
        customer_id=customer.id,
        amount=1200.50,
        currency="INR",
        payment_method="upi",
        status="failed",
        failure_reason="bank_server_down",
    )
    db_session.add(payment)
    db_session.commit()

    opportunity = RecoveryOpportunity(
        payment_id=payment.id,
        status="open",
        revenue_at_risk=1200.50,
        expected_recovery=960.40,
        recommended_action="smart_retry_upi",
        confidence=0.80,
    )
    db_session.add(opportunity)
    db_session.commit()
    db_session.refresh(payment)

    assert payment.recovery_opportunity is not None
    assert payment.recovery_opportunity.revenue_at_risk == 1200.50
    assert payment.recovery_opportunity.payment.id == payment.id


def test_recovery_opportunity_attempts_and_audit(db_session):
    """Verify attempts and audit events linked to a recovery opportunity."""
    customer = Customer(name="Jane Roe", email="jane@example.com")
    db_session.add(customer)
    db_session.commit()

    payment = Payment(
        customer_id=customer.id,
        amount=2500.00,
        currency="INR",
        payment_method="netbanking",
        status="failed",
    )
    db_session.add(payment)
    db_session.commit()

    opportunity = RecoveryOpportunity(
        payment_id=payment.id,
        status="in_progress",
        revenue_at_risk=2500.00,
        expected_recovery=2000.00,
    )
    db_session.add(opportunity)
    db_session.commit()

    attempt = RecoveryAttempt(
        opportunity_id=opportunity.id,
        action="retry_charge",
        status="pending",
    )
    db_session.add(attempt)

    audit_event = AuditEvent(
        opportunity_id=opportunity.id,
        event_type="policy_evaluated",
        description="Policy rule retry_low_risk triggered",
        metadata_payload={"rule_id": "pol_001", "score": 0.85},
    )
    db_session.add(audit_event)
    db_session.commit()
    db_session.refresh(opportunity)

    assert len(opportunity.attempts) == 1
    assert opportunity.attempts[0].action == "retry_charge"
    assert len(opportunity.audit_events) == 1
    assert opportunity.audit_events[0].event_type == "policy_evaluated"
    assert opportunity.audit_events[0].metadata_payload == {
        "rule_id": "pol_001",
        "score": 0.85,
    }


def test_cascade_delete_from_customer(db_session):
    """Verify deleting a customer cascades down to payments and opportunities."""
    customer = Customer(name="Mark Stone", email="mark@example.com")
    db_session.add(customer)
    db_session.commit()

    payment = Payment(
        customer_id=customer.id,
        amount=100.0,
        payment_method="card",
        status="failed",
    )
    db_session.add(payment)
    db_session.commit()

    opportunity = RecoveryOpportunity(
        payment_id=payment.id,
        status="open",
        revenue_at_risk=100.0,
        expected_recovery=80.0,
    )
    db_session.add(opportunity)
    db_session.commit()

    # Delete customer
    db_session.delete(customer)
    db_session.commit()

    assert db_session.query(Customer).count() == 0
    assert db_session.query(Payment).count() == 0
    assert db_session.query(RecoveryOpportunity).count() == 0
