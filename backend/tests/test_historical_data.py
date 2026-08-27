import csv
import pytest
import tempfile
from pathlib import Path
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.database import Base
from app.models import (
    Customer,
    Payment,
    RecoveryOpportunity,
    RecoveryAttempt,
    AuditEvent,
)
from app.historical_data import (
    load_historical_data,
    validate_dataset_schema,
    EXPECTED_CSV_COLUMNS,
)


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


@pytest.fixture
def sample_csv_file(tmp_path):
    """Create a temporary valid CSV file with controlled test records."""
    csv_file = tmp_path / "test_recovery_data.csv"
    rows = [
        {
            "record_id": "rec_001",
            "customer_id": "cust_101",
            "payment_id": "pay_201",
            "customer_payment_count": 10,
            "customer_success_rate": 0.8,
            "customer_previous_failures": 2,
            "payment_amount": 1500.0,
            "currency": "INR",
            "payment_method": "upi",
            "failure_reason": "bank_timeout",
            "attempt_number": 1,
            "hours_since_failure": 0.5,
            "action_taken": "RETRY",
            "previous_action": "",
            "previous_attempt_count": 0,
            "recovered": "True",
            "amount_recovered": 1500.0,
            "recovery_time_hours": 1.2,
        },
        {
            "record_id": "rec_002",
            "customer_id": "cust_101",  # Same customer, different payment
            "payment_id": "pay_202",
            "customer_payment_count": 10,
            "customer_success_rate": 0.8,
            "customer_previous_failures": 2,
            "payment_amount": 3000.0,
            "currency": "INR",
            "payment_method": "card",
            "failure_reason": "insufficient_funds",
            "attempt_number": 1,
            "hours_since_failure": 1.0,
            "action_taken": "PAYMENT_LINK",
            "previous_action": "",
            "previous_attempt_count": 0,
            "recovered": "False",
            "amount_recovered": 0.0,
            "recovery_time_hours": 0.0,
        },
        {
            "record_id": "rec_003",
            "customer_id": "cust_101",
            "payment_id": "pay_202",  # Same payment pay_202, second attempt
            "customer_payment_count": 10,
            "customer_success_rate": 0.8,
            "customer_previous_failures": 2,
            "payment_amount": 3000.0,
            "currency": "INR",
            "payment_method": "card",
            "failure_reason": "insufficient_funds",
            "attempt_number": 2,
            "hours_since_failure": 8.0,
            "action_taken": "REMINDER",
            "previous_action": "PAYMENT_LINK",
            "previous_attempt_count": 1,
            "recovered": "True",
            "amount_recovered": 3000.0,
            "recovery_time_hours": 9.5,
        },
        {
            "record_id": "rec_004",
            "customer_id": "cust_102",  # New customer, failed outcome only
            "payment_id": "pay_203",
            "customer_payment_count": 4,
            "customer_success_rate": 0.5,
            "customer_previous_failures": 2,
            "payment_amount": 4500.0,
            "currency": "INR",
            "payment_method": "netbanking",
            "failure_reason": "technical_error",
            "attempt_number": 1,
            "hours_since_failure": 2.0,
            "action_taken": "STOP",
            "previous_action": "",
            "previous_attempt_count": 0,
            "recovered": "False",
            "amount_recovered": 0.0,
            "recovery_time_hours": 0.0,
        },
    ]

    with open(csv_file, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=EXPECTED_CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    return csv_file


def test_schema_validation_valid(sample_csv_file):
    """Verify that a valid CSV schema passes validation."""
    # Should not raise
    validate_dataset_schema(sample_csv_file)


def test_schema_validation_missing_columns(tmp_path):
    """Verify that a CSV missing required columns raises ValueError."""
    bad_csv = tmp_path / "bad.csv"
    with open(bad_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["record_id", "customer_id", "payment_id"])  # Missing many fields
        writer.writerow(["rec_1", "cust_1", "pay_1"])

    with pytest.raises(ValueError, match="Missing required columns"):
        validate_dataset_schema(bad_csv)


def test_schema_validation_nonexistent_file(tmp_path):
    """Verify that a nonexistent CSV path raises FileNotFoundError."""
    missing_path = tmp_path / "does_not_exist.csv"
    with pytest.raises(FileNotFoundError):
        validate_dataset_schema(missing_path)


def test_successful_ingestion(db_session, sample_csv_file):
    """Verify successful ingestion of records into database entities."""
    summary = load_historical_data(db_session, csv_path=sample_csv_file)

    assert summary["total_records_processed"] == 4
    assert summary["customers_created"] == 2
    assert summary["payments_created"] == 3
    assert summary["opportunities_created"] == 3
    assert summary["attempts_created"] == 4
    assert summary["audit_events_created"] == 4
    assert summary["skipped_duplicate_attempts"] == 0

    # Verify counts in DB
    assert db_session.query(Customer).count() == 2
    assert db_session.query(Payment).count() == 3
    assert db_session.query(RecoveryOpportunity).count() == 3
    assert db_session.query(RecoveryAttempt).count() == 4
    assert db_session.query(AuditEvent).count() == 4


def test_customer_and_payment_deduplication(db_session, sample_csv_file):
    """
    Verify that multiple attempts belonging to the same payment and customer
    correctly deduplicate and link to the same Customer and Payment entities.
    """
    load_historical_data(db_session, csv_path=sample_csv_file)

    # cust_101 has 2 payments (pay_201 with 1 attempt, pay_202 with 2 attempts)
    customer_101 = db_session.query(Customer).filter_by(external_customer_id="cust_101").one()
    assert customer_101.total_payments == 10
    assert customer_101.successful_payments == 8
    assert customer_101.failed_payments == 2
    assert len(customer_101.payments) == 2

    # pay_202 has 2 attempts on its recovery opportunity
    payment_202 = db_session.query(Payment).filter_by(external_payment_id="pay_202").one()
    assert payment_202.customer_id == customer_101.id
    assert payment_202.recovery_opportunity is not None
    assert len(payment_202.recovery_opportunity.attempts) == 2
    attempt_refs = {a.external_reference for a in payment_202.recovery_opportunity.attempts}
    assert attempt_refs == {"rec_002", "rec_003"}


def test_recovery_attempt_relationships(db_session, sample_csv_file):
    """Verify foreign keys and bidirectional relationships among all entities."""
    load_historical_data(db_session, csv_path=sample_csv_file)

    attempt_1 = db_session.query(RecoveryAttempt).filter_by(external_reference="rec_001").one()
    assert attempt_1.opportunity is not None
    assert attempt_1.opportunity.payment is not None
    assert attempt_1.opportunity.payment.customer is not None
    assert attempt_1.opportunity.payment.customer.external_customer_id == "cust_101"
    assert attempt_1.opportunity.payment.external_payment_id == "pay_201"


def test_recovered_vs_failed_outcomes(db_session, sample_csv_file):
    """
    Verify the domain rule:
    Only recovered=True produces a successful recovery outcome.
    """
    load_historical_data(db_session, csv_path=sample_csv_file)

    # rec_001: recovered=True
    attempt_1 = db_session.query(RecoveryAttempt).filter_by(external_reference="rec_001").one()
    assert attempt_1.status == "succeeded"
    assert attempt_1.amount_recovered == 1500.0
    assert attempt_1.opportunity.status == "recovered"
    assert attempt_1.opportunity.payment.status == "succeeded"

    # rec_004: recovered=False
    attempt_4 = db_session.query(RecoveryAttempt).filter_by(external_reference="rec_004").one()
    assert attempt_4.status == "failed"
    assert attempt_4.amount_recovered == 0.0
    assert attempt_4.opportunity.status == "failed"
    assert attempt_4.opportunity.payment.status == "failed"

    # rec_002 (failed attempt 1) followed by rec_003 (recovered attempt 2) on pay_202
    attempt_2 = db_session.query(RecoveryAttempt).filter_by(external_reference="rec_002").one()
    attempt_3 = db_session.query(RecoveryAttempt).filter_by(external_reference="rec_003").one()
    assert attempt_2.status == "failed"
    assert attempt_3.status == "succeeded"
    # Overall opportunity & payment reflect recovery
    assert attempt_2.opportunity.status == "recovered"
    assert attempt_2.opportunity.payment.status == "succeeded"


def test_repeated_ingestion_idempotency(db_session, sample_csv_file):
    """
    Verify that running load_historical_data twice is safe and does not create
    duplicate customers, payments, opportunities, attempts, or audit events.
    """
    first_run = load_historical_data(db_session, csv_path=sample_csv_file)
    assert first_run["attempts_created"] == 4
    assert first_run["skipped_duplicate_attempts"] == 0

    counts_after_first = {
        "customers": db_session.query(Customer).count(),
        "payments": db_session.query(Payment).count(),
        "opportunities": db_session.query(RecoveryOpportunity).count(),
        "attempts": db_session.query(RecoveryAttempt).count(),
        "audit_events": db_session.query(AuditEvent).count(),
    }

    # Second run with same data
    second_run = load_historical_data(db_session, csv_path=sample_csv_file)
    assert second_run["customers_created"] == 0
    assert second_run["payments_created"] == 0
    assert second_run["opportunities_created"] == 0
    assert second_run["attempts_created"] == 0
    assert second_run["audit_events_created"] == 0
    assert second_run["skipped_duplicate_attempts"] == 4

    counts_after_second = {
        "customers": db_session.query(Customer).count(),
        "payments": db_session.query(Payment).count(),
        "opportunities": db_session.query(RecoveryOpportunity).count(),
        "attempts": db_session.query(RecoveryAttempt).count(),
        "audit_events": db_session.query(AuditEvent).count(),
    }

    assert counts_after_first == counts_after_second


def test_ingest_actual_historical_dataset(db_session):
    """
    Integration test: Ingest the actual 5,000-record synthetic historical dataset
    located at data/historical_recovery_data.csv.
    """
    csv_path = Path(__file__).resolve().parent.parent.parent / "data" / "historical_recovery_data.csv"
    if not csv_path.exists():
        pytest.skip("Historical dataset CSV not found at expected repo path.")

    summary = load_historical_data(db_session, csv_path=csv_path)

    assert summary["total_records_processed"] == 5000
    assert summary["attempts_created"] == 5000
    assert summary["customers_created"] > 0
    assert summary["payments_created"] > 0
    assert summary["opportunities_created"] == summary["payments_created"]
    assert summary["audit_events_created"] == 5000

    # Verify that recovered amounts match
    recovered_attempts = (
        db_session.query(RecoveryAttempt)
        .filter(RecoveryAttempt.status == "succeeded")
        .count()
    )
    assert recovered_attempts > 0
