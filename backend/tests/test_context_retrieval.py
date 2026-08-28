"""
Comprehensive Test Suite for Deterministic Context Retrieval.

Tests:
1. Existing customer + existing payment retrieval
2. Customer with multiple historical payments (current payment excluded)
3. Customer with multiple recovery attempts (deterministic chronological ordering)
4. New/cold-start customer with zero history (no division by zero)
5. Nonexistent customer raises CustomerNotFoundError
6. Nonexistent payment raises PaymentNotFoundError
7. Payment belonging to a different customer raises PaymentCustomerMismatchError (tenant isolation)
8. Missing recovery opportunity raises RecoveryOpportunityNotFoundError
9. Strict customer data isolation (no cross-customer leakage)
10. Dual identifier lookup (internal UUID and external string ID)
11. Recovery rate and statistics calculations
12. Deterministic ordering of historical records
"""

from datetime import datetime, timezone, timedelta
import uuid
import pytest
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
from app.context import (
    CustomerNotFoundError,
    PaymentNotFoundError,
    PaymentCustomerMismatchError,
    RecoveryOpportunityNotFoundError,
)
from app.context_retrieval import get_customer_context


@pytest.fixture
def db_session():
    """Create an isolated in-memory SQLite database session for each test."""
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


def test_existing_customer_and_payment_retrieval(db_session):
    """Verify basic retrieval for an existing customer and failed payment."""
    customer = Customer(
        external_customer_id="cust_101",
        name="Alice Walker",
        email="alice@example.com",
        total_payments=5,
        successful_payments=4,
        failed_payments=1,
    )
    db_session.add(customer)
    db_session.flush()

    payment = Payment(
        external_payment_id="pay_101",
        customer_id=customer.id,
        amount=1999.00,
        currency="INR",
        payment_method="card",
        status="failed",
        failure_reason="insufficient_funds",
    )
    db_session.add(payment)
    db_session.flush()

    opportunity = RecoveryOpportunity(
        payment_id=payment.id,
        status="open",
        revenue_at_risk=1999.00,
        expected_recovery=1500.00,
        recommended_action="smart_retry_card",
        confidence=0.8,
    )
    db_session.add(opportunity)
    db_session.commit()

    # Retrieve context via UUIDs
    ctx = get_customer_context(db_session, customer.id, payment.id)

    assert ctx.customer.customer_id == customer.id
    assert ctx.customer.name == "Alice Walker"
    assert ctx.customer.total_payments == 5
    assert ctx.customer.successful_payments == 4
    assert ctx.customer.failed_payments == 1
    assert ctx.customer.historical_success_rate == 0.8

    assert ctx.current_payment.payment_id == payment.id
    assert ctx.current_payment.amount == 1999.00
    assert ctx.current_payment.status == "failed"
    assert ctx.current_payment.failure_reason == "insufficient_funds"

    assert ctx.current_opportunity.opportunity_id == opportunity.id
    assert ctx.current_opportunity.revenue_at_risk == 1999.00
    assert ctx.current_opportunity.expected_recovery == 1500.00
    assert ctx.current_payment_attempts == []
    assert ctx.historical_payments == []


def test_dual_identifier_lookup(db_session):
    """Verify context retrieval works using both internal UUIDs and external string IDs."""
    customer = Customer(
        external_customer_id="cust_ext_alpha",
        name="Bob Stone",
        email="bob@example.com",
        total_payments=1,
        successful_payments=0,
        failed_payments=1,
    )
    db_session.add(customer)
    db_session.flush()

    payment = Payment(
        external_payment_id="pay_ext_beta",
        customer_id=customer.id,
        amount=500.0,
        currency="INR",
        payment_method="upi",
        status="failed",
        failure_reason="timeout",
    )
    db_session.add(payment)
    db_session.flush()

    opportunity = RecoveryOpportunity(
        payment_id=payment.id,
        status="open",
        revenue_at_risk=500.0,
        expected_recovery=400.0,
    )
    db_session.add(opportunity)
    db_session.commit()

    # 1. Query by external string IDs
    ctx_by_strings = get_customer_context(db_session, "cust_ext_alpha", "pay_ext_beta")
    assert ctx_by_strings.customer.customer_id == customer.id
    assert ctx_by_strings.current_payment.payment_id == payment.id

    # 2. Query by UUIDs
    ctx_by_uuids = get_customer_context(db_session, customer.id, payment.id)
    assert ctx_by_uuids.customer.external_customer_id == "cust_ext_alpha"
    assert ctx_by_uuids.current_payment.external_payment_id == "pay_ext_beta"


def test_customer_with_multiple_historical_payments_and_exclusion(db_session):
    """Verify historical payments belong to the customer, are chronologically sorted, and exclude current payment."""
    customer = Customer(
        external_customer_id="cust_hist_1",
        name="Charlie Brown",
        email="charlie@example.com",
        total_payments=4,
        successful_payments=3,
        failed_payments=1,
    )
    db_session.add(customer)
    db_session.flush()

    base_time = datetime(2026, 1, 1, 10, 0, tzinfo=timezone.utc)

    # Past payment 1 (recovered)
    p1 = Payment(
        external_payment_id="pay_past_1",
        customer_id=customer.id,
        amount=100.0,
        currency="INR",
        payment_method="card",
        status="succeeded",
        created_at=base_time,
    )
    db_session.add(p1)
    db_session.flush()

    opp1 = RecoveryOpportunity(
        payment_id=p1.id,
        status="recovered",
        revenue_at_risk=100.0,
        expected_recovery=100.0,
        recommended_action="smart_retry_card",
        created_at=base_time,
    )
    db_session.add(opp1)
    db_session.flush()

    att1 = RecoveryAttempt(
        opportunity_id=opp1.id,
        action="smart_retry_card",
        status="succeeded",
        amount_recovered=100.0,
        created_at=base_time + timedelta(minutes=5),
    )
    db_session.add(att1)

    # Past payment 2 (failed)
    p2 = Payment(
        external_payment_id="pay_past_2",
        customer_id=customer.id,
        amount=200.0,
        currency="INR",
        payment_method="upi",
        status="failed",
        failure_reason="user_cancelled",
        created_at=base_time + timedelta(days=1),
    )
    db_session.add(p2)
    db_session.flush()

    opp2 = RecoveryOpportunity(
        payment_id=p2.id,
        status="failed",
        revenue_at_risk=200.0,
        expected_recovery=0.0,
        recommended_action="customer_prompt_upi",
        created_at=base_time + timedelta(days=1),
    )
    db_session.add(opp2)
    db_session.flush()

    att2 = RecoveryAttempt(
        opportunity_id=opp2.id,
        action="customer_prompt_upi",
        status="failed",
        amount_recovered=0.0,
        created_at=base_time + timedelta(days=1, minutes=10),
    )
    db_session.add(att2)

    # Current failed payment
    current_p = Payment(
        external_payment_id="pay_current",
        customer_id=customer.id,
        amount=300.0,
        currency="INR",
        payment_method="netbanking",
        status="failed",
        failure_reason="bank_unavailable",
        created_at=base_time + timedelta(days=2),
    )
    db_session.add(current_p)
    db_session.flush()

    current_opp = RecoveryOpportunity(
        payment_id=current_p.id,
        status="open",
        revenue_at_risk=300.0,
        expected_recovery=240.0,
        created_at=base_time + timedelta(days=2),
    )
    db_session.add(current_opp)
    db_session.commit()

    ctx = get_customer_context(db_session, customer.id, current_p.id)

    # Current payment must NOT be in historical payments
    hist_payment_ids = [hp.payment_id for hp in ctx.historical_payments]
    assert current_p.id not in hist_payment_ids
    assert len(ctx.historical_payments) == 2

    # Deterministic chronological order: newest historical payment first (p2, then p1)
    assert ctx.historical_payments[0].payment_id == p2.id
    assert ctx.historical_payments[0].amount == 200.0
    assert ctx.historical_payments[0].was_recovered is False

    assert ctx.historical_payments[1].payment_id == p1.id
    assert ctx.historical_payments[1].amount == 100.0
    assert ctx.historical_payments[1].was_recovered is True
    assert ctx.historical_payments[1].recovery_action == "smart_retry_card"

    # Aggregates
    assert ctx.recovery_statistics.total_recovery_opportunities == 2
    assert ctx.recovery_statistics.recovered_opportunities == 1
    assert ctx.recovery_statistics.failed_opportunities == 1
    assert ctx.recovery_statistics.recovery_rate == 0.5
    assert ctx.recovery_statistics.total_amount_recovered == 100.0
    assert ctx.recovery_statistics.previously_successful_actions == ["smart_retry_card"]
    assert ctx.recovery_statistics.previously_failed_actions == ["customer_prompt_upi"]


def test_multiple_recovery_attempts_chronological_ordering(db_session):
    """Verify attempts for the current payment are retrieved in deterministic chronological order."""
    customer = Customer(
        name="Diana Prince",
        email="diana@example.com",
        total_payments=1,
        successful_payments=0,
        failed_payments=1,
    )
    db_session.add(customer)
    db_session.flush()

    payment = Payment(
        customer_id=customer.id,
        amount=1500.0,
        currency="INR",
        payment_method="card",
        status="failed",
    )
    db_session.add(payment)
    db_session.flush()

    opportunity = RecoveryOpportunity(
        payment_id=payment.id,
        status="in_progress",
        revenue_at_risk=1500.0,
        expected_recovery=1200.0,
    )
    db_session.add(opportunity)
    db_session.flush()

    now = datetime.now(timezone.utc)
    att1 = RecoveryAttempt(
        opportunity_id=opportunity.id,
        action="smart_retry_1",
        status="failed",
        error_code="insufficient_funds",
        created_at=now - timedelta(hours=2),
    )
    att2 = RecoveryAttempt(
        opportunity_id=opportunity.id,
        action="smart_retry_2",
        status="failed",
        error_code="temporary_hold",
        created_at=now - timedelta(hours=1),
    )
    att3 = RecoveryAttempt(
        opportunity_id=opportunity.id,
        action="customer_prompt",
        status="pending",
        created_at=now,
    )
    db_session.add_all([att2, att1, att3])  # added out of order intentionally
    db_session.commit()

    ctx = get_customer_context(db_session, customer.id, payment.id)

    assert len(ctx.current_payment_attempts) == 3
    assert ctx.current_payment_attempts[0].action == "smart_retry_1"
    assert ctx.current_payment_attempts[1].action == "smart_retry_2"
    assert ctx.current_payment_attempts[2].action == "customer_prompt"


def test_cold_start_new_customer(db_session):
    """Verify cold start customer with no historical payments or attempts succeeds without zero division."""
    customer = Customer(
        external_customer_id="cust_new_001",
        name="Eve FirstTimer",
        email="eve@example.com",
        total_payments=0,
        successful_payments=0,
        failed_payments=0,
    )
    db_session.add(customer)
    db_session.flush()

    payment = Payment(
        external_payment_id="pay_first_001",
        customer_id=customer.id,
        amount=799.0,
        currency="INR",
        payment_method="upi",
        status="failed",
        failure_reason="timeout",
    )
    db_session.add(payment)
    db_session.flush()

    opportunity = RecoveryOpportunity(
        payment_id=payment.id,
        status="open",
        revenue_at_risk=799.0,
        expected_recovery=0.0,
    )
    db_session.add(opportunity)
    db_session.commit()

    ctx = get_customer_context(db_session, customer.id, payment.id)

    assert ctx.customer.total_payments == 0
    assert ctx.customer.historical_success_rate == 0.0
    assert ctx.current_payment_attempts == []
    assert ctx.historical_payments == []
    assert ctx.recovery_statistics.total_recovery_opportunities == 0
    assert ctx.recovery_statistics.recovery_rate == 0.0
    assert ctx.recovery_statistics.total_amount_recovered == 0.0
    assert ctx.recovery_statistics.previously_successful_actions == []
    assert ctx.recovery_statistics.previously_failed_actions == []


def test_nonexistent_customer_raises_error(db_session):
    """Verify CustomerNotFoundError when customer does not exist."""
    fake_cust_id = uuid.uuid4()
    fake_pay_id = uuid.uuid4()

    with pytest.raises(CustomerNotFoundError) as exc_info:
        get_customer_context(db_session, fake_cust_id, fake_pay_id)

    assert str(fake_cust_id) in str(exc_info.value)


def test_nonexistent_payment_raises_error(db_session):
    """Verify PaymentNotFoundError when payment does not exist for an existing customer."""
    customer = Customer(name="Frank", email="frank@example.com")
    db_session.add(customer)
    db_session.commit()

    fake_pay_id = uuid.uuid4()
    with pytest.raises(PaymentNotFoundError) as exc_info:
        get_customer_context(db_session, customer.id, fake_pay_id)

    assert str(fake_pay_id) in str(exc_info.value)


def test_payment_customer_mismatch_raises_error(db_session):
    """Verify tenant isolation: PaymentCustomerMismatchError when payment belongs to another customer."""
    cust_a = Customer(name="Tenant A", email="a@example.com")
    cust_b = Customer(name="Tenant B", email="b@example.com")
    db_session.add_all([cust_a, cust_b])
    db_session.flush()

    # Payment belongs to Tenant B
    payment_b = Payment(
        customer_id=cust_b.id,
        amount=1000.0,
        currency="INR",
        payment_method="card",
        status="failed",
    )
    db_session.add(payment_b)
    db_session.flush()

    opportunity_b = RecoveryOpportunity(
        payment_id=payment_b.id,
        status="open",
        revenue_at_risk=1000.0,
        expected_recovery=800.0,
    )
    db_session.add(opportunity_b)
    db_session.commit()

    # Query payment_b with Tenant A's customer_id
    with pytest.raises(PaymentCustomerMismatchError) as exc_info:
        get_customer_context(db_session, cust_a.id, payment_b.id)

    assert str(payment_b.id) in str(exc_info.value)
    assert str(cust_a.id) in str(exc_info.value)
    assert str(cust_b.id) in str(exc_info.value)


def test_missing_recovery_opportunity_raises_error(db_session):
    """Verify RecoveryOpportunityNotFoundError when payment exists but lacks a recovery opportunity."""
    customer = Customer(name="George", email="george@example.com")
    db_session.add(customer)
    db_session.flush()

    payment = Payment(
        customer_id=customer.id,
        amount=200.0,
        currency="INR",
        payment_method="card",
        status="failed",
    )
    db_session.add(payment)
    db_session.commit()

    with pytest.raises(RecoveryOpportunityNotFoundError) as exc_info:
        get_customer_context(db_session, customer.id, payment.id)

    assert str(payment.id) in str(exc_info.value)


def test_strict_customer_data_isolation(db_session):
    """Verify historical payments and recovery attempts of other customers never leak into target context."""
    cust_target = Customer(name="Target Cust", email="target@example.com")
    cust_other = Customer(name="Other Cust", email="other@example.com")
    db_session.add_all([cust_target, cust_other])
    db_session.flush()

    # Other customer's history
    p_other = Payment(
        customer_id=cust_other.id,
        amount=9999.0,
        currency="INR",
        payment_method="card",
        status="succeeded",
    )
    db_session.add(p_other)
    db_session.flush()

    opp_other = RecoveryOpportunity(
        payment_id=p_other.id,
        status="recovered",
        revenue_at_risk=9999.0,
        expected_recovery=9999.0,
        recommended_action="exclusive_vip_action",
    )
    db_session.add(opp_other)
    db_session.flush()

    att_other = RecoveryAttempt(
        opportunity_id=opp_other.id,
        action="exclusive_vip_action",
        status="succeeded",
        amount_recovered=9999.0,
    )
    db_session.add(att_other)

    # Target customer's current payment
    p_target = Payment(
        customer_id=cust_target.id,
        amount=100.0,
        currency="INR",
        payment_method="upi",
        status="failed",
    )
    db_session.add(p_target)
    db_session.flush()

    opp_target = RecoveryOpportunity(
        payment_id=p_target.id,
        status="open",
        revenue_at_risk=100.0,
        expected_recovery=0.0,
    )
    db_session.add(opp_target)
    db_session.commit()

    ctx = get_customer_context(db_session, cust_target.id, p_target.id)

    # Ensure other customer's action and data are NOT in target's context
    assert ctx.historical_payments == []
    assert ctx.recovery_statistics.previously_successful_actions == []
    assert "exclusive_vip_action" not in ctx.recovery_statistics.previously_successful_actions
    assert ctx.recovery_statistics.total_amount_recovered == 0.0


def test_history_limit_parameter(db_session):
    """Verify that history_limit parameter bounds the retrieved historical payments."""
    customer = Customer(name="High Volume User", email="hv@example.com")
    db_session.add(customer)
    db_session.flush()

    # Create 5 historical payments
    for i in range(5):
        p = Payment(
            customer_id=customer.id,
            amount=100.0 * (i + 1),
            payment_method="card",
            status="succeeded",
        )
        db_session.add(p)

    # Current failed payment
    current_p = Payment(
        customer_id=customer.id,
        amount=999.0,
        payment_method="upi",
        status="failed",
    )
    db_session.add(current_p)
    db_session.flush()

    opp = RecoveryOpportunity(
        payment_id=current_p.id,
        status="open",
        revenue_at_risk=999.0,
        expected_recovery=0.0,
    )
    db_session.add(opp)
    db_session.commit()

    # Retrieve with history_limit=2
    ctx = get_customer_context(db_session, customer.id, current_p.id, history_limit=2)
    assert len(ctx.historical_payments) == 2


def test_retrieval_with_ingested_csv_data(db_session, tmp_path):
    """Verify context retrieval on data ingested via the historical ingestion pipeline."""
    import csv
    from app.historical_data import load_historical_data

    csv_file = tmp_path / "test_ingestion.csv"
    headers = [
        "record_id", "customer_id", "payment_id", "customer_payment_count",
        "customer_success_rate", "customer_previous_failures", "payment_amount",
        "currency", "payment_method", "failure_reason", "attempt_number",
        "hours_since_failure", "action_taken", "previous_action",
        "previous_attempt_count", "recovered", "amount_recovered", "recovery_time_hours"
    ]
    rows = [
        # Payment 1 (past attempt 1 - failed)
        [
            "rec_001", "cust_ingest_1", "pay_ingest_1", "5", "0.8", "1", "1200.0",
            "INR", "card", "insufficient_funds", "1", "1.0", "smart_retry", "none", "0",
            "false", "0.0", "0.0"
        ],
        # Payment 1 (past attempt 2 - succeeded)
        [
            "rec_002", "cust_ingest_1", "pay_ingest_1", "5", "0.8", "1", "1200.0",
            "INR", "card", "insufficient_funds", "2", "24.0", "customer_prompt", "smart_retry", "1",
            "true", "1200.0", "24.0"
        ],
        # Payment 2 (current failed payment)
        [
            "rec_003", "cust_ingest_1", "pay_ingest_2", "5", "0.8", "1", "2500.0",
            "INR", "upi", "bank_timeout", "1", "0.5", "smart_retry", "none", "0",
            "false", "0.0", "0.0"
        ],
    ]
    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(rows)

    load_historical_data(db_session, csv_path=csv_file)

    ctx = get_customer_context(db_session, "cust_ingest_1", "pay_ingest_2")

    assert ctx.customer.external_customer_id == "cust_ingest_1"
    assert ctx.current_payment.external_payment_id == "pay_ingest_2"
    assert ctx.current_payment.amount == 2500.0
    assert len(ctx.current_payment_attempts) == 1
    assert ctx.current_payment_attempts[0].action == "smart_retry"
    assert ctx.current_payment_attempts[0].status == "failed"

    # Historical payments: pay_ingest_1
    assert len(ctx.historical_payments) == 1
    assert ctx.historical_payments[0].external_payment_id == "pay_ingest_1"
    assert ctx.historical_payments[0].was_recovered is True
    assert ctx.historical_payments[0].recovery_attempts_count == 2

    # Aggregates from historical data
    assert ctx.recovery_statistics.total_recovery_opportunities == 1
    assert ctx.recovery_statistics.recovered_opportunities == 1
    assert ctx.recovery_statistics.recovery_rate == 1.0
    assert ctx.recovery_statistics.total_amount_recovered == 1200.0
    assert "customer_prompt" in ctx.recovery_statistics.previously_successful_actions

