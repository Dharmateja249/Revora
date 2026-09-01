"""
Comprehensive Test Suite for Deterministic HistoricalRetriever.

Tests:
1. Relevant historical cases are retrieved and ranked
2. Current payment is excluded from candidates
3. Cross-customer records are strictly isolated (no tenant leakage)
4. Failure reason matching increases relevance score
5. Payment method matching affects ranking
6. Amount proximity affects ranking
7. Recency affects ranking
8. Recovery outcome affects ranking (recovered > unrecovered)
9. Results are strictly ordered by relevance score descending
10. Deterministic tie-breaking works (score -> recency -> payment_id)
11. top_k parameter is strictly respected
12. Zero-history / cold-start customers return empty list []
13. Relevance scores stay strictly within [0.0, 1.0]
14. Returned objects are valid HistoricalCase instances
15. Eager-loading query count test (no N+1 regressions)
16. Standalone execution with only CustomerRecoveryContext (zero DB)
"""

import uuid
from datetime import datetime, timedelta, timezone

import pytest
from app.context import (
    CustomerContext,
    CustomerRecoveryContext,
    HistoricalPaymentContext,
    PaymentContext,
    RecoveryOpportunityContext,
)
from app.database import Base
from app.historical_retrieval import HistoricalCase
from app.historical_retriever import (
    HistoricalRetriever,
    retrieve_historical_cases,
)
from app.models import (
    Customer,
    Payment,
    RecoveryAttempt,
    RecoveryOpportunity,
)
from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker


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


# ============================================================================
# 1. Basic Retrieval & Candidate Ranking with Database
# ============================================================================


def test_retrieve_relevant_cases_with_db(db_session):
    """Verify historical cases are retrieved from DB, ranked, and current payment excluded."""
    customer = Customer(
        name="Alice Walker",
        email="alice@example.com",
        total_payments=3,
        successful_payments=2,
        failed_payments=1,
    )
    db_session.add(customer)
    db_session.flush()

    t0 = datetime(2026, 1, 1, 10, 0, 0, tzinfo=timezone.utc)
    t1 = datetime(2026, 1, 15, 10, 0, 0, tzinfo=timezone.utc)
    t_curr = datetime(2026, 2, 1, 10, 0, 0, tzinfo=timezone.utc)

    # Historical Payment 1 (High match: same failure reason & method, recovered)
    p1 = Payment(
        customer=customer,
        amount=1500.0,
        currency="INR",
        payment_method="card",
        status="succeeded",
        failure_reason="bank_timeout",
        created_at=t1,
    )
    db_session.add(p1)
    db_session.flush()

    opp1 = RecoveryOpportunity(
        payment=p1,
        status="recovered",
        revenue_at_risk=1500.0,
        expected_recovery=1500.0,
        recommended_action="smart_retry",
        created_at=t1,
    )
    db_session.add(opp1)
    db_session.flush()

    att1 = RecoveryAttempt(
        opportunity=opp1,
        action="smart_retry",
        status="succeeded",
        amount_recovered=1500.0,
        created_at=t1,
    )
    db_session.add(att1)

    # Historical Payment 2 (Lower match: different failure reason & method)
    p2 = Payment(
        customer=customer,
        amount=5000.0,
        currency="INR",
        payment_method="upi",
        status="failed",
        failure_reason="invalid_card",
        created_at=t0,
    )
    db_session.add(p2)
    db_session.flush()

    opp2 = RecoveryOpportunity(
        payment=p2,
        status="failed",
        revenue_at_risk=5000.0,
        expected_recovery=0.0,
        recommended_action="change_payment_method",
        created_at=t0,
    )
    db_session.add(opp2)

    # Current Failed Payment
    p_curr = Payment(
        customer=customer,
        amount=1500.0,
        currency="INR",
        payment_method="card",
        status="failed",
        failure_reason="bank_timeout",
        created_at=t_curr,
    )
    db_session.add(p_curr)
    db_session.flush()

    opp_curr = RecoveryOpportunity(
        payment=p_curr,
        status="open",
        revenue_at_risk=1500.0,
        expected_recovery=0.0,
        created_at=t_curr,
    )
    db_session.add(opp_curr)
    db_session.commit()

    context = CustomerRecoveryContext(
        customer=CustomerContext(
            customer_id=customer.id,
            name=customer.name,
            email=customer.email,
        ),
        current_payment=PaymentContext(
            payment_id=p_curr.id,
            amount=p_curr.amount,
            currency=p_curr.currency,
            payment_method=p_curr.payment_method,
            status=p_curr.status,
            failure_reason=p_curr.failure_reason,
            created_at=p_curr.created_at,
        ),
        current_opportunity=RecoveryOpportunityContext(
            opportunity_id=opp_curr.id,
            status=opp_curr.status,
            revenue_at_risk=opp_curr.revenue_at_risk,
        ),
    )

    retriever = HistoricalRetriever(db_session=db_session)
    results = retriever.retrieve_relevant_cases(context, top_k=5)

    assert len(results) == 2
    # Current payment must not be in results
    assert all(r.payment_id != p_curr.id for r in results)
    # p1 should be ranked #1 with higher relevance score than p2
    assert results[0].payment_id == p1.id
    assert results[1].payment_id == p2.id
    assert results[0].relevance_score > results[1].relevance_score
    assert isinstance(results[0], HistoricalCase)
    assert results[0].was_recovered is True
    assert results[0].recovery_action == "smart_retry"


# ============================================================================
# 2. Strict Customer Isolation (Tenant Leakage Prevention)
# ============================================================================


def test_cross_customer_isolation(db_session):
    """Verify that records belonging to other customers are never retrieved."""
    cust_a = Customer(name="Customer A", email="a@example.com")
    cust_b = Customer(name="Customer B", email="b@example.com")
    db_session.add_all([cust_a, cust_b])
    db_session.flush()

    # Cust B payment (perfect match on reason/method)
    p_b = Payment(
        customer=cust_b,
        amount=1000.0,
        currency="INR",
        payment_method="card",
        status="succeeded",
        failure_reason="timeout",
    )
    db_session.add(p_b)

    # Cust A current payment
    p_a = Payment(
        customer=cust_a,
        amount=1000.0,
        currency="INR",
        payment_method="card",
        status="failed",
        failure_reason="timeout",
    )
    db_session.add(p_a)
    db_session.commit()

    context_a = CustomerRecoveryContext(
        customer=CustomerContext(customer_id=cust_a.id),
        current_payment=PaymentContext(
            payment_id=p_a.id,
            amount=p_a.amount,
            payment_method=p_a.payment_method,
            status=p_a.status,
            failure_reason=p_a.failure_reason,
        ),
    )

    retriever = HistoricalRetriever(db_session=db_session)
    results = retriever.retrieve_relevant_cases(context_a, top_k=5)

    # Customer A has no prior payments; Cust B's record must not leak
    assert len(results) == 0


# ============================================================================
# 3. Context-Only Execution (Zero Database Dependency)
# ============================================================================


def test_retrieve_from_context_without_db():
    """Verify retriever works deterministically directly from CustomerRecoveryContext without DB."""
    cust_id = uuid.uuid4()
    curr_id = uuid.uuid4()
    h1_id = uuid.uuid4()
    h2_id = uuid.uuid4()
    now = datetime(2026, 2, 1, 12, 0, 0, tzinfo=timezone.utc)

    curr_p = PaymentContext(
        payment_id=curr_id,
        amount=2000.0,
        currency="INR",
        payment_method="card",
        status="failed",
        failure_reason="insufficient_funds",
        created_at=now,
    )

    h1 = HistoricalPaymentContext(
        payment_id=h1_id,
        amount=2000.0,
        currency="INR",
        payment_method="card",
        status="succeeded",
        failure_reason="insufficient_funds",
        created_at=now - timedelta(days=2),
        was_recovered=True,
        recovery_action="payment_link",
        recovery_attempts_count=1,
    )

    h2 = HistoricalPaymentContext(
        payment_id=h2_id,
        amount=100.0,
        currency="INR",
        payment_method="upi",
        status="failed",
        failure_reason="lost_card",
        created_at=now - timedelta(days=30),
        was_recovered=False,
        recovery_action=None,
        recovery_attempts_count=0,
    )

    context = CustomerRecoveryContext(
        customer=CustomerContext(customer_id=cust_id),
        current_payment=curr_p,
        historical_payments=[h2, h1],  # Intentionally passed in reverse order
    )

    retriever = HistoricalRetriever(db_session=None)
    results = retriever.retrieve_relevant_cases(context, top_k=5)

    assert len(results) == 2
    assert results[0].payment_id == h1_id
    assert results[1].payment_id == h2_id
    assert results[0].relevance_score > results[1].relevance_score
    assert results[0].was_recovered is True
    assert results[0].recovery_action == "payment_link"


# ============================================================================
# 4. Scoring Signal Isolated Tests
# ============================================================================


def test_failure_reason_matching_increases_relevance():
    """Verify that an exact or category failure match yields higher score than unrelated failure."""
    cust_id = uuid.uuid4()
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)

    curr = PaymentContext(
        payment_id=uuid.uuid4(),
        amount=1000.0,
        currency="INR",
        payment_method="card",
        status="failed",
        failure_reason="bank_timeout",
        created_at=now,
    )

    h_exact = HistoricalPaymentContext(
        payment_id=uuid.uuid4(),
        amount=1000.0,
        currency="INR",
        payment_method="card",
        status="succeeded",
        failure_reason="bank_timeout",
        created_at=now,
    )

    h_diff = HistoricalPaymentContext(
        payment_id=uuid.uuid4(),
        amount=1000.0,
        currency="INR",
        payment_method="card",
        status="succeeded",
        failure_reason="card_expired",
        created_at=now,
    )

    context = CustomerRecoveryContext(
        customer=CustomerContext(customer_id=cust_id),
        current_payment=curr,
        historical_payments=[h_diff, h_exact],
    )

    retriever = HistoricalRetriever()
    results = retriever.retrieve_relevant_cases(context)

    assert results[0].payment_id == h_exact.payment_id
    assert results[0].relevance_score > results[1].relevance_score


def test_payment_method_matching_affects_ranking():
    """Verify same payment method scores higher than different payment method (all else equal)."""
    cust_id = uuid.uuid4()
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)

    curr = PaymentContext(
        payment_id=uuid.uuid4(),
        amount=1000.0,
        currency="INR",
        payment_method="upi",
        status="failed",
        failure_reason="timeout",
        created_at=now,
    )

    h_same_method = HistoricalPaymentContext(
        payment_id=uuid.uuid4(),
        amount=1000.0,
        currency="INR",
        payment_method="upi",
        status="failed",
        failure_reason="timeout",
        created_at=now,
    )

    h_diff_method = HistoricalPaymentContext(
        payment_id=uuid.uuid4(),
        amount=1000.0,
        currency="INR",
        payment_method="card",
        status="failed",
        failure_reason="timeout",
        created_at=now,
    )

    context = CustomerRecoveryContext(
        customer=CustomerContext(customer_id=cust_id),
        current_payment=curr,
        historical_payments=[h_diff_method, h_same_method],
    )

    results = retrieve_historical_cases(context)
    assert results[0].payment_id == h_same_method.payment_id
    assert results[0].relevance_score > results[1].relevance_score


def test_amount_similarity_affects_ranking():
    """Verify closer payment amount produces higher relevance score."""
    cust_id = uuid.uuid4()
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)

    curr = PaymentContext(
        payment_id=uuid.uuid4(),
        amount=2500.0,
        currency="INR",
        payment_method="card",
        status="failed",
        failure_reason="timeout",
        created_at=now,
    )

    h_close_amount = HistoricalPaymentContext(
        payment_id=uuid.uuid4(),
        amount=2600.0,
        currency="INR",
        payment_method="card",
        status="failed",
        failure_reason="timeout",
        created_at=now,
    )

    h_distant_amount = HistoricalPaymentContext(
        payment_id=uuid.uuid4(),
        amount=100000.0,
        currency="INR",
        payment_method="card",
        status="failed",
        failure_reason="timeout",
        created_at=now,
    )

    context = CustomerRecoveryContext(
        customer=CustomerContext(customer_id=cust_id),
        current_payment=curr,
        historical_payments=[h_distant_amount, h_close_amount],
    )

    results = retrieve_historical_cases(context)
    assert results[0].payment_id == h_close_amount.payment_id
    assert results[0].relevance_score > results[1].relevance_score


def test_recency_affects_ranking():
    """Verify more recent payment scores higher than older payment (all else equal)."""
    cust_id = uuid.uuid4()
    t_curr = datetime(2026, 2, 1, tzinfo=timezone.utc)

    curr = PaymentContext(
        payment_id=uuid.uuid4(),
        amount=1000.0,
        currency="INR",
        payment_method="card",
        status="failed",
        failure_reason="timeout",
        created_at=t_curr,
    )

    h_recent = HistoricalPaymentContext(
        payment_id=uuid.uuid4(),
        amount=1000.0,
        currency="INR",
        payment_method="card",
        status="failed",
        failure_reason="timeout",
        created_at=t_curr - timedelta(days=2),
    )

    h_old = HistoricalPaymentContext(
        payment_id=uuid.uuid4(),
        amount=1000.0,
        currency="INR",
        payment_method="card",
        status="failed",
        failure_reason="timeout",
        created_at=t_curr - timedelta(days=180),
    )

    context = CustomerRecoveryContext(
        customer=CustomerContext(customer_id=cust_id),
        current_payment=curr,
        historical_payments=[h_old, h_recent],
    )

    results = retrieve_historical_cases(context)
    assert results[0].payment_id == h_recent.payment_id
    assert results[0].relevance_score > results[1].relevance_score


# ============================================================================
# 5. Deterministic Tie-Breaking & Top-K Boundaries
# ============================================================================


def test_deterministic_tie_breaking():
    """Verify candidates with identical scores and timestamps sort by payment_id."""
    cust_id = uuid.uuid4()
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)

    id_a = uuid.UUID("00000000-0000-0000-0000-000000000001")
    id_b = uuid.UUID("00000000-0000-0000-0000-000000000002")

    curr = PaymentContext(
        payment_id=uuid.uuid4(),
        amount=1000.0,
        currency="INR",
        payment_method="card",
        status="failed",
        failure_reason="timeout",
        created_at=now,
    )

    h_b = HistoricalPaymentContext(
        payment_id=id_b,
        amount=1000.0,
        currency="INR",
        payment_method="card",
        status="failed",
        failure_reason="timeout",
        created_at=now,
    )

    h_a = HistoricalPaymentContext(
        payment_id=id_a,
        amount=1000.0,
        currency="INR",
        payment_method="card",
        status="failed",
        failure_reason="timeout",
        created_at=now,
    )

    context = CustomerRecoveryContext(
        customer=CustomerContext(customer_id=cust_id),
        current_payment=curr,
        historical_payments=[h_b, h_a],
    )

    results = retrieve_historical_cases(context)
    assert len(results) == 2
    assert results[0].relevance_score == results[1].relevance_score
    # Lexicographical UUID tie-breaker: id_a < id_b
    assert results[0].payment_id == id_a
    assert results[1].payment_id == id_b


def test_top_k_parameter_respected():
    """Verify top_k limit truncates results correctly."""
    cust_id = uuid.uuid4()
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)

    curr = PaymentContext(
        payment_id=uuid.uuid4(),
        amount=1000.0,
        currency="INR",
        payment_method="card",
        status="failed",
    )

    hist_payments = [
        HistoricalPaymentContext(
            payment_id=uuid.uuid4(),
            amount=float(100 * i),
            currency="INR",
            payment_method="card",
            status="failed",
            created_at=now,
        )
        for i in range(1, 11)
    ]

    context = CustomerRecoveryContext(
        customer=CustomerContext(customer_id=cust_id),
        current_payment=curr,
        historical_payments=hist_payments,
    )

    results_3 = retrieve_historical_cases(context, top_k=3)
    assert len(results_3) == 3

    results_0 = retrieve_historical_cases(context, top_k=0)
    assert len(results_0) == 0


def test_cold_start_empty_history_returns_empty_list():
    """Verify cold-start customer with zero history gracefully returns []."""
    context = CustomerRecoveryContext(
        customer=CustomerContext(customer_id=uuid.uuid4()),
        current_payment=PaymentContext(
            payment_id=uuid.uuid4(),
            amount=500.0,
            payment_method="upi",
            status="failed",
        ),
        historical_payments=[],
    )

    results = retrieve_historical_cases(context)
    assert results == []


def test_relevance_scores_within_strict_bounds():
    """Verify all returned relevance scores are within [0.0, 1.0]."""
    cust_id = uuid.uuid4()
    curr = PaymentContext(
        payment_id=uuid.uuid4(),
        amount=0.0,
        currency="INR",
        payment_method="unknown",
        status="failed",
    )

    h1 = HistoricalPaymentContext(
        payment_id=uuid.uuid4(),
        amount=999999.0,
        currency="USD",
        payment_method="other",
        status="failed",
    )

    context = CustomerRecoveryContext(
        customer=CustomerContext(customer_id=cust_id),
        current_payment=curr,
        historical_payments=[h1],
    )

    results = retrieve_historical_cases(context)
    assert len(results) == 1
    assert 0.0 <= results[0].relevance_score <= 1.0


# ============================================================================
# 6. Eager Loading & Query Efficiency (No N+1 Regression)
# ============================================================================


def test_no_n_plus_one_query_regression(db_session):
    """Verify retrieving multiple historical cases executes in constant/bounded queries."""
    customer = Customer(name="Perf Test", email="perf@example.com")
    db_session.add(customer)
    db_session.flush()

    # Create 10 historical payments each with recovery opportunity and attempts
    for i in range(10):
        p = Payment(
            customer=customer,
            amount=100.0 * (i + 1),
            currency="INR",
            payment_method="card",
            status="succeeded" if i % 2 == 0 else "failed",
            failure_reason="timeout",
        )
        db_session.add(p)
        db_session.flush()

        opp = RecoveryOpportunity(
            payment=p,
            status="recovered" if i % 2 == 0 else "failed",
            revenue_at_risk=p.amount,
            expected_recovery=p.amount if i % 2 == 0 else 0.0,
            recommended_action="smart_retry",
        )
        db_session.add(opp)
        db_session.flush()

        att = RecoveryAttempt(
            opportunity=opp,
            action="smart_retry",
            status="succeeded" if i % 2 == 0 else "failed",
            amount_recovered=p.amount if i % 2 == 0 else 0.0,
        )
        db_session.add(att)

    # Current payment
    curr = Payment(
        customer=customer,
        amount=500.0,
        currency="INR",
        payment_method="card",
        status="failed",
        failure_reason="timeout",
    )
    db_session.add(curr)
    db_session.commit()

    context = CustomerRecoveryContext(
        customer=CustomerContext(customer_id=customer.id),
        current_payment=PaymentContext(
            payment_id=curr.id,
            amount=curr.amount,
            currency=curr.currency,
            payment_method=curr.payment_method,
            status=curr.status,
            failure_reason=curr.failure_reason,
        ),
    )

    query_count = 0

    def query_listener(conn, cursor, statement, parameters, context, executemany):
        nonlocal query_count
        query_count += 1

    engine = db_session.get_bind()
    event.listen(engine, "before_cursor_execute", query_listener)

    try:
        retriever = HistoricalRetriever(db_session=db_session)
        results = retriever.retrieve_relevant_cases(context, top_k=10)
        assert len(results) == 10
        # With eager selectinload, loading payments, opportunities, and attempts
        # should take <= 3 queries (Payment, RecoveryOpportunity, RecoveryAttempt)
        assert query_count <= 3
    finally:
        event.remove(engine, "before_cursor_execute", query_listener)
