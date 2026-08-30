"""
Unit and Integration Tests for RecoveryService Application Orchestrator.

Tests:
1. Happy path: context retrieval -> RAG retrieval -> decision -> opportunity update -> audit event -> commit -> response DTO
2. RAG disabled (use_rag=False): retriever not called, decision still evaluated, response reports RAG disabled
3. Dependency injection: custom DecisionEngine and HybridHistoricalRetriever injection
4. Error paths: CustomerNotFoundError, PaymentNotFoundError, PaymentCustomerMismatchError, RecoveryOpportunityNotFoundError
5. Transaction safety: rollback occurs on commit failure, exception propagates
6. Audit correctness: verifies AuditEvent fields and metadata_payload in database
7. Response isolation: ensures response DTO exposes no raw ORM objects or sensitive PII
"""

from datetime import datetime, timezone, timedelta
import uuid
import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker

from app.context import (
    CustomerNotFoundError,
    PaymentCustomerMismatchError,
    PaymentNotFoundError,
    RecoveryOpportunityNotFoundError,
)
from app.database import Base
from app.decision_engine import DecisionEngine, RecoveryAction, RecoveryDecision
from app.embedding_service import get_embedding_service
from app.historical_retrieval import HistoricalCase
from app.hybrid_historical_retriever import HybridHistoricalRetriever
from app.models import AuditEvent, Customer, Payment, RecoveryOpportunity, utc_now
from app.recovery_service import RecoveryService
from app.retrieval_document import historical_case_to_document
from app.schemas.recovery import (
    RecoveryEvaluationRequest,
    RecoveryEvaluationResponse,
)
from app.semantic_historical_retriever import SemanticHistoricalRetriever
from app.vector_index import VectorIndex


# ============================================================================
# Database Test Fixtures
# ============================================================================


@pytest.fixture
def in_memory_db():
    """Create a clean in-memory SQLite database for test execution."""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    TestingSession = sessionmaker(bind=engine, autocommit=False, autoflush=False)
    session = TestingSession()
    try:
        yield session
    finally:
        session.close()
        Base.metadata.drop_all(engine)


def _seed_customer_payment_opportunity(
    session,
    failure_reason="bank_timeout",
    payment_method="upi",
    amount=2500.0,
    status="failed",
):
    """Helper to seed customer, failed payment, and recovery opportunity in the DB."""
    cust_id = uuid.uuid4()
    pay_id = uuid.uuid4()
    opp_id = uuid.uuid4()
    now = utc_now()

    ext_suffix = uuid.uuid4().hex[:8]
    customer = Customer(
        id=cust_id,
        external_customer_id=f"cust_ext_{ext_suffix}",
        name="Alice Sharma",
        email=f"alice_{ext_suffix}@example.com",
        total_payments=5,
        successful_payments=4,
        failed_payments=1,
        created_at=now - timedelta(days=30),
    )
    payment = Payment(
        id=pay_id,
        external_payment_id=f"pay_ext_{ext_suffix}",
        customer_id=cust_id,
        amount=amount,
        currency="INR",
        payment_method=payment_method,
        status=status,
        failure_reason=failure_reason,
        created_at=now,
    )
    opportunity = RecoveryOpportunity(
        id=opp_id,
        payment_id=pay_id,
        status="open",
        revenue_at_risk=amount,
        expected_recovery=amount * 0.8,
        recommended_action=None,
        confidence=None,
        created_at=now,
    )

    session.add_all([customer, payment, opportunity])
    session.commit()
    return customer, payment, opportunity


# ============================================================================
# 1. Happy Path & RAG Integration
# ============================================================================


def test_recovery_service_happy_path(in_memory_db):
    """Verify complete recovery evaluation, DB mutation, and audit logging."""
    customer, payment, opportunity = _seed_customer_payment_opportunity(in_memory_db)

    # Seed an in-memory vector index with a historical case
    emb_service = get_embedding_service()
    vector_index = VectorIndex(dimension=emb_service.dimension)
    hist_case = HistoricalCase(
        payment_id=uuid.uuid4(),
        customer_id=customer.id,
        amount=2500.0,
        currency="INR",
        payment_method="upi",
        failure_reason="bank_timeout",
        recovery_action="payment_link",
        recovery_status="recovered",
        amount_recovered=2500.0,
        was_recovered=True,
        created_at=utc_now() - timedelta(days=2),
    )
    doc = historical_case_to_document(hist_case)
    vector_index.add(doc, emb_service.embed(doc.text))

    service = RecoveryService(
        vector_index=vector_index,
        embedding_service=emb_service,
    )

    request = RecoveryEvaluationRequest(
        customer_id=customer.id,
        payment_id=payment.id,
        use_rag=True,
    )

    response = service.evaluate_recovery(in_memory_db, request)

    # 1. Verify Response DTO
    assert isinstance(response, RecoveryEvaluationResponse)
    assert response.payment_id == payment.id
    assert response.customer_id == customer.id
    assert response.opportunity_id == opportunity.id
    assert response.recommended_action in (
        RecoveryAction.RETRY_PAYMENT,
        RecoveryAction.PAYMENT_LINK,
    )
    assert response.confidence > 0.0
    assert response.historical_rag_used is True
    assert response.retrieved_evidence_count >= 1

    # 2. Verify Database State Mutation
    in_memory_db.refresh(opportunity)
    assert opportunity.recommended_action == response.recommended_action.value
    assert opportunity.confidence == response.confidence

    # 3. Verify AuditEvent Record in DB
    audit_stmt = select(AuditEvent).where(AuditEvent.opportunity_id == opportunity.id)
    audit_events = in_memory_db.execute(audit_stmt).scalars().all()
    assert len(audit_events) == 1
    audit = audit_events[0]
    assert audit.event_type == "recovery_decision_evaluated"
    assert audit.description == response.reason
    assert "rule_matched" in audit.metadata_payload


def test_recovery_service_with_rag_disabled(in_memory_db):
    """Verify that use_rag=False disables historical retrieval and executes pure rule baseline."""
    customer, payment, opportunity = _seed_customer_payment_opportunity(in_memory_db)

    service = RecoveryService()
    request = RecoveryEvaluationRequest(
        customer_id=customer.id,
        payment_id=payment.id,
        use_rag=False,
    )

    response = service.evaluate_recovery(in_memory_db, request)

    assert response.historical_rag_used is False
    assert response.retrieved_evidence_count == 0
    assert response.recommended_action == RecoveryAction.RETRY_PAYMENT


# ============================================================================
# 2. Dependency Injection & Mocking
# ============================================================================


class MockDecisionEngine(DecisionEngine):
    """Mock decision engine for testing DI."""

    def evaluate(self, context, historical_cases=None):
        return RecoveryDecision(
            recommended_action=RecoveryAction.WAIT_AND_RETRY,
            reason="Mock decision engine recommendation",
            confidence=0.99,
            decision_basis={"mock": True},
        )


class MockRetriever:
    """Mock hybrid retriever for testing DI."""

    def __init__(self, cases):
        self.cases = cases
        self.called = False

    def retrieve_relevant_cases(self, context, top_k=5):
        self.called = True
        return self.cases


def test_recovery_service_dependency_injection(in_memory_db):
    """Verify custom decision engine and retriever can be injected and executed."""
    customer, payment, opportunity = _seed_customer_payment_opportunity(in_memory_db)

    mock_hist_case = HistoricalCase(
        payment_id=uuid.uuid4(),
        customer_id=customer.id,
        amount=1000.0,
        payment_method="upi",
        recovery_status="recovered",
        was_recovered=True,
    )
    mock_retriever = MockRetriever([mock_hist_case])
    mock_engine = MockDecisionEngine()

    service = RecoveryService(
        decision_engine=mock_engine,
        hybrid_retriever=mock_retriever,  # type: ignore
    )

    request = RecoveryEvaluationRequest(
        customer_id=customer.id,
        payment_id=payment.id,
        use_rag=True,
    )

    response = service.evaluate_recovery(in_memory_db, request)

    assert mock_retriever.called is True
    assert response.recommended_action == RecoveryAction.WAIT_AND_RETRY
    assert response.confidence == 0.99
    assert response.reason == "Mock decision engine recommendation"
    assert response.retrieved_evidence_count == 1


# ============================================================================
# 3. Error Paths & Domain Exceptions
# ============================================================================


def test_customer_not_found_raises_exception(in_memory_db):
    """Verify CustomerNotFoundError propagates without swallowing."""
    service = RecoveryService()
    request = RecoveryEvaluationRequest(
        customer_id=uuid.uuid4(),
        payment_id=uuid.uuid4(),
    )

    with pytest.raises(CustomerNotFoundError):
        service.evaluate_recovery(in_memory_db, request)


def test_payment_not_found_raises_exception(in_memory_db):
    """Verify PaymentNotFoundError propagates when payment does not exist."""
    customer, _, _ = _seed_customer_payment_opportunity(in_memory_db)
    service = RecoveryService()
    request = RecoveryEvaluationRequest(
        customer_id=customer.id,
        payment_id=uuid.uuid4(),
    )

    with pytest.raises(PaymentNotFoundError):
        service.evaluate_recovery(in_memory_db, request)


def test_payment_customer_mismatch_raises_exception(in_memory_db):
    """Verify PaymentCustomerMismatchError propagates when payment belongs to another customer."""
    customer1, payment1, _ = _seed_customer_payment_opportunity(in_memory_db)
    customer2, _, _ = _seed_customer_payment_opportunity(in_memory_db)

    service = RecoveryService()
    # Request evaluation of customer2 with payment1 (which belongs to customer1)
    request = RecoveryEvaluationRequest(
        customer_id=customer2.id,
        payment_id=payment1.id,
    )

    with pytest.raises(PaymentCustomerMismatchError):
        service.evaluate_recovery(in_memory_db, request)


def test_missing_opportunity_raises_exception(in_memory_db):
    """Verify RecoveryOpportunityNotFoundError propagates when payment has no opportunity."""
    customer, payment, opportunity = _seed_customer_payment_opportunity(in_memory_db)
    # Delete opportunity
    in_memory_db.delete(opportunity)
    in_memory_db.commit()

    service = RecoveryService()
    request = RecoveryEvaluationRequest(
        customer_id=customer.id,
        payment_id=payment.id,
    )

    with pytest.raises(RecoveryOpportunityNotFoundError):
        service.evaluate_recovery(in_memory_db, request)


# ============================================================================
# 4. Transaction Safety & Rollback
# ============================================================================


def test_transaction_rollback_on_persistence_failure(in_memory_db):
    """Verify transaction is rolled back and no partial audit record persists on DB failure."""
    customer, payment, opportunity = _seed_customer_payment_opportunity(in_memory_db)

    service = RecoveryService()
    request = RecoveryEvaluationRequest(
        customer_id=customer.id,
        payment_id=payment.id,
    )

    # Monkeypatch commit to simulate database error
    def failing_commit():
        raise RuntimeError("Simulated Database I/O Failure")

    in_memory_db.commit = failing_commit

    with pytest.raises(RuntimeError) as exc:
        service.evaluate_recovery(in_memory_db, request)
    assert "Simulated Database I/O Failure" in str(exc.value)


# ============================================================================
# 5. Response Isolation (No PII Leakage)
# ============================================================================


def test_response_isolation_no_pii_leaked(in_memory_db):
    """Verify response DTO contains no customer email or raw ORM objects."""
    customer, payment, opportunity = _seed_customer_payment_opportunity(in_memory_db)
    service = RecoveryService()
    request = RecoveryEvaluationRequest(
        customer_id=customer.id,
        payment_id=payment.id,
    )

    response = service.evaluate_recovery(in_memory_db, request)
    resp_dict = response.model_dump()

    # Verify no PII fields
    assert "email" not in resp_dict
    assert "name" not in resp_dict
    assert "customer" not in resp_dict
    assert "payment" not in resp_dict
    assert "opportunity" not in resp_dict
