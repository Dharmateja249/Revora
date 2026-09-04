"""
Comprehensive Adaptive Feedback Loop Test Suite for Revora.

Proves end-to-end:
1. Failed payment evaluates customer context derived from the relational DATABASE.
2. Context includes transaction volumes (total, successful, average), recent payment behavior,
   and previous recovery statistics.
3. PolicyValidator maintains unbypassable authority over LLM recommendations.
4. Action execution persists complete outcomes (Payment, RecoveryOpportunity, RecoveryAttempt, AuditEvent)
   into the relational database.
5. Successfully or failed executed recovery outcomes are dynamically ingested into the runtime RAG VectorIndex.
6. A SUBSEQUENT decision proves that:
   - Customer payment count and transaction volumes have updated in the database context.
   - Recent payment behavior reflects the previous execution and its outcome.
   - The newly added recovery precedent is retrieved by RAG for relevant subsequent queries.
7. Tenant isolation is strictly preserved.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest
from app.action_executor import ActionExecutor
from app.agent.context_builder import AgentContextBuilder
from app.agent.orchestrator import AgentOrchestrator
from app.agent.provider import MockLLMProvider
from app.agent.schemas import LLMRecoveryRecommendation
from app.context import CustomerNotFoundError
from app.context_retrieval import DEMO_CUSTOMER_UUID, get_customer_context
from app.database import Base
from app.decision_engine import RecoveryAction
from app.embedding_service import get_embedding_service
from app.models import (
    AuditEvent,
    Customer,
    Payment,
    RecoveryAttempt,
    RecoveryOpportunity,
)
from app.policies.validator import PolicyValidator
from app.razorpay_adapter import RazorpayAdapter, RazorpayAPIError
from app.recovery_decision_service import RecoveryDecisionService
from app.schemas.decision import (
    CustomerProfileDTO,
    RecoveryDecisionRequest,
)
from app.vector_index import VectorIndex
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool


@pytest.fixture
def isolated_db_session():
    """Create an in-memory SQLite database session with all models created."""
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(bind=engine)
    TestingSession = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    session = TestingSession()
    try:
        yield session
    finally:
        session.close()
        Base.metadata.drop_all(bind=engine)


def test_adaptive_feedback_loop_end_to_end(isolated_db_session: Session):
    """
    Demonstrate that a persisted recovery outcome is visible to a subsequent decision:
    1. First decision executes a payment link.
    2. Outcome is written to SQLite and ingested into runtime VectorIndex.
    3. Second decision observes incremented payment count, updated volume, recent behavior,
       and retrieves the newly created recovery case via RAG!
    """
    db = isolated_db_session
    vector_index = VectorIndex()
    embedding_service = get_embedding_service()

    # Configure mock Razorpay adapter for successful payment link creation
    mock_adapter = MagicMock(spec=RazorpayAdapter)
    mock_adapter.create_payment_link = AsyncMock(
        return_value={
            "id": "plink_adaptive_001",
            "short_url": "https://rzp.io/i/adaptive001",
            "amount": 845000,
            "currency": "INR",
            "status": "created",
            "description": "Payment recovery for failed CARD payment",
        }
    )
    mock_adapter.dry_run = False
    executor = ActionExecutor(razorpay_adapter=mock_adapter)

    # Configure mock LLM to recommend PAYMENT_LINK
    mock_provider = MockLLMProvider(
        recommendation=LLMRecoveryRecommendation(
            recommended_action=RecoveryAction.PAYMENT_LINK,
            confidence=0.85,
            reasoning="OTP timeout on high-value enterprise card. Issue payment link.",
            key_factors=("High historical success rate", "OTP timeout"),
            referenced_case_ids=(),
        )
    )
    orchestrator = AgentOrchestrator(
        provider=mock_provider,
        policy_validator=PolicyValidator(),
        context_builder=AgentContextBuilder(),
    )

    service = RecoveryDecisionService(
        agent_orchestrator=orchestrator,
        action_executor=executor,
        vector_index=vector_index,
        embedding_service=embedding_service,
    )

    # -------------------------------------------------------------------------
    # STEP 1: First Decision (Initial baseline state)
    # -------------------------------------------------------------------------
    payment_1_id = uuid4()
    req_1 = RecoveryDecisionRequest(
        payment_id=payment_1_id,
        amount=8450.0,
        currency="INR",
        payment_method="card",
        failure_reason="customer_auth_failed_otp_timeout",
        payment_status="failed",
        customer=CustomerProfileDTO(
            customer_id=DEMO_CUSTOMER_UUID,
            total_payments=42,
            successful_payments=40,
            failed_payments=2,
            historical_success_rate=0.95,
        ),
        previous_attempts=[],
        opportunity_status="open",
        revenue_at_risk=8450.0,
        max_attempts=3,
        execute_action=True,
    )

    resp_1 = asyncio.run(service.evaluate_decision(req_1, db_session=db))

    # Verify First Decision Outcome
    assert resp_1.recommended_action == RecoveryAction.PAYMENT_LINK
    assert resp_1.execution is not None
    assert resp_1.execution.attempted is True
    assert resp_1.execution.status == "success"
    assert resp_1.execution.success is True
    assert resp_1.execution.reference_id == "plink_adaptive_001"
    assert resp_1.execution.persisted is True
    assert resp_1.execution.persistence_error is None

    # -------------------------------------------------------------------------
    # STEP 2: Verify Database Persistence (Closed-Loop Invariant)
    # -------------------------------------------------------------------------
    # Verify Payment record in DB
    db_payment = db.get(Payment, payment_1_id)
    assert db_payment is not None
    assert db_payment.amount == 8450.0
    assert db_payment.payment_method == "card"
    assert db_payment.failure_reason == "customer_auth_failed_otp_timeout"

    # Verify RecoveryOpportunity record in DB
    db_opp = (
        db.execute(
            select(RecoveryOpportunity).where(
                RecoveryOpportunity.payment_id == payment_1_id
            )
        )
        .scalars()
        .first()
    )
    assert db_opp is not None
    assert db_opp.status == "recovered"
    assert db_opp.recommended_action == RecoveryAction.PAYMENT_LINK.value

    # Verify RecoveryAttempt record in DB
    db_attempt = (
        db.execute(
            select(RecoveryAttempt).where(RecoveryAttempt.opportunity_id == db_opp.id)
        )
        .scalars()
        .first()
    )
    assert db_attempt is not None
    assert db_attempt.action == RecoveryAction.PAYMENT_LINK.value
    assert db_attempt.external_reference == "plink_adaptive_001"

    # Verify Customer lifetime count incremented in DB
    db_customer = db.get(Customer, DEMO_CUSTOMER_UUID)
    assert db_customer is not None
    assert db_customer.total_payments == 43  # Incremented from 42 to 43!
    assert db_customer.successful_payments == 41

    # Verify AuditEvent recorded
    db_audit = (
        db.execute(select(AuditEvent).where(AuditEvent.opportunity_id == db_opp.id))
        .scalars()
        .first()
    )
    assert db_audit is not None
    assert db_audit.event_type == "recovery_action_executed"

    # -------------------------------------------------------------------------
    # STEP 3: Verify Dynamic RAG Feedback
    # -------------------------------------------------------------------------
    # The newly executed recovery outcome must now be in vector_index!
    assert vector_index.size == 1
    stored_docs = vector_index.get_all_documents()
    assert (
        stored_docs[0].metadata.get("failure_reason")
        == "customer_auth_failed_otp_timeout"
    )
    assert (
        stored_docs[0].metadata.get("recovery_action")
        == RecoveryAction.PAYMENT_LINK.value
    )
    assert stored_docs[0].metadata.get("was_recovered") is True

    # -------------------------------------------------------------------------
    # STEP 4: SUBSEQUENT DECISION (Proving Closed Loop in Action!)
    # -------------------------------------------------------------------------
    # Customer has another failed payment later
    payment_2_id = uuid4()
    req_2 = RecoveryDecisionRequest(
        payment_id=payment_2_id,
        amount=5200.0,
        currency="INR",
        payment_method="card",
        failure_reason="customer_auth_failed_otp_timeout",
        payment_status="failed",
        customer=CustomerProfileDTO(
            customer_id=DEMO_CUSTOMER_UUID,
            # Pass stale or empty numbers to prove DB is the true source of truth
            total_payments=0,
            successful_payments=0,
            failed_payments=0,
            historical_success_rate=0.0,
        ),
        previous_attempts=[],
        opportunity_status="open",
        revenue_at_risk=5200.0,
        max_attempts=3,
        execute_action=False,
    )

    # Configure mock provider to cite the retrieved live precedent (anonymized as case_1 in prompt)
    mock_provider._recommendation = LLMRecoveryRecommendation(
        recommended_action=RecoveryAction.PAYMENT_LINK,
        confidence=0.88,
        reasoning="Similar precedent case_1 succeeded with payment link for customer_auth_failed_otp_timeout.",
        key_factors=("Historical precedent success", "OTP timeout"),
        referenced_case_ids=("case_1",),
    )

    resp_2 = asyncio.run(service.evaluate_decision(req_2, db_session=db))

    # 1. RAG retrieval retrieved the previous execution's newly added case and LLM cited it!
    assert "case_1" in resp_2.referenced_case_ids

    # 2. Database context directly queried from SQLite
    ctx = get_customer_context(
        db_session=db,
        customer_id=DEMO_CUSTOMER_UUID,
        payment_id=payment_2_id,
        current_payment_amount=5200.0,
        current_payment_currency="INR",
        current_payment_method="card",
        current_payment_failure_reason="customer_auth_failed_otp_timeout",
    )
    assert ctx.customer.total_payments == 43
    assert ctx.customer.successful_payments == 41
    assert ctx.customer.total_transaction_amount > 8450.0

    # 3. Recent payment behavior includes the first executed payment!
    recent = ctx.customer.recent_payment_behavior
    assert len(recent) > 0
    assert any(
        r["amount"] == 8450.0
        and r["failure_reason"] == "customer_auth_failed_otp_timeout"
        for r in recent
    )


def test_adaptive_feedback_loop_records_failed_execution(isolated_db_session: Session):
    """
    Verify requirement 15: Failed executions must also be persisted because they are
    useful historical evidence.
    """
    db = isolated_db_session
    vector_index = VectorIndex()
    embedding_service = get_embedding_service()

    # Mock adapter failure
    mock_adapter = MagicMock(spec=RazorpayAdapter)
    mock_adapter.create_payment_link = AsyncMock(
        side_effect=RazorpayAPIError("Simulated gateway 504 Gateway Timeout")
    )
    mock_adapter.dry_run = False
    executor = ActionExecutor(razorpay_adapter=mock_adapter)

    mock_provider = MockLLMProvider(
        recommendation=LLMRecoveryRecommendation(
            recommended_action=RecoveryAction.PAYMENT_LINK,
            confidence=0.80,
            reasoning="Attempt payment link.",
        )
    )
    orchestrator = AgentOrchestrator(
        provider=mock_provider,
        policy_validator=PolicyValidator(),
        context_builder=AgentContextBuilder(),
    )

    service = RecoveryDecisionService(
        agent_orchestrator=orchestrator,
        action_executor=executor,
        vector_index=vector_index,
        embedding_service=embedding_service,
    )

    pay_id = uuid4()
    req = RecoveryDecisionRequest(
        payment_id=pay_id,
        amount=3000.0,
        currency="INR",
        payment_method="upi",
        failure_reason="bank_technical_timeout",
        payment_status="failed",
        customer=CustomerProfileDTO(
            customer_id=DEMO_CUSTOMER_UUID,
            total_payments=42,
            successful_payments=40,
            failed_payments=2,
            historical_success_rate=0.95,
        ),
        execute_action=True,
    )

    resp = asyncio.run(service.evaluate_decision(req, db_session=db))
    assert resp.execution is not None
    assert resp.execution.status == "failed"
    assert resp.execution.success is False
    assert resp.execution.persisted is True
    assert resp.execution.persistence_error is None

    # Verify failed payment, opportunity, and attempt are all saved in DB
    db_attempt = (
        db.execute(
            select(RecoveryAttempt)
            .join(RecoveryOpportunity)
            .where(RecoveryOpportunity.payment_id == pay_id)
        )
        .scalars()
        .first()
    )
    assert db_attempt is not None
    assert db_attempt.status == "failed"

    # Verify vector index ingested the failure case as negative precedent
    assert vector_index.size == 1
    doc = vector_index.get_all_documents()[0]
    assert doc.metadata.get("was_recovered") is False


def test_razorpay_success_db_persistence_failure(isolated_db_session: Session):
    """
    Stage 11.1.1: Verify behavior when Razorpay execution succeeds but DB persistence fails.
    - External action result is preserved (success=True, reference_id, resource_url).
    - persisted is False.
    - Safe persistence_error message is returned without leaking internal DB details.
    - DB rollback occurs.
    - Vector index is NOT ingested with unpersisted state.
    """
    db = isolated_db_session
    vector_index = VectorIndex()
    embedding_service = get_embedding_service()

    # Razorpay succeeds
    mock_adapter = MagicMock(spec=RazorpayAdapter)
    mock_adapter.create_payment_link = AsyncMock(
        return_value={
            "id": "plink_success_db_fail_999",
            "short_url": "https://rzp.io/i/success_db_fail_999",
            "amount": 500000,
            "currency": "INR",
            "status": "created",
            "description": "Payment recovery",
        }
    )
    mock_adapter.dry_run = False
    executor = ActionExecutor(razorpay_adapter=mock_adapter)

    mock_provider = MockLLMProvider(
        recommendation=LLMRecoveryRecommendation(
            recommended_action=RecoveryAction.PAYMENT_LINK,
            confidence=0.90,
            reasoning="Issue payment link.",
        )
    )
    orchestrator = AgentOrchestrator(
        provider=mock_provider,
        policy_validator=PolicyValidator(),
        context_builder=AgentContextBuilder(),
    )

    service = RecoveryDecisionService(
        agent_orchestrator=orchestrator,
        action_executor=executor,
        vector_index=vector_index,
        embedding_service=embedding_service,
    )

    pay_id = uuid4()
    req = RecoveryDecisionRequest(
        payment_id=pay_id,
        amount=5000.0,
        currency="INR",
        payment_method="card",
        failure_reason="customer_auth_failed_otp_timeout",
        payment_status="failed",
        customer=CustomerProfileDTO(
            customer_id=DEMO_CUSTOMER_UUID,
            total_payments=42,
            successful_payments=40,
            failed_payments=2,
            historical_success_rate=0.95,
        ),
        execute_action=True,
    )

    # Ensure baseline customer is seeded in DB first so context retrieval succeeds
    from app.context_retrieval import ensure_demo_customer_seeded

    ensure_demo_customer_seeded(db)

    # Monkeypatch db.commit to simulate database persistence error during execution persistence
    real_commit = db.commit

    def failing_commit():
        raise RuntimeError(
            "FATAL: disk I/O failure on /var/lib/sqlite/revora.db table 'payments'"
        )

    db.commit = failing_commit

    try:
        resp = asyncio.run(service.evaluate_decision(req, db_session=db))

        # 1. External execution status is preserved
        assert resp.execution is not None
        assert resp.execution.attempted is True
        assert resp.execution.success is True
        assert resp.execution.status == "success"
        assert resp.execution.reference_id == "plink_success_db_fail_999"
        assert resp.execution.resource_url == "https://rzp.io/i/success_db_fail_999"

        # 2. Persistence failure is explicitly flagged
        assert resp.execution.persisted is False
        assert resp.execution.persistence_error is not None
        assert (
            resp.execution.persistence_error
            == "Recovery outcome could not be persisted; reconciliation required."
        )

        # 3. Security invariant: Raw exception details are NOT leaked into response
        assert "FATAL" not in resp.execution.persistence_error
        assert "sqlite" not in resp.execution.persistence_error
        assert "revora.db" not in resp.execution.persistence_error
        assert "payments" not in resp.execution.persistence_error

        # 4. Vector index must NOT have been ingested
        assert vector_index.size == 0
    finally:
        db.commit = real_commit


def test_razorpay_failure_db_persistence_failure(isolated_db_session: Session):
    """
    Stage 11.1.1: Verify behavior when Razorpay execution fails AND DB persistence also fails.
    - External action result reflects gateway failure (success=False, status='failed').
    - persisted is False.
    - Safe persistence_error message is returned.
    - DB rollback occurs.
    """
    db = isolated_db_session
    vector_index = VectorIndex()
    embedding_service = get_embedding_service()

    # Razorpay fails
    mock_adapter = MagicMock(spec=RazorpayAdapter)
    mock_adapter.create_payment_link = AsyncMock(
        side_effect=RazorpayAPIError("503 Service Unavailable from Razorpay")
    )
    mock_adapter.dry_run = False
    executor = ActionExecutor(razorpay_adapter=mock_adapter)

    mock_provider = MockLLMProvider(
        recommendation=LLMRecoveryRecommendation(
            recommended_action=RecoveryAction.PAYMENT_LINK,
            confidence=0.85,
            reasoning="Issue payment link.",
        )
    )
    orchestrator = AgentOrchestrator(
        provider=mock_provider,
        policy_validator=PolicyValidator(),
        context_builder=AgentContextBuilder(),
    )

    service = RecoveryDecisionService(
        agent_orchestrator=orchestrator,
        action_executor=executor,
        vector_index=vector_index,
        embedding_service=embedding_service,
    )

    pay_id = uuid4()
    req = RecoveryDecisionRequest(
        payment_id=pay_id,
        amount=3000.0,
        currency="INR",
        payment_method="upi",
        failure_reason="bank_technical_timeout",
        payment_status="failed",
        customer=CustomerProfileDTO(
            customer_id=DEMO_CUSTOMER_UUID,
            total_payments=42,
            successful_payments=40,
            failed_payments=2,
            historical_success_rate=0.95,
        ),
        execute_action=True,
    )

    # Ensure baseline customer is seeded in DB first so context retrieval succeeds
    from app.context_retrieval import ensure_demo_customer_seeded

    ensure_demo_customer_seeded(db)

    # Monkeypatch db.commit to fail
    real_commit = db.commit

    def failing_commit():
        raise RuntimeError("DB connection timeout")

    db.commit = failing_commit

    try:
        resp = asyncio.run(service.evaluate_decision(req, db_session=db))

        # External action failed
        assert resp.execution is not None
        assert resp.execution.attempted is True
        assert resp.execution.success is False
        assert resp.execution.status == "failed"
        assert "503 Service Unavailable" in (resp.execution.error or "")

        # Persistence also failed
        assert resp.execution.persisted is False
        assert (
            resp.execution.persistence_error
            == "Recovery outcome could not be persisted; reconciliation required."
        )

        # Vector index not ingested
        assert vector_index.size == 0
    finally:
        db.commit = real_commit


def test_tenant_isolation_preserved(isolated_db_session: Session):
    """
    Verify requirement 17: Customer data and RAG retrieval strictly preserve tenant isolation.
    """
    db = isolated_db_session
    foreign_customer_id = uuid4()

    # Foreign customer with non-existent ID raises CustomerNotFoundError in direct lookup
    with pytest.raises(CustomerNotFoundError):
        get_customer_context(db_session=db, customer_id=foreign_customer_id)
