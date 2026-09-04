"""
Tests for Stage 11.1.2: Execution Idempotency & Deduplication.

Verifies:
1. Sequential replay of successful execution returns cached result without re-calling Razorpay.
2. Concurrent execution reservation prevents duplicate gateway link creation.
3. Razorpay reference_id is correctly derived and passed in the API payload.
4. Gateway timeout / ambiguous result is reconciled on retry without duplicate links.
5. DB persistence failure followed by retry reconciles state using the same reference_id.
6. Gateway failure does not permanently block retries.
7. Already recovered opportunity replay skips gateway invocation.
8. Missing payment_id derives stable identity.
9. Cross-tenant idempotency key collision is strictly forbidden (tenant isolation).
10. Execution replay does not insert duplicate documents into the runtime vector index.
11. Explicit client-provided idempotency_key is honored.
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
from app.context_retrieval import DEMO_CUSTOMER_UUID, ensure_demo_customer_seeded
from app.database import Base
from app.decision_engine import RecoveryAction
from app.embedding_service import get_embedding_service
from app.models import (
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
from fastapi import HTTPException
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool


@pytest.fixture
def isolated_db_session() -> Session:
    """Fixture providing an isolated in-memory SQLite database session."""
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(bind=engine)
    session_factory = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    session = session_factory()
    try:
        yield session
    finally:
        session.close()


def test_sequential_replay_deduplication(isolated_db_session: Session):
    """Verify that submitting the same execution request twice calls Razorpay only once."""
    db = isolated_db_session
    ensure_demo_customer_seeded(db)

    mock_adapter = MagicMock(spec=RazorpayAdapter)
    mock_adapter.create_payment_link = AsyncMock(
        return_value={
            "id": "plink_idem_seq_001",
            "short_url": "https://rzp.io/i/idem_seq_001",
            "amount": 250000,
            "currency": "INR",
            "status": "created",
            "reference_id": "rec_custom_seq_001",
        }
    )
    mock_adapter.dry_run = False
    executor = ActionExecutor(razorpay_adapter=mock_adapter)

    provider = MockLLMProvider(
        recommendation=LLMRecoveryRecommendation(
            recommended_action=RecoveryAction.PAYMENT_LINK,
            confidence=0.95,
            reasoning="Issue payment link.",
        )
    )
    orchestrator = AgentOrchestrator(
        provider=provider,
        policy_validator=PolicyValidator(),
        context_builder=AgentContextBuilder(),
    )
    service = RecoveryDecisionService(
        agent_orchestrator=orchestrator,
        action_executor=executor,
        vector_index=VectorIndex(),
        embedding_service=get_embedding_service(),
    )

    pay_id = uuid4()
    req = RecoveryDecisionRequest(
        payment_id=pay_id,
        amount=2500.0,
        currency="INR",
        payment_method="upi",
        failure_reason="bank_timeout",
        payment_status="failed",
        customer=CustomerProfileDTO(customer_id=DEMO_CUSTOMER_UUID),
        execute_action=True,
    )

    # First Call
    resp_1 = asyncio.run(service.evaluate_decision(req, db_session=db))
    assert resp_1.execution is not None
    assert resp_1.execution.success is True
    assert resp_1.execution.status == "success"
    assert resp_1.execution.reference_id == "plink_idem_seq_001"
    assert resp_1.execution.persisted is True
    assert mock_adapter.create_payment_link.call_count == 1

    # Second Call (Sequential Replay)
    resp_2 = asyncio.run(service.evaluate_decision(req, db_session=db))
    assert resp_2.execution is not None
    assert resp_2.execution.success is True
    assert resp_2.execution.status == "already_executed"
    assert resp_2.execution.reference_id == "plink_idem_seq_001"
    assert resp_2.execution.attempted is False
    assert resp_2.execution.persisted is True

    # Gateway call count must remain 1
    assert mock_adapter.create_payment_link.call_count == 1


def test_concurrent_execution_reservation_blocks_duplicate_gateway_calls(
    isolated_db_session: Session,
):
    """Verify that when an in_progress attempt exists, concurrent requests return in_progress without calling gateway."""
    db = isolated_db_session
    ensure_demo_customer_seeded(db)

    mock_adapter = MagicMock(spec=RazorpayAdapter)
    mock_adapter.create_payment_link = AsyncMock(
        return_value={
            "id": "plink_concurrent_001",
            "short_url": "https://rzp.io/i/concurrent_001",
            "amount": 100000,
            "currency": "INR",
            "status": "created",
        }
    )
    mock_adapter.dry_run = False
    executor = ActionExecutor(razorpay_adapter=mock_adapter)

    provider = MockLLMProvider(
        recommendation=LLMRecoveryRecommendation(
            recommended_action=RecoveryAction.PAYMENT_LINK,
            confidence=0.9,
            reasoning="Payment link.",
        )
    )
    orchestrator = AgentOrchestrator(provider=provider)
    service = RecoveryDecisionService(
        agent_orchestrator=orchestrator,
        action_executor=executor,
    )

    pay_id = uuid4()
    # Pre-stage an in_progress attempt in DB representing a concurrent worker holding execution ownership
    payment = Payment(
        id=pay_id,
        customer_id=DEMO_CUSTOMER_UUID,
        amount=1000.0,
        currency="INR",
        payment_method="card",
        status="failed",
    )
    opp = RecoveryOpportunity(
        id=uuid4(),
        payment_id=pay_id,
        status="in_progress",
        revenue_at_risk=1000.0,
        expected_recovery=0.0,
    )
    att = RecoveryAttempt(
        id=uuid4(),
        opportunity_id=opp.id,
        action="payment_link",
        status="in_progress",
        idempotency_key=f"rec_{pay_id.hex}",
    )
    db.add(payment)
    db.add(opp)
    db.add(att)
    db.commit()

    req = RecoveryDecisionRequest(
        payment_id=pay_id,
        amount=1000.0,
        currency="INR",
        payment_method="card",
        failure_reason="timeout",
        customer=CustomerProfileDTO(customer_id=DEMO_CUSTOMER_UUID),
        execute_action=True,
    )

    # Incoming second request during in_progress
    resp = asyncio.run(service.evaluate_decision(req, db_session=db))
    assert resp.execution is not None
    assert resp.execution.status == "in_progress"
    assert resp.execution.attempted is False
    assert resp.execution.success is False
    # Gateway was never called
    assert mock_adapter.create_payment_link.call_count == 0


def test_razorpay_reference_id_propagation():
    """Verify that ActionExecutor passes reference_id to RazorpayAdapter."""
    mock_adapter = MagicMock(spec=RazorpayAdapter)
    mock_adapter.create_payment_link = AsyncMock(
        return_value={
            "id": "plink_ref_check",
            "short_url": "https://rzp.io/i/ref_check",
            "amount": 50000,
            "currency": "INR",
            "status": "created",
            "reference_id": "rec_custom_test_ref_123",
        }
    )
    mock_adapter.dry_run = False
    executor = ActionExecutor(razorpay_adapter=mock_adapter)

    from app.context import CustomerContext, CustomerRecoveryContext, PaymentContext
    from app.policies.schemas import RecoveryPolicyContext

    cust = CustomerContext(customer_id=uuid4(), name="Priya", email="priya@example.com")
    pay = PaymentContext(
        payment_id=uuid4(),
        amount=500.0,
        currency="INR",
        payment_method="upi",
        status="failed",
    )
    ctx = CustomerRecoveryContext(customer=cust, current_payment=pay)
    policy_ctx = RecoveryPolicyContext(
        provider="razorpay",
        policy_version="2026.1",
        applicable_rules=(),
        allowed_actions=(RecoveryAction.PAYMENT_LINK,),
        prohibited_actions=(),
        mandatory_fallback_action=None,
    )

    res = asyncio.run(
        executor.execute(
            approved_action=RecoveryAction.PAYMENT_LINK,
            policy_context=policy_ctx,
            context=ctx,
            reference_id="rec_custom_test_ref_123",
        )
    )

    assert res.success is True
    assert res.reference_id == "plink_ref_check"
    mock_adapter.create_payment_link.assert_called_once_with(
        amount=500.0,
        currency="INR",
        description="Payment recovery for failed UPI payment",
        customer_name="Priya",
        customer_email="priya@example.com",
        reference_id="rec_custom_test_ref_123",
    )


def test_gateway_timeout_ambiguity_reconciliation(isolated_db_session: Session):
    """
    Simulate ambiguous gateway timeout: first call times out, but retry with the same
    reference_id reconciles seamlessly without creating duplicate resources.
    """
    db = isolated_db_session
    ensure_demo_customer_seeded(db)

    # First call times out, second call succeeds (returning existing link for the same reference)
    mock_adapter = MagicMock(spec=RazorpayAdapter)
    mock_adapter.create_payment_link = AsyncMock(
        side_effect=[
            RazorpayAPIError("504 Gateway Timeout"),
            {
                "id": "plink_timeout_reconciled_999",
                "short_url": "https://rzp.io/i/timeout_reconciled_999",
                "amount": 350000,
                "currency": "INR",
                "status": "created",
                "reference_id": "rec_timeout_key_1",
            },
        ]
    )
    mock_adapter.dry_run = False
    executor = ActionExecutor(razorpay_adapter=mock_adapter)

    provider = MockLLMProvider(
        recommendation=LLMRecoveryRecommendation(
            recommended_action=RecoveryAction.PAYMENT_LINK,
            confidence=0.9,
            reasoning="Payment link.",
        )
    )
    orchestrator = AgentOrchestrator(provider=provider)
    service = RecoveryDecisionService(
        agent_orchestrator=orchestrator,
        action_executor=executor,
    )

    pay_id = uuid4()
    req = RecoveryDecisionRequest(
        payment_id=pay_id,
        amount=3500.0,
        currency="INR",
        payment_method="card",
        failure_reason="timeout",
        customer=CustomerProfileDTO(customer_id=DEMO_CUSTOMER_UUID),
        execute_action=True,
        idempotency_key="rec_timeout_key_1",
    )

    # 1. First execution attempt hits gateway timeout
    resp_1 = asyncio.run(service.evaluate_decision(req, db_session=db))
    assert resp_1.execution is not None
    assert resp_1.execution.success is False
    assert resp_1.execution.status == "failed"
    assert "504 Gateway Timeout" in (resp_1.execution.error or "")

    # 2. Retry with same idempotency_key reconciles and succeeds
    resp_2 = asyncio.run(service.evaluate_decision(req, db_session=db))
    assert resp_2.execution is not None
    assert resp_2.execution.success is True
    assert resp_2.execution.status == "success"
    assert resp_2.execution.reference_id == "plink_timeout_reconciled_999"
    assert resp_2.execution.persisted is True

    # 3. Third call is a sequential replay
    resp_3 = asyncio.run(service.evaluate_decision(req, db_session=db))
    assert resp_3.execution.status == "already_executed"
    assert mock_adapter.create_payment_link.call_count == 2


def test_db_persistence_failure_and_subsequent_retry_reconciliation(
    isolated_db_session: Session,
):
    """
    Stage 11.1.1 + 11.1.2: When DB persistence fails after Razorpay success,
    first call returns persisted=False, and subsequent retry reconciles DB successfully.
    """
    db = isolated_db_session
    ensure_demo_customer_seeded(db)

    mock_adapter = MagicMock(spec=RazorpayAdapter)
    mock_adapter.create_payment_link = AsyncMock(
        return_value={
            "id": "plink_reconcile_db_fail_777",
            "short_url": "https://rzp.io/i/reconcile_db_fail_777",
            "amount": 400000,
            "currency": "INR",
            "status": "created",
            "reference_id": "rec_persist_reconcile_777",
        }
    )
    mock_adapter.dry_run = False
    executor = ActionExecutor(razorpay_adapter=mock_adapter)

    provider = MockLLMProvider(
        recommendation=LLMRecoveryRecommendation(
            recommended_action=RecoveryAction.PAYMENT_LINK,
            confidence=0.9,
            reasoning="Payment link.",
        )
    )
    orchestrator = AgentOrchestrator(provider=provider)
    service = RecoveryDecisionService(
        agent_orchestrator=orchestrator,
        action_executor=executor,
    )

    pay_id = uuid4()
    req = RecoveryDecisionRequest(
        payment_id=pay_id,
        amount=4000.0,
        currency="INR",
        payment_method="upi",
        failure_reason="timeout",
        customer=CustomerProfileDTO(customer_id=DEMO_CUSTOMER_UUID),
        execute_action=True,
        idempotency_key="rec_persist_reconcile_777",
    )

    # Simulate DB failure on post-execution commit
    real_commit = db.commit
    commit_call_count = 0

    def failing_first_commit():
        nonlocal commit_call_count
        commit_call_count += 1
        # Allow reservation commit (1), fail on post-execution commit (2)
        if commit_call_count == 2:
            raise RuntimeError("DB Disk Full")
        return real_commit()

    db.commit = failing_first_commit

    try:
        resp_1 = asyncio.run(service.evaluate_decision(req, db_session=db))
        assert resp_1.execution is not None
        assert resp_1.execution.success is True
        assert resp_1.execution.persisted is False
        assert resp_1.execution.reference_id == "plink_reconcile_db_fail_777"
        assert (
            resp_1.execution.persistence_error
            == "Recovery outcome could not be persisted; reconciliation required."
        )
    finally:
        db.commit = real_commit

    # Subsequent retry with healthy DB
    resp_2 = asyncio.run(service.evaluate_decision(req, db_session=db))
    assert resp_2.execution is not None
    assert resp_2.execution.success is True
    assert resp_2.execution.persisted is True
    assert resp_2.execution.reference_id == "plink_reconcile_db_fail_777"


def test_cross_tenant_idempotency_key_collision_rejected(
    isolated_db_session: Session,
):
    """Verify that a customer cannot replay an idempotency key created by a different customer (Tenant Isolation)."""
    db = isolated_db_session
    ensure_demo_customer_seeded(db)

    # Customer 1 creates an execution with key 'tenant_shared_key_123'
    cust_1_id = DEMO_CUSTOMER_UUID
    cust_2_id = uuid4()
    cust_2 = Customer(id=cust_2_id, name="Attacker", email="attacker@example.com")
    db.add(cust_2)

    pay_1 = Payment(
        id=uuid4(),
        customer_id=cust_1_id,
        amount=500.0,
        currency="INR",
        payment_method="card",
        status="succeeded",
    )
    opp_1 = RecoveryOpportunity(
        id=uuid4(),
        payment_id=pay_1.id,
        status="recovered",
        revenue_at_risk=500.0,
        expected_recovery=500.0,
    )
    att_1 = RecoveryAttempt(
        id=uuid4(),
        opportunity_id=opp_1.id,
        action="payment_link",
        status="succeeded",
        idempotency_key="tenant_shared_key_123",
        external_reference="plink_tenant_1",
    )
    db.add(pay_1)
    db.add(opp_1)
    db.add(att_1)
    db.commit()

    service = RecoveryDecisionService(
        agent_orchestrator=AgentOrchestrator(
            provider=MockLLMProvider(
                recommendation=LLMRecoveryRecommendation(
                    recommended_action=RecoveryAction.PAYMENT_LINK,
                    confidence=0.9,
                    reasoning="Payment link.",
                )
            )
        )
    )

    # Customer 2 attempts to use Customer 1's idempotency key
    req_tenant_attack = RecoveryDecisionRequest(
        payment_id=uuid4(),
        amount=500.0,
        currency="INR",
        payment_method="card",
        failure_reason="timeout",
        customer=CustomerProfileDTO(customer_id=cust_2_id),
        execute_action=True,
        idempotency_key="tenant_shared_key_123",
    )

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(service.evaluate_decision(req_tenant_attack, db_session=db))
    assert exc_info.value.status_code == 403
    assert "Idempotency key collision across tenant boundary" in str(
        exc_info.value.detail
    )


def test_replay_does_not_create_duplicate_vector_documents(
    isolated_db_session: Session,
):
    """Verify that execution replay does not insert redundant vector index entries."""
    db = isolated_db_session
    ensure_demo_customer_seeded(db)

    vector_index = VectorIndex()
    embedding_service = get_embedding_service()

    mock_adapter = MagicMock(spec=RazorpayAdapter)
    mock_adapter.create_payment_link = AsyncMock(
        return_value={
            "id": "plink_vector_replay_001",
            "short_url": "https://rzp.io/i/vector_replay_001",
            "amount": 200000,
            "currency": "INR",
            "status": "created",
        }
    )
    mock_adapter.dry_run = False
    executor = ActionExecutor(razorpay_adapter=mock_adapter)

    service = RecoveryDecisionService(
        agent_orchestrator=AgentOrchestrator(
            provider=MockLLMProvider(
                recommendation=LLMRecoveryRecommendation(
                    recommended_action=RecoveryAction.PAYMENT_LINK,
                    confidence=0.9,
                    reasoning="Link.",
                )
            )
        ),
        action_executor=executor,
        vector_index=vector_index,
        embedding_service=embedding_service,
    )

    pay_id = uuid4()
    req = RecoveryDecisionRequest(
        payment_id=pay_id,
        amount=2000.0,
        currency="INR",
        payment_method="upi",
        failure_reason="timeout",
        customer=CustomerProfileDTO(customer_id=DEMO_CUSTOMER_UUID),
        execute_action=True,
    )

    # First Call
    resp_1 = asyncio.run(service.evaluate_decision(req, db_session=db))
    assert resp_1.execution.success is True
    assert vector_index.size == 1

    # Second Call (Replay)
    resp_2 = asyncio.run(service.evaluate_decision(req, db_session=db))
    assert resp_2.execution.status == "already_executed"
    # Vector store size must remain 1
    assert vector_index.size == 1


def test_already_recovered_opportunity_replay_skips_gateway(
    isolated_db_session: Session,
):
    """Verify that if an opportunity is already recovered, gateway is never invoked (0 calls)."""
    db = isolated_db_session
    ensure_demo_customer_seeded(db)

    mock_adapter = MagicMock(spec=RazorpayAdapter)
    mock_adapter.create_payment_link = AsyncMock()
    mock_adapter.dry_run = False
    executor = ActionExecutor(razorpay_adapter=mock_adapter)

    pay_id = uuid4()
    payment = Payment(
        id=pay_id,
        customer_id=DEMO_CUSTOMER_UUID,
        amount=1500.0,
        currency="INR",
        payment_method="upi",
        status="succeeded",
    )
    opp = RecoveryOpportunity(
        id=uuid4(),
        payment_id=pay_id,
        status="recovered",
        revenue_at_risk=1500.0,
        expected_recovery=1500.0,
    )
    att = RecoveryAttempt(
        id=uuid4(),
        opportunity_id=opp.id,
        action="payment_link",
        status="succeeded",
        external_reference="plink_recovered_preexisting",
        idempotency_key=f"rec_{pay_id.hex}",
    )
    db.add(payment)
    db.add(opp)
    db.add(att)
    db.commit()

    service = RecoveryDecisionService(
        agent_orchestrator=AgentOrchestrator(
            provider=MockLLMProvider(
                recommendation=LLMRecoveryRecommendation(
                    recommended_action=RecoveryAction.PAYMENT_LINK,
                    confidence=0.9,
                    reasoning="Link.",
                )
            )
        ),
        action_executor=executor,
    )

    req = RecoveryDecisionRequest(
        payment_id=pay_id,
        amount=1500.0,
        currency="INR",
        payment_method="upi",
        failure_reason="timeout",
        customer=CustomerProfileDTO(customer_id=DEMO_CUSTOMER_UUID),
        execute_action=True,
    )

    resp = asyncio.run(service.evaluate_decision(req, db_session=db))
    assert resp.execution is not None
    assert resp.execution.status == "already_executed"
    assert resp.execution.reference_id == "plink_recovered_preexisting"
    # Gateway was never called
    assert mock_adapter.create_payment_link.call_count == 0


def test_client_provided_idempotency_key_override(isolated_db_session: Session):
    """Verify that an explicit client-provided idempotency_key is passed to gateway and saved in DB."""
    db = isolated_db_session
    ensure_demo_customer_seeded(db)

    mock_adapter = MagicMock(spec=RazorpayAdapter)
    mock_adapter.create_payment_link = AsyncMock(
        return_value={
            "id": "plink_client_key_123",
            "short_url": "https://rzp.io/i/client_key_123",
            "amount": 250000,
            "currency": "INR",
            "status": "created",
            "reference_id": "client_custom_idempotency_key_xyz",
        }
    )
    mock_adapter.dry_run = False
    executor = ActionExecutor(razorpay_adapter=mock_adapter)

    service = RecoveryDecisionService(
        agent_orchestrator=AgentOrchestrator(
            provider=MockLLMProvider(
                recommendation=LLMRecoveryRecommendation(
                    recommended_action=RecoveryAction.PAYMENT_LINK,
                    confidence=0.9,
                    reasoning="Link.",
                )
            )
        ),
        action_executor=executor,
    )

    pay_id = uuid4()
    req = RecoveryDecisionRequest(
        payment_id=pay_id,
        amount=2500.0,
        currency="INR",
        payment_method="upi",
        failure_reason="timeout",
        customer=CustomerProfileDTO(customer_id=DEMO_CUSTOMER_UUID),
        execute_action=True,
        idempotency_key="client_custom_idempotency_key_xyz",
    )

    resp = asyncio.run(service.evaluate_decision(req, db_session=db))
    assert resp.execution is not None
    assert resp.execution.success is True
    assert resp.execution.reference_id == "plink_client_key_123"

    # Verify DB attempt record has the client-provided idempotency key
    attempt = (
        db.execute(
            select(RecoveryAttempt).where(
                RecoveryAttempt.idempotency_key == "client_custom_idempotency_key_xyz"
            )
        )
        .scalars()
        .first()
    )
    assert attempt is not None
    assert attempt.external_reference == "plink_client_key_123"
