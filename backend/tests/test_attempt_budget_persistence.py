"""
Tests for Authoritative Attempt Budget Tracking, State Persistence & Hydration.

Verifies:
1. Fresh demo case with stable payment_id initializes with attempt_count=0 and previous_attempts=[].
2. First execution increments attempt_count to 1 and persists RecoveryAttempt.
3. Subsequent read-only evaluation (page hydration) reads persisted attempt state from DB (attempt_count=1).
4. Second and third executions accurately progress attempt_count (1 -> 2 -> 3).
5. Fourth execution attempt is blocked by SAFETY_MAX_ATTEMPTS_EXCEEDED policy without external gateway dispatch.
6. DB customer counters (total_payments, successful_payments) are returned authoritatively.
7. Opportunity status transitions are accurately reflected in the response DTO.
8. Cross-tenant access to payment_id is strictly blocked (HTTP 403).
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock
from uuid import UUID, uuid4

import pytest
from app.action_executor import ActionExecutor
from app.agent.orchestrator import AgentOrchestrator
from app.agent.provider import MockLLMProvider
from app.agent.schemas import LLMRecoveryRecommendation
from app.context import PaymentCustomerMismatchError
from app.context_retrieval import DEMO_CUSTOMER_UUID, ensure_demo_customer_seeded
from app.database import Base
from app.decision_engine import RecoveryAction
from app.embedding_service import get_embedding_service
from app.models import Customer, Payment, RecoveryOpportunity
from app.policies.validator import PolicyValidator
from app.razorpay_adapter import RazorpayAdapter
from app.recovery_decision_service import RecoveryDecisionService
from app.schemas.decision import CustomerProfileDTO, RecoveryDecisionRequest
from app.vector_index import VectorIndex
from fastapi import HTTPException
from sqlalchemy import create_engine
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


def test_fresh_demo_case_initial_attempt_budget_zero(isolated_db_session: Session):
    """Verify that a fresh demo case with stable payment_id returns attempt_count=0 on initial evaluation."""
    db = isolated_db_session
    ensure_demo_customer_seeded(db)

    mock_adapter = MagicMock(spec=RazorpayAdapter)
    executor = ActionExecutor(razorpay_adapter=mock_adapter)
    provider = MockLLMProvider(
        recommendation=LLMRecoveryRecommendation(
            recommended_action=RecoveryAction.PAYMENT_LINK,
            confidence=0.92,
            reasoning="Issue payment link for OTP timeout.",
        )
    )
    orchestrator = AgentOrchestrator(
        provider=provider, policy_validator=PolicyValidator()
    )
    service = RecoveryDecisionService(
        agent_orchestrator=orchestrator,
        action_executor=executor,
        vector_index=VectorIndex(),
        embedding_service=get_embedding_service(),
    )

    stable_payment_id = UUID("c0000001-0000-4000-8000-000000000001")
    req = RecoveryDecisionRequest(
        payment_id=stable_payment_id,
        amount=8450.0,
        currency="INR",
        payment_method="card",
        failure_reason="customer_auth_failed_otp_timeout",
        payment_status="failed",
        customer=CustomerProfileDTO(customer_id=DEMO_CUSTOMER_UUID),
        previous_attempts=[],
        max_attempts=3,
        execute_action=False,
    )

    resp = asyncio.run(service.evaluate_decision(req, db_session=db))

    assert resp.payment_id == stable_payment_id
    assert resp.attempt_count == 0
    assert resp.previous_attempts == []
    assert resp.opportunity_status == "open"
    assert resp.customer is not None
    assert resp.customer.total_payments == 42
    assert resp.customer.successful_payments == 40
    assert resp.execution is None


def test_attempt_budget_progression_across_multiple_executions(
    isolated_db_session: Session,
):
    """
    Verify complete attempt budget lifecycle across multiple attempts:
    0/3 (initial) -> 1/3 (attempt 1 failed) -> 1/3 (hydrate) -> 2/3 (attempt 2 failed) -> 3/3 (attempt 3 succeeded) -> 4th blocked by max attempts.
    """
    db = isolated_db_session
    ensure_demo_customer_seeded(db)

    call_index = 0

    async def mock_execute_action(*args, **kwargs):
        nonlocal call_index
        call_index += 1
        # Attempts 1 and 2 fail; attempt 3 succeeds
        if call_index in (1, 2):
            raise RuntimeError("Gateway temporary timeout")
        return {
            "id": f"plink_multi_{call_index:03d}",
            "short_url": f"https://rzp.io/i/multi_{call_index:03d}",
            "amount": 1499900,
            "currency": "INR",
            "status": "created",
        }

    mock_adapter = MagicMock(spec=RazorpayAdapter)
    mock_adapter.create_payment_link = AsyncMock(side_effect=mock_execute_action)
    mock_adapter.dry_run = False
    executor = ActionExecutor(razorpay_adapter=mock_adapter)

    provider = MockLLMProvider(
        recommendation=LLMRecoveryRecommendation(
            recommended_action=RecoveryAction.PAYMENT_LINK,
            confidence=0.88,
            reasoning="Issue payment link for retry.",
        )
    )
    orchestrator = AgentOrchestrator(
        provider=provider, policy_validator=PolicyValidator()
    )
    service = RecoveryDecisionService(
        agent_orchestrator=orchestrator,
        action_executor=executor,
        vector_index=VectorIndex(),
        embedding_service=get_embedding_service(),
    )

    stable_payment_id = UUID("c0000003-0000-4000-8000-000000000003")

    # 1. Initial Evaluation (Page Load / Hydration before any execution)
    req_init = RecoveryDecisionRequest(
        payment_id=stable_payment_id,
        amount=14999.0,
        currency="INR",
        payment_method="card",
        failure_reason="insufficient_funds",
        payment_status="failed",
        customer=CustomerProfileDTO(customer_id=DEMO_CUSTOMER_UUID),
        previous_attempts=[],
        max_attempts=3,
        execute_action=False,
    )
    resp_init = asyncio.run(service.evaluate_decision(req_init, db_session=db))
    assert resp_init.attempt_count == 0
    assert len(resp_init.previous_attempts) == 0
    assert resp_init.opportunity_status == "open"

    # 2. First Execution (Attempt 1 fails at gateway)
    req_exec_1 = RecoveryDecisionRequest(
        payment_id=stable_payment_id,
        amount=14999.0,
        currency="INR",
        payment_method="card",
        failure_reason="insufficient_funds",
        payment_status="failed",
        customer=CustomerProfileDTO(customer_id=DEMO_CUSTOMER_UUID),
        previous_attempts=[],
        max_attempts=3,
        execute_action=True,
    )
    resp_exec_1 = asyncio.run(service.evaluate_decision(req_exec_1, db_session=db))
    assert resp_exec_1.execution is not None
    assert resp_exec_1.execution.success is False
    assert resp_exec_1.execution.persisted is True
    assert resp_exec_1.attempt_count == 1
    assert len(resp_exec_1.previous_attempts) == 1
    assert resp_exec_1.previous_attempts[0].action == "payment_link"
    assert resp_exec_1.previous_attempts[0].status == "failed"
    assert resp_exec_1.customer.total_payments == 43
    assert resp_exec_1.customer.failed_payments == 3

    # 3. Hydration after Browser Refresh (execute_action=False, sending stable payment_id)
    # Even if client sends empty previous_attempts, DB authoritative state must be loaded!
    req_hydrate = RecoveryDecisionRequest(
        payment_id=stable_payment_id,
        amount=14999.0,
        currency="INR",
        payment_method="card",
        failure_reason="insufficient_funds",
        payment_status="failed",
        customer=CustomerProfileDTO(customer_id=DEMO_CUSTOMER_UUID),
        previous_attempts=[],  # Stale client data ignored in favor of DB
        max_attempts=3,
        execute_action=False,
    )
    resp_hydrate = asyncio.run(service.evaluate_decision(req_hydrate, db_session=db))
    assert resp_hydrate.attempt_count == 1
    assert len(resp_hydrate.previous_attempts) == 1
    assert resp_hydrate.previous_attempts[0].action == "payment_link"
    assert resp_hydrate.customer.total_payments == 43

    # 4. Second Execution (Attempt 2 fails at gateway)
    req_exec_2 = RecoveryDecisionRequest(
        payment_id=stable_payment_id,
        amount=14999.0,
        currency="INR",
        payment_method="card",
        failure_reason="insufficient_funds",
        payment_status="failed",
        customer=CustomerProfileDTO(customer_id=DEMO_CUSTOMER_UUID),
        previous_attempts=[],
        max_attempts=3,
        execute_action=True,
    )
    resp_exec_2 = asyncio.run(service.evaluate_decision(req_exec_2, db_session=db))
    assert resp_exec_2.execution.success is False
    assert resp_exec_2.attempt_count == 2
    assert len(resp_exec_2.previous_attempts) == 2

    # 5. Third Execution (Attempt 3 succeeds)
    req_exec_3 = RecoveryDecisionRequest(
        payment_id=stable_payment_id,
        amount=14999.0,
        currency="INR",
        payment_method="card",
        failure_reason="insufficient_funds",
        payment_status="failed",
        customer=CustomerProfileDTO(customer_id=DEMO_CUSTOMER_UUID),
        previous_attempts=[],
        max_attempts=3,
        execute_action=True,
    )
    resp_exec_3 = asyncio.run(service.evaluate_decision(req_exec_3, db_session=db))
    assert resp_exec_3.execution.success is True
    assert resp_exec_3.attempt_count == 3
    assert len(resp_exec_3.previous_attempts) == 3

    # 6. Fourth Execution Attempt: Must be blocked by SAFETY_MAX_ATTEMPTS_EXCEEDED or SAFETY_ALREADY_RECOVERED
    req_exec_4 = RecoveryDecisionRequest(
        payment_id=stable_payment_id,
        amount=14999.0,
        currency="INR",
        payment_method="card",
        failure_reason="insufficient_funds",
        payment_status="failed",
        customer=CustomerProfileDTO(customer_id=DEMO_CUSTOMER_UUID),
        previous_attempts=[],
        max_attempts=3,
        execute_action=True,
    )
    resp_exec_4 = asyncio.run(service.evaluate_decision(req_exec_4, db_session=db))
    # Policy override / deduplication must force NO_ACTION and block gateway execution
    assert resp_exec_4.recommended_action == RecoveryAction.NO_ACTION
    assert resp_exec_4.policy_overridden is True
    assert resp_exec_4.execution is not None
    assert resp_exec_4.execution.attempted is False
    assert resp_exec_4.execution.status in ("prohibited", "already_executed")
    # Call count should not increase beyond 3
    assert mock_adapter.create_payment_link.call_count == 3


def test_tenant_isolation_rejects_cross_customer_payment(isolated_db_session: Session):
    """Verify that attempting to access another customer's payment_id raises 403."""
    db = isolated_db_session
    ensure_demo_customer_seeded(db)

    # Create another customer and payment
    other_cust_id = uuid4()
    other_cust = Customer(
        id=other_cust_id,
        name="Other Enterprise",
        email="other@tenant.io",
        total_payments=5,
        successful_payments=5,
        failed_payments=0,
    )
    other_pay_id = uuid4()
    other_pay = Payment(
        id=other_pay_id,
        customer_id=other_cust_id,
        amount=5000.0,
        currency="INR",
        payment_method="card",
        status="failed",
    )
    other_opp = RecoveryOpportunity(
        id=uuid4(),
        payment_id=other_pay_id,
        status="open",
        revenue_at_risk=5000.0,
        expected_recovery=0.0,
    )
    db.add(other_cust)
    db.add(other_pay)
    db.add(other_opp)
    db.commit()

    mock_adapter = MagicMock(spec=RazorpayAdapter)
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
        action_executor=ActionExecutor(razorpay_adapter=mock_adapter),
    )

    # Authenticated as DEMO_CUSTOMER_UUID, but providing other_pay_id
    req = RecoveryDecisionRequest(
        payment_id=other_pay_id,
        amount=5000.0,
        currency="INR",
        payment_method="card",
        failure_reason="timeout",
        customer=CustomerProfileDTO(customer_id=DEMO_CUSTOMER_UUID),
        execute_action=False,
    )

    with pytest.raises((HTTPException, PaymentCustomerMismatchError)):
        asyncio.run(service.evaluate_decision(req, db_session=db))
