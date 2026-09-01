"""
Integration Tests for RecoveryService Agent Core Integration (Milestone 2 — Stage 5.2).

Verifies:
1. Service default uses deterministic DecisionEngine (agent_used=False, is_fallback=False).
2. Service with agent enabled uses AgentOrchestrator and maps AgentDecisionResult.
3. Request-level agent toggle overrides service defaults in all 3 combinations.
4. Policy override by agent orchestrator propagates to service response and AuditEvent.
5. Provider failure produces sanitized deterministic fallback without leaking raw exceptions.
6. Unexpected agent exceptions trigger service-level fallback to DecisionEngine safely.
7. AuditEvent metadata payload contains sanitized agent telemetry and no PII/raw prompts.
8. Historical RAG evidence is retrieved and passed intact to AgentOrchestrator.
9. Resolved RecoveryPolicyContext is passed intact to AgentOrchestrator.
10. Default deterministic execution preserves existing behavior identically.
"""

from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional
import uuid
import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.pool import StaticPool
from sqlalchemy.orm import sessionmaker

from app.agent.orchestrator import AgentOrchestrator
from app.agent.provider import LLMProviderError, MockLLMProvider
from app.agent.schemas import LLMRecoveryRecommendation
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
from app.models import AuditEvent, Customer, Payment, RecoveryAttempt, RecoveryOpportunity, utc_now
from app.policies.schemas import RecoveryPolicyContext
from app.recovery_service import RecoveryService
from app.schemas.recovery import (
    RecoveryEvaluationRequest,
    RecoveryEvaluationResponse,
)
from app.semantic_historical_retriever import SemanticHistoricalRetriever
from app.vector_index import VectorIndex


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def in_memory_db():
    """Create a clean in-memory SQLite database for test execution."""
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
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
        name="Test Customer",
        email=f"customer_{ext_suffix}@example.com",
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
        expected_recovery=amount * 0.7,
        created_at=now,
    )
    session.add_all([customer, payment, opportunity])
    session.commit()
    return cust_id, pay_id, opp_id


# ============================================================================
# Stage 5.2 Integration Tests
# ============================================================================


def test_service_default_uses_deterministic_engine(in_memory_db):
    """1. Verify default service instantiation uses deterministic engine with agent disabled."""
    cust_id, pay_id, opp_id = _seed_customer_payment_opportunity(in_memory_db)
    service = RecoveryService()

    assert service.use_agent is False
    assert service.agent_orchestrator is None

    req = RecoveryEvaluationRequest(
        customer_id=cust_id,
        payment_id=pay_id,
        use_rag=False,
    )

    resp = service.evaluate_recovery(db_session=in_memory_db, request=req)

    assert resp.agent_used is False
    assert resp.is_fallback is False
    assert resp.fallback_reason is None
    assert resp.recommended_action in (
        RecoveryAction.RETRY_PAYMENT,
        RecoveryAction.WAIT_AND_RETRY,
        RecoveryAction.PAYMENT_LINK,
        RecoveryAction.CHANGE_PAYMENT_METHOD,
    )


@pytest.mark.anyio
async def test_service_with_agent_enabled_uses_orchestrator(in_memory_db):
    """2. Verify service with agent enabled executes AgentOrchestrator and maps AgentDecisionResult."""
    cust_id, pay_id, opp_id = _seed_customer_payment_opportunity(in_memory_db)

    rec = LLMRecoveryRecommendation(
        recommended_action=RecoveryAction.PAYMENT_LINK,
        confidence=0.88,
        reasoning="High probability recovery via personalized payment link.",
        key_factors=("customer_affinity", "transient_timeout"),
        referenced_case_ids=(),
    )
    provider = MockLLMProvider(recommendation=rec)
    orchestrator = AgentOrchestrator(provider=provider)

    service = RecoveryService(
        agent_orchestrator=orchestrator,
        use_agent=True,
    )

    req = RecoveryEvaluationRequest(
        customer_id=cust_id,
        payment_id=pay_id,
        use_rag=False,
    )

    resp = await service.evaluate_recovery_async(db_session=in_memory_db, request=req)

    assert resp.agent_used is True
    assert resp.is_fallback is False
    assert resp.fallback_reason is None
    assert resp.recommended_action == RecoveryAction.PAYMENT_LINK
    assert resp.confidence == 0.88
    assert resp.reason == "High probability recovery via personalized payment link."
    assert resp.decision_basis["agent_used"] is True
    assert resp.decision_basis["key_factors"] == ["customer_affinity", "transient_timeout"]

    # Verify DB opportunity update
    opp = in_memory_db.get(RecoveryOpportunity, opp_id)
    assert opp is not None
    assert opp.recommended_action == "payment_link"
    assert opp.confidence == 0.88


def test_service_request_level_agent_toggle(in_memory_db):
    """3. Verify request-level use_agent toggle overrides service defaults in all 3 cases."""
    cust_id, pay_id, opp_id = _seed_customer_payment_opportunity(in_memory_db)

    rec = LLMRecoveryRecommendation(
        recommended_action=RecoveryAction.CHANGE_PAYMENT_METHOD,
        confidence=0.92,
        reasoning="Agent recommended payment method update.",
        key_factors=("persistent_card_decline",),
        referenced_case_ids=(),
    )
    provider = MockLLMProvider(recommendation=rec)
    orchestrator = AgentOrchestrator(provider=provider)

    # Case A: service=False + request=True -> Agent path
    service_disabled = RecoveryService(agent_orchestrator=orchestrator, use_agent=False)
    req_enable = RecoveryEvaluationRequest(customer_id=cust_id, payment_id=pay_id, use_rag=False, use_agent=True)
    resp_a = service_disabled.evaluate_recovery(db_session=in_memory_db, request=req_enable)
    assert resp_a.agent_used is True
    assert resp_a.recommended_action == RecoveryAction.CHANGE_PAYMENT_METHOD

    # Case B: service=True + request=False -> Deterministic path
    service_enabled = RecoveryService(agent_orchestrator=orchestrator, use_agent=True)
    req_disable = RecoveryEvaluationRequest(customer_id=cust_id, payment_id=pay_id, use_rag=False, use_agent=False)
    resp_b = service_enabled.evaluate_recovery(db_session=in_memory_db, request=req_disable)
    assert resp_b.agent_used is False
    assert resp_b.is_fallback is False

    # Case C: service=True + request=None -> Agent path
    req_default = RecoveryEvaluationRequest(customer_id=cust_id, payment_id=pay_id, use_rag=False, use_agent=None)
    resp_c = service_enabled.evaluate_recovery(db_session=in_memory_db, request=req_default)
    assert resp_c.agent_used is True
    assert resp_c.recommended_action == RecoveryAction.CHANGE_PAYMENT_METHOD


@pytest.mark.anyio
async def test_service_agent_policy_override_telemetry(in_memory_db):
    """4. Verify policy override applied by AgentOrchestrator propagates to service response and DB."""
    # Seed payment with authentication_failed error where retry_payment is prohibited
    cust_id, pay_id, opp_id = _seed_customer_payment_opportunity(
        in_memory_db, failure_reason="authentication_failed"
    )

    # LLM recommends prohibited action: RETRY_PAYMENT
    rec = LLMRecoveryRecommendation(
        recommended_action=RecoveryAction.RETRY_PAYMENT,
        confidence=0.90,
        reasoning="Retry the authentication flow immediately.",
        key_factors=("auth_retry",),
        referenced_case_ids=(),
    )
    provider = MockLLMProvider(recommendation=rec)
    orchestrator = AgentOrchestrator(provider=provider)
    service = RecoveryService(agent_orchestrator=orchestrator, use_agent=True)

    req = RecoveryEvaluationRequest(customer_id=cust_id, payment_id=pay_id, use_rag=False)
    resp = await service.evaluate_recovery_async(db_session=in_memory_db, request=req)

    # Policy validator inside orchestrator must have overridden prohibited RETRY_PAYMENT
    assert resp.recommended_action != RecoveryAction.RETRY_PAYMENT
    assert resp.policy_overridden is True
    assert resp.agent_used is True
    assert resp.is_fallback is False
    assert "RZP_CUSTOMER_AUTH_2FA_REQUIRED" in resp.applied_policy_ids

    # Verify DB persistence reflects overridden action
    opp = in_memory_db.get(RecoveryOpportunity, opp_id)
    assert opp is not None
    assert opp.recommended_action != "retry_payment"


def test_service_agent_provider_failure_applies_sanitized_fallback(in_memory_db):
    """5. Verify provider failure triggers deterministic fallback with sanitized reasons and metadata."""
    cust_id, pay_id, opp_id = _seed_customer_payment_opportunity(in_memory_db)

    dummy_rec = LLMRecoveryRecommendation(
        recommended_action=RecoveryAction.PAYMENT_LINK,
        confidence=0.8,
        reasoning="Dummy rec.",
        key_factors=(),
        referenced_case_ids=(),
    )
    sensitive_markers = (
        "provider-test-api-marker",
        "provider-test-token-marker",
        "provider-test-card-marker",
    )
    sensitive_error = f"CRITICAL: {' '.join(sensitive_markers)}"
    provider = MockLLMProvider(
        recommendation=dummy_rec,
        should_fail=True,
        failure_exception=LLMProviderError(sensitive_error),
    )
    orchestrator = AgentOrchestrator(provider=provider)
    service = RecoveryService(agent_orchestrator=orchestrator, use_agent=True)

    req = RecoveryEvaluationRequest(customer_id=cust_id, payment_id=pay_id, use_rag=False)
    resp = service.evaluate_recovery(db_session=in_memory_db, request=req)

    assert resp.agent_used is False
    assert resp.is_fallback is True
    assert resp.fallback_reason == "LLM provider failure; deterministic fallback applied"
    assert resp.decision_basis["error_type"] == "LLMProviderError"

    # Verify sensitive markers are strictly absent from serialized response and audit event
    dumped_json = resp.model_dump_json()
    assert "CRITICAL" not in dumped_json
    for marker in sensitive_markers:
        assert marker not in dumped_json

    # Audit event verification
    stmt = select(AuditEvent).where(AuditEvent.opportunity_id == opp_id)
    audit = in_memory_db.execute(stmt).scalars().first()
    assert audit is not None
    assert "CRITICAL" not in str(audit.metadata_payload)
    for marker in sensitive_markers:
        assert marker not in str(audit.metadata_payload)
    assert audit.metadata_payload["error_type"] == "LLMProviderError"


@pytest.mark.anyio
async def test_service_unexpected_agent_exception_falls_back_to_deterministic_engine(
    in_memory_db, monkeypatch
):
    """6. Verify unexpected agent exceptions trigger service-level fallback to DecisionEngine."""
    cust_id, pay_id, opp_id = _seed_customer_payment_opportunity(in_memory_db)

    dummy_rec = LLMRecoveryRecommendation(
        recommended_action=RecoveryAction.PAYMENT_LINK,
        confidence=0.8,
        reasoning="Dummy rec.",
        key_factors=(),
        referenced_case_ids=(),
    )
    provider = MockLLMProvider(recommendation=dummy_rec)
    orchestrator = AgentOrchestrator(provider=provider)

    async def _failing_decide(*args, **kwargs):
        raise RuntimeError("Unexpected memory corruption in sub-process")

    monkeypatch.setattr(orchestrator, "decide", _failing_decide)

    service = RecoveryService(agent_orchestrator=orchestrator, use_agent=True)
    req = RecoveryEvaluationRequest(customer_id=cust_id, payment_id=pay_id, use_rag=False)

    resp = await service.evaluate_recovery_async(db_session=in_memory_db, request=req)

    assert resp.agent_used is False
    assert resp.is_fallback is True
    assert resp.fallback_reason == "Unexpected agent orchestration failure; deterministic fallback applied"
    assert resp.decision_basis["error_type"] == "RuntimeError"

    # Confirm raw error message is not leaked in response
    assert "memory corruption" not in resp.model_dump_json()


def test_service_audit_event_contains_sanitized_agent_telemetry(in_memory_db):
    """7. Verify AuditEvent contains sanitized agent telemetry and no PII or raw prompts."""
    cust_id, pay_id, opp_id = _seed_customer_payment_opportunity(in_memory_db)

    rec = LLMRecoveryRecommendation(
        recommended_action=RecoveryAction.WAIT_AND_RETRY,
        confidence=0.80,
        reasoning="Wait 24h for banking window opening.",
        key_factors=("banking_hours",),
        referenced_case_ids=(),
    )
    provider = MockLLMProvider(recommendation=rec)
    orchestrator = AgentOrchestrator(provider=provider)
    service = RecoveryService(agent_orchestrator=orchestrator, use_agent=True)

    req = RecoveryEvaluationRequest(customer_id=cust_id, payment_id=pay_id, use_rag=False)
    service.evaluate_recovery(db_session=in_memory_db, request=req)

    stmt = select(AuditEvent).where(AuditEvent.opportunity_id == opp_id)
    audit = in_memory_db.execute(stmt).scalars().first()
    assert audit is not None
    assert audit.event_type == "recovery_decision_evaluated"
    assert audit.metadata_payload["agent_used"] is True
    assert audit.metadata_payload["is_fallback"] is False
    assert audit.metadata_payload["key_factors"] == ["banking_hours"]
    # Verify no raw prompt or PII in audit metadata
    assert "prompt" not in audit.metadata_payload
    assert "Alice" not in str(audit.metadata_payload)


@pytest.mark.anyio
async def test_agent_path_preserves_rag_evidence(in_memory_db, monkeypatch):
    """8. Verify historical cases retrieved via RAG are passed intact to AgentOrchestrator."""
    cust_id, pay_id, opp_id = _seed_customer_payment_opportunity(in_memory_db)

    captured_cases = None

    rec = LLMRecoveryRecommendation(
        recommended_action=RecoveryAction.PAYMENT_LINK,
        confidence=0.85,
        reasoning="RAG evidence supports link.",
        key_factors=("rag_supported",),
        referenced_case_ids=(),
    )
    provider = MockLLMProvider(recommendation=rec)
    orchestrator = AgentOrchestrator(provider=provider)
    orig_decide = orchestrator.decide

    async def _inspecting_decide(context, policy_context, historical_cases=None):
        nonlocal captured_cases
        captured_cases = historical_cases
        return await orig_decide(context, policy_context, historical_cases)

    monkeypatch.setattr(orchestrator, "decide", _inspecting_decide)

    # Mock hybrid retriever returning a known historical case
    mock_case = HistoricalCase(
        payment_id=uuid.uuid4(),
        customer_id=cust_id,
        amount=2500.0,
        currency="INR",
        payment_method="upi",
        failure_reason="bank_timeout",
        recovery_action="payment_link",
        recovery_status="recovered",
        amount_recovered=2500.0,
        was_recovered=True,
        relevance_score=0.95,
    )

    class MockRetriever:
        def retrieve_relevant_cases(self, context, top_k=5):
            return [mock_case]

    service = RecoveryService(
        agent_orchestrator=orchestrator,
        hybrid_retriever=MockRetriever(),  # type: ignore
        use_agent=True,
    )

    req = RecoveryEvaluationRequest(customer_id=cust_id, payment_id=pay_id, use_rag=True)
    resp = await service.evaluate_recovery_async(db_session=in_memory_db, request=req)

    assert resp.historical_rag_used is True
    assert resp.retrieved_evidence_count == 1
    assert captured_cases is not None
    assert len(captured_cases) == 1
    assert captured_cases[0].relevance_score == 0.95


@pytest.mark.anyio
async def test_agent_path_preserves_policy_context(in_memory_db, monkeypatch):
    """9. Verify resolved RecoveryPolicyContext is passed unchanged to AgentOrchestrator."""
    cust_id, pay_id, opp_id = _seed_customer_payment_opportunity(in_memory_db)

    captured_policy = None

    rec = LLMRecoveryRecommendation(
        recommended_action=RecoveryAction.RETRY_PAYMENT,
        confidence=0.85,
        reasoning="Safe retry.",
        key_factors=(),
        referenced_case_ids=(),
    )
    provider = MockLLMProvider(recommendation=rec)
    orchestrator = AgentOrchestrator(provider=provider)
    orig_decide = orchestrator.decide

    async def _inspecting_decide(context, policy_context, historical_cases=None):
        nonlocal captured_policy
        captured_policy = policy_context
        return await orig_decide(context, policy_context, historical_cases)

    monkeypatch.setattr(orchestrator, "decide", _inspecting_decide)

    service = RecoveryService(agent_orchestrator=orchestrator, use_agent=True)
    req = RecoveryEvaluationRequest(customer_id=cust_id, payment_id=pay_id, use_rag=False)

    await service.evaluate_recovery_async(db_session=in_memory_db, request=req)

    assert captured_policy is not None
    assert isinstance(captured_policy, RecoveryPolicyContext)
    assert captured_policy.provider == "razorpay"
    assert len(captured_policy.allowed_actions) > 0


def test_agent_disabled_preserves_existing_behavior(in_memory_db):
    """10. Verify deterministic execution with agent disabled matches pre-agent baseline."""
    cust_id, pay_id, opp_id = _seed_customer_payment_opportunity(
        in_memory_db, failure_reason="bank_timeout", payment_method="upi"
    )

    service = RecoveryService(use_agent=False)
    req = RecoveryEvaluationRequest(customer_id=cust_id, payment_id=pay_id, use_rag=False)

    resp = service.evaluate_recovery(db_session=in_memory_db, request=req)

    assert resp.agent_used is False
    assert resp.is_fallback is False
    assert resp.fallback_reason is None
    assert resp.recommended_action == RecoveryAction.RETRY_PAYMENT
    assert resp.payment_id == pay_id
    assert resp.customer_id == cust_id
    assert resp.opportunity_id == opp_id


# ============================================================================
# Stage 5.4 Comprehensive Matrix & Safety Tests
# ============================================================================


def test_service_disabled_with_request_use_agent_true_falls_back_safely(in_memory_db):
    """11. Verify service without orchestrator safely uses DecisionEngine when request.use_agent=True."""
    cust_id, pay_id, opp_id = _seed_customer_payment_opportunity(in_memory_db)
    service = RecoveryService(agent_orchestrator=None, use_agent=False)

    req = RecoveryEvaluationRequest(
        customer_id=cust_id,
        payment_id=pay_id,
        use_rag=False,
        use_agent=True,
    )

    resp = service.evaluate_recovery(db_session=in_memory_db, request=req)

    assert resp.agent_used is False
    assert resp.is_fallback is False
    assert resp.fallback_reason is None
    assert resp.recommended_action == RecoveryAction.RETRY_PAYMENT


@pytest.mark.anyio
async def test_service_sync_and_async_evaluations_are_identical(in_memory_db):
    """12. Verify synchronous evaluate_recovery and asynchronous evaluate_recovery_async produce identical output."""
    rec = LLMRecoveryRecommendation(
        recommended_action=RecoveryAction.PAYMENT_LINK,
        confidence=0.88,
        reasoning="High probability recovery via payment link.",
        key_factors=("test_factor",),
        referenced_case_ids=(),
    )
    provider = MockLLMProvider(recommendation=rec)
    orchestrator = AgentOrchestrator(provider=provider)
    service = RecoveryService(agent_orchestrator=orchestrator, use_agent=True)

    cust_id_1, pay_id_1, _ = _seed_customer_payment_opportunity(in_memory_db)
    cust_id_2, pay_id_2, _ = _seed_customer_payment_opportunity(in_memory_db)

    req_sync = RecoveryEvaluationRequest(customer_id=cust_id_1, payment_id=pay_id_1, use_rag=False)
    req_async = RecoveryEvaluationRequest(customer_id=cust_id_2, payment_id=pay_id_2, use_rag=False)

    resp_sync = service.evaluate_recovery(db_session=in_memory_db, request=req_sync)
    resp_async = await service.evaluate_recovery_async(db_session=in_memory_db, request=req_async)

    assert resp_sync.recommended_action == resp_async.recommended_action
    assert resp_sync.confidence == resp_async.confidence
    assert resp_sync.reason == resp_async.reason
    assert resp_sync.agent_used == resp_async.agent_used == True
    assert resp_sync.is_fallback == resp_async.is_fallback == False
    assert resp_sync.decision_basis["key_factors"] == resp_async.decision_basis["key_factors"]


def test_service_transaction_rollback_on_persistence_failure_with_agent(in_memory_db, monkeypatch):
    """13. Verify transaction rollback occurs and opportunity is not corrupted on persistence failure."""
    cust_id, pay_id, opp_id = _seed_customer_payment_opportunity(in_memory_db)

    rec = LLMRecoveryRecommendation(
        recommended_action=RecoveryAction.PAYMENT_LINK,
        confidence=0.90,
        reasoning="Valid recommendation.",
        key_factors=(),
        referenced_case_ids=(),
    )
    provider = MockLLMProvider(recommendation=rec)
    orchestrator = AgentOrchestrator(provider=provider)
    service = RecoveryService(agent_orchestrator=orchestrator, use_agent=True)

    def _failing_commit():
        raise RuntimeError("Database connection lost during commit")

    monkeypatch.setattr(in_memory_db, "commit", _failing_commit)

    req = RecoveryEvaluationRequest(customer_id=cust_id, payment_id=pay_id, use_rag=False)

    with pytest.raises(RuntimeError, match="Database connection lost"):
        service.evaluate_recovery(db_session=in_memory_db, request=req)

    # Opportunity recommended_action should remain unchanged (None/open)
    opp = in_memory_db.get(RecoveryOpportunity, opp_id)
    assert opp is not None
    assert opp.recommended_action is None


def test_service_no_recovery_attempt_created_during_agent_evaluation(in_memory_db):
    """14. Verify RecoveryAttempt is never created during evaluation (evaluation only updates opportunity & audit)."""
    cust_id, pay_id, opp_id = _seed_customer_payment_opportunity(in_memory_db)

    rec = LLMRecoveryRecommendation(
        recommended_action=RecoveryAction.PAYMENT_LINK,
        confidence=0.88,
        reasoning="Valid recommendation.",
        key_factors=(),
        referenced_case_ids=(),
    )
    provider = MockLLMProvider(recommendation=rec)
    orchestrator = AgentOrchestrator(provider=provider)
    service = RecoveryService(agent_orchestrator=orchestrator, use_agent=True)

    req = RecoveryEvaluationRequest(customer_id=cust_id, payment_id=pay_id, use_rag=False)
    service.evaluate_recovery(db_session=in_memory_db, request=req)

    stmt = select(RecoveryAttempt).where(RecoveryAttempt.opportunity_id == opp_id)
    attempts = in_memory_db.execute(stmt).scalars().all()
    assert len(attempts) == 0


@pytest.mark.anyio
async def test_service_agent_malformed_response_triggers_deterministic_fallback(in_memory_db):
    """15. Verify LLMResponseValidationError produces clean deterministic fallback without leaking internal error."""
    from app.agent.provider import LLMResponseValidationError

    cust_id, pay_id, opp_id = _seed_customer_payment_opportunity(in_memory_db)

    dummy_rec = LLMRecoveryRecommendation(
        recommended_action=RecoveryAction.PAYMENT_LINK,
        confidence=0.8,
        reasoning="Dummy rec.",
        key_factors=(),
        referenced_case_ids=(),
    )
    provider = MockLLMProvider(
        recommendation=dummy_rec,
        should_fail=True,
        failure_exception=LLMResponseValidationError("Missing required JSON field 'confidence' in response payload"),
    )
    orchestrator = AgentOrchestrator(provider=provider)
    service = RecoveryService(agent_orchestrator=orchestrator, use_agent=True)

    req = RecoveryEvaluationRequest(customer_id=cust_id, payment_id=pay_id, use_rag=False)
    resp = await service.evaluate_recovery_async(db_session=in_memory_db, request=req)

    assert resp.agent_used is False
    assert resp.is_fallback is True
    assert resp.fallback_reason == "LLM provider failure; deterministic fallback applied"
    assert resp.decision_basis["error_type"] == "LLMResponseValidationError"
    assert "Missing required JSON field" not in resp.model_dump_json()

