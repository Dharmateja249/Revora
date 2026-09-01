"""
Unit tests for Revora Adaptive Recovery Agent Decision Orchestrator.
"""

import uuid
from datetime import datetime, timezone

import pytest
from app.agent.orchestrator import AgentOrchestrator
from app.agent.provider import (
    LLMProviderError,
    LLMResponseValidationError,
    MockLLMProvider,
)
from app.agent.schemas import (
    AgentDecisionResult,
    LLMRecoveryRecommendation,
)
from app.context import (
    CustomerContext,
    CustomerRecoveryContext,
    PaymentContext,
    RecoveryAttemptContext,
    RecoveryOpportunityContext,
)
from app.decision_engine import RecoveryAction
from app.historical_retrieval import HistoricalCase
from app.policies.registry import (
    RZP_CUSTOMER_AUTH_2FA_REQUIRED_RULE,
    RZP_PERMANENT_CREDENTIAL_ERROR_RULE,
)
from app.policies.schemas import RecoveryPolicyContext


@pytest.fixture
def sample_customer_context() -> CustomerRecoveryContext:
    """Build a rich sample CustomerRecoveryContext with sensitive identifiers."""
    now = datetime(2026, 9, 1, 12, 0, 0, tzinfo=timezone.utc)
    cust_id = uuid.UUID("99999999-8888-7777-6666-555555555555")
    pay_id = uuid.UUID("22222222-3333-4444-5555-666666666666")
    opp_id = uuid.uuid4()

    customer = CustomerContext(
        customer_id=cust_id,
        external_customer_id="CUST_SECRET_EXT",
        name="Sensitive Customer",
        email="sensitive.user@example.com",
        total_payments=6,
        successful_payments=5,
        failed_payments=1,
        historical_success_rate=0.8333,
    )

    payment = PaymentContext(
        payment_id=pay_id,
        external_payment_id="PAY_SECRET_EXT",
        amount=1250.0,
        currency="INR",
        payment_method="card",
        status="failed",
        failure_reason="authentication_failed",
        created_at=now,
    )

    opportunity = RecoveryOpportunityContext(
        opportunity_id=opp_id,
        status="open",
        revenue_at_risk=1250.0,
        expected_recovery=1000.0,
        created_at=now,
    )

    return CustomerRecoveryContext(
        customer=customer,
        current_payment=payment,
        current_opportunity=opportunity,
        current_payment_attempts=[
            RecoveryAttemptContext(
                attempt_id=uuid.uuid4(),
                action="retry_payment",
                status="failed",
                amount_recovered=0.0,
                error_code="3ds_timeout",
                created_at=now,
            )
        ],
        historical_payments=[],
        retrieved_at=now,
    )


@pytest.fixture
def sample_policy_context() -> RecoveryPolicyContext:
    """Build a standard 2FA-mandated RecoveryPolicyContext."""
    return RecoveryPolicyContext(
        provider="razorpay",
        policy_version="2026.1",
        applicable_rules=(RZP_CUSTOMER_AUTH_2FA_REQUIRED_RULE,),
        allowed_actions=(
            RecoveryAction.PAYMENT_LINK,
            RecoveryAction.CHANGE_PAYMENT_METHOD,
        ),
        prohibited_actions=(
            RecoveryAction.RETRY_PAYMENT,
            RecoveryAction.WAIT_AND_RETRY,
        ),
        mandatory_fallback_action=RecoveryAction.PAYMENT_LINK,
    )


@pytest.fixture
def valid_llm_recommendation() -> LLMRecoveryRecommendation:
    """Build a compliant candidate recommendation."""
    return LLMRecoveryRecommendation(
        recommended_action=RecoveryAction.PAYMENT_LINK,
        confidence=0.88,
        reasoning="Authentication failure requires interactive customer verification via payment link.",
        key_factors=["2fa_required", "high_customer_trust"],
        referenced_case_ids=["case_1"],
    )


# ============================================================================
# 1. Successful Decision Pipeline & Telemetry Tests
# ============================================================================


@pytest.mark.anyio
async def test_orchestrator_successful_llm_decision(
    sample_customer_context,
    sample_policy_context,
    valid_llm_recommendation,
):
    """Verify happy-path decision pipeline returns valid AgentDecisionResult."""
    provider = MockLLMProvider(recommendation=valid_llm_recommendation)
    orchestrator = AgentOrchestrator(provider=provider)

    result = await orchestrator.decide(
        context=sample_customer_context,
        policy_context=sample_policy_context,
    )

    assert isinstance(result, AgentDecisionResult)
    assert result.agent_used is True
    assert result.is_fallback is False
    assert result.fallback_reason is None
    assert result.recommendation.recommended_action == RecoveryAction.PAYMENT_LINK
    assert result.recommendation.confidence == 0.88
    assert result.recommendation.reasoning == valid_llm_recommendation.reasoning
    assert result.recommendation.key_factors == ("2fa_required", "high_customer_trust")
    assert result.recommendation.referenced_case_ids == ("case_1",)
    assert result.latency_ms is not None
    assert result.latency_ms >= 0.0
    assert result.evaluated_at.tzinfo is not None
    assert result.evaluated_at.tzinfo == timezone.utc
    assert result.metadata["policy_overridden"] is False
    assert "RZP_CUSTOMER_AUTH_2FA_REQUIRED" in result.metadata["applied_policy_ids"]


# ============================================================================
# 2. Policy Enforcement & Override Tests
# ============================================================================


@pytest.mark.anyio
async def test_orchestrator_overrides_prohibited_llm_action(
    sample_customer_context,
    sample_policy_context,
):
    """Verify PolicyValidator overrides a prohibited LLM recommendation while preserving agent_used=True."""
    prohibited_rec = LLMRecoveryRecommendation(
        recommended_action=RecoveryAction.RETRY_PAYMENT,
        confidence=0.92,
        reasoning="Attempting silent background retry for fast recovery.",
        key_factors=["fast_recovery"],
        referenced_case_ids=["case_1"],
    )

    provider = MockLLMProvider(recommendation=prohibited_rec)
    orchestrator = AgentOrchestrator(provider=provider)

    result = await orchestrator.decide(
        context=sample_customer_context,
        policy_context=sample_policy_context,
    )

    assert isinstance(result, AgentDecisionResult)
    # Policy override is NOT a fallback!
    assert result.agent_used is True
    assert result.is_fallback is False
    assert result.fallback_reason is None

    # Effective action must be overridden to payment_link by policy
    assert result.recommendation.recommended_action == RecoveryAction.PAYMENT_LINK
    assert result.recommendation.confidence == 0.92
    assert "Policy override" in result.recommendation.reasoning
    assert "policy_override:payment_link" in result.recommendation.key_factors
    assert result.recommendation.referenced_case_ids == ("case_1",)

    assert result.metadata["policy_overridden"] is True
    assert result.metadata["original_candidate_action"] == "retry_payment"
    assert "RZP_CUSTOMER_AUTH_2FA_REQUIRED" in result.metadata["violated_policy_ids"]


@pytest.mark.anyio
async def test_orchestrator_respects_mandatory_fallback_on_policy_violation(
    sample_customer_context,
):
    """Verify mandatory fallback action is strictly selected when a rule violation occurs."""
    policy_ctx = RecoveryPolicyContext(
        provider="razorpay",
        policy_version="2026.1",
        applicable_rules=(RZP_PERMANENT_CREDENTIAL_ERROR_RULE,),
        allowed_actions=(
            RecoveryAction.CHANGE_PAYMENT_METHOD,
            RecoveryAction.PAYMENT_LINK,
        ),
        prohibited_actions=(
            RecoveryAction.RETRY_PAYMENT,
            RecoveryAction.WAIT_AND_RETRY,
        ),
        mandatory_fallback_action=RecoveryAction.CHANGE_PAYMENT_METHOD,
    )

    prohibited_rec = LLMRecoveryRecommendation(
        recommended_action=RecoveryAction.RETRY_PAYMENT,
        confidence=0.75,
        reasoning="Try retrying expired card.",
    )

    provider = MockLLMProvider(recommendation=prohibited_rec)
    orchestrator = AgentOrchestrator(provider=provider)

    result = await orchestrator.decide(
        context=sample_customer_context,
        policy_context=policy_ctx,
    )

    assert result.agent_used is True
    assert result.is_fallback is False
    assert (
        result.recommendation.recommended_action == RecoveryAction.CHANGE_PAYMENT_METHOD
    )
    assert result.metadata["policy_overridden"] is True


# ============================================================================
# 3. Provider Failure & Deterministic Fallback Tests
# ============================================================================


@pytest.mark.anyio
async def test_orchestrator_handles_provider_execution_error(
    sample_customer_context,
    sample_policy_context,
    valid_llm_recommendation,
):
    """Verify LLMProviderError triggers deterministic fallback with sanitized reason and metadata."""
    provider = MockLLMProvider(
        recommendation=valid_llm_recommendation,
        should_fail=True,
        failure_exception=LLMProviderError("Remote API connection timeout."),
    )
    orchestrator = AgentOrchestrator(provider=provider)

    result = await orchestrator.decide(
        context=sample_customer_context,
        policy_context=sample_policy_context,
    )

    assert isinstance(result, AgentDecisionResult)
    assert result.agent_used is False
    assert result.is_fallback is True
    assert (
        result.fallback_reason == "LLM provider failure; deterministic fallback applied"
    )
    assert result.recommendation.recommended_action == RecoveryAction.PAYMENT_LINK
    assert result.recommendation.confidence == 0.0
    assert (
        result.recommendation.reasoning
        == "Deterministic fallback triggered due to LLM provider failure."
    )
    assert result.recommendation.key_factors == (
        "deterministic_fallback",
        "provider_error",
    )
    assert result.metadata["error_type"] == "LLMProviderError"
    assert result.latency_ms is not None
    assert result.latency_ms >= 0.0
    assert "Remote API connection timeout" not in str(result.model_dump(mode="json"))


@pytest.mark.anyio
async def test_orchestrator_handles_malformed_response_error(
    sample_customer_context,
    sample_policy_context,
    valid_llm_recommendation,
):
    """Verify LLMResponseValidationError triggers deterministic fallback with sanitized reason."""
    provider = MockLLMProvider(
        recommendation=valid_llm_recommendation,
        should_fail=True,
        failure_exception=LLMResponseValidationError(
            "Response JSON could not be decoded."
        ),
    )
    orchestrator = AgentOrchestrator(provider=provider)

    result = await orchestrator.decide(
        context=sample_customer_context,
        policy_context=sample_policy_context,
    )

    assert result.agent_used is False
    assert result.is_fallback is True
    assert (
        result.fallback_reason == "LLM provider failure; deterministic fallback applied"
    )
    assert (
        result.recommendation.reasoning
        == "Deterministic fallback triggered due to LLM provider failure."
    )
    assert result.metadata["error_type"] == "LLMResponseValidationError"
    assert "Response JSON could not be decoded" not in str(
        result.model_dump(mode="json")
    )


# ============================================================================
# 4. Fail-Closed Boundaries Tests
# ============================================================================


@pytest.mark.anyio
async def test_orchestrator_rejects_missing_policy_context(
    sample_customer_context,
    valid_llm_recommendation,
):
    """Verify policy_context=None raises ValueError and provider.generate() is never called."""
    provider = MockLLMProvider(
        recommendation=valid_llm_recommendation, record_messages=True
    )
    orchestrator = AgentOrchestrator(provider=provider)

    with pytest.raises(ValueError, match="policy_context is required"):
        await orchestrator.decide(
            context=sample_customer_context,
            policy_context=None,  # type: ignore
        )

    # Assert provider was never invoked
    assert provider.recorded_messages == []


@pytest.mark.anyio
async def test_orchestrator_rejects_empty_allowed_actions(
    sample_customer_context,
    valid_llm_recommendation,
):
    """Verify empty allowed_actions in policy context fails closed with ValueError and zero LLM calls."""
    empty_policy = RecoveryPolicyContext(
        provider="razorpay",
        policy_version="2026.1",
        applicable_rules=(),
        allowed_actions=(),
        prohibited_actions=(RecoveryAction.RETRY_PAYMENT,),
    )

    provider = MockLLMProvider(
        recommendation=valid_llm_recommendation, record_messages=True
    )
    orchestrator = AgentOrchestrator(provider=provider)

    with pytest.raises(ValueError, match="at least one allowed action"):
        await orchestrator.decide(
            context=sample_customer_context,
            policy_context=empty_policy,
        )

    assert provider.recorded_messages == []


# ============================================================================
# 5. Dependency Validation & Immutability Tests
# ============================================================================


def test_orchestrator_constructor_dependency_validation(valid_llm_recommendation):
    """Verify constructor rejects invalid provider, context_builder, or policy_validator objects."""
    with pytest.raises(TypeError, match="Expected provider implementing LLMProvider"):
        AgentOrchestrator(provider="not_a_provider")  # type: ignore

    provider = MockLLMProvider(recommendation=valid_llm_recommendation)

    with pytest.raises(
        TypeError, match="Expected context_builder to be AgentContextBuilder"
    ):
        AgentOrchestrator(provider=provider, context_builder="invalid_builder")  # type: ignore

    with pytest.raises(
        TypeError, match="Expected policy_validator to be PolicyValidator"
    ):
        AgentOrchestrator(provider=provider, policy_validator="invalid_validator")  # type: ignore


@pytest.mark.anyio
async def test_orchestrator_does_not_mutate_inputs(
    sample_customer_context,
    sample_policy_context,
    valid_llm_recommendation,
):
    """Verify CustomerRecoveryContext and RecoveryPolicyContext remain unmutated."""
    provider = MockLLMProvider(recommendation=valid_llm_recommendation)
    orchestrator = AgentOrchestrator(provider=provider)

    original_amount = sample_customer_context.current_payment.amount
    original_allowed = tuple(sample_policy_context.allowed_actions)

    await orchestrator.decide(
        context=sample_customer_context,
        policy_context=sample_policy_context,
    )

    assert sample_customer_context.current_payment.amount == original_amount
    assert sample_policy_context.allowed_actions == original_allowed


# ============================================================================
# 6. Privacy & Security Invariant Tests
# ============================================================================


@pytest.mark.anyio
async def test_orchestrator_does_not_leak_payment_ids(
    sample_customer_context,
    sample_policy_context,
    valid_llm_recommendation,
):
    """Verify historical cases with explicit payment UUIDs are sanitized before reaching the provider."""
    raw_payment_uuid = uuid.UUID("11111111-2222-3333-4444-555555555555")
    raw_customer_uuid = uuid.UUID("77777777-8888-9999-0000-111111111111")

    hist_case = HistoricalCase(
        payment_id=raw_payment_uuid,
        customer_id=raw_customer_uuid,
        amount=1250.0,
        currency="INR",
        payment_method="card",
        recovery_action="payment_link",
        recovery_status="recovered",
        amount_recovered=1250.0,
        was_recovered=True,
        relevance_score=0.97,
    )

    provider = MockLLMProvider(
        recommendation=valid_llm_recommendation, record_messages=True
    )
    orchestrator = AgentOrchestrator(provider=provider)

    await orchestrator.decide(
        context=sample_customer_context,
        policy_context=sample_policy_context,
        historical_cases=[hist_case],
    )

    recorded = provider.last_messages
    assert recorded is not None
    user_message = recorded[1]["content"]

    # Raw UUIDs must NOT be in user message
    assert str(raw_payment_uuid) not in user_message
    assert str(raw_customer_uuid) not in user_message
    assert str(sample_customer_context.customer.customer_id) not in user_message
    assert str(sample_customer_context.current_payment.payment_id) not in user_message
    assert "sensitive.user@example.com" not in user_message
    assert "Sensitive Customer" not in user_message

    # Anonymous token must be present
    assert "case_1" in user_message


@pytest.mark.anyio
async def test_policy_override_is_not_marked_as_fallback(
    sample_customer_context,
    sample_policy_context,
):
    """Explicitly test distinction: Policy override is agent_used=True and is_fallback=False."""
    prohibited_rec = LLMRecoveryRecommendation(
        recommended_action=RecoveryAction.RETRY_PAYMENT,
        confidence=0.8,
        reasoning="Attempting retry.",
    )
    provider = MockLLMProvider(recommendation=prohibited_rec)
    orchestrator = AgentOrchestrator(provider=provider)

    result = await orchestrator.decide(
        context=sample_customer_context,
        policy_context=sample_policy_context,
    )

    assert result.agent_used is True
    assert result.is_fallback is False
    assert result.fallback_reason is None
    assert result.metadata["policy_overridden"] is True


@pytest.mark.anyio
async def test_orchestrator_preserves_recommendation_metadata(
    sample_customer_context,
    sample_policy_context,
):
    """Verify confidence, key factors, and referenced cases are preserved on policy override."""
    rec = LLMRecoveryRecommendation(
        recommended_action=RecoveryAction.RETRY_PAYMENT,
        confidence=0.91,
        reasoning="Historical conversion affinity for retry.",
        key_factors=["high_affinity", "recurring_user"],
        referenced_case_ids=["case_1", "case_2"],
    )
    provider = MockLLMProvider(recommendation=rec)
    orchestrator = AgentOrchestrator(provider=provider)

    result = await orchestrator.decide(
        context=sample_customer_context,
        policy_context=sample_policy_context,
    )

    assert result.recommendation.confidence == 0.91
    assert "high_affinity" in result.recommendation.key_factors
    assert "recurring_user" in result.recommendation.key_factors
    assert result.recommendation.referenced_case_ids == ("case_1", "case_2")
    assert result.recommendation.recommended_action == RecoveryAction.PAYMENT_LINK


@pytest.mark.anyio
async def test_provider_failure_does_not_expose_sensitive_data(
    sample_customer_context,
    sample_policy_context,
    valid_llm_recommendation,
):
    """Verify fallback reasoning and metadata do not contain PII or raw payment UUIDs."""
    provider = MockLLMProvider(
        recommendation=valid_llm_recommendation,
        should_fail=True,
        failure_exception=LLMProviderError("Transient upstream gateway timeout."),
    )
    orchestrator = AgentOrchestrator(provider=provider)

    result = await orchestrator.decide(
        context=sample_customer_context,
        policy_context=sample_policy_context,
    )

    dumped = result.model_dump(mode="json")
    json_str = str(dumped)

    assert str(sample_customer_context.customer.customer_id) not in json_str
    assert str(sample_customer_context.current_payment.payment_id) not in json_str
    assert "sensitive.user@example.com" not in json_str
    assert "Sensitive Customer" not in json_str
    assert "CUST_SECRET_EXT" not in json_str
    assert "PAY_SECRET_EXT" not in json_str


@pytest.mark.anyio
async def test_fallback_action_raises_when_no_compliant_action_available(
    sample_customer_context,
    valid_llm_recommendation,
):
    """Regression test for Finding 1: fails closed with ValueError when all allowed actions are prohibited."""
    all_prohibited_policy = RecoveryPolicyContext(
        provider="razorpay",
        policy_version="2026.1",
        applicable_rules=(),
        allowed_actions=(RecoveryAction.RETRY_PAYMENT, RecoveryAction.NO_ACTION),
        prohibited_actions=(RecoveryAction.RETRY_PAYMENT, RecoveryAction.NO_ACTION),
    )
    provider = MockLLMProvider(
        recommendation=valid_llm_recommendation,
        should_fail=True,
        failure_exception=LLMProviderError("Upstream failure"),
    )
    orchestrator = AgentOrchestrator(provider=provider)

    with pytest.raises(
        ValueError, match="no policy-compliant recovery action available"
    ):
        await orchestrator.decide(
            context=sample_customer_context,
            policy_context=all_prohibited_policy,
        )


@pytest.mark.anyio
async def test_fallback_action_selects_first_compliant_when_allowed_contains_prohibited(
    sample_customer_context,
    valid_llm_recommendation,
):
    """Regression test for Finding 1: selects first allowed action that is not in prohibited_actions."""
    mixed_policy = RecoveryPolicyContext(
        provider="razorpay",
        policy_version="2026.1",
        applicable_rules=(),
        allowed_actions=(
            RecoveryAction.RETRY_PAYMENT,
            RecoveryAction.PAYMENT_LINK,
            RecoveryAction.CHANGE_PAYMENT_METHOD,
        ),
        prohibited_actions=(RecoveryAction.RETRY_PAYMENT,),
    )
    provider = MockLLMProvider(
        recommendation=valid_llm_recommendation,
        should_fail=True,
        failure_exception=LLMProviderError("Upstream failure"),
    )
    orchestrator = AgentOrchestrator(provider=provider)

    result = await orchestrator.decide(
        context=sample_customer_context,
        policy_context=mixed_policy,
    )

    assert result.is_fallback is True
    assert result.recommendation.recommended_action == RecoveryAction.PAYMENT_LINK
    assert (
        result.recommendation.recommended_action not in mixed_policy.prohibited_actions
    )


@pytest.mark.anyio
async def test_fallback_action_ignores_prohibited_mandatory_fallback(
    sample_customer_context,
    valid_llm_recommendation,
):
    """Regression test for Finding 1: mandatory_fallback is skipped if it is prohibited."""
    policy_with_prohibited_mandatory = RecoveryPolicyContext(
        provider="razorpay",
        policy_version="2026.1",
        applicable_rules=(),
        allowed_actions=(RecoveryAction.RETRY_PAYMENT, RecoveryAction.PAYMENT_LINK),
        prohibited_actions=(RecoveryAction.RETRY_PAYMENT,),
        mandatory_fallback_action=RecoveryAction.RETRY_PAYMENT,
    )
    provider = MockLLMProvider(
        recommendation=valid_llm_recommendation,
        should_fail=True,
        failure_exception=LLMProviderError("Upstream failure"),
    )
    orchestrator = AgentOrchestrator(provider=provider)

    result = await orchestrator.decide(
        context=sample_customer_context,
        policy_context=policy_with_prohibited_mandatory,
    )

    assert result.is_fallback is True
    assert result.recommendation.recommended_action == RecoveryAction.PAYMENT_LINK
    assert result.recommendation.recommended_action != RecoveryAction.RETRY_PAYMENT


@pytest.mark.anyio
async def test_provider_failure_never_produces_prohibited_action(
    sample_customer_context,
    valid_llm_recommendation,
):
    """Regression test for Finding 1: invariant check that fallback action is never in prohibited_actions."""
    policy = RecoveryPolicyContext(
        provider="razorpay",
        policy_version="2026.1",
        applicable_rules=(),
        allowed_actions=(
            RecoveryAction.WAIT_AND_RETRY,
            RecoveryAction.CHANGE_PAYMENT_METHOD,
        ),
        prohibited_actions=(
            RecoveryAction.WAIT_AND_RETRY,
            RecoveryAction.RETRY_PAYMENT,
        ),
    )
    provider = MockLLMProvider(
        recommendation=valid_llm_recommendation,
        should_fail=True,
        failure_exception=LLMProviderError("Upstream failure"),
    )
    orchestrator = AgentOrchestrator(provider=provider)

    result = await orchestrator.decide(
        context=sample_customer_context,
        policy_context=policy,
    )

    assert result.recommendation.recommended_action not in policy.prohibited_actions
    assert result.recommendation.recommended_action in policy.allowed_actions
    assert (
        result.recommendation.recommended_action == RecoveryAction.CHANGE_PAYMENT_METHOD
    )


@pytest.mark.anyio
async def test_provider_failure_sanitizes_fake_sensitive_exception_string(
    sample_customer_context,
    sample_policy_context,
    valid_llm_recommendation,
):
    """Regression test for Finding 2: raw sensitive provider exception strings are never exposed."""
    sensitive_markers = (
        "provider-test-api-marker",
        "provider-test-token-marker",
        "provider-test-card-marker",
    )
    provider_error_message = f"CRITICAL_LEAK: {' '.join(sensitive_markers)}"
    provider = MockLLMProvider(
        recommendation=valid_llm_recommendation,
        should_fail=True,
        failure_exception=LLMProviderError(provider_error_message),
    )
    orchestrator = AgentOrchestrator(provider=provider)

    result = await orchestrator.decide(
        context=sample_customer_context,
        policy_context=sample_policy_context,
    )

    dumped = result.model_dump(mode="json")
    json_str = str(dumped)

    # Asserts sensitive text is NOT leaked
    assert "CRITICAL_LEAK" not in json_str
    for marker in sensitive_markers:
        assert marker not in json_str

    # Asserts fixed sanitized strings and safe diagnostic metadata
    assert (
        result.fallback_reason == "LLM provider failure; deterministic fallback applied"
    )
    assert (
        result.recommendation.reasoning
        == "Deterministic fallback triggered due to LLM provider failure."
    )
    assert result.metadata["error_type"] == "LLMProviderError"
