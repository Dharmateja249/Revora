"""
Unit tests for Revora Adaptive Recovery Agent Schemas.
"""

from datetime import datetime, timedelta, timezone
import pytest
from pydantic import ValidationError

from app.agent.schemas import (
    AgentDecisionPromptContext,
    AgentDecisionResult,
    LLMRecoveryRecommendation,
)
from app.decision_engine import RecoveryAction


# ============================================================================
# 1. LLMRecoveryRecommendation Tests
# ============================================================================


def test_valid_llm_recommendation_creation():
    """Verify construction of a valid LLMRecoveryRecommendation with default lists."""
    rec = LLMRecoveryRecommendation(
        recommended_action=RecoveryAction.RETRY_PAYMENT,
        confidence=0.85,
        reasoning="Transient bank timeout detected with strong customer recovery history.",
        key_factors=["transient_failure", "high_customer_recovery_rate"],
        referenced_case_ids=["case_123", "case_456"],
    )

    assert rec.recommended_action == RecoveryAction.RETRY_PAYMENT
    assert rec.confidence == 0.85
    assert rec.reasoning == "Transient bank timeout detected with strong customer recovery history."
    assert rec.key_factors == ("transient_failure", "high_customer_recovery_rate")
    assert rec.referenced_case_ids == ("case_123", "case_456")


def test_valid_llm_recommendation_string_coercion():
    """Verify string action is converted to RecoveryAction enum if valid."""
    rec = LLMRecoveryRecommendation(
        recommended_action="payment_link",
        confidence=0.75,
        reasoning="Interactive link required for OTP expiration.",
    )
    assert rec.recommended_action == RecoveryAction.PAYMENT_LINK
    assert rec.key_factors == ()
    assert rec.referenced_case_ids == ()


def test_confidence_boundary_zero():
    """Verify confidence = 0.0 succeeds."""
    rec = LLMRecoveryRecommendation(
        recommended_action=RecoveryAction.NO_ACTION,
        confidence=0.0,
        reasoning="Zero recovery expectation.",
    )
    assert rec.confidence == 0.0


def test_confidence_boundary_one():
    """Verify confidence = 1.0 succeeds."""
    rec = LLMRecoveryRecommendation(
        recommended_action=RecoveryAction.PAYMENT_LINK,
        confidence=1.0,
        reasoning="Absolute certainty from empirical data.",
    )
    assert rec.confidence == 1.0


def test_confidence_below_zero_rejected():
    """Verify confidence < 0.0 raises ValidationError."""
    with pytest.raises(ValidationError):
        LLMRecoveryRecommendation(
            recommended_action=RecoveryAction.RETRY_PAYMENT,
            confidence=-0.01,
            reasoning="Invalid negative confidence.",
        )


def test_confidence_above_one_rejected():
    """Verify confidence > 1.0 raises ValidationError."""
    with pytest.raises(ValidationError):
        LLMRecoveryRecommendation(
            recommended_action=RecoveryAction.RETRY_PAYMENT,
            confidence=1.01,
            reasoning="Invalid excessive confidence.",
        )


def test_invalid_action_rejected():
    """Verify arbitrary unknown action string is rejected."""
    with pytest.raises(ValidationError):
        LLMRecoveryRecommendation(
            recommended_action="send_email_prompt",
            confidence=0.5,
            reasoning="Custom non-existent action.",
        )


def test_missing_or_empty_reasoning_rejected():
    """Verify empty or whitespace-only reasoning raises ValidationError."""
    with pytest.raises(ValidationError):
        LLMRecoveryRecommendation(
            recommended_action=RecoveryAction.RETRY_PAYMENT,
            confidence=0.8,
            reasoning="",
        )

    with pytest.raises(ValidationError):
        LLMRecoveryRecommendation(
            recommended_action=RecoveryAction.RETRY_PAYMENT,
            confidence=0.8,
            reasoning="   \n\t  ",
        )


def test_llm_recommendation_immutability():
    """Verify LLMRecoveryRecommendation is frozen."""
    rec = LLMRecoveryRecommendation(
        recommended_action=RecoveryAction.WAIT_AND_RETRY,
        confidence=0.6,
        reasoning="Cooldown needed.",
    )

    with pytest.raises(ValidationError):
        rec.confidence = 0.9

    with pytest.raises(ValidationError):
        rec.recommended_action = RecoveryAction.PAYMENT_LINK


def test_no_duplicate_action_enum():
    """Verify recommendation uses the exact RecoveryAction from decision_engine."""
    rec = LLMRecoveryRecommendation(
        recommended_action=RecoveryAction.CHANGE_PAYMENT_METHOD,
        confidence=0.7,
        reasoning="Card expired.",
    )
    assert isinstance(rec.recommended_action, RecoveryAction)
    assert rec.recommended_action is RecoveryAction.CHANGE_PAYMENT_METHOD


# ============================================================================
# 2. AgentDecisionPromptContext Tests
# ============================================================================


def test_valid_agent_prompt_context_creation():
    """Verify creation and serialization of AgentDecisionPromptContext."""
    ctx = AgentDecisionPromptContext(
        current_payment={
            "amount": 500.0,
            "currency": "INR",
            "payment_method": "upi",
            "failure_reason": "bank_timeout",
        },
        customer_profile={
            "historical_success_rate": 0.85,
            "recovery_rate": 0.8,
            "lifetime_recovered": 2500.0,
        },
        recovery_attempt_history=[
            {"action": "retry_payment", "status": "failed", "attempt_number": 1}
        ],
        historical_cases=[
            {
                "case_id": "c1",
                "payment_method": "upi",
                "failure_reason": "bank_timeout",
                "recovery_action": "retry_payment",
                "was_recovered": True,
            }
        ],
        allowed_actions=["retry_payment", "payment_link", "wait_and_retry"],
        prohibited_actions=["change_payment_method"],
        mandatory_fallback="retry_payment",
        policy_constraints=["Transient gateway timeout permits automated retry."],
    )

    assert ctx.current_payment["amount"] == 500.0
    assert ctx.customer_profile["historical_success_rate"] == 0.85
    assert len(ctx.recovery_attempt_history) == 1
    assert len(ctx.historical_cases) == 1
    assert ctx.allowed_actions == ("retry_payment", "payment_link", "wait_and_retry")
    assert ctx.prohibited_actions == ("change_payment_method",)
    assert ctx.mandatory_fallback == "retry_payment"


def test_agent_prompt_context_immutability():
    """Verify AgentDecisionPromptContext and its nested dictionaries are frozen."""
    ctx = AgentDecisionPromptContext(
        current_payment={"amount": 100.0},
        customer_profile={"rate": 0.9},
    )

    with pytest.raises(ValidationError):
        ctx.mandatory_fallback = "payment_link"

    # Attempt nested mutation on frozen mapping
    with pytest.raises(TypeError):
        ctx.current_payment["amount"] = 200.0


def test_agent_prompt_context_serialization():
    """Verify clean JSON serialization without MappingProxyType artifacts."""
    ctx = AgentDecisionPromptContext(
        current_payment={"amount": 100.0},
        customer_profile={"rate": 0.9},
        allowed_actions=["payment_link"],
    )

    data = ctx.model_dump(mode="json")
    assert isinstance(data["current_payment"], dict)
    assert data["current_payment"]["amount"] == 100.0
    assert isinstance(data["allowed_actions"], list)
    assert data["allowed_actions"] == ["payment_link"]


# ============================================================================
# 3. AgentDecisionResult Tests
# ============================================================================


def test_agent_decision_result_llm_used():
    """Verify AgentDecisionResult representing successful LLM decision."""
    rec = LLMRecoveryRecommendation(
        recommended_action=RecoveryAction.PAYMENT_LINK,
        confidence=0.88,
        reasoning="High link conversion historical affinity.",
    )

    now = datetime(2026, 8, 31, 12, 0, 0, tzinfo=timezone.utc)
    result = AgentDecisionResult(
        recommendation=rec,
        agent_used=True,
        provider="openai",
        model_name="gpt-4o",
        is_fallback=False,
        latency_ms=245.5,
        evaluated_at=now,
        metadata={"prompt_tokens": 350, "completion_tokens": 60},
    )

    assert result.agent_used is True
    assert result.provider == "openai"
    assert result.model_name == "gpt-4o"
    assert result.is_fallback is False
    assert result.fallback_reason is None
    assert result.latency_ms == 245.5
    assert result.recommendation.recommended_action == RecoveryAction.PAYMENT_LINK


def test_agent_decision_result_deterministic_fallback():
    """Verify AgentDecisionResult representing deterministic fallback when LLM fails."""
    rec = LLMRecoveryRecommendation(
        recommended_action=RecoveryAction.WAIT_AND_RETRY,
        confidence=0.5,
        reasoning="Deterministic fallback rule triggered due to LLM timeout.",
    )

    result = AgentDecisionResult(
        recommendation=rec,
        agent_used=False,
        provider="mock_deterministic_fallback",
        is_fallback=True,
        fallback_reason="LLM provider timed out after 2000ms",
        latency_ms=2005.0,
    )

    assert result.agent_used is False
    assert result.is_fallback is True
    assert result.fallback_reason == "LLM provider timed out after 2000ms"
    assert result.recommendation.recommended_action == RecoveryAction.WAIT_AND_RETRY


def test_agent_decision_result_immutability():
    """Verify AgentDecisionResult is frozen."""
    rec = LLMRecoveryRecommendation(
        recommended_action=RecoveryAction.NO_ACTION,
        confidence=0.95,
        reasoning="Max attempts exceeded.",
    )

    result = AgentDecisionResult(
        recommendation=rec,
        agent_used=True,
        provider="test",
    )

    with pytest.raises(ValidationError):
        result.agent_used = False


def test_agent_decision_result_rejects_agent_used_fallback():
    """Verify that agent_used=True with is_fallback=True raises ValidationError."""
    rec = LLMRecoveryRecommendation(
        recommended_action=RecoveryAction.PAYMENT_LINK,
        confidence=0.8,
        reasoning="Reason.",
    )

    with pytest.raises(ValidationError, match="agent_used cannot be True when is_fallback is True"):
        AgentDecisionResult(
            recommendation=rec,
            agent_used=True,
            is_fallback=True,
            fallback_reason="Some error",
        )


def test_agent_decision_result_rejects_fallback_without_reason():
    """Verify that is_fallback=True with empty/missing fallback_reason raises ValidationError."""
    rec = LLMRecoveryRecommendation(
        recommended_action=RecoveryAction.WAIT_AND_RETRY,
        confidence=0.5,
        reasoning="Reason.",
    )

    with pytest.raises(ValidationError, match="fallback_reason is required and cannot be empty"):
        AgentDecisionResult(
            recommendation=rec,
            agent_used=False,
            is_fallback=True,
            fallback_reason=None,
        )

    with pytest.raises(ValidationError, match="fallback_reason is required and cannot be empty"):
        AgentDecisionResult(
            recommendation=rec,
            agent_used=False,
            is_fallback=True,
            fallback_reason="   \t  ",
        )


def test_agent_decision_result_rejects_inconsistent_non_fallback_without_agent():
    """Verify that agent_used=False with is_fallback=False raises ValidationError."""
    rec = LLMRecoveryRecommendation(
        recommended_action=RecoveryAction.PAYMENT_LINK,
        confidence=0.8,
        reasoning="Reason.",
    )

    with pytest.raises(ValidationError, match="agent_used must be True when is_fallback is False"):
        AgentDecisionResult(
            recommendation=rec,
            agent_used=False,
            is_fallback=False,
        )


def test_agent_decision_result_accepts_valid_llm_result():
    """Verify standard valid LLM result state (agent_used=True, is_fallback=False)."""
    rec = LLMRecoveryRecommendation(
        recommended_action=RecoveryAction.PAYMENT_LINK,
        confidence=0.9,
        reasoning="Valid LLM generated reasoning.",
    )

    result = AgentDecisionResult(
        recommendation=rec,
        agent_used=True,
        is_fallback=False,
        provider="mock_llm",
        model_name="test-model",
    )

    assert result.agent_used is True
    assert result.is_fallback is False
    assert result.fallback_reason is None


def test_agent_decision_result_accepts_valid_fallback_result():
    """Verify standard valid fallback result state (agent_used=False, is_fallback=True, with reason)."""
    rec = LLMRecoveryRecommendation(
        recommended_action=RecoveryAction.WAIT_AND_RETRY,
        confidence=0.5,
        reasoning="Deterministic fallback rule applied.",
    )

    result = AgentDecisionResult(
        recommendation=rec,
        agent_used=False,
        is_fallback=True,
        fallback_reason="LLM provider timed out after 2000ms",
        provider="deterministic_engine",
    )

    assert result.agent_used is False
    assert result.is_fallback is True
    assert result.fallback_reason == "LLM provider timed out after 2000ms"


def test_agent_decision_result_rejects_naive_evaluated_at():
    """Verify that a naive datetime (without timezone info) raises ValidationError."""
    rec = LLMRecoveryRecommendation(
        recommended_action=RecoveryAction.PAYMENT_LINK,
        confidence=0.85,
        reasoning="Reason.",
    )

    naive_dt = datetime(2026, 8, 31, 10, 30, 0)  # No tzinfo

    with pytest.raises(ValidationError, match="evaluated_at must be a timezone-aware datetime"):
        AgentDecisionResult(
            recommendation=rec,
            agent_used=True,
            evaluated_at=naive_dt,
        )


def test_agent_decision_result_normalizes_non_utc_evaluated_at():
    """Verify that a timezone-aware non-UTC datetime is converted to timezone.utc."""
    rec = LLMRecoveryRecommendation(
        recommended_action=RecoveryAction.PAYMENT_LINK,
        confidence=0.85,
        reasoning="Reason.",
    )

    # IST timezone (+05:30)
    ist_tz = timezone(timedelta(hours=5, minutes=30))
    ist_dt = datetime(2026, 8, 31, 15, 30, 0, tzinfo=ist_tz)

    result = AgentDecisionResult(
        recommendation=rec,
        agent_used=True,
        evaluated_at=ist_dt,
    )

    # Expected equivalent UTC: 2026-08-31 10:00:00 UTC
    assert result.evaluated_at.tzinfo == timezone.utc
    assert result.evaluated_at.hour == 10
    assert result.evaluated_at.minute == 0
    assert result.evaluated_at.year == 2026
    assert result.evaluated_at.month == 8
    assert result.evaluated_at.day == 31


def test_agent_decision_result_default_evaluated_at_is_utc():
    """Verify default factory evaluates to timezone.utc datetime."""
    rec = LLMRecoveryRecommendation(
        recommended_action=RecoveryAction.PAYMENT_LINK,
        confidence=0.85,
        reasoning="Reason.",
    )

    result = AgentDecisionResult(
        recommendation=rec,
        agent_used=True,
    )

    assert result.evaluated_at.tzinfo == timezone.utc
