"""
Unit and Integration Tests for Revora Deterministic Decision Engine.

Verifies:
1. Basic recommendation behavior for known failure archetypes
2. Transient technical failure progression (retry_payment -> wait_and_retry -> payment_link)
3. Permanent credential failure (change_payment_method)
4. Customer interaction / authentication failure (payment_link)
5. Insufficient funds rule and escalation
6. Customer historical recovery affinity (successful actions favored)
7. Customer historical failed action avoidance
8. Repeated attempts on current payment and max attempts limit (no_action)
9. Already recovered opportunity handling (no_action)
10. Cold-start customer with zero history
11. Unsupported / missing failure reason fallback
12. Strict determinism across repeated executions
13. Schema immutability and confidence score bounds [0.0, 1.0]
"""

import uuid
from datetime import datetime, timezone

import pytest
from app.context import (
    CustomerContext,
    CustomerRecoveryContext,
    CustomerRecoveryStatsContext,
    PaymentContext,
    RecoveryAttemptContext,
    RecoveryOpportunityContext,
)
from app.decision_engine import (
    RecoveryAction,
    RecoveryDecision,
    evaluate_recovery_decision,
)
from pydantic import ValidationError


def _build_test_context(
    failure_reason: str = "insufficient_funds",
    payment_method: str = "card",
    opp_status: str = "open",
    payment_status: str = "failed",
    current_attempts: list | None = None,
    hist_successful_actions: list | None = None,
    hist_failed_actions: list | None = None,
    recovery_rate: float = 0.0,
    total_cust_payments: int = 5,
) -> CustomerRecoveryContext:
    """Helper to construct controlled test contexts."""
    now = datetime.now(timezone.utc)
    cust_id = uuid.uuid4()
    pay_id = uuid.uuid4()
    opp_id = uuid.uuid4()

    customer = CustomerContext(
        customer_id=cust_id,
        external_customer_id="cust_test_101",
        name="Test User",
        email="test@example.com",
        total_payments=total_cust_payments,
        successful_payments=total_cust_payments - 1 if total_cust_payments > 0 else 0,
        failed_payments=1 if total_cust_payments > 0 else 0,
        historical_success_rate=0.8 if total_cust_payments > 0 else 0.0,
        created_at=now,
    )

    payment = PaymentContext(
        payment_id=pay_id,
        external_payment_id="pay_test_101",
        amount=1500.0,
        currency="INR",
        payment_method=payment_method,
        status=payment_status,
        failure_reason=failure_reason,
        created_at=now,
    )

    opportunity = RecoveryOpportunityContext(
        opportunity_id=opp_id,
        status=opp_status,
        revenue_at_risk=1500.0,
        expected_recovery=1200.0,
        created_at=now,
    )

    stats = CustomerRecoveryStatsContext(
        total_recovery_opportunities=1 if recovery_rate > 0 else 0,
        recovered_opportunities=1 if recovery_rate > 0 else 0,
        failed_opportunities=0,
        recovery_rate=recovery_rate,
        previously_successful_actions=hist_successful_actions or [],
        previously_failed_actions=hist_failed_actions or [],
        total_amount_recovered=1200.0 if recovery_rate > 0 else 0.0,
    )

    return CustomerRecoveryContext(
        customer=customer,
        current_payment=payment,
        current_opportunity=opportunity,
        current_payment_attempts=current_attempts or [],
        historical_payments=[],
        recovery_statistics=stats,
        retrieved_at=now,
    )


def test_already_recovered_opportunity_returns_no_action():
    """Verify that an already recovered opportunity produces NO_ACTION with 1.0 confidence."""
    context = _build_test_context(opp_status="recovered", payment_status="succeeded")
    decision = evaluate_recovery_decision(context)

    assert decision.recommended_action == RecoveryAction.NO_ACTION
    assert decision.confidence == 1.0
    assert "already resolved" in decision.reason
    assert decision.decision_basis["rule_matched"] == "AlreadyRecoveredRule"


def test_max_attempts_exceeded_returns_no_action():
    """Verify that exceeding max retry attempts yields NO_ACTION."""
    now = datetime.now(timezone.utc)
    attempts = [
        RecoveryAttemptContext(action="retry_payment", status="failed", created_at=now),
        RecoveryAttemptContext(
            action="wait_and_retry", status="failed", created_at=now
        ),
        RecoveryAttemptContext(action="payment_link", status="failed", created_at=now),
    ]
    context = _build_test_context(current_attempts=attempts)
    decision = evaluate_recovery_decision(context, max_attempts=3)

    assert decision.recommended_action == RecoveryAction.NO_ACTION
    assert decision.confidence == 0.95
    assert "Maximum recovery attempts" in decision.reason
    assert decision.decision_basis["rule_matched"] == "MaxAttemptsExceededRule"


def test_transient_technical_failure_progression():
    """Verify progression: retry_payment (1st attempt) -> wait_and_retry (2nd) -> payment_link (3rd)."""
    now = datetime.now(timezone.utc)

    # 1st attempt: immediate retry
    ctx1 = _build_test_context(failure_reason="bank_server_down")
    d1 = evaluate_recovery_decision(ctx1)
    assert d1.recommended_action == RecoveryAction.RETRY_PAYMENT
    assert d1.confidence == 0.85

    # 2nd attempt: wait and retry
    att1 = [
        RecoveryAttemptContext(action="retry_payment", status="failed", created_at=now)
    ]
    ctx2 = _build_test_context(failure_reason="bank_server_down", current_attempts=att1)
    d2 = evaluate_recovery_decision(ctx2)
    assert d2.recommended_action == RecoveryAction.WAIT_AND_RETRY
    assert d2.confidence == 0.75

    # 3rd attempt: payment link fallback
    att2 = att1 + [
        RecoveryAttemptContext(action="wait_and_retry", status="failed", created_at=now)
    ]
    ctx3 = _build_test_context(failure_reason="bank_server_down", current_attempts=att2)
    d3 = evaluate_recovery_decision(ctx3)
    assert d3.recommended_action == RecoveryAction.PAYMENT_LINK
    assert d3.confidence == 0.70


def test_permanent_credential_failure():
    """Verify permanent card expiration or account closure requests CHANGE_PAYMENT_METHOD."""
    for reason in ["card_expired", "invalid_cvv", "account_closed", "lost_card"]:
        ctx = _build_test_context(failure_reason=reason)
        decision = evaluate_recovery_decision(ctx)
        assert decision.recommended_action == RecoveryAction.CHANGE_PAYMENT_METHOD
        assert decision.confidence == 0.90
        assert (
            decision.decision_basis["rule_matched"] == "PermanentCredentialFailureRule"
        )


def test_customer_authentication_failure():
    """Verify 3DS / OTP / user cancellation triggers PAYMENT_LINK for interactive re-entry."""
    for reason in ["authentication_failed", "otp_expired", "user_cancelled"]:
        ctx = _build_test_context(failure_reason=reason)
        decision = evaluate_recovery_decision(ctx)
        assert decision.recommended_action == RecoveryAction.PAYMENT_LINK
        assert decision.confidence == 0.80
        assert (
            decision.decision_basis["rule_matched"]
            == "CustomerInteractionPaymentLinkRule"
        )


def test_otp_timeout_classified_as_customer_interaction_regression():
    """
    Regression Test: otp_timeout contains 'timeout' (which is in TRANSIENT_TECHNICAL_REASONS),
    but because otp_timeout is an explicit customer interaction reason, exact category matching
    must take precedence and classify it as customer_interaction producing PAYMENT_LINK.
    """
    ctx = _build_test_context(failure_reason="otp_timeout")
    decision = evaluate_recovery_decision(ctx)

    assert decision.recommended_action == RecoveryAction.PAYMENT_LINK
    assert decision.recommended_action != RecoveryAction.RETRY_PAYMENT
    assert decision.confidence == 0.80
    assert (
        decision.decision_basis["rule_matched"] == "CustomerInteractionPaymentLinkRule"
    )


def test_insufficient_funds_with_historical_affinity():
    """Verify customer with previous successful payment link recovery receives payment link on insufficient funds."""
    ctx = _build_test_context(
        failure_reason="insufficient_funds",
        hist_successful_actions=["payment_link"],
        recovery_rate=0.8,
    )
    decision = evaluate_recovery_decision(ctx)
    assert decision.recommended_action == RecoveryAction.PAYMENT_LINK
    assert decision.confidence == 0.80
    assert "historical" in decision.reason.lower()


def test_insufficient_funds_cold_start_initial_wait():
    """Verify new customer with insufficient funds receives WAIT_AND_RETRY first."""
    ctx = _build_test_context(
        failure_reason="insufficient_funds",
        total_cust_payments=0,
        recovery_rate=0.0,
    )
    decision = evaluate_recovery_decision(ctx)
    assert decision.recommended_action == RecoveryAction.WAIT_AND_RETRY
    assert decision.confidence == 0.75


def test_historical_affinity_general_rule():
    """Verify general historical affinity boosts recommendation for established customers."""
    ctx = _build_test_context(
        failure_reason="generic_decline",
        hist_successful_actions=["smart_retry"],
        recovery_rate=0.8,
    )
    decision = evaluate_recovery_decision(ctx)
    assert decision.recommended_action == RecoveryAction.RETRY_PAYMENT
    assert decision.decision_basis["rule_matched"] == "HistoricalAffinityRule"
    assert decision.confidence >= 0.75


def test_cold_start_customer_fallback():
    """Verify cold-start customer with unknown failure reason safely falls back to PAYMENT_LINK."""
    ctx = _build_test_context(
        failure_reason="",
        total_cust_payments=0,
        recovery_rate=0.0,
    )
    decision = evaluate_recovery_decision(ctx)
    assert decision.recommended_action == RecoveryAction.PAYMENT_LINK
    assert decision.confidence == 0.55
    assert decision.decision_basis["rule_matched"] == "DefaultFallbackRule"


def test_unsupported_failure_reason_fallback():
    """Verify uncommon/unknown failure reason produces an explainable fallback."""
    ctx = _build_test_context(failure_reason="mysterious_merchant_custom_code_99")
    decision = evaluate_recovery_decision(ctx)
    assert decision.recommended_action == RecoveryAction.PAYMENT_LINK
    assert "Unsupported failure reason" in decision.reason
    assert decision.confidence == 0.50


def test_decision_determinism():
    """Verify running the engine multiple times with identical context returns exact same output."""
    ctx = _build_test_context(failure_reason="network_timeout")
    d1 = evaluate_recovery_decision(ctx)

    for _ in range(50):
        d_next = evaluate_recovery_decision(ctx)
        assert d_next.recommended_action == d1.recommended_action
        assert d_next.confidence == d1.confidence
        assert d_next.reason == d1.reason
        assert d_next.decision_basis == d1.decision_basis


def test_recovery_decision_immutability():
    """Verify that RecoveryDecision is frozen and cannot be mutated."""
    decision = RecoveryDecision(
        recommended_action=RecoveryAction.RETRY_PAYMENT,
        reason="Test reason",
        confidence=0.85,
    )

    with pytest.raises(ValidationError):
        decision.recommended_action = RecoveryAction.NO_ACTION  # type: ignore

    with pytest.raises(ValidationError):
        decision.confidence = 0.5  # type: ignore


def test_confidence_bounds_enforcement():
    """Verify that confidence scores outside [0.0, 1.0] are rejected by schema."""
    with pytest.raises(ValidationError):
        RecoveryDecision(
            recommended_action=RecoveryAction.RETRY_PAYMENT,
            reason="Invalid confidence",
            confidence=1.5,
        )

    with pytest.raises(ValidationError):
        RecoveryDecision(
            recommended_action=RecoveryAction.RETRY_PAYMENT,
            reason="Invalid confidence",
            confidence=-0.1,
        )


def test_decision_basis_recursive_immutability():
    """Verify direct, nested dictionary, and nested sequence mutation prevention on decision_basis."""
    mutable_input_basis = {
        "rule_matched": "TestRule",
        "attempted_actions": ["retry_payment", "wait_and_retry"],
        "nested_metadata": {
            "gateway": "stripe",
            "flags": ["high_risk", "auto_retry"],
        },
    }

    decision = RecoveryDecision(
        recommended_action=RecoveryAction.RETRY_PAYMENT,
        reason="Test recursive immutability",
        confidence=0.85,
        decision_basis=mutable_input_basis,
    )

    # 1. Direct mutation of decision_basis
    with pytest.raises(TypeError):
        decision.decision_basis["rule_matched"] = "MutatedRule"  # type: ignore

    with pytest.raises(TypeError):
        decision.decision_basis["new_key"] = "new_val"  # type: ignore

    # 2. Nested dictionary mutation
    with pytest.raises(TypeError):
        decision.decision_basis["nested_metadata"]["gateway"] = "razorpay"  # type: ignore

    with pytest.raises(TypeError):
        decision.decision_basis["nested_metadata"]["new_field"] = 123  # type: ignore

    # 3. Nested sequence mutation
    with pytest.raises((AttributeError, TypeError)):
        decision.decision_basis["attempted_actions"].append("payment_link")  # type: ignore

    with pytest.raises(TypeError):
        decision.decision_basis["attempted_actions"][0] = "mutated_action"  # type: ignore

    with pytest.raises((AttributeError, TypeError)):
        decision.decision_basis["nested_metadata"]["flags"].append("fraud")  # type: ignore

    # 4. Confirmation that original decision metadata remains unchanged when input dict is mutated
    mutable_input_basis["rule_matched"] = "ExternallyMutated"
    mutable_input_basis["attempted_actions"].append("external_append")
    mutable_input_basis["nested_metadata"]["gateway"] = "external_gateway"

    assert decision.decision_basis["rule_matched"] == "TestRule"
    assert decision.decision_basis["attempted_actions"] == (
        "retry_payment",
        "wait_and_retry",
    )
    assert decision.decision_basis["nested_metadata"]["gateway"] == "stripe"


def test_default_empty_decision_basis_is_immutable():
    """Verify that default decision_basis is also an immutable mapping."""
    decision = RecoveryDecision(
        recommended_action=RecoveryAction.NO_ACTION,
        reason="Default test",
        confidence=1.0,
    )
    with pytest.raises(TypeError):
        decision.decision_basis["new_key"] = "val"  # type: ignore


def test_explicit_none_decision_basis_is_normalized_to_immutable_mapping():
    """
    Regression Test: explicitly passing decision_basis=None must be normalized
    before Pydantic type validation into an empty immutable mapping.
    """
    decision = RecoveryDecision(
        recommended_action=RecoveryAction.NO_ACTION,
        reason="Explicit None test",
        confidence=1.0,
        decision_basis=None,  # type: ignore
    )

    assert decision.decision_basis == {}
    with pytest.raises(TypeError):
        decision.decision_basis["new_key"] = "val"  # type: ignore
