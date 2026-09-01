"""
Unit tests for Revora Policy Resolver.
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
from app.decision_engine import RecoveryAction
from app.policies.resolver import resolve_policy_context


@pytest.fixture
def base_context():
    """Helper fixture to create a valid baseline CustomerRecoveryContext."""
    cust_id = uuid.uuid4()
    pay_id = uuid.uuid4()
    opp_id = uuid.uuid4()
    now = datetime(2026, 6, 15, 12, 0, 0, tzinfo=timezone.utc)

    return CustomerRecoveryContext(
        customer=CustomerContext(
            customer_id=cust_id,
            email="cust@example.com",
            total_payments=5,
            successful_payments=4,
            failed_payments=1,
            historical_success_rate=0.8,
        ),
        current_payment=PaymentContext(
            payment_id=pay_id,
            amount=500.0,
            currency="INR",
            payment_method="upi",
            status="failed",
            failure_reason="bank_timeout",
            created_at=now,
        ),
        current_opportunity=RecoveryOpportunityContext(
            opportunity_id=opp_id,
            status="open",
            revenue_at_risk=500.0,
            expected_recovery=0.0,
            created_at=now,
        ),
        current_payment_attempts=[],
        historical_payments=[],
        recovery_statistics=CustomerRecoveryStatsContext(
            total_recovery_opportunities=1,
            recovered_opportunities=1,
            failed_opportunities=0,
            recovery_rate=1.0,
            previously_successful_actions=["retry_payment"],
            previously_failed_actions=[],
            total_amount_recovered=500.0,
        ),
        retrieved_at=now,
    )


def test_resolve_transient_failure_policy(base_context):
    """Verify transient gateway failure resolves business retry rule and permits retries."""
    policy_ctx = resolve_policy_context(base_context, provider="razorpay")

    assert policy_ctx.provider == "razorpay"
    assert RecoveryAction.RETRY_PAYMENT in policy_ctx.allowed_actions
    assert RecoveryAction.PAYMENT_LINK in policy_ctx.allowed_actions
    assert len(policy_ctx.prohibited_actions) == 0
    assert policy_ctx.mandatory_fallback_action == RecoveryAction.RETRY_PAYMENT

    rule_ids = [r.policy_id for r in policy_ctx.applicable_rules]
    assert "REVORA_TRANSIENT_GATEWAY_RETRY" in rule_ids


def test_resolve_permanent_card_decline_policy(base_context):
    """Verify permanent credential failure applies provider constraint prohibiting silent retries."""
    context = base_context.model_copy(
        update={
            "current_payment": base_context.current_payment.model_copy(
                update={"payment_method": "card", "failure_reason": "card_expired"}
            )
        }
    )

    policy_ctx = resolve_policy_context(context, provider="razorpay")

    assert RecoveryAction.RETRY_PAYMENT in policy_ctx.prohibited_actions
    assert RecoveryAction.WAIT_AND_RETRY in policy_ctx.prohibited_actions
    assert RecoveryAction.CHANGE_PAYMENT_METHOD in policy_ctx.allowed_actions
    assert policy_ctx.mandatory_fallback_action == RecoveryAction.CHANGE_PAYMENT_METHOD

    rule_ids = [r.policy_id for r in policy_ctx.applicable_rules]
    assert "RZP_PERMANENT_CREDENTIAL_DECLINE" in rule_ids


def test_resolve_customer_auth_2fa_policy(base_context):
    """Verify customer authentication failure applies 2FA provider constraint."""
    context = base_context.model_copy(
        update={
            "current_payment": base_context.current_payment.model_copy(
                update={"failure_reason": "authentication_failed"}
            )
        }
    )

    policy_ctx = resolve_policy_context(context, provider="razorpay")

    assert RecoveryAction.RETRY_PAYMENT in policy_ctx.prohibited_actions
    assert RecoveryAction.PAYMENT_LINK in policy_ctx.allowed_actions
    assert policy_ctx.mandatory_fallback_action == RecoveryAction.PAYMENT_LINK

    rule_ids = [r.policy_id for r in policy_ctx.applicable_rules]
    assert "RZP_CUSTOMER_AUTH_2FA_REQUIRED" in rule_ids


def test_resolve_insufficient_funds_policy(base_context):
    """Verify insufficient funds failure prohibits immediate retry and mandates cooldown/link."""
    context = base_context.model_copy(
        update={
            "current_payment": base_context.current_payment.model_copy(
                update={"failure_reason": "insufficient_funds"}
            )
        }
    )

    policy_ctx = resolve_policy_context(context, provider="razorpay")

    assert RecoveryAction.RETRY_PAYMENT in policy_ctx.prohibited_actions
    assert RecoveryAction.WAIT_AND_RETRY in policy_ctx.allowed_actions
    assert RecoveryAction.PAYMENT_LINK in policy_ctx.allowed_actions

    rule_ids = [r.policy_id for r in policy_ctx.applicable_rules]
    assert "REVORA_INSUFFICIENT_FUNDS_PROGRESSIVE" in rule_ids


def test_resolve_max_attempts_safety_rule_overrides_all(base_context):
    """Verify max recovery attempts safety rule strictly overrides any domain rule."""
    # Add 3 prior attempts to payment
    attempts = [
        RecoveryAttemptContext(action="retry_payment", status="failed"),
        RecoveryAttemptContext(action="wait_and_retry", status="failed"),
        RecoveryAttemptContext(action="payment_link", status="failed"),
    ]
    context = base_context.model_copy(update={"current_payment_attempts": attempts})

    policy_ctx = resolve_policy_context(context, provider="razorpay", max_attempts=3)

    assert policy_ctx.allowed_actions == (RecoveryAction.NO_ACTION,)
    assert RecoveryAction.RETRY_PAYMENT in policy_ctx.prohibited_actions
    assert RecoveryAction.PAYMENT_LINK in policy_ctx.prohibited_actions
    assert policy_ctx.mandatory_fallback_action == RecoveryAction.NO_ACTION

    rule_ids = [r.policy_id for r in policy_ctx.applicable_rules]
    assert "SAFETY_MAX_ATTEMPTS_EXCEEDED" in rule_ids


def test_resolve_already_recovered_safety_rule(base_context):
    """Verify already recovered opportunity restricts actions to NO_ACTION."""
    context = base_context.model_copy(
        update={
            "current_opportunity": base_context.current_opportunity.model_copy(
                update={"status": "recovered"}
            )
        }
    )

    policy_ctx = resolve_policy_context(context, provider="razorpay")

    assert policy_ctx.allowed_actions == (RecoveryAction.NO_ACTION,)
    assert policy_ctx.mandatory_fallback_action == RecoveryAction.NO_ACTION

    rule_ids = [r.policy_id for r in policy_ctx.applicable_rules]
    assert "SAFETY_ALREADY_RECOVERED" in rule_ids


def test_resolve_unknown_failure_uses_default_envelope(base_context):
    """Verify unrecognized failure reason resolves to default safe envelope."""
    context = base_context.model_copy(
        update={
            "current_payment": base_context.current_payment.model_copy(
                update={"failure_reason": "unrecognized_custom_decline_code_99"}
            )
        }
    )

    policy_ctx = resolve_policy_context(context, provider="razorpay")

    assert RecoveryAction.PAYMENT_LINK in policy_ctx.allowed_actions
    assert policy_ctx.mandatory_fallback_action == RecoveryAction.PAYMENT_LINK

    rule_ids = [r.policy_id for r in policy_ctx.applicable_rules]
    assert "REVORA_DEFAULT_SAFE_ENVELOPE" in rule_ids


def test_invalid_context_type_raises_type_error():
    """Verify TypeError raised if invalid context passed."""
    with pytest.raises(TypeError):
        resolve_policy_context({"not": "a context"})
