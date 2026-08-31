"""
Unit tests for Revora Deterministic Policy Validator.
"""

import pytest

from app.decision_engine import RecoveryAction
from app.policies.registry import (
    RZP_CUSTOMER_AUTH_2FA_REQUIRED_RULE,
    RZP_PERMANENT_CREDENTIAL_ERROR_RULE,
    SAFETY_MAX_ATTEMPTS_RULE,
)
from app.policies.schemas import RecoveryPolicyContext
from app.policies.validator import PolicyValidator


@pytest.fixture
def validator():
    return PolicyValidator()


def test_allowed_candidate_action_accepted(validator):
    """Verify an allowed candidate action is accepted without modification."""
    policy_ctx = RecoveryPolicyContext(
        provider="razorpay",
        policy_version="2026.1",
        applicable_rules=(RZP_CUSTOMER_AUTH_2FA_REQUIRED_RULE,),
        allowed_actions=(RecoveryAction.PAYMENT_LINK, RecoveryAction.CHANGE_PAYMENT_METHOD),
        prohibited_actions=(RecoveryAction.RETRY_PAYMENT, RecoveryAction.WAIT_AND_RETRY),
        mandatory_fallback_action=RecoveryAction.PAYMENT_LINK,
    )

    result = validator.validate_decision(
        candidate_action=RecoveryAction.PAYMENT_LINK,
        policy_context=policy_ctx,
    )

    assert result.is_valid is True
    assert result.was_overridden is False
    assert result.effective_action == RecoveryAction.PAYMENT_LINK
    assert len(result.violated_policy_ids) == 0
    assert result.applied_policy_ids == ("RZP_CUSTOMER_AUTH_2FA_REQUIRED",)


def test_prohibited_action_overridden_for_auth_failure(validator):
    """Verify retry_payment on authentication failure is overridden by policy validator to payment_link."""
    policy_ctx = RecoveryPolicyContext(
        provider="razorpay",
        policy_version="2026.1",
        applicable_rules=(RZP_CUSTOMER_AUTH_2FA_REQUIRED_RULE,),
        allowed_actions=(RecoveryAction.PAYMENT_LINK, RecoveryAction.CHANGE_PAYMENT_METHOD),
        prohibited_actions=(RecoveryAction.RETRY_PAYMENT, RecoveryAction.WAIT_AND_RETRY),
        mandatory_fallback_action=RecoveryAction.PAYMENT_LINK,
    )

    # Candidate proposing retry_payment (e.g. from historical precedent or prompt hallucination)
    result = validator.validate_decision(
        candidate_action=RecoveryAction.RETRY_PAYMENT,
        policy_context=policy_ctx,
    )

    assert result.is_valid is False
    assert result.was_overridden is True
    assert result.candidate_action == RecoveryAction.RETRY_PAYMENT
    assert result.effective_action == RecoveryAction.PAYMENT_LINK
    assert "RZP_CUSTOMER_AUTH_2FA_REQUIRED" in result.violated_policy_ids


def test_prohibited_action_overridden_for_card_expiry(validator):
    """Verify retry_payment on expired card is overridden to change_payment_method."""
    policy_ctx = RecoveryPolicyContext(
        provider="razorpay",
        policy_version="2026.1",
        applicable_rules=(RZP_PERMANENT_CREDENTIAL_ERROR_RULE,),
        allowed_actions=(RecoveryAction.CHANGE_PAYMENT_METHOD, RecoveryAction.PAYMENT_LINK),
        prohibited_actions=(RecoveryAction.RETRY_PAYMENT, RecoveryAction.WAIT_AND_RETRY),
        mandatory_fallback_action=RecoveryAction.CHANGE_PAYMENT_METHOD,
    )

    result = validator.validate_decision(
        candidate_action=RecoveryAction.RETRY_PAYMENT,
        policy_context=policy_ctx,
    )

    assert result.is_valid is False
    assert result.was_overridden is True
    assert result.effective_action == RecoveryAction.CHANGE_PAYMENT_METHOD
    assert "RZP_PERMANENT_CREDENTIAL_DECLINE" in result.violated_policy_ids


def test_max_attempts_exceeded_overrides_to_no_action(validator):
    """Verify any action proposed when max attempts exceeded is overridden to NO_ACTION."""
    policy_ctx = RecoveryPolicyContext(
        provider="system",
        policy_version="2026.1",
        applicable_rules=(SAFETY_MAX_ATTEMPTS_RULE,),
        allowed_actions=(RecoveryAction.NO_ACTION,),
        prohibited_actions=(
            RecoveryAction.RETRY_PAYMENT,
            RecoveryAction.WAIT_AND_RETRY,
            RecoveryAction.PAYMENT_LINK,
            RecoveryAction.CHANGE_PAYMENT_METHOD,
        ),
        mandatory_fallback_action=RecoveryAction.NO_ACTION,
    )

    result = validator.validate_decision(
        candidate_action=RecoveryAction.PAYMENT_LINK,
        policy_context=policy_ctx,
    )

    assert result.is_valid is False
    assert result.was_overridden is True
    assert result.effective_action == RecoveryAction.NO_ACTION
    assert "SAFETY_MAX_ATTEMPTS_EXCEEDED" in result.violated_policy_ids


def test_empty_allowed_actions_fails_closed_to_no_action(validator):
    """Verify validator fails closed to NO_ACTION if no allowed actions exist."""
    policy_ctx = RecoveryPolicyContext(
        provider="razorpay",
        policy_version="2026.1",
        applicable_rules=(),
        allowed_actions=(),
        prohibited_actions=(RecoveryAction.RETRY_PAYMENT,),
        mandatory_fallback_action=None,
    )

    result = validator.validate_decision(
        candidate_action=RecoveryAction.RETRY_PAYMENT,
        policy_context=policy_ctx,
    )

    assert result.is_valid is False
    assert result.was_overridden is True
    assert result.effective_action == RecoveryAction.NO_ACTION


def test_invalid_types_raise_type_error(validator):
    """Verify TypeError on invalid argument types."""
    policy_ctx = RecoveryPolicyContext(
        provider="razorpay",
        policy_version="2026.1",
        applicable_rules=(),
        allowed_actions=(RecoveryAction.PAYMENT_LINK,),
    )

    with pytest.raises(TypeError):
        validator.validate_decision("not_an_action", policy_ctx)

    with pytest.raises(TypeError):
        validator.validate_decision(RecoveryAction.PAYMENT_LINK, {"not": "a_context"})
