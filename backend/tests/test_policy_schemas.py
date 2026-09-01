"""
Unit tests for Revora Policy Schemas and Data Contracts.
"""

import pytest
from app.decision_engine import RecoveryAction
from app.policies.schemas import (
    PolicyRule,
    PolicyType,
    PolicyValidationResult,
    RecoveryPolicyContext,
)
from pydantic import ValidationError


def test_policy_type_values():
    """Verify enum members of PolicyType."""
    assert PolicyType.SAFETY.value == "safety"
    assert PolicyType.PROVIDER_CONSTRAINT.value == "provider_constraint"
    assert PolicyType.BUSINESS_RULE.value == "business_rule"


def test_policy_rule_valid_construction():
    """Verify valid PolicyRule construction and immutability."""
    rule = PolicyRule(
        policy_id="TEST_RULE_01",
        provider="razorpay",
        version="2026.1",
        policy_type=PolicyType.PROVIDER_CONSTRAINT,
        description="Test description",
        applicable_failure_reasons=("bank_timeout", "network_timeout"),
        applicable_payment_methods=("upi", "card"),
        allowed_actions=(RecoveryAction.RETRY_PAYMENT, RecoveryAction.PAYMENT_LINK),
        prohibited_actions=(RecoveryAction.CHANGE_PAYMENT_METHOD,),
        mandatory_fallback=RecoveryAction.RETRY_PAYMENT,
        priority=800,
        metadata={"foo": "bar"},
    )

    assert rule.policy_id == "TEST_RULE_01"
    assert rule.provider == "razorpay"
    assert rule.version == "2026.1"
    assert rule.policy_type == PolicyType.PROVIDER_CONSTRAINT
    assert rule.allowed_actions == (
        RecoveryAction.RETRY_PAYMENT,
        RecoveryAction.PAYMENT_LINK,
    )
    assert rule.prohibited_actions == (RecoveryAction.CHANGE_PAYMENT_METHOD,)
    assert rule.mandatory_fallback == RecoveryAction.RETRY_PAYMENT
    assert rule.priority == 800
    assert rule.metadata["foo"] == "bar"

    # Immutability
    with pytest.raises(ValidationError):
        rule.policy_id = "MUTATED"


def test_policy_rule_string_normalization_and_coercion():
    """Verify strings are trimmed and lowercased where appropriate."""
    rule = PolicyRule(
        policy_id=" RZP_RULE ",
        provider=" Razorpay ",
        version=" 1.0 ",
        policy_type=PolicyType.SAFETY,
        description=" Trimmed description ",
        applicable_failure_reasons=[" Bank_Timeout ", "NETWORK_ERROR"],
        applicable_payment_methods=[" UPI ", "CARD"],
        allowed_actions=["retry_payment", "payment_link"],
        prohibited_actions=["change_payment_method"],
        mandatory_fallback="retry_payment",
    )
    assert rule.policy_id == "RZP_RULE"
    assert rule.provider == "Razorpay"
    assert rule.version == "1.0"
    assert rule.description == "Trimmed description"
    assert rule.applicable_failure_reasons == ("bank_timeout", "network_error")
    assert rule.applicable_payment_methods == ("upi", "card")
    assert rule.allowed_actions == (
        RecoveryAction.RETRY_PAYMENT,
        RecoveryAction.PAYMENT_LINK,
    )
    assert rule.prohibited_actions == (RecoveryAction.CHANGE_PAYMENT_METHOD,)
    assert rule.mandatory_fallback == RecoveryAction.RETRY_PAYMENT


def test_policy_rule_invalid_fields_rejected():
    """Verify validation on empty strings, negative priority, and invalid actions."""
    with pytest.raises(ValidationError):
        PolicyRule(
            policy_id="",
            provider="razorpay",
            version="2026.1",
            policy_type=PolicyType.SAFETY,
            description="desc",
        )

    with pytest.raises(ValidationError):
        PolicyRule(
            policy_id="ID",
            provider="razorpay",
            version="2026.1",
            policy_type=PolicyType.SAFETY,
            description="desc",
            priority=-10,
        )

    with pytest.raises(ValidationError):
        PolicyRule(
            policy_id="ID",
            provider="razorpay",
            version="2026.1",
            policy_type=PolicyType.SAFETY,
            description="desc",
            allowed_actions=["invalid_action_name"],
        )


def test_policy_rule_inconsistency_rejected():
    """Verify an action cannot be both allowed and prohibited in the same rule."""
    with pytest.raises(ValidationError, match="cannot be both allowed and prohibited"):
        PolicyRule(
            policy_id="CONFLICT_RULE",
            provider="razorpay",
            version="2026.1",
            policy_type=PolicyType.BUSINESS_RULE,
            description="desc",
            allowed_actions=[RecoveryAction.RETRY_PAYMENT, RecoveryAction.PAYMENT_LINK],
            prohibited_actions=[RecoveryAction.RETRY_PAYMENT],
        )

    with pytest.raises(ValidationError, match="which is listed in prohibited_actions"):
        PolicyRule(
            policy_id="FALLBACK_CONFLICT",
            provider="razorpay",
            version="2026.1",
            policy_type=PolicyType.BUSINESS_RULE,
            description="desc",
            allowed_actions=[RecoveryAction.PAYMENT_LINK],
            prohibited_actions=[RecoveryAction.RETRY_PAYMENT],
            mandatory_fallback=RecoveryAction.RETRY_PAYMENT,
        )


def test_recovery_policy_context_immutability():
    """Verify RecoveryPolicyContext contract and immutability."""
    rule = PolicyRule(
        policy_id="SAFETY_MAX",
        provider="system",
        version="2026.1",
        policy_type=PolicyType.SAFETY,
        description="desc",
        allowed_actions=(RecoveryAction.NO_ACTION,),
        prohibited_actions=(RecoveryAction.RETRY_PAYMENT,),
        mandatory_fallback=RecoveryAction.NO_ACTION,
    )

    ctx = RecoveryPolicyContext(
        provider="razorpay",
        policy_version="2026.1",
        applicable_rules=(rule,),
        allowed_actions=(RecoveryAction.NO_ACTION,),
        prohibited_actions=(RecoveryAction.RETRY_PAYMENT,),
        mandatory_fallback_action=RecoveryAction.NO_ACTION,
        metadata={"key": "val"},
    )

    assert ctx.provider == "razorpay"
    assert ctx.policy_version == "2026.1"
    assert len(ctx.applicable_rules) == 1
    assert ctx.mandatory_fallback_action == RecoveryAction.NO_ACTION

    with pytest.raises(ValidationError):
        ctx.provider = "stripe"


def test_policy_validation_result_serialization():
    """Verify PolicyValidationResult JSON serialization."""
    res = PolicyValidationResult(
        is_valid=False,
        candidate_action=RecoveryAction.RETRY_PAYMENT,
        effective_action=RecoveryAction.PAYMENT_LINK,
        was_overridden=True,
        violated_policy_ids=("RZP_CUSTOMER_AUTH_2FA_REQUIRED",),
        applied_policy_ids=("RZP_CUSTOMER_AUTH_2FA_REQUIRED", "SAFETY_MAX_ATTEMPTS"),
        explanation="Action retry_payment prohibited by auth rule.",
        metadata={"foo": "bar"},
    )

    data = res.model_dump(mode="json")
    assert data["is_valid"] is False
    assert data["candidate_action"] == "retry_payment"
    assert data["effective_action"] == "payment_link"
    assert data["was_overridden"] is True
    assert data["violated_policy_ids"] == ["RZP_CUSTOMER_AUTH_2FA_REQUIRED"]
    assert data["applied_policy_ids"] == [
        "RZP_CUSTOMER_AUTH_2FA_REQUIRED",
        "SAFETY_MAX_ATTEMPTS",
    ]
