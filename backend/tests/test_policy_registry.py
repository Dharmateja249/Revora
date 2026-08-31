"""
Unit tests for Revora Policy Registry.
"""

import pytest

from app.decision_engine import RecoveryAction
from app.policies.registry import (
    DEFAULT_POLICY_VERSION,
    PolicyRegistry,
    get_policy_registry,
)
from app.policies.schemas import PolicyRule, PolicyType


def test_default_registry_initialization():
    """Verify default policy registry registers safety, provider, and business rules."""
    registry = PolicyRegistry()
    rules = registry.list_rules()
    assert len(rules) >= 7

    # Sorted by priority descending
    priorities = [r.priority for r in rules]
    assert priorities == sorted(priorities, reverse=True)

    # Check presence of primary rules
    max_attempts = registry.get_rule("SAFETY_MAX_ATTEMPTS_EXCEEDED")
    assert max_attempts is not None
    assert max_attempts.policy_type == PolicyType.SAFETY
    assert max_attempts.priority == 1000
    assert max_attempts.allowed_actions == (RecoveryAction.NO_ACTION,)

    already_recovered = registry.get_rule("SAFETY_ALREADY_RECOVERED")
    assert already_recovered is not None
    assert already_recovered.policy_type == PolicyType.SAFETY

    perm_decline = registry.get_rule("RZP_PERMANENT_CREDENTIAL_DECLINE")
    assert perm_decline is not None
    assert perm_decline.policy_type == PolicyType.PROVIDER_CONSTRAINT
    assert perm_decline.priority == 800
    assert RecoveryAction.RETRY_PAYMENT in perm_decline.prohibited_actions

    auth_2fa = registry.get_rule("RZP_CUSTOMER_AUTH_2FA_REQUIRED")
    assert auth_2fa is not None
    assert auth_2fa.policy_type == PolicyType.PROVIDER_CONSTRAINT
    assert RecoveryAction.PAYMENT_LINK in auth_2fa.allowed_actions

    insufficient = registry.get_rule("REVORA_INSUFFICIENT_FUNDS_PROGRESSIVE")
    assert insufficient is not None
    assert insufficient.policy_type == PolicyType.BUSINESS_RULE
    assert insufficient.priority == 500


def test_custom_rule_registration():
    """Verify custom rule registration and override in registry."""
    registry = PolicyRegistry(version="2026.2")
    custom_rule = PolicyRule(
        policy_id="CUSTOM_MERCHANT_RULE",
        provider="custom_merchant",
        version="2026.2",
        policy_type=PolicyType.BUSINESS_RULE,
        description="Custom rule",
        applicable_failure_reasons=("custom_reason",),
        allowed_actions=(RecoveryAction.PAYMENT_LINK,),
        prohibited_actions=(RecoveryAction.RETRY_PAYMENT,),
        priority=600,
    )
    registry.register_rule(custom_rule)

    retrieved = registry.get_rule("CUSTOM_MERCHANT_RULE")
    assert retrieved == custom_rule
    assert custom_rule in registry.list_rules()


def test_global_get_policy_registry_singleton():
    """Verify get_policy_registry returns a stable singleton."""
    reg1 = get_policy_registry()
    reg2 = get_policy_registry()
    assert reg1 is reg2
    assert reg1.version == DEFAULT_POLICY_VERSION
