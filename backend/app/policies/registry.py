"""
Revora Policy Registry.

Defines deterministic, versioned policy rules across Safety invariants, verified
Provider constraints, and Revora Business rules.
"""

from typing import Dict, List, Mapping, Optional, Sequence, Tuple

from app.decision_engine import RecoveryAction
from app.policies.schemas import PolicyRule, PolicyType

DEFAULT_POLICY_VERSION: str = "2026.1"

# ============================================================================
# 1. HARD SAFETY CONSTRAINTS (Provider = "system", Priority = 1000)
# Invariants of Revora that must never be violated.
# ============================================================================

SAFETY_MAX_ATTEMPTS_RULE = PolicyRule(
    policy_id="SAFETY_MAX_ATTEMPTS_EXCEEDED",
    provider="system",
    version=DEFAULT_POLICY_VERSION,
    policy_type=PolicyType.SAFETY,
    description="Maximum recovery attempts reached or opportunity is in terminal state; all recovery actions are prohibited.",
    applicable_failure_reasons=(),
    applicable_payment_methods=(),
    allowed_actions=(RecoveryAction.NO_ACTION,),
    prohibited_actions=(
        RecoveryAction.RETRY_PAYMENT,
        RecoveryAction.WAIT_AND_RETRY,
        RecoveryAction.PAYMENT_LINK,
        RecoveryAction.CHANGE_PAYMENT_METHOD,
    ),
    mandatory_fallback=RecoveryAction.NO_ACTION,
    priority=1000,
    metadata={"scope": "safety_invariant"},
)

SAFETY_ALREADY_RECOVERED_RULE = PolicyRule(
    policy_id="SAFETY_ALREADY_RECOVERED",
    provider="system",
    version=DEFAULT_POLICY_VERSION,
    policy_type=PolicyType.SAFETY,
    description="Payment or recovery opportunity is already resolved and recovered; no further recovery actions permitted.",
    applicable_failure_reasons=(),
    applicable_payment_methods=(),
    allowed_actions=(RecoveryAction.NO_ACTION,),
    prohibited_actions=(
        RecoveryAction.RETRY_PAYMENT,
        RecoveryAction.WAIT_AND_RETRY,
        RecoveryAction.PAYMENT_LINK,
        RecoveryAction.CHANGE_PAYMENT_METHOD,
    ),
    mandatory_fallback=RecoveryAction.NO_ACTION,
    priority=1000,
    metadata={"scope": "safety_invariant"},
)

SAFETY_NO_ACTIVE_OPPORTUNITY_RULE = PolicyRule(
    policy_id="SAFETY_NO_ACTIVE_OPPORTUNITY",
    provider="system",
    version=DEFAULT_POLICY_VERSION,
    policy_type=PolicyType.SAFETY,
    description="No active payment or recovery opportunity present in evaluation context.",
    applicable_failure_reasons=(),
    applicable_payment_methods=(),
    allowed_actions=(RecoveryAction.NO_ACTION,),
    prohibited_actions=(
        RecoveryAction.RETRY_PAYMENT,
        RecoveryAction.WAIT_AND_RETRY,
        RecoveryAction.PAYMENT_LINK,
        RecoveryAction.CHANGE_PAYMENT_METHOD,
    ),
    mandatory_fallback=RecoveryAction.NO_ACTION,
    priority=1000,
    metadata={"scope": "safety_invariant"},
)


# ============================================================================
# 2. VERIFIED PAYMENT PROVIDER CONSTRAINTS (Provider = "razorpay", Priority = 800)
# Rules established by payment gateway / card network / 2FA mandates.
# ============================================================================

RZP_PERMANENT_CREDENTIAL_ERROR_RULE = PolicyRule(
    policy_id="RZP_PERMANENT_CREDENTIAL_DECLINE",
    provider="razorpay",
    version=DEFAULT_POLICY_VERSION,
    policy_type=PolicyType.PROVIDER_CONSTRAINT,
    description="Card/Account permanent decline (expired, invalid, stolen, closed). Automated retries are prohibited by provider/network regulations.",
    applicable_failure_reasons=(
        "card_expired",
        "expired_card",
        "invalid_card",
        "invalid_card_number",
        "invalid_cvv",
        "account_closed",
        "invalid_account",
        "lost_card",
        "stolen_card",
        "pickup_card",
        "blocked_account",
        "do_not_honor",
    ),
    applicable_payment_methods=(),
    allowed_actions=(
        RecoveryAction.CHANGE_PAYMENT_METHOD,
        RecoveryAction.PAYMENT_LINK,
        RecoveryAction.NO_ACTION,
    ),
    prohibited_actions=(
        RecoveryAction.RETRY_PAYMENT,
        RecoveryAction.WAIT_AND_RETRY,
    ),
    mandatory_fallback=RecoveryAction.CHANGE_PAYMENT_METHOD,
    priority=800,
    metadata={"regulatory_basis": "card_network_anti_spam_and_credential_expiry"},
)

RZP_CUSTOMER_AUTH_2FA_REQUIRED_RULE = PolicyRule(
    policy_id="RZP_CUSTOMER_AUTH_2FA_REQUIRED",
    provider="razorpay",
    version=DEFAULT_POLICY_VERSION,
    policy_type=PolicyType.PROVIDER_CONSTRAINT,
    description="Customer 2FA / 3DS interactive authentication required. Silent server-side retry cannot complete cardholder verification.",
    applicable_failure_reasons=(
        "authentication_failed",
        "otp_expired",
        "otp_timeout",
        "user_cancelled",
        "customer_cancelled",
        "declined_by_user",
        "3ds_failed",
        "pin_incorrect",
    ),
    applicable_payment_methods=(),
    allowed_actions=(
        RecoveryAction.PAYMENT_LINK,
        RecoveryAction.CHANGE_PAYMENT_METHOD,
        RecoveryAction.NO_ACTION,
    ),
    prohibited_actions=(
        RecoveryAction.RETRY_PAYMENT,
        RecoveryAction.WAIT_AND_RETRY,
    ),
    mandatory_fallback=RecoveryAction.PAYMENT_LINK,
    priority=800,
    metadata={"regulatory_basis": "rbi_and_3ds_cardholder_presence_mandate"},
)


# ============================================================================
# 3. REVORA BUSINESS RULES (Provider = "revora", Priority = 500)
# Merchant recovery strategy rules.
# ============================================================================

REVORA_INSUFFICIENT_FUNDS_RULE = PolicyRule(
    policy_id="REVORA_INSUFFICIENT_FUNDS_PROGRESSIVE",
    provider="revora",
    version=DEFAULT_POLICY_VERSION,
    policy_type=PolicyType.BUSINESS_RULE,
    description="Progressive recovery for balance deficits: immediate automated retry is prohibited to prevent bank decline spam.",
    applicable_failure_reasons=(
        "insufficient_funds",
        "low_balance",
        "balance_insufficient",
        "limit_exceeded",
        "daily_limit_exceeded",
        "exceeds_limit",
    ),
    applicable_payment_methods=(),
    allowed_actions=(
        RecoveryAction.WAIT_AND_RETRY,
        RecoveryAction.PAYMENT_LINK,
        RecoveryAction.CHANGE_PAYMENT_METHOD,
        RecoveryAction.NO_ACTION,
    ),
    prohibited_actions=(RecoveryAction.RETRY_PAYMENT,),
    mandatory_fallback=RecoveryAction.WAIT_AND_RETRY,
    priority=500,
    metadata={"strategy": "balance_deficit_cooldown"},
)

REVORA_TRANSIENT_GATEWAY_RULE = PolicyRule(
    policy_id="REVORA_TRANSIENT_GATEWAY_RETRY",
    provider="revora",
    version=DEFAULT_POLICY_VERSION,
    policy_type=PolicyType.BUSINESS_RULE,
    description="Transient gateway and bank network timeouts permit automated retry, scheduled retry, and interactive payment link.",
    applicable_failure_reasons=(
        "bank_server_down",
        "bank_timeout",
        "network_timeout",
        "system_error",
        "gateway_timeout",
        "internal_server_error",
        "service_unavailable",
        "connection_reset",
        "timeout",
    ),
    applicable_payment_methods=(),
    allowed_actions=(
        RecoveryAction.RETRY_PAYMENT,
        RecoveryAction.WAIT_AND_RETRY,
        RecoveryAction.PAYMENT_LINK,
        RecoveryAction.CHANGE_PAYMENT_METHOD,
        RecoveryAction.NO_ACTION,
    ),
    prohibited_actions=(),
    mandatory_fallback=RecoveryAction.RETRY_PAYMENT,
    priority=500,
    metadata={"strategy": "transient_retry_permitted"},
)

REVORA_DEFAULT_FALLBACK_RULE = PolicyRule(
    policy_id="REVORA_DEFAULT_SAFE_ENVELOPE",
    provider="revora",
    version=DEFAULT_POLICY_VERSION,
    policy_type=PolicyType.BUSINESS_RULE,
    description="Default safe envelope for unclassified or cold-start failure states: interactive links and method updates are permitted.",
    applicable_failure_reasons=(),
    applicable_payment_methods=(),
    allowed_actions=(
        RecoveryAction.PAYMENT_LINK,
        RecoveryAction.WAIT_AND_RETRY,
        RecoveryAction.CHANGE_PAYMENT_METHOD,
        RecoveryAction.NO_ACTION,
    ),
    prohibited_actions=(),
    mandatory_fallback=RecoveryAction.PAYMENT_LINK,
    priority=100,
    metadata={"strategy": "cold_start_safe_envelope"},
)


class PolicyRegistry:
    """
    Registry of versioned recovery policy rules.
    """

    def __init__(self, version: str = DEFAULT_POLICY_VERSION):
        self.version = version
        self._rules: Dict[str, PolicyRule] = {}
        self._register_defaults()

    def _register_defaults(self) -> None:
        """Register default safety, provider, and business rules."""
        default_rules = [
            SAFETY_MAX_ATTEMPTS_RULE,
            SAFETY_ALREADY_RECOVERED_RULE,
            SAFETY_NO_ACTIVE_OPPORTUNITY_RULE,
            RZP_PERMANENT_CREDENTIAL_ERROR_RULE,
            RZP_CUSTOMER_AUTH_2FA_REQUIRED_RULE,
            REVORA_INSUFFICIENT_FUNDS_RULE,
            REVORA_TRANSIENT_GATEWAY_RULE,
            REVORA_DEFAULT_FALLBACK_RULE,
        ]
        for rule in default_rules:
            self.register_rule(rule)

    def register_rule(self, rule: PolicyRule) -> None:
        """Register or update a policy rule in the registry."""
        if not isinstance(rule, PolicyRule):
            raise TypeError(f"Expected PolicyRule, got {type(rule).__name__}")
        self._rules[rule.policy_id] = rule

    def get_rule(self, policy_id: str) -> Optional[PolicyRule]:
        """Retrieve a rule by its policy_id."""
        return self._rules.get(policy_id)

    def list_rules(self) -> Tuple[PolicyRule, ...]:
        """Return all registered policy rules sorted by priority descending."""
        return tuple(sorted(self._rules.values(), key=lambda r: -r.priority))


# Global default registry instance
_default_registry: Optional[PolicyRegistry] = None


def get_policy_registry() -> PolicyRegistry:
    """Retrieve or initialize the global default PolicyRegistry instance."""
    global _default_registry
    if _default_registry is None:
        _default_registry = PolicyRegistry()
    return _default_registry
