"""
Revora Policy Resolver.

Deterministically resolves the applicable PolicyRule set and computes the immutable
RecoveryPolicyContext envelope for a given CustomerRecoveryContext.
"""

from app.context import CustomerRecoveryContext
from app.decision_engine import RecoveryAction
from app.policies.registry import (
    PolicyRegistry,
    get_policy_registry,
)
from app.policies.schemas import PolicyRule, PolicyType, RecoveryPolicyContext


def _matches_failure_reason(
    rule_reasons: tuple[str, ...], failure_reason: str | None
) -> bool:
    """Check if failure reason matches any pattern defined in the rule."""
    if not rule_reasons:
        return False
    if not failure_reason:
        return False
    clean_reason = failure_reason.strip().lower()
    if clean_reason in rule_reasons:
        return True
    return any(term in clean_reason for term in rule_reasons)


def _matches_payment_method(
    rule_methods: tuple[str, ...], payment_method: str | None
) -> bool:
    """Check if payment method matches any pattern defined in the rule."""
    if not rule_methods:
        return True  # Empty means all methods applicable
    if not payment_method:
        return False
    clean_method = payment_method.strip().lower()
    return clean_method in rule_methods or "*" in rule_methods


def resolve_policy_context(
    context: CustomerRecoveryContext,
    provider: str = "razorpay",
    registry: PolicyRegistry | None = None,
    max_attempts: int = 3,
) -> RecoveryPolicyContext:
    """
    Deterministically resolve the active policy envelope for the evaluation context.

    Evaluates:
    1. Hard Safety Constraints (highest priority: terminal opportunity, max attempts, missing context)
    2. Provider Constraints (Razorpay/network rules for card permanent failures, 2FA auth requirements)
    3. Revora Business Rules (insufficient funds progressive escalation, transient timeout policies)
    4. Fallback Envelope

    Returns:
        RecoveryPolicyContext containing applicable rules, allowed actions, prohibited actions,
        and mandatory fallback action.
    """
    if not isinstance(context, CustomerRecoveryContext):
        raise TypeError(
            f"Expected CustomerRecoveryContext, got {type(context).__name__}"
        )

    reg = registry or get_policy_registry()

    current_payment = context.current_payment
    current_opportunity = context.current_opportunity
    raw_attempts = context.current_payment_attempts or []
    attempt_count = len(raw_attempts)

    applicable_rules: list[PolicyRule] = []

    # =========================================================================
    # Phase 1: Safety Invariant Checks (Priority = 1000)
    # =========================================================================

    if current_payment is None or current_opportunity is None:
        no_active_rule = reg.get_rule("SAFETY_NO_ACTIVE_OPPORTUNITY")
        if no_active_rule:
            applicable_rules.append(no_active_rule)
        return RecoveryPolicyContext(
            provider=provider,
            policy_version=reg.version,
            applicable_rules=tuple(applicable_rules),
            allowed_actions=(RecoveryAction.NO_ACTION,),
            prohibited_actions=(
                RecoveryAction.RETRY_PAYMENT,
                RecoveryAction.WAIT_AND_RETRY,
                RecoveryAction.PAYMENT_LINK,
                RecoveryAction.CHANGE_PAYMENT_METHOD,
            ),
            mandatory_fallback_action=RecoveryAction.NO_ACTION,
            metadata={"resolved_by": "SafetyNoActiveOpportunity"},
        )

    if (
        current_opportunity.status == "recovered"
        or current_payment.status == "succeeded"
    ):
        recovered_rule = reg.get_rule("SAFETY_ALREADY_RECOVERED")
        if recovered_rule:
            applicable_rules.append(recovered_rule)
        return RecoveryPolicyContext(
            provider=provider,
            policy_version=reg.version,
            applicable_rules=tuple(applicable_rules),
            allowed_actions=(RecoveryAction.NO_ACTION,),
            prohibited_actions=(
                RecoveryAction.RETRY_PAYMENT,
                RecoveryAction.WAIT_AND_RETRY,
                RecoveryAction.PAYMENT_LINK,
                RecoveryAction.CHANGE_PAYMENT_METHOD,
            ),
            mandatory_fallback_action=RecoveryAction.NO_ACTION,
            metadata={"resolved_by": "SafetyAlreadyRecovered"},
        )

    if attempt_count >= max_attempts or current_opportunity.status in (
        "failed",
        "abandoned",
    ):
        max_attempts_rule = reg.get_rule("SAFETY_MAX_ATTEMPTS_EXCEEDED")
        if max_attempts_rule:
            applicable_rules.append(max_attempts_rule)
        return RecoveryPolicyContext(
            provider=provider,
            policy_version=reg.version,
            applicable_rules=tuple(applicable_rules),
            allowed_actions=(RecoveryAction.NO_ACTION,),
            prohibited_actions=(
                RecoveryAction.RETRY_PAYMENT,
                RecoveryAction.WAIT_AND_RETRY,
                RecoveryAction.PAYMENT_LINK,
                RecoveryAction.CHANGE_PAYMENT_METHOD,
            ),
            mandatory_fallback_action=RecoveryAction.NO_ACTION,
            metadata={"resolved_by": "SafetyMaxAttemptsExceeded"},
        )

    fraud_rule = reg.get_rule("SAFETY_FRAUD_SECURITY_DECLINE")
    if fraud_rule and _matches_failure_reason(
        fraud_rule.applicable_failure_reasons, current_payment.failure_reason
    ):
        applicable_rules.append(fraud_rule)
        return RecoveryPolicyContext(
            provider=provider,
            policy_version=reg.version,
            applicable_rules=tuple(applicable_rules),
            allowed_actions=(RecoveryAction.NO_ACTION,),
            prohibited_actions=(
                RecoveryAction.RETRY_PAYMENT,
                RecoveryAction.WAIT_AND_RETRY,
                RecoveryAction.PAYMENT_LINK,
                RecoveryAction.CHANGE_PAYMENT_METHOD,
            ),
            mandatory_fallback_action=RecoveryAction.NO_ACTION,
            metadata={
                "resolved_by": "SafetyFraudSecurityDecline",
                "primary_rule_id": fraud_rule.policy_id,
            },
        )

    # =========================================================================
    # Phase 2: Domain Rule Matching (Provider Constraints & Business Rules)
    # =========================================================================

    failure_reason = current_payment.failure_reason or ""
    payment_method = current_payment.payment_method or ""

    # Check all registered rules
    all_registered = reg.list_rules()
    domain_rules_matched: list[PolicyRule] = []

    for rule in all_registered:
        if rule.policy_type == PolicyType.SAFETY:
            continue  # Safety already evaluated in Phase 1

        if (
            rule.applicable_failure_reasons
            and _matches_failure_reason(rule.applicable_failure_reasons, failure_reason)
            and _matches_payment_method(rule.applicable_payment_methods, payment_method)
        ):
            domain_rules_matched.append(rule)

    if not domain_rules_matched:
        # Fallback to default safe envelope
        fallback_rule = reg.get_rule("REVORA_DEFAULT_SAFE_ENVELOPE")
        if fallback_rule:
            domain_rules_matched.append(fallback_rule)

    applicable_rules.extend(domain_rules_matched)

    # Sort applicable rules by priority descending
    applicable_rules.sort(key=lambda r: -r.priority)

    # =========================================================================
    # Phase 3: Precedence-Aware Action Envelope Calculation
    # =========================================================================

    # 1. Accumulate all prohibited actions across all matched rules
    all_prohibited: set[RecoveryAction] = set()
    for rule in applicable_rules:
        all_prohibited.update(rule.prohibited_actions)

    # 2. Determine allowed actions from the highest-priority matching rule
    primary_rule = applicable_rules[0]
    candidate_allowed: list[RecoveryAction] = [
        act for act in primary_rule.allowed_actions if act not in all_prohibited
    ]

    # Always ensure NO_ACTION is available if no recovery actions are possible
    if (
        RecoveryAction.NO_ACTION not in candidate_allowed
        and RecoveryAction.NO_ACTION not in all_prohibited
    ):
        candidate_allowed.append(RecoveryAction.NO_ACTION)

    # 3. Determine mandatory fallback
    mandatory_fallback: RecoveryAction | None = None
    if (
        primary_rule.mandatory_fallback
        and primary_rule.mandatory_fallback not in all_prohibited
    ):
        mandatory_fallback = primary_rule.mandatory_fallback
    elif candidate_allowed:
        # Pick first non-NO_ACTION allowed action if available, else NO_ACTION
        non_stop = [a for a in candidate_allowed if a != RecoveryAction.NO_ACTION]
        mandatory_fallback = non_stop[0] if non_stop else RecoveryAction.NO_ACTION
    else:
        # Fail closed
        candidate_allowed = [RecoveryAction.NO_ACTION]
        mandatory_fallback = RecoveryAction.NO_ACTION

    return RecoveryPolicyContext(
        provider=provider,
        policy_version=reg.version,
        applicable_rules=tuple(applicable_rules),
        allowed_actions=tuple(candidate_allowed),
        prohibited_actions=tuple(sorted(all_prohibited, key=lambda a: a.value)),
        mandatory_fallback_action=mandatory_fallback,
        metadata={
            "primary_rule_id": primary_rule.policy_id,
            "matched_rules_count": len(applicable_rules),
        },
    )
