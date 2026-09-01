"""
Revora Deterministic Policy Validator.

Validates candidate recovery recommendations against an immutable RecoveryPolicyContext,
guaranteeing fail-closed deterministic policy enforcement before action persistence.
"""

from app.decision_engine import RecoveryAction
from app.policies.schemas import PolicyValidationResult, RecoveryPolicyContext


class PolicyValidator:
    """
    Deterministic policy and safety validator for recovery decisions.

    Acts as the unbypassable gate ensuring that no candidate action (whether produced
    by a rule engine, historical evidence, or a future LLM) violates active policy rules.
    """

    def validate_decision(
        self,
        candidate_action: RecoveryAction,
        policy_context: RecoveryPolicyContext,
    ) -> PolicyValidationResult:
        """
        Validate a candidate recovery action against the active policy envelope.

        Args:
            candidate_action: Candidate RecoveryAction proposed by decision engine or LLM.
            policy_context: Resolved RecoveryPolicyContext defining allowed and prohibited actions.

        Returns:
            PolicyValidationResult with validation status, effective action, override telemetry,
            and violated/applied policy IDs.
        """
        if not isinstance(candidate_action, RecoveryAction):
            raise TypeError(
                f"Expected candidate_action to be RecoveryAction, got {type(candidate_action).__name__}"
            )
        if not isinstance(policy_context, RecoveryPolicyContext):
            raise TypeError(
                f"Expected policy_context to be RecoveryPolicyContext, got {type(policy_context).__name__}"
            )

        applied_ids: tuple[str, ...] = tuple(
            r.policy_id for r in policy_context.applicable_rules
        )

        # 1. Check if candidate is explicitly prohibited or missing from allowed actions
        is_prohibited = candidate_action in policy_context.prohibited_actions
        is_allowed = candidate_action in policy_context.allowed_actions

        if is_allowed and not is_prohibited:
            return PolicyValidationResult(
                is_valid=True,
                candidate_action=candidate_action,
                effective_action=candidate_action,
                was_overridden=False,
                violated_policy_ids=(),
                applied_policy_ids=applied_ids,
                explanation=f"Candidate action '{candidate_action.value}' strictly complies with active policy rules.",
                metadata={
                    "policy_version": policy_context.policy_version,
                    "provider": policy_context.provider,
                },
            )

        # 2. Candidate action violated active policy constraints -> Deterministic override
        violated_rules: list[str] = []
        for rule in policy_context.applicable_rules:
            if candidate_action in rule.prohibited_actions or (
                rule.allowed_actions and candidate_action not in rule.allowed_actions
            ):
                violated_rules.append(rule.policy_id)

        violated_tuple = (
            tuple(violated_rules)
            if violated_rules
            else ("POLICY_ACTION_NOT_PERMITTED",)
        )

        # 3. Select effective fallback action deterministically
        effective_action: RecoveryAction
        if policy_context.mandatory_fallback_action is not None and (
            policy_context.mandatory_fallback_action in policy_context.allowed_actions
            and policy_context.mandatory_fallback_action
            not in policy_context.prohibited_actions
        ):
            effective_action = policy_context.mandatory_fallback_action
        elif policy_context.allowed_actions:
            # Pick first permitted non-NO_ACTION if possible, else NO_ACTION
            non_stop = [
                a
                for a in policy_context.allowed_actions
                if a != RecoveryAction.NO_ACTION
            ]
            effective_action = non_stop[0] if non_stop else RecoveryAction.NO_ACTION
        else:
            # Strict fail-closed
            effective_action = RecoveryAction.NO_ACTION

        explanation = (
            f"Candidate action '{candidate_action.value}' violated policy rules "
            f"({', '.join(violated_tuple)}); deterministically overridden with compliant action '{effective_action.value}'."
        )

        return PolicyValidationResult(
            is_valid=False,
            candidate_action=candidate_action,
            effective_action=effective_action,
            was_overridden=True,
            violated_policy_ids=violated_tuple,
            applied_policy_ids=applied_ids,
            explanation=explanation,
            metadata={
                "policy_version": policy_context.policy_version,
                "provider": policy_context.provider,
                "original_candidate": candidate_action.value,
            },
        )
