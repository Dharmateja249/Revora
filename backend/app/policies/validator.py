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
        try:
            effective_action = self.select_fallback_action(policy_context)
        except ValueError:
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

    def select_fallback_action(
        self,
        policy_context: RecoveryPolicyContext,
    ) -> RecoveryAction:
        """
        Deterministically select a safe, policy-compliant fallback recovery action.

        Evaluation order:
        1. Explicit mandatory_fallback_action if present, allowed, and not prohibited.
        2. Next highest-priority applicable rule's mandatory_fallback that is allowed and not prohibited.
        3. First compliant non-NO_ACTION action from policy_context.allowed_actions.
        4. RecoveryAction.NO_ACTION if allowed and not prohibited (fail-closed).
        5. If allowed_actions is empty and NO_ACTION is not prohibited, fails closed to NO_ACTION.

        Raises:
            TypeError: If policy_context is not a RecoveryPolicyContext.
            ValueError: If no policy-compliant fallback action can be established.
        """
        if not isinstance(policy_context, RecoveryPolicyContext):
            raise TypeError(
                f"Expected policy_context to be RecoveryPolicyContext, got {type(policy_context).__name__}"
            )

        # 1. Primary mandatory fallback action
        if (
            policy_context.mandatory_fallback_action is not None
            and policy_context.mandatory_fallback_action
            in policy_context.allowed_actions
            and policy_context.mandatory_fallback_action
            not in policy_context.prohibited_actions
        ):
            return policy_context.mandatory_fallback_action

        # 2. Check applicable rules in priority order for a compliant mandatory fallback
        for rule in policy_context.applicable_rules:
            if (
                rule.mandatory_fallback is not None
                and rule.mandatory_fallback in policy_context.allowed_actions
                and rule.mandatory_fallback not in policy_context.prohibited_actions
            ):
                return rule.mandatory_fallback

        # 3. Filter compliant actions (allowed and not prohibited)
        compliant_actions = [
            a
            for a in policy_context.allowed_actions
            if a not in policy_context.prohibited_actions
        ]

        if compliant_actions:
            # Prefer first compliant non-NO_ACTION action if permitted
            non_stop = [a for a in compliant_actions if a != RecoveryAction.NO_ACTION]
            if non_stop:
                return non_stop[0]
            if RecoveryAction.NO_ACTION in compliant_actions:
                return RecoveryAction.NO_ACTION

        # 4. If allowed_actions is empty, check if NO_ACTION is safe (fail-closed)
        if (
            not policy_context.allowed_actions
            and RecoveryAction.NO_ACTION not in policy_context.prohibited_actions
        ):
            return RecoveryAction.NO_ACTION

        # 5. Strict fail-closed: No policy-compliant fallback exists
        raise ValueError(
            "Cannot establish deterministic fallback: no policy-compliant recovery action available in policy context."
        )
