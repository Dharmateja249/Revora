"""
Revora Adaptive Recovery Agent Decision Orchestrator.

Connects context synthesis, prompt formatting, LLM provider execution,
deterministic policy validation, and error fallback handling into an
immutable AgentDecisionResult outcome.
"""

from datetime import timezone
import time
from typing import Any, Dict, List, Optional, Sequence

from app.agent.context_builder import AgentContextBuilder
from app.agent.prompts import build_agent_messages
from app.agent.provider import LLMProvider, LLMProviderError
from app.agent.schemas import (
    AgentDecisionPromptContext,
    AgentDecisionResult,
    LLMRecoveryRecommendation,
    utc_now,
)
from app.context import CustomerRecoveryContext
from app.decision_engine import RecoveryAction
from app.historical_retrieval import HistoricalCase
from app.policies.schemas import PolicyValidationResult, RecoveryPolicyContext
from app.policies.validator import PolicyValidator


class AgentOrchestrator:
    """
    Orchestrates the end-to-end adaptive recovery decision pipeline.

    Coordinates:
    CustomerRecoveryContext
            ↓
    AgentContextBuilder
            ↓
    build_agent_messages()
            ↓
    LLMProvider.generate()
            ↓
    LLMRecoveryRecommendation
            ↓
    PolicyValidator.validate_decision()
            ↓
    AgentDecisionResult
    """

    def __init__(
        self,
        provider: LLMProvider,
        context_builder: Optional[AgentContextBuilder] = None,
        policy_validator: Optional[PolicyValidator] = None,
    ):
        """
        Initialize the AgentOrchestrator with required and optional components.

        Args:
            provider: Component implementing the LLMProvider protocol.
            context_builder: Optional custom AgentContextBuilder (defaults to standard builder).
            policy_validator: Optional custom PolicyValidator (defaults to standard validator).

        Raises:
            TypeError: If any injected dependency violates expected types/contracts.
        """
        if not isinstance(provider, LLMProvider) or not hasattr(provider, "generate") or not callable(getattr(provider, "generate")):
            raise TypeError(
                f"Expected provider implementing LLMProvider protocol, got {type(provider).__name__}"
            )

        if context_builder is not None and not isinstance(context_builder, AgentContextBuilder):
            raise TypeError(
                f"Expected context_builder to be AgentContextBuilder, got {type(context_builder).__name__}"
            )

        if policy_validator is not None and not isinstance(policy_validator, PolicyValidator):
            raise TypeError(
                f"Expected policy_validator to be PolicyValidator, got {type(policy_validator).__name__}"
            )

        self._provider = provider
        self._context_builder = context_builder or AgentContextBuilder()
        self._policy_validator = policy_validator or PolicyValidator()

    @property
    def provider(self) -> LLMProvider:
        """Return the active LLM provider."""
        return self._provider

    @property
    def context_builder(self) -> AgentContextBuilder:
        """Return the active context builder."""
        return self._context_builder

    @property
    def policy_validator(self) -> PolicyValidator:
        """Return the active policy validator."""
        return self._policy_validator

    async def decide(
        self,
        context: CustomerRecoveryContext,
        policy_context: RecoveryPolicyContext,
        historical_cases: Optional[Sequence[HistoricalCase]] = None,
    ) -> AgentDecisionResult:
        """
        Execute the adaptive recovery decision pipeline.

        Args:
            context: Top-level CustomerRecoveryContext domain model.
            policy_context: Resolved RecoveryPolicyContext defining allowed and prohibited actions.
            historical_cases: Optional sequence of retrieved historical recovery cases.

        Returns:
            Immutable AgentDecisionResult containing the validated recommendation and telemetry.

        Raises:
            TypeError: If input arguments violate expected types.
            ValueError: If policy_context is missing, or allowed_actions is empty (fails closed).
        """
        # 1. Build sanitized prompt context (fails closed on invalid policy or context)
        prompt_context: AgentDecisionPromptContext = self._context_builder.build_prompt_context(
            context=context,
            historical_cases=historical_cases,
            policy_context=policy_context,
        )

        # 2. Build deterministic chat messages
        messages = build_agent_messages(prompt_context)

        # 3. Execute LLM Reasoning with monotonic high-resolution latency timer
        start_time = time.perf_counter()
        try:
            raw_recommendation: LLMRecoveryRecommendation = await self._provider.generate(messages)
            elapsed_ms = max(0.0, (time.perf_counter() - start_time) * 1000.0)
        except LLMProviderError as exc:
            elapsed_ms = max(0.0, (time.perf_counter() - start_time) * 1000.0)
            return self._build_deterministic_fallback_result(
                policy_context=policy_context,
                exception=exc,
                latency_ms=elapsed_ms,
            )

        # 4. Enforce deterministic policy validation on candidate action
        validation_result: PolicyValidationResult = self._policy_validator.validate_decision(
            candidate_action=raw_recommendation.recommended_action,
            policy_context=policy_context,
        )

        provider_name = getattr(self._provider, "provider_name", None)
        model_name = getattr(self._provider, "model_name", None)

        if not validation_result.was_overridden:
            # Candidate action complies with policy
            return AgentDecisionResult(
                recommendation=raw_recommendation,
                agent_used=True,
                provider=provider_name,
                model_name=model_name,
                is_fallback=False,
                fallback_reason=None,
                latency_ms=elapsed_ms,
                evaluated_at=utc_now(),
                metadata={
                    "policy_overridden": False,
                    "applied_policy_ids": list(validation_result.applied_policy_ids),
                    "violated_policy_ids": list(validation_result.violated_policy_ids),
                },
            )

        # 5. Policy violation override (Deterministic policy enforcement; LLM was used)
        effective_recommendation = LLMRecoveryRecommendation(
            recommended_action=validation_result.effective_action,
            confidence=raw_recommendation.confidence,
            reasoning=(
                f"{raw_recommendation.reasoning} "
                f"[Policy override: candidate '{raw_recommendation.recommended_action.value}' "
                f"overridden to '{validation_result.effective_action.value}']"
            ),
            key_factors=raw_recommendation.key_factors + (
                f"policy_override:{validation_result.effective_action.value}",
            ),
            referenced_case_ids=raw_recommendation.referenced_case_ids,
        )

        return AgentDecisionResult(
            recommendation=effective_recommendation,
            agent_used=True,
            provider=provider_name,
            model_name=model_name,
            is_fallback=False,
            fallback_reason=None,
            latency_ms=elapsed_ms,
            evaluated_at=utc_now(),
            metadata={
                "policy_overridden": True,
                "original_candidate_action": raw_recommendation.recommended_action.value,
                "applied_policy_ids": list(validation_result.applied_policy_ids),
                "violated_policy_ids": list(validation_result.violated_policy_ids),
            },
        )

    def _select_deterministic_fallback_action(
        self,
        policy_context: RecoveryPolicyContext,
    ) -> RecoveryAction:
        """
        Deterministically select a safe fallback recovery action strictly compliant with policy.

        Enforces the invariant that the fallback action must be in allowed_actions
        and must not be in prohibited_actions.

        Raises:
            ValueError: If no policy-compliant fallback action can be determined.
        """
        if (
            policy_context.mandatory_fallback_action is not None
            and policy_context.mandatory_fallback_action in policy_context.allowed_actions
            and policy_context.mandatory_fallback_action not in policy_context.prohibited_actions
        ):
            return policy_context.mandatory_fallback_action

        compliant_actions = [
            a
            for a in policy_context.allowed_actions
            if a not in policy_context.prohibited_actions
        ]
        if compliant_actions:
            return compliant_actions[0]

        raise ValueError(
            "Cannot establish deterministic fallback: no policy-compliant recovery action available in policy context."
        )

    def _build_deterministic_fallback_result(
        self,
        policy_context: RecoveryPolicyContext,
        exception: Exception,
        latency_ms: float,
    ) -> AgentDecisionResult:
        """
        Construct a deterministic fallback AgentDecisionResult when provider execution fails.

        Preserves privacy and security boundaries by sanitizing externally visible fallback reasons
        and reasoning messages, ensuring no raw exception details, customer PII, or prompt data are leaked.
        """
        fallback_action = self._select_deterministic_fallback_action(policy_context)
        error_type_name = type(exception).__name__

        fallback_rec = LLMRecoveryRecommendation(
            recommended_action=fallback_action,
            confidence=0.0,
            reasoning="Deterministic fallback triggered due to LLM provider failure.",
            key_factors=("deterministic_fallback", "provider_error"),
            referenced_case_ids=(),
        )

        provider_name = getattr(self._provider, "provider_name", None)
        model_name = getattr(self._provider, "model_name", None)

        return AgentDecisionResult(
            recommendation=fallback_rec,
            agent_used=False,
            provider=provider_name,
            model_name=model_name,
            is_fallback=True,
            fallback_reason="LLM provider failure; deterministic fallback applied",
            latency_ms=latency_ms,
            evaluated_at=utc_now(),
            metadata={
                "error_type": error_type_name,
                "policy_overridden": False,
                "applied_policy_ids": [r.policy_id for r in policy_context.applicable_rules],
                "violated_policy_ids": [],
            },
        )
