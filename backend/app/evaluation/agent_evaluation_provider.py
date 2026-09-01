"""
Revora Offline Evaluation LLM Provider & Factory.

Provides a deterministic, offline, zero-external-API LLMProvider implementation
for evaluating AgentOrchestrator and AgentRAGPipeline across benchmarks without live LLM calls.
"""

import json
from collections.abc import Mapping, Sequence
from typing import Any

from app.agent.orchestrator import AgentOrchestrator
from app.agent.provider import LLMProvider, validate_chat_messages
from app.agent.schemas import LLMRecoveryRecommendation
from app.decision_engine import RecoveryAction


class EvaluationAgentLLMProvider(LLMProvider):
    """
    Deterministic offline reasoning provider for synthetic recovery evaluation.
    Simulates LLM agentic reasoning over structured prompt context, historical precedents,
    and policy envelopes without external network dependencies.
    """

    async def generate(
        self,
        messages: Sequence[Mapping[str, str]],
    ) -> LLMRecoveryRecommendation:
        """
        Execute deterministic evaluation reasoning from formatted chat messages.

        Args:
            messages: Chat messages containing system instructions and user JSON context.

        Returns:
            Validated LLMRecoveryRecommendation.
        """
        validate_chat_messages(messages)

        # Parse user message JSON payload (handling markdown fence if present)
        user_msg = messages[-1]["content"]
        payload: dict[str, Any] = {}
        try:
            if "```json" in user_msg:
                json_part = user_msg.split("```json", 1)[1].split("```", 1)[0].strip()
                payload = json.loads(json_part)
            elif "```" in user_msg:
                json_part = user_msg.split("```", 1)[1].split("```", 1)[0].strip()
                payload = json.loads(json_part)
            else:
                payload = json.loads(user_msg)
        except Exception:  # noqa: BLE001
            payload = {}

        policy_envelope = payload.get("policy_envelope", {})
        allowed_actions = policy_envelope.get("allowed_actions", [])
        prohibited_actions = policy_envelope.get("prohibited_actions", [])

        current_payment = payload.get("current_payment", {})
        failure_reason = str(current_payment.get("failure_reason", "")).lower()
        amount = float(current_payment.get("amount", 0.0))

        customer_profile = payload.get("customer_recovery_profile", {})
        risk_score = float(customer_profile.get("risk_score", 0.05))

        historical_cases = payload.get("historical_evidence", [])
        referenced_case_ids: list[str] = []

        # Find relevant historical precedents
        for case in historical_cases:
            case_id = str(case.get("payment_id", ""))
            if case.get("was_recovered") and case_id:
                referenced_case_ids.append(case_id)

        # Agentic decision heuristic based on failure domain, risk, and policy bounds
        chosen_action = RecoveryAction.NO_ACTION
        reasoning = "Evaluated failure profile and constraints."
        key_factors: list[str] = [f"failure_reason:{failure_reason}"]
        confidence = 0.85

        # 1. High risk / fraud declines
        if "stolen" in failure_reason or "fraud" in failure_reason or risk_score > 0.8:
            chosen_action = RecoveryAction.NO_ACTION
            reasoning = "High risk or fraudulent instrument decline; action prohibited."
            key_factors.append("high_fraud_risk")
            confidence = 0.95

        # 2. Expired card / instrument details
        elif "expired" in failure_reason or "invalid" in failure_reason:
            if RecoveryAction.PAYMENT_LINK.value in allowed_actions:
                chosen_action = RecoveryAction.PAYMENT_LINK
                reasoning = (
                    "Expired payment method requires customer payment link update."
                )
                key_factors.append("expired_instrument")
                confidence = 0.92
            elif RecoveryAction.CHANGE_PAYMENT_METHOD.value in allowed_actions:
                chosen_action = RecoveryAction.CHANGE_PAYMENT_METHOD
                reasoning = (
                    "Customer instrument expired; requesting payment method update."
                )
                confidence = 0.90

        # 3. Gateway / routing failure
        elif "gateway" in failure_reason or "routing" in failure_reason:
            if RecoveryAction.WAIT_AND_RETRY.value in allowed_actions:
                chosen_action = RecoveryAction.WAIT_AND_RETRY
                reasoning = "Gateway routing defect; deferred retry scheduled."
                key_factors.append("gateway_defect")
                confidence = 0.94
            elif RecoveryAction.RETRY_PAYMENT.value in allowed_actions:
                chosen_action = RecoveryAction.RETRY_PAYMENT
                reasoning = "Immediate retry on alternative route."
                confidence = 0.88

        # 4. High-value dispute / VIP customer
        elif "dispute" in failure_reason or amount >= 50000.0:
            if RecoveryAction.CHANGE_PAYMENT_METHOD.value in allowed_actions:
                chosen_action = RecoveryAction.CHANGE_PAYMENT_METHOD
                reasoning = (
                    "High-value VIP payment intervention via alternative method."
                )
                key_factors.append("high_value_transaction")
                confidence = 0.91
            elif RecoveryAction.PAYMENT_LINK.value in allowed_actions:
                chosen_action = RecoveryAction.PAYMENT_LINK
                reasoning = "Direct payment link outreach for high-value transaction."
                confidence = 0.88

        # 5. Network / Timeout / Insufficient funds
        else:
            if (
                RecoveryAction.RETRY_PAYMENT.value in allowed_actions
                and RecoveryAction.RETRY_PAYMENT.value not in prohibited_actions
            ):
                chosen_action = RecoveryAction.RETRY_PAYMENT
                reasoning = "Soft failure with positive retry precedent."
                key_factors.append("soft_recoverable_decline")
                confidence = 0.89
            elif RecoveryAction.PAYMENT_LINK.value in allowed_actions:
                chosen_action = RecoveryAction.PAYMENT_LINK
                reasoning = "Fallback payment link for unrecoverable soft decline."
                confidence = 0.82

        # Final safety check against allowed actions
        if allowed_actions and chosen_action.value not in allowed_actions:
            # Pick first available allowed action
            chosen_action = RecoveryAction(allowed_actions[0])
            reasoning = f"Policy constraint forced fallback to allowed action: {chosen_action.value}."
            confidence = 0.75

        return LLMRecoveryRecommendation(
            recommended_action=chosen_action,
            confidence=confidence,
            reasoning=reasoning,
            key_factors=tuple(key_factors),
            referenced_case_ids=tuple(referenced_case_ids[:3]),
        )


def create_evaluation_agent_orchestrator(
    provider: LLMProvider | None = None,
) -> AgentOrchestrator:
    """
    Construct an AgentOrchestrator instance preconfigured for evaluation and benchmarking.

    Args:
        provider: Optional custom LLMProvider (defaults to EvaluationAgentLLMProvider).

    Returns:
        Configured AgentOrchestrator instance.
    """
    active_provider = provider or EvaluationAgentLLMProvider()
    return AgentOrchestrator(provider=active_provider)
