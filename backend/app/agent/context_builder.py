"""
Revora Agent Context Builder.

Transforms domain contexts (CustomerRecoveryContext, HistoricalCase, RecoveryPolicyContext)
into sanitized, bounded, immutable AgentDecisionPromptContext instances with strict PII protection
and deterministic ordering.
"""

from typing import Any, Dict, List, Optional, Sequence, Tuple

from app.agent.schemas import AgentDecisionPromptContext
from app.context import (
    CustomerContext,
    CustomerRecoveryContext,
    CustomerRecoveryStatsContext,
    PaymentContext,
    RecoveryAttemptContext,
)
from app.decision_engine import RecoveryAction
from app.historical_retrieval import HistoricalCase
from app.policies.schemas import RecoveryPolicyContext


class AgentContextBuilder:
    """
    Transforms domain context, historical RAG evidence, and policy boundaries
    into a sanitized, PII-free AgentDecisionPromptContext for LLM consumption.
    """

    def __init__(self, max_historical_cases: int = 5):
        if not isinstance(max_historical_cases, int) or isinstance(max_historical_cases, bool):
            raise TypeError(f"max_historical_cases must be an integer, got {type(max_historical_cases).__name__}")
        if max_historical_cases <= 0:
            raise ValueError(f"max_historical_cases must be a positive integer, got {max_historical_cases}")
        self.max_historical_cases = max_historical_cases

    def build_prompt_context(
        self,
        context: CustomerRecoveryContext,
        historical_cases: Optional[Sequence[HistoricalCase]] = None,
        policy_context: Optional[RecoveryPolicyContext] = None,
    ) -> AgentDecisionPromptContext:
        """
        Build an immutable, sanitized AgentDecisionPromptContext from domain models.

        Args:
            context: Top-level CustomerRecoveryContext from context retrieval engine.
            historical_cases: Optional sequence of retrieved historical recovery cases.
            policy_context: Optional resolved RecoveryPolicyContext.

        Returns:
            Immutable AgentDecisionPromptContext safe for LLM prompt generation.
        """
        if not isinstance(context, CustomerRecoveryContext):
            raise TypeError(f"Expected CustomerRecoveryContext, got {type(context).__name__}")

        current_payment_payload = self._build_payment_context(context.current_payment)
        customer_profile_payload = self._build_customer_profile(
            context.customer,
            context.recovery_statistics,
        )
        attempt_history_payload = self._build_attempt_history(context.current_payment_attempts)
        historical_cases_payload = self._build_historical_cases(historical_cases)
        allowed_actions, prohibited_actions, mandatory_fallback, policy_constraints = (
            self._build_policy_boundary(policy_context)
        )

        return AgentDecisionPromptContext(
            current_payment=current_payment_payload,
            customer_profile=customer_profile_payload,
            recovery_attempt_history=attempt_history_payload,
            historical_cases=historical_cases_payload,
            allowed_actions=allowed_actions,
            prohibited_actions=prohibited_actions,
            mandatory_fallback=mandatory_fallback,
            policy_constraints=policy_constraints,
        )

    def _build_payment_context(
        self,
        payment: Optional[PaymentContext],
    ) -> Dict[str, Any]:
        """
        Extract only approved payment fields using an explicit allowlist.
        Excludes payment_id, external_payment_id, and internal database attributes.
        """
        if payment is None:
            return {
                "amount": 0.0,
                "currency": "INR",
                "payment_method": "unspecified",
                "failure_reason": "unspecified",
            }

        return {
            "amount": float(payment.amount),
            "currency": str(payment.currency),
            "payment_method": str(payment.payment_method),
            "failure_reason": str(payment.failure_reason) if payment.failure_reason else "unspecified",
        }

    def _build_customer_profile(
        self,
        customer: Optional[CustomerContext],
        stats: Optional[CustomerRecoveryStatsContext],
    ) -> Dict[str, Any]:
        """
        Extract anonymized customer aggregates and recovery statistics.
        Strictly excludes customer_id, external_customer_id, name, and email.
        """
        total_payments = int(customer.total_payments) if customer else 0
        successful_payments = int(customer.successful_payments) if customer else 0
        failed_payments = int(customer.failed_payments) if customer else 0
        historical_success_rate = float(customer.historical_success_rate) if customer else 0.0

        total_recovery_opportunities = int(stats.total_recovery_opportunities) if stats else 0
        recovered_opportunities = int(stats.recovered_opportunities) if stats else 0
        failed_opportunities = int(stats.failed_opportunities) if stats else 0
        recovery_rate = float(stats.recovery_rate) if stats else 0.0
        previously_successful_actions = (
            [str(a) for a in stats.previously_successful_actions] if stats else []
        )
        previously_failed_actions = (
            [str(a) for a in stats.previously_failed_actions] if stats else []
        )
        lifetime_amount_recovered = float(stats.total_amount_recovered) if stats else 0.0

        return {
            "total_payments": total_payments,
            "successful_payments": successful_payments,
            "failed_payments": failed_payments,
            "historical_success_rate": historical_success_rate,
            "total_recovery_opportunities": total_recovery_opportunities,
            "recovered_opportunities": recovered_opportunities,
            "failed_opportunities": failed_opportunities,
            "recovery_rate": recovery_rate,
            "previously_successful_actions": previously_successful_actions,
            "previously_failed_actions": previously_failed_actions,
            "lifetime_amount_recovered": lifetime_amount_recovered,
        }

    def _build_attempt_history(
        self,
        attempts: Optional[Sequence[RecoveryAttemptContext]],
    ) -> List[Dict[str, Any]]:
        """
        Extract chronological attempt sequence with sanitized fields.
        Excludes attempt_id, external_reference, and raw timestamps.
        """
        if not attempts:
            return []

        payload = []
        for idx, attempt in enumerate(attempts):
            payload.append(
                {
                    "attempt_number": idx + 1,
                    "action": str(attempt.action),
                    "status": str(attempt.status),
                    "error_code": str(attempt.error_code) if attempt.error_code else "none",
                }
            )
        return payload

    def _build_historical_cases(
        self,
        cases: Optional[Sequence[HistoricalCase]],
    ) -> List[Dict[str, Any]]:
        """
        Sort, cap, and format historical RAG evidence.
        Applies relevance score DESC with case_id ASC tie-breaker.
        Excludes customer_id, raw metadata, and internal database references.
        """
        if not cases:
            return []

        # Deterministic sorting: relevance_score DESC, payment_id ASC
        sorted_cases = sorted(
            cases,
            key=lambda c: (
                -(c.relevance_score if c.relevance_score is not None else 0.0),
                str(c.payment_id),
            ),
        )

        capped_cases = sorted_cases[: self.max_historical_cases]
        payload = []
        for case in capped_cases:
            payload.append(
                {
                    "case_id": str(case.payment_id),
                    "amount": float(case.amount),
                    "currency": str(case.currency),
                    "payment_method": str(case.payment_method),
                    "failure_reason": str(case.failure_reason) if case.failure_reason else "unspecified",
                    "recovery_action": str(case.recovery_action) if case.recovery_action else "none",
                    "recovery_status": str(case.recovery_status),
                    "amount_recovered": float(case.amount_recovered),
                    "was_recovered": bool(case.was_recovered),
                    "relevance_score": float(case.relevance_score) if case.relevance_score is not None else 0.0,
                }
            )
        return payload

    def _build_policy_boundary(
        self,
        policy_context: Optional[RecoveryPolicyContext],
    ) -> Tuple[List[str], List[str], Optional[str], List[str]]:
        """
        Transform active policy envelope into sanitized string collections.
        Extracts allowed actions, prohibited actions, mandatory fallback, and rule descriptions.
        """
        if policy_context is None:
            return [], [], None, []

        allowed = sorted(
            [a.value if isinstance(a, RecoveryAction) else str(a) for a in policy_context.allowed_actions]
        )
        prohibited = sorted(
            [a.value if isinstance(a, RecoveryAction) else str(a) for a in policy_context.prohibited_actions]
        )

        fallback = None
        if policy_context.mandatory_fallback_action is not None:
            fallback = (
                policy_context.mandatory_fallback_action.value
                if isinstance(policy_context.mandatory_fallback_action, RecoveryAction)
                else str(policy_context.mandatory_fallback_action)
            )

        constraints = [str(rule.description) for rule in policy_context.applicable_rules if rule.description]

        return allowed, prohibited, fallback, constraints
