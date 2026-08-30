from __future__ import annotations

"""
Deterministic Recovery Decision Engine for Revora with Historical RAG Integration.

Consumes an immutable CustomerRecoveryContext and evaluates prioritized, explainable,
deterministic rules augmented by empirical HistoricalCase evidence to produce a
RecoveryDecision recommendation without LLMs.
"""

from collections import defaultdict
from enum import Enum
import types
from typing import TYPE_CHECKING, Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple
from uuid import UUID
from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.context import CustomerRecoveryContext
from app.historical_retrieval import HistoricalCase

if TYPE_CHECKING:
    from app.hybrid_historical_retriever import HybridHistoricalRetriever


def _freeze_nested(value: Any) -> Any:
    """Recursively freeze dictionaries to MappingProxyType and sequences to tuples."""
    if isinstance(value, (dict, types.MappingProxyType)):
        return types.MappingProxyType({k: _freeze_nested(v) for k, v in value.items()})
    if isinstance(value, (list, set)):
        return tuple(_freeze_nested(v) for v in value)
    if isinstance(value, tuple):
        return tuple(_freeze_nested(v) for v in value)
    return value


class RecoveryAction(str, Enum):
    """Supported recovery actions in the Revora decision domain."""

    PAYMENT_LINK = "payment_link"
    RETRY_PAYMENT = "retry_payment"
    CHANGE_PAYMENT_METHOD = "change_payment_method"
    WAIT_AND_RETRY = "wait_and_retry"
    NO_ACTION = "no_action"


class RecoveryDecision(BaseModel):
    """
    Immutable decision contract produced by the deterministic recovery decision engine.
    """

    model_config = ConfigDict(
        frozen=True,
        from_attributes=True,
        arbitrary_types_allowed=True,
        validate_default=True,
    )

    recommended_action: RecoveryAction
    reason: str
    confidence: float = Field(ge=0.0, le=1.0)
    decision_basis: Mapping[str, Any] = Field(default_factory=dict)

    @field_validator("decision_basis", mode="before")
    @classmethod
    def _normalize_none_decision_basis(cls, v: Any) -> Any:
        if v is None:
            return {}
        return v

    @field_validator("decision_basis", mode="after")
    @classmethod
    def _ensure_immutable_decision_basis(cls, v: Any) -> Mapping[str, Any]:
        if v is None:
            return types.MappingProxyType({})
        return _freeze_nested(v)


# Failure Reason Classification Sets (Normalized lowercase)
PERMANENT_FAILURE_REASONS: Set[str] = {
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
}

TRANSIENT_TECHNICAL_REASONS: Set[str] = {
    "bank_server_down",
    "bank_timeout",
    "network_timeout",
    "system_error",
    "gateway_timeout",
    "internal_server_error",
    "service_unavailable",
    "connection_reset",
    "timeout",
}

INSUFFICIENT_FUNDS_REASONS: Set[str] = {
    "insufficient_funds",
    "low_balance",
    "balance_insufficient",
    "limit_exceeded",
    "daily_limit_exceeded",
    "exceeds_limit",
}

CUSTOMER_INTERACTION_REASONS: Set[str] = {
    "authentication_failed",
    "otp_expired",
    "otp_timeout",
    "user_cancelled",
    "customer_cancelled",
    "declined_by_user",
    "3ds_failed",
    "pin_incorrect",
}


def normalize_action_name(action_str: Optional[str]) -> Optional[RecoveryAction]:
    """
    Map raw or historical action strings to standard RecoveryAction enum members.
    """
    if not action_str:
        return None
    cleaned = action_str.strip().lower()
    if cleaned in (
        RecoveryAction.RETRY_PAYMENT.value,
        "smart_retry",
        "retry",
        "retry_charge",
        "smart_retry_card",
        "smart_retry_upi",
    ):
        return RecoveryAction.RETRY_PAYMENT
    if cleaned in (RecoveryAction.WAIT_AND_RETRY.value, "wait_retry", "delayed_retry"):
        return RecoveryAction.WAIT_AND_RETRY
    if cleaned in (
        RecoveryAction.PAYMENT_LINK.value,
        "customer_prompt",
        "customer_prompt_upi",
        "prompt",
        "sms_link",
        "whatsapp_prompt",
    ):
        return RecoveryAction.PAYMENT_LINK
    if cleaned in (
        RecoveryAction.CHANGE_PAYMENT_METHOD.value,
        "update_card",
        "switch_method",
    ):
        return RecoveryAction.CHANGE_PAYMENT_METHOD
    if cleaned in (RecoveryAction.NO_ACTION.value, "abandon", "none"):
        return RecoveryAction.NO_ACTION
    return None


class HistoricalEvidenceSynthesizer:
    """
    Synthesizes retrieved HistoricalCase evidence into structured action scores.
    """

    MIN_RELEVANCE_THRESHOLD: float = 0.40

    @classmethod
    def synthesize(
        cls,
        cases: Optional[Sequence[HistoricalCase]],
    ) -> Dict[str, Any]:
        """
        Aggregate and score retrieved cases by action.
        """
        if not cases:
            return {
                "has_evidence": False,
                "retrieved_cases_count": 0,
                "top_case": None,
                "action_net_scores": {},
                "best_action": None,
                "best_action_score": 0.0,
            }

        filtered = [
            c
            for c in cases
            if c.relevance_score is not None and c.relevance_score >= cls.MIN_RELEVANCE_THRESHOLD
        ]
        if not filtered:
            return {
                "has_evidence": False,
                "retrieved_cases_count": len(cases),
                "top_case": None,
                "action_net_scores": {},
                "best_action": None,
                "best_action_score": 0.0,
            }

        action_scores: Dict[RecoveryAction, float] = defaultdict(float)
        action_success_counts: Dict[RecoveryAction, int] = defaultdict(int)
        action_failure_counts: Dict[RecoveryAction, int] = defaultdict(int)

        for case in filtered:
            norm_action = normalize_action_name(case.recovery_action)
            if not norm_action or norm_action == RecoveryAction.NO_ACTION:
                continue

            rel = case.relevance_score or 0.5
            if case.was_recovered:
                action_scores[norm_action] += rel
                action_success_counts[norm_action] += 1
            else:
                action_scores[norm_action] -= rel * 0.75
                action_failure_counts[norm_action] += 1

        top_case = filtered[0]
        top_case_summary = {
            "payment_id": str(top_case.payment_id),
            "relevance_score": top_case.relevance_score,
            "was_recovered": top_case.was_recovered,
            "action": top_case.recovery_action,
            "failure_reason": top_case.failure_reason,
        }

        # Find best net positive action
        best_action: Optional[RecoveryAction] = None
        best_score: float = 0.0
        for act, net_score in action_scores.items():
            if net_score > best_score:
                best_score = net_score
                best_action = act

        return {
            "has_evidence": True,
            "retrieved_cases_count": len(cases),
            "valid_cases_count": len(filtered),
            "top_case": top_case_summary,
            "action_net_scores": {k.value: round(v, 4) for k, v in action_scores.items()},
            "action_success_counts": {k.value: v for k, v in action_success_counts.items()},
            "action_failure_counts": {k.value: v for k, v in action_failure_counts.items()},
            "best_action": best_action,
            "best_action_score": round(best_score, 4),
        }


class DecisionEngine:
    """
    Deterministic payment recovery decision engine with Historical RAG integration.

    Evaluates CustomerRecoveryContext augmented by empirical HistoricalCase evidence
    to recommend an optimal recovery action with full explainability.
    """

    def __init__(
        self,
        max_attempts: int = 3,
        retriever: Optional[HybridHistoricalRetriever] = None,
    ):
        self.max_attempts = max_attempts
        self.retriever = retriever

    def evaluate(
        self,
        context: CustomerRecoveryContext,
        historical_cases: Optional[List[HistoricalCase]] = None,
    ) -> RecoveryDecision:
        """
        Evaluate context and historical evidence to produce an immutable RecoveryDecision.
        """
        if not isinstance(context, CustomerRecoveryContext):
            raise TypeError(
                f"Expected context to be CustomerRecoveryContext, got {type(context).__name__}"
            )

        current_payment = context.current_payment
        current_opportunity = context.current_opportunity

        # 1. Retrieve Historical Evidence if retriever is configured and not pre-supplied
        retrieved_cases = historical_cases
        if retrieved_cases is None and self.retriever is not None:
            retrieved_cases = self.retriever.retrieve_relevant_cases(context, top_k=5)

        # 2. Synthesize Historical Evidence
        evidence = HistoricalEvidenceSynthesizer.synthesize(retrieved_cases)

        # 3. Extract normalized attempt history on this payment
        raw_attempts = context.current_payment_attempts or []
        attempt_count = len(raw_attempts)
        attempted_actions_normalized: Set[RecoveryAction] = set()
        for att in raw_attempts:
            norm = normalize_action_name(att.action)
            if norm:
                attempted_actions_normalized.add(norm)

        # 4. Build base signals for explainability
        failure_reason_raw = (
            current_payment.failure_reason
            if current_payment and current_payment.failure_reason
            else ""
        )
        failure_reason = failure_reason_raw.strip().lower()
        payment_method = current_payment.payment_method if current_payment else "unknown"

        base_basis: Dict[str, Any] = {
            "failure_reason": failure_reason_raw,
            "payment_method": payment_method,
            "current_attempt_count": attempt_count,
            "attempted_actions": [att.action for att in raw_attempts],
            "customer_historical_recovery_rate": context.recovery_statistics.recovery_rate,
            "historical_successful_actions": context.recovery_statistics.previously_successful_actions,
            "historical_failed_actions": context.recovery_statistics.previously_failed_actions,
            "historical_rag_evidence": evidence,
        }

        # 5. Category classification
        exact_category: Optional[str] = None
        if failure_reason in PERMANENT_FAILURE_REASONS:
            exact_category = "permanent"
        elif failure_reason in CUSTOMER_INTERACTION_REASONS:
            exact_category = "customer_interaction"
        elif failure_reason in INSUFFICIENT_FUNDS_REASONS:
            exact_category = "insufficient_funds"
        elif failure_reason in TRANSIENT_TECHNICAL_REASONS:
            exact_category = "transient"

        is_permanent = (exact_category == "permanent") or (
            exact_category is None
            and any(term in failure_reason for term in PERMANENT_FAILURE_REASONS)
        )
        is_transient = (exact_category == "transient") or (
            exact_category is None
            and any(term in failure_reason for term in TRANSIENT_TECHNICAL_REASONS)
        )
        is_insufficient_funds = (exact_category == "insufficient_funds") or (
            exact_category is None
            and any(term in failure_reason for term in INSUFFICIENT_FUNDS_REASONS)
        )
        is_customer_interaction = (exact_category == "customer_interaction") or (
            exact_category is None
            and any(term in failure_reason for term in CUSTOMER_INTERACTION_REASONS)
        )

        # =========================================================================
        # HARD SAFETY RULES (Cannot be overridden by historical evidence)
        # =========================================================================

        # Rule 1: No Active Opportunity or Succeeded Payment
        if current_payment is None or current_opportunity is None:
            return RecoveryDecision(
                recommended_action=RecoveryAction.NO_ACTION,
                reason="No active payment or recovery opportunity present in context.",
                confidence=1.0,
                decision_basis={**base_basis, "rule_matched": "NoActiveOpportunityRule"},
            )

        if current_opportunity.status == "recovered" or current_payment.status == "succeeded":
            return RecoveryDecision(
                recommended_action=RecoveryAction.NO_ACTION,
                reason="Payment or recovery opportunity is already resolved and recovered.",
                confidence=1.0,
                decision_basis={**base_basis, "rule_matched": "AlreadyRecoveredRule"},
            )

        # Rule 2: Max Attempts Exceeded or Opportunity in Terminal State
        if (
            attempt_count >= self.max_attempts
            or current_opportunity.status in ("failed", "abandoned")
        ):
            return RecoveryDecision(
                recommended_action=RecoveryAction.NO_ACTION,
                reason=f"Maximum recovery attempts ({self.max_attempts}) reached or opportunity is in terminal state.",
                confidence=0.95,
                decision_basis={**base_basis, "rule_matched": "MaxAttemptsExceededRule"},
            )

        # Rule 3: Permanent Credential Failure
        if is_permanent:
            return RecoveryDecision(
                recommended_action=RecoveryAction.CHANGE_PAYMENT_METHOD,
                reason=f"Permanent payment failure detected ({failure_reason_raw}); customer must update payment method.",
                confidence=0.90,
                decision_basis={**base_basis, "rule_matched": "PermanentCredentialFailureRule"},
            )

        # =========================================================================
        # ADAPTIVE RECOVERY RULES (Empirically augmented by Historical Evidence)
        # =========================================================================

        # Rule 4: Transient Technical / Bank Failure
        if is_transient:
            if RecoveryAction.RETRY_PAYMENT not in attempted_actions_normalized:
                base_conf = 0.85
                reason = "Transient network or bank gateway timeout detected; immediate automatic retry is recommended."
                rule_name = "TransientTechnicalImmediateRetryRule"
                if evidence["has_evidence"] and evidence["best_action"] == RecoveryAction.RETRY_PAYMENT:
                    base_conf = min(0.95, base_conf + 0.05)
                    reason += f" (Supported by {evidence['valid_cases_count']} similar historical recovery cases)."
                return RecoveryDecision(
                    recommended_action=RecoveryAction.RETRY_PAYMENT,
                    reason=reason,
                    confidence=base_conf,
                    decision_basis={**base_basis, "rule_matched": rule_name},
                )
            elif RecoveryAction.WAIT_AND_RETRY not in attempted_actions_normalized:
                base_conf = 0.75
                reason = "Immediate retry failed for bank timeout; scheduled wait-and-retry recommended to allow gateway stabilization."
                rule_name = "TransientTechnicalWaitAndRetryRule"
                # If historical evidence strongly favors direct payment link over wait-and-retry
                if (
                    evidence["has_evidence"]
                    and evidence["best_action"] == RecoveryAction.PAYMENT_LINK
                    and RecoveryAction.PAYMENT_LINK not in attempted_actions_normalized
                    and evidence["best_action_score"] > 0.8
                ):
                    return RecoveryDecision(
                        recommended_action=RecoveryAction.PAYMENT_LINK,
                        reason="Immediate retry failed; empirical historical evidence indicates high payment link recovery rate for this customer.",
                        confidence=0.80,
                        decision_basis={**base_basis, "rule_matched": "TransientTechnicalHistoricalLinkPivotRule"},
                    )
                return RecoveryDecision(
                    recommended_action=RecoveryAction.WAIT_AND_RETRY,
                    reason=reason,
                    confidence=base_conf,
                    decision_basis={**base_basis, "rule_matched": rule_name},
                )
            else:
                return RecoveryDecision(
                    recommended_action=RecoveryAction.PAYMENT_LINK,
                    reason="Automated technical retries exhausted; issuing direct payment link for manual re-attempt.",
                    confidence=0.70,
                    decision_basis={**base_basis, "rule_matched": "TransientTechnicalFallbackLinkRule"},
                )

        # Rule 5: Insufficient Funds
        if is_insufficient_funds:
            # Check historical customer affinity from stats or retrieved evidence
            hist_successful = [
                normalize_action_name(a)
                for a in context.recovery_statistics.previously_successful_actions
            ]
            has_hist_link = (
                RecoveryAction.PAYMENT_LINK in hist_successful
                or (
                    evidence["has_evidence"]
                    and evidence["best_action"] == RecoveryAction.PAYMENT_LINK
                    and evidence["best_action_score"] > 0.4
                )
            )

            if has_hist_link and RecoveryAction.PAYMENT_LINK not in attempted_actions_normalized:
                conf = 0.80
                if evidence["has_evidence"] and evidence["best_action"] == RecoveryAction.PAYMENT_LINK:
                    conf = min(0.90, conf + 0.05)
                return RecoveryDecision(
                    recommended_action=RecoveryAction.PAYMENT_LINK,
                    reason="Insufficient funds detected; customer has historically recovered successfully via payment link.",
                    confidence=conf,
                    decision_basis={**base_basis, "rule_matched": "InsufficientFundsHistoricalLinkRule"},
                )

            if (
                RecoveryAction.WAIT_AND_RETRY not in attempted_actions_normalized
                and attempt_count == 0
            ):
                return RecoveryDecision(
                    recommended_action=RecoveryAction.WAIT_AND_RETRY,
                    reason="Insufficient funds detected; wait-and-retry recommended to allow account replenishment.",
                    confidence=0.75,
                    decision_basis={**base_basis, "rule_matched": "InsufficientFundsInitialWaitRule"},
                )
            elif RecoveryAction.PAYMENT_LINK not in attempted_actions_normalized:
                return RecoveryDecision(
                    recommended_action=RecoveryAction.PAYMENT_LINK,
                    reason="Insufficient funds persisted after wait; issuing interactive payment link to customer.",
                    confidence=0.70,
                    decision_basis={**base_basis, "rule_matched": "InsufficientFundsEscalateLinkRule"},
                )
            else:
                return RecoveryDecision(
                    recommended_action=RecoveryAction.CHANGE_PAYMENT_METHOD,
                    reason="Repeated insufficient funds failures; requesting customer change payment method.",
                    confidence=0.65,
                    decision_basis={**base_basis, "rule_matched": "InsufficientFundsChangeMethodRule"},
                )

        # Rule 6: Customer Interaction / Authentication Failure
        if is_customer_interaction:
            if RecoveryAction.PAYMENT_LINK not in attempted_actions_normalized:
                base_conf = 0.80
                reason = "Customer authentication error or cancellation detected; sending interactive payment link for user re-entry."
                if evidence["has_evidence"] and evidence["best_action"] == RecoveryAction.PAYMENT_LINK:
                    base_conf = min(0.90, base_conf + 0.05)
                return RecoveryDecision(
                    recommended_action=RecoveryAction.PAYMENT_LINK,
                    reason=reason,
                    confidence=base_conf,
                    decision_basis={**base_basis, "rule_matched": "CustomerInteractionPaymentLinkRule"},
                )
            else:
                return RecoveryDecision(
                    recommended_action=RecoveryAction.CHANGE_PAYMENT_METHOD,
                    reason="Authentication issue unresolved via payment link; recommending change of payment method.",
                    confidence=0.70,
                    decision_basis={**base_basis, "rule_matched": "CustomerInteractionChangeMethodRule"},
                )

        # Rule 7: Empirical Historical Evidence Selection (for cold/unclassified failure reasons)
        if evidence["has_evidence"] and evidence["best_action"] is not None:
            best_act = evidence["best_action"]
            if best_act not in attempted_actions_normalized and best_act != RecoveryAction.NO_ACTION:
                score = evidence["best_action_score"]
                conf = round(min(0.85, 0.65 + 0.15 * min(1.0, score)), 2)
                return RecoveryDecision(
                    recommended_action=best_act,
                    reason=f"Selected '{best_act.value}' based on {evidence['valid_cases_count']} similar historical recovery cases.",
                    confidence=conf,
                    decision_basis={**base_basis, "rule_matched": "HistoricalRAGSelectedRule"},
                )

        # Rule 8: Historical Affinity from Recovery Statistics
        if context.recovery_statistics.recovery_rate > 0.0:
            for hist_action_str in context.recovery_statistics.previously_successful_actions:
                norm_action = normalize_action_name(hist_action_str)
                if (
                    norm_action
                    and norm_action not in attempted_actions_normalized
                    and norm_action != RecoveryAction.NO_ACTION
                ):
                    confidence_score = round(
                        min(0.85, 0.60 + 0.25 * context.recovery_statistics.recovery_rate), 2
                    )
                    return RecoveryDecision(
                        recommended_action=norm_action,
                        reason=f"Selected '{norm_action.value}' based on customer's historical recovery success track record.",
                        confidence=confidence_score,
                        decision_basis={**base_basis, "rule_matched": "HistoricalAffinityRule"},
                    )

        # Rule 9: Default Fallback for Cold-Start / Unknown Failure Reason
        if RecoveryAction.PAYMENT_LINK not in attempted_actions_normalized:
            default_action = RecoveryAction.PAYMENT_LINK
            reason = (
                "Unspecified failure reason or cold-start customer; standard interactive payment link recommended."
                if not failure_reason
                else f"Unsupported failure reason '{failure_reason_raw}'; fallback to interactive payment link."
            )
            confidence = 0.55 if not failure_reason else 0.50
        elif RecoveryAction.WAIT_AND_RETRY not in attempted_actions_normalized:
            default_action = RecoveryAction.WAIT_AND_RETRY
            reason = "Payment link previously attempted; falling back to scheduled wait-and-retry."
            confidence = 0.50
        else:
            default_action = RecoveryAction.CHANGE_PAYMENT_METHOD
            reason = "Standard automated and interactive options attempted; requesting payment method update."
            confidence = 0.45

        return RecoveryDecision(
            recommended_action=default_action,
            reason=reason,
            confidence=confidence,
            decision_basis={**base_basis, "rule_matched": "DefaultFallbackRule"},
        )


def evaluate_recovery_decision(
    context: CustomerRecoveryContext,
    max_attempts: int = 3,
    retriever: Optional[HybridHistoricalRetriever] = None,
    historical_cases: Optional[List[HistoricalCase]] = None,
) -> RecoveryDecision:
    """
    Public entrypoint for the deterministic recovery decision engine with optional historical RAG.
    """
    engine = DecisionEngine(max_attempts=max_attempts, retriever=retriever)
    return engine.evaluate(context, historical_cases=historical_cases)
