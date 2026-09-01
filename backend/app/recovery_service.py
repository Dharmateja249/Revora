"""
Revora Recovery Application Service.

Orchestrates the end-to-end recovery evaluation workflow:
context retrieval -> RAG evidence synthesis -> decision evaluation -> transactional persistence -> audit logging.
"""

import asyncio
import concurrent.futures
import inspect
import logging
from typing import Any

from sqlalchemy.orm import Session

from app.agent.orchestrator import AgentOrchestrator
from app.agent.schemas import AgentDecisionResult
from app.context import (
    CustomerNotFoundError,
    PaymentCustomerMismatchError,
    PaymentNotFoundError,
    RecoveryOpportunityNotFoundError,
)
from app.context_retrieval import get_customer_context
from app.decision_engine import DecisionEngine, RecoveryAction, RecoveryDecision
from app.embedding_service import EmbeddingService, get_embedding_service
from app.historical_retrieval import HistoricalCase
from app.historical_retriever import HistoricalRetriever
from app.hybrid_historical_retriever import HybridHistoricalRetriever
from app.models import AuditEvent, RecoveryOpportunity, utc_now
from app.policies.resolver import resolve_policy_context
from app.policies.schemas import RecoveryPolicyContext
from app.schemas.recovery import (
    RecoveryEvaluationRequest,
    RecoveryEvaluationResponse,
)
from app.semantic_historical_retriever import SemanticHistoricalRetriever
from app.vector_index import VectorIndex, get_vector_index

logger = logging.getLogger("revora.recovery_service")


def _supports_policy_context(engine: Any) -> bool:
    """Determine whether the decision engine accepts a policy_context argument."""
    try:
        sig = inspect.signature(engine.evaluate)
        return "policy_context" in sig.parameters or any(
            p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
        )
    except (TypeError, ValueError, AttributeError):
        return False


def _unfreeze_for_json(data: Any) -> Any:
    """Recursively convert MappingProxyType and tuples to standard dicts and lists for JSON serialization."""
    if hasattr(data, "items"):
        return {k: _unfreeze_for_json(v) for k, v in data.items()}
    if isinstance(data, (list, tuple, set)):
        return [_unfreeze_for_json(v) for v in data]
    return data


class RecoveryService:
    """
    Application Service for payment recovery evaluation and audit logging.
    Supports both deterministic rule-based evaluation and adaptive agent orchestration.
    """

    def __init__(
        self,
        decision_engine: DecisionEngine | None = None,
        agent_orchestrator: AgentOrchestrator | None = None,
        use_agent: bool = False,
        hybrid_retriever: HybridHistoricalRetriever | None = None,
        vector_index: VectorIndex | None = None,
        embedding_service: EmbeddingService | None = None,
    ):
        self.decision_engine = decision_engine or DecisionEngine()
        self.agent_orchestrator = agent_orchestrator
        self.use_agent = use_agent
        self.hybrid_retriever = hybrid_retriever
        self.vector_index = (
            vector_index
            if isinstance(vector_index, VectorIndex)
            else get_vector_index()
        )
        self.embedding_service = (
            embedding_service
            if isinstance(embedding_service, EmbeddingService)
            else get_embedding_service()
        )

    def _resolve_retriever(self, db_session: Session) -> HybridHistoricalRetriever:
        """Resolve or construct the HybridHistoricalRetriever bound to the active database session."""
        if self.hybrid_retriever is not None:
            return self.hybrid_retriever

        det_retriever = HistoricalRetriever(db_session=db_session)
        sem_retriever = SemanticHistoricalRetriever(
            vector_index=self.vector_index,
            embedding_service=self.embedding_service,
        )
        return HybridHistoricalRetriever(
            deterministic_retriever=det_retriever,
            semantic_retriever=sem_retriever,
            rrf_k=60,
        )

    def _evaluate_deterministic(
        self,
        context: Any,
        historical_cases: list[HistoricalCase] | None,
        policy_context: RecoveryPolicyContext,
    ) -> RecoveryDecision:
        """Helper to invoke DecisionEngine safely while distinguishing policy capability."""
        if _supports_policy_context(self.decision_engine):
            return self.decision_engine.evaluate(
                context=context,
                historical_cases=historical_cases,
                policy_context=policy_context,
            )
        return self.decision_engine.evaluate(
            context=context,
            historical_cases=historical_cases,
        )

    async def evaluate_recovery_async(
        self,
        db_session: Session,
        request: RecoveryEvaluationRequest,
    ) -> RecoveryEvaluationResponse:
        """
        Asynchronously execute the full recovery evaluation workflow for a failed payment.

        Args:
            db_session: Active SQLAlchemy database session.
            request: RecoveryEvaluationRequest containing customer and payment identifiers.

        Returns:
            RecoveryEvaluationResponse: Structured recovery decision and audit summary.

        Raises:
            CustomerNotFoundError: If the customer does not exist.
            PaymentNotFoundError: If the payment does not exist.
            PaymentCustomerMismatchError: If the payment does not belong to the customer.
            RecoveryOpportunityNotFoundError: If the payment lacks a recovery opportunity.
            Exception: Any persistence or infrastructure failure.
        """
        # 1. Retrieve Deterministic Customer Context
        context = get_customer_context(
            db_session=db_session,
            customer_id=request.customer_id,
            payment_id=request.payment_id,
        )

        # 2. Resolve Structured Recovery Policy Context
        max_attempts = getattr(self.decision_engine, "max_attempts", 3)
        policy_context = resolve_policy_context(
            context=context,
            provider="razorpay",
            max_attempts=max_attempts,
        )

        # 3. Execute RAG Retrieval
        retrieved_cases: list[HistoricalCase] | None = None
        historical_rag_used = False
        retrieved_evidence_count = 0

        if request.use_rag:
            retriever = self._resolve_retriever(db_session)
            retrieved_cases = retriever.retrieve_relevant_cases(context, top_k=5)
            historical_rag_used = True
            retrieved_evidence_count = len(retrieved_cases) if retrieved_cases else 0

        # 4. Evaluate Decision (Agent Orchestrator vs. Deterministic Engine)
        effective_use_agent = (
            request.use_agent if request.use_agent is not None else self.use_agent
        )

        recommended_action: RecoveryAction
        confidence: float
        reason: str
        decision_basis: dict[str, Any]
        agent_used: bool = False
        is_fallback: bool = False
        fallback_reason: str | None = None
        policy_overridden: bool = False

        if effective_use_agent and self.agent_orchestrator is not None:
            try:
                agent_result: AgentDecisionResult = (
                    await self.agent_orchestrator.decide(
                        context=context,
                        policy_context=policy_context,
                        historical_cases=retrieved_cases,
                    )
                )
                recommended_action = agent_result.recommendation.recommended_action
                confidence = agent_result.recommendation.confidence
                reason = agent_result.recommendation.reasoning
                agent_used = agent_result.agent_used
                is_fallback = agent_result.is_fallback
                fallback_reason = agent_result.fallback_reason

                policy_overridden = bool(
                    agent_result.metadata.get("policy_overridden", False)
                )

                raw_metadata = dict(agent_result.metadata)
                decision_basis = {
                    "agent_used": agent_used,
                    "is_fallback": is_fallback,
                    "fallback_reason": fallback_reason,
                    "provider": agent_result.provider,
                    "model_name": agent_result.model_name,
                    "latency_ms": agent_result.latency_ms,
                    "key_factors": list(agent_result.recommendation.key_factors),
                    "referenced_case_ids": list(
                        agent_result.recommendation.referenced_case_ids
                    ),
                    "policy_overridden": policy_overridden,
                }
                if "error_type" in raw_metadata:
                    decision_basis["error_type"] = raw_metadata["error_type"]
                if "violated_policy_ids" in raw_metadata:
                    decision_basis["violated_policy_ids"] = list(
                        raw_metadata["violated_policy_ids"]
                    )
                if "original_candidate_action" in raw_metadata:
                    decision_basis["original_candidate_action"] = raw_metadata[
                        "original_candidate_action"
                    ]
            except (
                CustomerNotFoundError,
                PaymentNotFoundError,
                PaymentCustomerMismatchError,
                RecoveryOpportunityNotFoundError,
                ValueError,
                TypeError,
            ):
                # Domain & strict policy validation errors must propagate fail-closed
                raise
            except Exception as unexpected_err:  # noqa: BLE001
                # Fall back safely to deterministic DecisionEngine on any unhandled agent failure
                logger.error(
                    "Unexpected failure during agent decision orchestration for payment %s: %s; falling back to deterministic DecisionEngine",
                    request.payment_id,
                    type(unexpected_err).__name__,
                )
                decision = self._evaluate_deterministic(
                    context=context,
                    historical_cases=retrieved_cases,
                    policy_context=policy_context,
                )
                recommended_action = decision.recommended_action
                confidence = decision.confidence
                reason = decision.reason
                agent_used = False
                is_fallback = True
                fallback_reason = "Unexpected agent orchestration failure; deterministic fallback applied"
                decision_basis = dict(_unfreeze_for_json(decision.decision_basis))
                decision_basis["agent_used"] = False
                decision_basis["is_fallback"] = True
                decision_basis["fallback_reason"] = fallback_reason
                decision_basis["error_type"] = type(unexpected_err).__name__
                policy_overridden = bool(decision_basis.get("policy_overridden", False))
        else:
            # Deterministic DecisionEngine evaluation
            decision = self._evaluate_deterministic(
                context=context,
                historical_cases=retrieved_cases,
                policy_context=policy_context,
            )
            recommended_action = decision.recommended_action
            confidence = decision.confidence
            reason = decision.reason
            agent_used = False
            is_fallback = False
            fallback_reason = None
            decision_basis = dict(_unfreeze_for_json(decision.decision_basis))
            policy_overridden = bool(decision_basis.get("policy_overridden", False))

        # 5. Atomic Persistence & Audit Logging
        unfrozen_basis = _unfreeze_for_json(decision_basis)
        try:
            opp_id = context.current_opportunity.opportunity_id
            opportunity = db_session.get(RecoveryOpportunity, opp_id)
            if opportunity is not None:
                opportunity.recommended_action = recommended_action.value
                opportunity.confidence = confidence
                opportunity.updated_at = utc_now()

            audit_event = AuditEvent(
                opportunity_id=opp_id,
                event_type="recovery_decision_evaluated",
                description=reason,
                metadata_payload=unfrozen_basis,
            )
            db_session.add(audit_event)
            db_session.commit()
        except Exception:
            db_session.rollback()
            logger.exception(
                "Failed to persist recovery decision and audit event for payment %s",
                request.payment_id,
            )
            raise

        # 6. Build Clean Client-Facing Response DTO with Policy Telemetry
        applied_policy_ids = [r.policy_id for r in policy_context.applicable_rules]

        return RecoveryEvaluationResponse(
            payment_id=context.current_payment.payment_id,
            customer_id=context.customer.customer_id,
            opportunity_id=context.current_opportunity.opportunity_id,
            recommended_action=recommended_action,
            reason=reason,
            confidence=confidence,
            decision_basis=unfrozen_basis,
            historical_rag_used=historical_rag_used,
            retrieved_evidence_count=retrieved_evidence_count,
            provider=policy_context.provider,
            policy_version=policy_context.policy_version,
            applied_policy_ids=applied_policy_ids,
            policy_overridden=policy_overridden,
            agent_used=agent_used,
            is_fallback=is_fallback,
            fallback_reason=fallback_reason,
            evaluated_at=utc_now(),
        )

    def evaluate_recovery(
        self,
        db_session: Session,
        request: RecoveryEvaluationRequest,
    ) -> RecoveryEvaluationResponse:
        """
        Synchronously execute the full recovery evaluation workflow for a failed payment.
        Bridges safely to evaluate_recovery_async without creating event-loop nesting errors.
        """
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                return pool.submit(
                    lambda: asyncio.run(
                        self.evaluate_recovery_async(
                            db_session=db_session, request=request
                        )
                    )
                ).result()

        return asyncio.run(
            self.evaluate_recovery_async(db_session=db_session, request=request)
        )
