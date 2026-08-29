"""
Revora Recovery Application Service.

Orchestrates the end-to-end recovery evaluation workflow:
context retrieval -> RAG evidence synthesis -> decision evaluation -> transactional persistence -> audit logging.
"""

from datetime import datetime, timezone
import logging
from typing import Any, Dict, List, Optional
from uuid import UUID
from sqlalchemy.orm import Session

from app.context_retrieval import get_customer_context
from app.decision_engine import DecisionEngine, RecoveryDecision
from app.embedding_service import EmbeddingService, get_embedding_service
from app.historical_retrieval import HistoricalCase
from app.historical_retriever import HistoricalRetriever
from app.hybrid_historical_retriever import HybridHistoricalRetriever
from app.models import AuditEvent, RecoveryOpportunity, utc_now
from app.schemas.recovery import (
    RecoveryEvaluationRequest,
    RecoveryEvaluationResponse,
)
from app.semantic_historical_retriever import SemanticHistoricalRetriever
from app.vector_index import VectorIndex

logger = logging.getLogger("revora.recovery_service")


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
    """

    def __init__(
        self,
        decision_engine: Optional[DecisionEngine] = None,
        hybrid_retriever: Optional[HybridHistoricalRetriever] = None,
        vector_index: Optional[VectorIndex] = None,
        embedding_service: Optional[EmbeddingService] = None,
    ):
        self.decision_engine = decision_engine or DecisionEngine()
        self.hybrid_retriever = hybrid_retriever
        self.vector_index = vector_index or VectorIndex()
        self.embedding_service = embedding_service or get_embedding_service()

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

    def evaluate_recovery(
        self,
        db_session: Session,
        request: RecoveryEvaluationRequest,
    ) -> RecoveryEvaluationResponse:
        """
        Execute the full recovery evaluation workflow for a failed payment.

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
        # 1. Retrieve Deterministic Customer Context (Propagates domain errors if missing/mismatched)
        context = get_customer_context(
            db_session=db_session,
            customer_id=request.customer_id,
            payment_id=request.payment_id,
        )

        # 2. Execute RAG Retrieval or Pure Deterministic Flow
        retrieved_cases: Optional[List[HistoricalCase]] = None
        historical_rag_used = False
        retrieved_evidence_count = 0

        if request.use_rag:
            retriever = self._resolve_retriever(db_session)
            retrieved_cases = retriever.retrieve_relevant_cases(context, top_k=5)
            historical_rag_used = True
            retrieved_evidence_count = len(retrieved_cases) if retrieved_cases else 0

        # 3. Evaluate Decision
        decision: RecoveryDecision = self.decision_engine.evaluate(
            context=context,
            historical_cases=retrieved_cases,
        )

        # 4. Atomic Persistence & Audit Logging
        try:
            # Resolve opportunity directly by UUID
            opp_id = context.current_opportunity.opportunity_id
            opportunity = db_session.get(RecoveryOpportunity, opp_id)
            if opportunity is not None:
                opportunity.recommended_action = decision.recommended_action.value
                opportunity.confidence = decision.confidence
                opportunity.updated_at = utc_now()

            # Record Audit Event with un-frozen metadata payload
            unfrozen_basis = _unfreeze_for_json(decision.decision_basis)
            audit_event = AuditEvent(
                opportunity_id=opp_id,
                event_type="recovery_decision_evaluated",
                description=decision.reason,
                metadata_payload=unfrozen_basis,
            )
            db_session.add(audit_event)

            # Single transactional commit
            db_session.commit()
        except Exception as exc:
            db_session.rollback()
            logger.error(
                "Failed to persist recovery decision and audit event for payment %s: %s",
                request.payment_id,
                exc,
                exc_info=True,
            )
            raise

        # 5. Build Clean Client-Facing Response DTO
        return RecoveryEvaluationResponse(
            payment_id=context.current_payment.payment_id,
            customer_id=context.customer.customer_id,
            opportunity_id=context.current_opportunity.opportunity_id,
            recommended_action=decision.recommended_action,
            reason=decision.reason,
            confidence=decision.confidence,
            decision_basis=unfrozen_basis,
            historical_rag_used=historical_rag_used,
            retrieved_evidence_count=retrieved_evidence_count,
            evaluated_at=utc_now(),
        )
