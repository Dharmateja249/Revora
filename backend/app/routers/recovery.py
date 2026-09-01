"""
FastAPI Recovery Evaluation Router.

Provides HTTP endpoint for evaluating failed payment recovery decisions.
Handles HTTP request parsing, dependency injection, and domain exception translation.
"""

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.agent.orchestrator import AgentOrchestrator
from app.agent.provider import MockLLMProvider
from app.agent.schemas import LLMRecoveryRecommendation
from app.auth import AuthenticatedPrincipal, get_current_principal
from app.config import Settings, get_settings
from app.context import (
    CustomerNotFoundError,
    PaymentCustomerMismatchError,
    PaymentNotFoundError,
    RecoveryOpportunityNotFoundError,
)
from app.database import get_db
from app.decision_engine import RecoveryAction
from app.embedding_service import EmbeddingService, get_embedding_service
from app.recovery_service import RecoveryService
from app.schemas.recovery import (
    RecoveryEvaluationRequest,
    RecoveryEvaluationResponse,
)
from app.vector_index import VectorIndex, get_vector_index

router = APIRouter()


def get_recovery_service(
    vector_index: Optional[VectorIndex] = Depends(get_vector_index),
    embedding_service: Optional[EmbeddingService] = Depends(get_embedding_service),
    settings: Optional[Settings] = Depends(get_settings),
) -> RecoveryService:
    """Dependency provider for RecoveryService instance with application-scoped vector index and agent configuration."""
    resolved_index = (
        vector_index if isinstance(vector_index, VectorIndex) else get_vector_index()
    )
    resolved_embedding = (
        embedding_service
        if isinstance(embedding_service, EmbeddingService)
        else get_embedding_service()
    )
    resolved_settings = (
        settings if isinstance(settings, Settings) else get_settings()
    )
    use_agent = resolved_settings.ENABLE_AGENT_DECISION_ENGINE
    agent_orchestrator: Optional[AgentOrchestrator] = None

    if use_agent:
        default_rec = LLMRecoveryRecommendation(
            recommended_action=RecoveryAction.PAYMENT_LINK,
            confidence=0.8,
            reasoning="Default adaptive recovery recommendation.",
            key_factors=("automated_recovery",),
            referenced_case_ids=(),
        )
        provider = MockLLMProvider(recommendation=default_rec)
        agent_orchestrator = AgentOrchestrator(provider=provider)

    return RecoveryService(
        vector_index=resolved_index,
        embedding_service=resolved_embedding,
        agent_orchestrator=agent_orchestrator,
        use_agent=use_agent,
    )


@router.post(
    "/evaluate-decision",
    response_model=RecoveryEvaluationResponse,
    status_code=status.HTTP_200_OK,
    summary="Evaluate Failed Payment Recovery Decision",
    description=(
        "Retrieves deterministic customer context, optionally synthesizes empirical historical "
        "RAG evidence, executes the decision engine, updates the recovery opportunity state, "
        "persists an audit event, and returns a clean recovery recommendation."
    ),
)
async def evaluate_decision(
    request: RecoveryEvaluationRequest,
    db: Session = Depends(get_db),
    principal: AuthenticatedPrincipal = Depends(get_current_principal),
    recovery_service: RecoveryService = Depends(get_recovery_service),
) -> RecoveryEvaluationResponse:
    """
    Evaluate a recovery decision for a specific failed payment and customer.
    Requires caller authentication and strict tenant authorization prior to evaluation.
    """
    if principal.customer_id != request.customer_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Cross-tenant access forbidden: authenticated customer cannot evaluate recovery for another customer.",
        )

    try:
        if hasattr(recovery_service, "evaluate_recovery_async"):
            return await recovery_service.evaluate_recovery_async(db_session=db, request=request)
        return recovery_service.evaluate_recovery(db_session=db, request=request)
    except CustomerNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc
    except PaymentNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc
    except RecoveryOpportunityNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc
    except PaymentCustomerMismatchError as exc:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=str(exc),
        ) from exc
