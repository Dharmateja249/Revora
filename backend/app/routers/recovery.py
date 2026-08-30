"""
FastAPI Recovery Evaluation Router.

Provides HTTP endpoint for evaluating failed payment recovery decisions.
Handles HTTP request parsing, dependency injection, and domain exception translation.
"""

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.auth import AuthenticatedPrincipal, get_current_principal
from app.context import (
    CustomerNotFoundError,
    PaymentCustomerMismatchError,
    PaymentNotFoundError,
    RecoveryOpportunityNotFoundError,
)
from app.database import get_db
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
) -> RecoveryService:
    """Dependency provider for RecoveryService instance with application-scoped vector index."""
    resolved_index = (
        vector_index if isinstance(vector_index, VectorIndex) else get_vector_index()
    )
    resolved_embedding = (
        embedding_service
        if isinstance(embedding_service, EmbeddingService)
        else get_embedding_service()
    )
    return RecoveryService(
        vector_index=resolved_index,
        embedding_service=resolved_embedding,
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
def evaluate_decision(
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
