"""
FastAPI Recovery Evaluation Router.

Provides HTTP endpoint for evaluating failed payment recovery decisions.
Handles HTTP request parsing, dependency injection, and domain exception translation.
"""

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.context import (
    CustomerNotFoundError,
    PaymentCustomerMismatchError,
    PaymentNotFoundError,
    RecoveryOpportunityNotFoundError,
)
from app.database import get_db
from app.recovery_service import RecoveryService
from app.schemas.recovery import (
    RecoveryEvaluationRequest,
    RecoveryEvaluationResponse,
)

router = APIRouter()


def get_recovery_service() -> RecoveryService:
    """Dependency provider for RecoveryService instance."""
    return RecoveryService()


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
    recovery_service: RecoveryService = Depends(get_recovery_service),
) -> RecoveryEvaluationResponse:
    """
    Evaluate a recovery decision for a specific failed payment and customer.
    """
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
