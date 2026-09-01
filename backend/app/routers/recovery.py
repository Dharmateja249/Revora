from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.auth import AuthenticatedPrincipal, get_current_principal
from app.config import Settings, get_settings
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
    vector_index: VectorIndex | None = Depends(get_vector_index),  # noqa: B008
    embedding_service: EmbeddingService | None = Depends(get_embedding_service),  # noqa: B008
    settings: Settings | None = Depends(get_settings),  # noqa: B008
) -> RecoveryService:
    """
    Dependency provider for RecoveryService instance with application-scoped vector index and configuration.

    In the current milestone, no production LLM provider is registered by default. MockLLMProvider is strictly
    for test and preview harnesses and is never bound to production routes. If ENABLE_AGENT_DECISION_ENGINE is
    enabled without an explicitly injected orchestrator, RecoveryService cleanly falls back to the deterministic
    DecisionEngine.
    """
    resolved_index = (
        vector_index if isinstance(vector_index, VectorIndex) else get_vector_index()
    )
    resolved_embedding = (
        embedding_service
        if isinstance(embedding_service, EmbeddingService)
        else get_embedding_service()
    )
    resolved_settings = settings if isinstance(settings, Settings) else get_settings()
    use_agent = resolved_settings.ENABLE_AGENT_DECISION_ENGINE

    return RecoveryService(
        vector_index=resolved_index,
        embedding_service=resolved_embedding,
        agent_orchestrator=None,
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
def evaluate_decision(
    request: RecoveryEvaluationRequest,
    db: Session = Depends(get_db),  # noqa: B008
    principal: AuthenticatedPrincipal = Depends(get_current_principal),  # noqa: B008
    recovery_service: RecoveryService = Depends(get_recovery_service),  # noqa: B008
) -> RecoveryEvaluationResponse:
    """
    Evaluate a recovery decision for a specific failed payment and customer.
    Requires caller authentication and strict tenant authorization prior to evaluation.
    Dispatched synchronously to FastAPI's worker threadpool to prevent blocking the ASGI event loop
    during synchronous SQLAlchemy database, vector retrieval, and persistence operations.
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
