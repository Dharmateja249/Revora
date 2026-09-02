"""
FastAPI Router for Revora Recovery Decision API (/api/recovery/decision).

Exposes a thin HTTP interface that delegates to the RecoveryDecisionService.
Contains no RAG, LLM, or policy evaluation logic.
"""

import logging

from fastapi import APIRouter, Depends, HTTPException, status

from app.auth import AuthenticatedPrincipal, get_current_principal
from app.recovery_decision_service import (
    RecoveryDecisionService,
    get_recovery_decision_service,
)
from app.schemas.decision import (
    RecoveryDecisionRequest,
    RecoveryDecisionResponse,
)

logger = logging.getLogger("revora.decision_router")

router = APIRouter(prefix="/api/recovery", tags=["Recovery Decision"])


@router.post(
    "/decision",
    response_model=RecoveryDecisionResponse,
    status_code=status.HTTP_200_OK,
    summary="Evaluate Payment Recovery Decision",
    description=(
        "Evaluates a failed payment recovery decision by synthesizing customer profile, "
        "previous attempts, policy constraints, and agent intelligence into an actionable recommendation."
    ),
)
async def evaluate_recovery_decision(
    request: RecoveryDecisionRequest,
    principal: AuthenticatedPrincipal = Depends(get_current_principal),  # noqa: B008
    service: RecoveryDecisionService = Depends(get_recovery_decision_service),  # noqa: B008
) -> RecoveryDecisionResponse:
    """
    Evaluate recovery decision for a failed payment.

    Requires valid caller authentication. Enforces tenant authorization if customer_id is provided.
    Validates request payload, invokes application service orchestration,
    and returns sanitized decision telemetry without internal database coupling or secrets.
    """
    if (
        request.customer.customer_id is not None
        and principal.customer_id != request.customer.customer_id
    ):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Cross-tenant access forbidden: authenticated principal cannot evaluate recovery for another customer.",
        )

    if request.customer.customer_id is None:
        request.customer.customer_id = principal.customer_id
    try:
        return await service.evaluate_decision(request)
    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc
    except Exception as exc:
        logger.exception("Unexpected error while evaluating recovery decision.")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred while evaluating the recovery decision.",
        ) from exc
