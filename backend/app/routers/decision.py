"""
FastAPI Router for Revora Recovery Decision API (/api/recovery/decision).

Exposes a thin HTTP interface that delegates to the RecoveryDecisionService.
Contains no RAG, LLM, or policy evaluation logic.
"""

import logging

from fastapi import APIRouter, Depends, HTTPException, status

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
    service: RecoveryDecisionService = Depends(get_recovery_decision_service),  # noqa: B008
) -> RecoveryDecisionResponse:
    """
    Evaluate recovery decision for a failed payment.

    Validates request payload, invokes application service orchestration,
    and returns sanitized decision telemetry without internal database coupling or secrets.
    """
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
