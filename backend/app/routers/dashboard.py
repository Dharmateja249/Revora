"""
FastAPI Router for Revora Dashboard Metrics API (/api/dashboard/metrics).

Exposes real-time recovery and AI performance KPIs scoped to the authenticated tenant.
"""

import logging

from fastapi import APIRouter, Depends, status
from sqlalchemy.orm import Session

from app.auth import AuthenticatedPrincipal, get_current_principal
from app.dashboard_service import DashboardService, get_dashboard_service
from app.database import get_db
from app.schemas.dashboard import DashboardMetricsResponse

logger = logging.getLogger("revora.dashboard_router")

router = APIRouter(prefix="/api/dashboard", tags=["Dashboard"])


@router.get(
    "/metrics",
    response_model=DashboardMetricsResponse,
    status_code=status.HTTP_200_OK,
    summary="Get Real-Time Recovery & AI Dashboard Metrics",
    description=(
        "Returns aggregated recovery, financial, execution, and AI performance metrics "
        "calculated directly from the authenticated tenant's database records and runtime RAG index."
    ),
)
def get_dashboard_metrics(
    principal: AuthenticatedPrincipal = Depends(get_current_principal),  # noqa: B008
    service: DashboardService = Depends(get_dashboard_service),  # noqa: B008
    db: Session = Depends(get_db),  # noqa: B008
) -> DashboardMetricsResponse:
    """
    Retrieve real-time metrics for the authenticated customer.

    Enforces strict tenant isolation: metrics are computed exclusively from records
    belonging to the principal's customer_id.
    """
    logger.info("Fetching dashboard metrics for customer_id=%s", principal.customer_id)
    return service.get_dashboard_metrics(db=db, customer_id=principal.customer_id)
