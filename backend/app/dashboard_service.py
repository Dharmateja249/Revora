"""
Revora Dashboard Metrics Application Service.

Aggregates real-time recovery, financial, and AI operational telemetry directly
from relational database state and runtime VectorIndex with strict tenant isolation.
"""

import logging
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import AuditEvent, Payment, RecoveryAttempt, RecoveryOpportunity
from app.schemas.dashboard import DashboardMetricsResponse
from app.vector_index import VectorIndex, get_vector_index

logger = logging.getLogger("revora.dashboard_service")


class DashboardService:
    """
    Application service computing tenant-scoped recovery and AI KPIs.
    Guarantees that metrics are derived solely from persisted database records
    and active runtime index state without hardcoded mock constants.
    """

    def __init__(self, vector_index: VectorIndex | None = None) -> None:
        self._vector_index = vector_index

    def get_dashboard_metrics(
        self,
        db: Session,
        customer_id: UUID,
    ) -> DashboardMetricsResponse:
        """
        Compute real-time dashboard KPIs for the authenticated customer/tenant.

        Args:
            db: Active SQLAlchemy database session.
            customer_id: Authenticated customer UUID (tenant boundary).

        Returns:
            DashboardMetricsResponse DTO containing verified metric totals and percentages.
        """
        # 1. Tenant-scoped Recovery Opportunities
        opp_stmt = (
            select(RecoveryOpportunity)
            .join(Payment, RecoveryOpportunity.payment_id == Payment.id)
            .where(Payment.customer_id == customer_id)
        )
        opportunities = list(db.execute(opp_stmt).scalars().all())

        total_cases = len(opportunities)
        recovered_cases = sum(1 for o in opportunities if o.status == "recovered")
        failed_cases = sum(
            1 for o in opportunities if o.status in ("failed", "abandoned")
        )
        pending_cases = sum(
            1 for o in opportunities if o.status in ("open", "in_progress")
        )
        revenue_at_risk = sum(
            float(o.revenue_at_risk)
            for o in opportunities
            if o.status in ("open", "in_progress")
        )

        recovery_rate = (
            round((recovered_cases / total_cases * 100.0), 1)
            if total_cases > 0
            else 0.0
        )

        confidences = [
            float(o.confidence) for o in opportunities if o.confidence is not None
        ]
        average_confidence = (
            round(sum(confidences) / len(confidences), 2) if confidences else 0.0
        )

        # 2. Tenant-scoped Recovery Execution Attempts
        att_stmt = (
            select(RecoveryAttempt)
            .join(
                RecoveryOpportunity,
                RecoveryAttempt.opportunity_id == RecoveryOpportunity.id,
            )
            .join(Payment, RecoveryOpportunity.payment_id == Payment.id)
            .where(Payment.customer_id == customer_id)
        )
        attempts = list(db.execute(att_stmt).scalars().all())

        total_executions = len(attempts)
        successful_executions = sum(
            1 for a in attempts if a.status in ("succeeded", "success", "simulated")
        )
        failed_executions = sum(1 for a in attempts if a.status == "failed")
        pending_executions = sum(
            1 for a in attempts if a.status in ("pending", "in_progress")
        )
        amount_recovered = sum(
            float(a.amount_recovered)
            for a in attempts
            if a.status in ("succeeded", "success", "simulated")
        )

        execution_success_rate = (
            round((successful_executions / total_executions * 100.0), 1)
            if total_executions > 0
            else 0.0
        )

        # 3. Tenant-scoped Audit Events (Policy Overrides & Fallbacks)
        audit_stmt = (
            select(AuditEvent)
            .join(
                RecoveryOpportunity,
                AuditEvent.opportunity_id == RecoveryOpportunity.id,
            )
            .join(Payment, RecoveryOpportunity.payment_id == Payment.id)
            .where(Payment.customer_id == customer_id)
        )
        audit_events = list(db.execute(audit_stmt).scalars().all())

        policy_overrides = sum(
            1
            for event in audit_events
            if isinstance(event.metadata_payload, dict)
            and bool(event.metadata_payload.get("policy_overridden"))
        )
        fallback_decisions = sum(
            1
            for event in audit_events
            if isinstance(event.metadata_payload, dict)
            and (
                bool(event.metadata_payload.get("is_fallback"))
                or (event.metadata_payload.get("agent_used") is False)
            )
        )

        # 4. Runtime VectorIndex Precedent Count
        rag_precedents = 0
        resolved_index = self._vector_index or get_vector_index()
        if resolved_index is not None:
            try:
                rag_precedents = resolved_index.size
            except Exception as index_err:  # noqa: BLE001
                logger.warning(
                    "Unable to read runtime vector index size: %s", index_err
                )
                rag_precedents = 0

        return DashboardMetricsResponse(
            recovery_rate=recovery_rate,
            amount_recovered=amount_recovered,
            total_cases=total_cases,
            recovered_cases=recovered_cases,
            failed_cases=failed_cases,
            pending_cases=pending_cases,
            revenue_at_risk=revenue_at_risk,
            execution_success_rate=execution_success_rate,
            total_executions=total_executions,
            successful_executions=successful_executions,
            failed_executions=failed_executions,
            pending_executions=pending_executions,
            policy_overrides=policy_overrides,
            rag_precedents=rag_precedents,
            average_confidence=average_confidence,
            fallback_decisions=fallback_decisions,
        )


def get_dashboard_service() -> DashboardService:
    """FastAPI dependency provider for DashboardService."""
    return DashboardService(vector_index=get_vector_index())
