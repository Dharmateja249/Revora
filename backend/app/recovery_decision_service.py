"""
Revora Recovery Decision Application Service.

Coordinates the end-to-end recovery decision pipeline without database coupling:
request mapping -> policy resolution -> RAG evidence synthesis -> AgentOrchestrator -> response DTO.
"""

import logging
from collections.abc import Sequence
from uuid import uuid4

from fastapi import Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.action_executor import ActionExecutor
from app.agent.factory import create_llm_provider
from app.agent.orchestrator import AgentOrchestrator
from app.agent.schemas import AgentDecisionResult
from app.config import Settings, get_settings
from app.context import (
    CustomerContext,
    CustomerNotFoundError,
    CustomerRecoveryContext,
    PaymentContext,
    RecoveryAttemptContext,
    RecoveryOpportunityContext,
)
from app.context_retrieval import get_customer_context
from app.embedding_service import EmbeddingService, get_embedding_service
from app.historical_retrieval import HistoricalCase
from app.hybrid_historical_retriever import HybridHistoricalRetriever
from app.models import (
    AuditEvent,
    Customer,
    Payment,
    RecoveryAttempt,
    RecoveryOpportunity,
    utc_now,
)
from app.observability import get_request_id
from app.policies.resolver import resolve_policy_context
from app.retrieval_document import historical_case_to_document
from app.schemas.decision import (
    ActionExecutionResultDTO,
    CustomerProfileDTO,
    RecoveryAttemptDTO,
    RecoveryDecisionRequest,
    RecoveryDecisionResponse,
)
from app.semantic_historical_retriever import SemanticHistoricalRetriever
from app.vector_index import VectorIndex, get_vector_index

logger = logging.getLogger("revora.recovery_decision_service")


class RecoveryDecisionService:
    """
    Application Service that executes recovery decisions for HTTP API callers.
    Acts as the boundary between FastAPI routers and domain agent orchestration.
    """

    def __init__(
        self,
        agent_orchestrator: AgentOrchestrator,
        action_executor: ActionExecutor | None = None,
        hybrid_retriever: HybridHistoricalRetriever | None = None,
        vector_index: VectorIndex | None = None,
        embedding_service: EmbeddingService | None = None,
    ):
        if not isinstance(agent_orchestrator, AgentOrchestrator):
            raise TypeError(
                f"Expected AgentOrchestrator instance, got {type(agent_orchestrator).__name__}"
            )
        self._orchestrator = agent_orchestrator
        self._action_executor = action_executor or ActionExecutor()
        self._hybrid_retriever = hybrid_retriever
        self._vector_index = vector_index
        self._embedding_service = embedding_service

    @property
    def action_executor(self) -> ActionExecutor:
        """Return the bound ActionExecutor instance."""
        return self._action_executor

    @property
    def orchestrator(self) -> AgentOrchestrator:
        """Return the bound AgentOrchestrator instance."""
        return self._orchestrator

    def _resolve_rag_cases(
        self, context: CustomerRecoveryContext
    ) -> Sequence[HistoricalCase] | None:
        """
        Synthesize relevant historical cases from available retrievers if configured.
        """
        if self._hybrid_retriever is not None:
            try:
                return self._hybrid_retriever.retrieve_relevant_cases(context, top_k=5)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "Hybrid retriever failed; proceeding without RAG: %s", exc
                )
                return None

        if self._vector_index is not None and self._embedding_service is not None:
            try:
                retriever = SemanticHistoricalRetriever(
                    vector_index=self._vector_index,
                    embedding_service=self._embedding_service,
                )
                return retriever.retrieve(context, top_k=5)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "Semantic retriever failed; proceeding without RAG: %s", exc
                )
                return None

        return None

    async def evaluate_decision(
        self,
        request: RecoveryDecisionRequest,
        db_session: Session | None = None,
    ) -> RecoveryDecisionResponse:
        """
        Execute the adaptive recovery decision workflow for the incoming request payload.
        Derives customer context and historical payments from the relational database when
        db_session is available, falling back safely to request DTOs.

        Args:
            request: Validated RecoveryDecisionRequest DTO.
            db_session: Optional active SQLAlchemy database session.

        Returns:
            RecoveryDecisionResponse containing effective action, confidence, reasoning,
            policy status, and telemetry.
        """
        customer_id = request.customer.customer_id or uuid4()
        payment_id = request.payment_id or uuid4()
        opportunity_id = uuid4()
        req_id = get_request_id()

        opp_rec: RecoveryOpportunity | None = None
        payment_rec: Payment | None = None
        customer: Customer | None = None
        attempt_rec: RecoveryAttempt | None = None

        attempts_ctx = [
            RecoveryAttemptContext(
                action=attempt.action,
                status=attempt.status,
                amount_recovered=attempt.amount_recovered,
                error_code=attempt.error_code,
            )
            for attempt in request.previous_attempts
        ]

        # 1. Derive CustomerRecoveryContext from Database (Canonical Source of Truth)
        if db_session is not None:
            try:
                context = get_customer_context(
                    db_session=db_session,
                    customer_id=customer_id,
                    payment_id=request.payment_id,
                    current_payment_amount=request.amount,
                    current_payment_currency=request.currency,
                    current_payment_method=request.payment_method,
                    current_payment_failure_reason=request.failure_reason,
                    current_payment_status=request.payment_status,
                    current_opportunity_status=request.opportunity_status,
                    current_revenue_at_risk=request.revenue_at_risk,
                    current_attempts=attempts_ctx,
                )
            except CustomerNotFoundError:
                new_cust = Customer(
                    id=customer_id,
                    name=f"Customer {str(customer_id)[:8]}",
                    email=f"customer_{str(customer_id)[:8]}@revora.internal",
                    total_payments=request.customer.total_payments,
                    successful_payments=request.customer.successful_payments,
                    failed_payments=request.customer.failed_payments,
                )
                db_session.add(new_cust)
                db_session.commit()
                context = get_customer_context(
                    db_session=db_session,
                    customer_id=customer_id,
                    payment_id=request.payment_id,
                    current_payment_amount=request.amount,
                    current_payment_currency=request.currency,
                    current_payment_method=request.payment_method,
                    current_payment_failure_reason=request.failure_reason,
                    current_payment_status=request.payment_status,
                    current_opportunity_status=request.opportunity_status,
                    current_revenue_at_risk=request.revenue_at_risk,
                    current_attempts=attempts_ctx,
                )
        else:
            customer_ctx = CustomerContext(
                customer_id=customer_id,
                total_payments=request.customer.total_payments,
                successful_payments=request.customer.successful_payments,
                failed_payments=request.customer.failed_payments,
                historical_success_rate=request.customer.historical_success_rate,
            )
            payment_ctx = PaymentContext(
                payment_id=payment_id,
                amount=request.amount,
                currency=request.currency,
                payment_method=request.payment_method,
                status=request.payment_status,
                failure_reason=request.failure_reason,
            )
            revenue_at_risk = (
                request.revenue_at_risk
                if request.revenue_at_risk is not None
                else request.amount
            )
            opportunity_ctx = RecoveryOpportunityContext(
                opportunity_id=opportunity_id,
                status=request.opportunity_status,
                revenue_at_risk=revenue_at_risk,
                expected_recovery=0.0,
            )
            context = CustomerRecoveryContext(
                customer=customer_ctx,
                current_payment=payment_ctx,
                current_opportunity=opportunity_ctx,
                current_payment_attempts=attempts_ctx,
            )

        logger.info(
            "customer_context_built request_id=%s customer_id=%s total_payments=%d successful_payments=%d payment_count=%d",
            req_id,
            customer_id,
            context.customer.total_payments,
            context.customer.successful_payments,
            len(context.current_payment_attempts or []),
        )

        # 2. Resolve Structured Recovery Policy Context
        policy_context = resolve_policy_context(
            context=context,
            provider="razorpay",
            max_attempts=request.max_attempts,
        )

        # 3. Retrieve Empirical Historical Evidence
        historical_cases = self._resolve_rag_cases(context)
        logger.info(
            "rag_retrieval_completed request_id=%s precedent_count=%d",
            req_id,
            len(historical_cases or []),
        )

        # 4. Invoke Agent Decision Pipeline
        agent_result: AgentDecisionResult = await self._orchestrator.decide(
            context=context,
            policy_context=policy_context,
            historical_cases=historical_cases,
        )
        approved_action = agent_result.recommendation.recommended_action
        policy_overridden = bool(agent_result.metadata.get("policy_overridden", False))

        logger.info(
            "llm_decision_generated request_id=%s candidate_action=%s confidence=%.2f agent_used=%s is_fallback=%s",
            req_id,
            agent_result.recommendation.recommended_action.value,
            agent_result.recommendation.confidence,
            agent_result.agent_used,
            agent_result.is_fallback,
        )
        logger.info(
            "policy_validation_completed request_id=%s approved_action=%s policy_overridden=%s",
            req_id,
            approved_action.value,
            policy_overridden,
        )

        # 5. Execute Action if Requested (Only after PolicyValidator approval)
        approved_action = agent_result.recommendation.recommended_action
        execution_dto: ActionExecutionResultDTO | None = None

        if request.execute_action:
            prior_count = len(context.current_payment_attempts or [])
            attempt_idx = prior_count + 1
            # Derive stable execution identity (idempotency key)
            if request.idempotency_key:
                derived_key = str(request.idempotency_key).strip()[:64]
            elif request.payment_id:
                derived_key = (
                    f"rec_{str(request.payment_id).replace('-', '')}_{attempt_idx}"
                )
            else:
                derived_key = f"rec_{payment_id.hex}_{attempt_idx}"

            # 5a. Pre-Execution Idempotency & Tenant Boundary Verification
            attempt_rec: RecoveryAttempt | None = None
            is_replay: bool = False

            if db_session is not None:
                # Check for cross-tenant key collision
                existing_key_attempt = (
                    db_session.execute(
                        select(RecoveryAttempt)
                        .join(
                            RecoveryOpportunity,
                            RecoveryAttempt.opportunity_id == RecoveryOpportunity.id,
                        )
                        .join(Payment, RecoveryOpportunity.payment_id == Payment.id)
                        .where(
                            (RecoveryAttempt.idempotency_key == derived_key)
                            | (
                                RecoveryAttempt.idempotency_key
                                == f"rec_{payment_id.hex}"
                            )
                        )
                    )
                    .scalars()
                    .first()
                )
                if existing_key_attempt is not None:
                    # Check tenant isolation
                    existing_payment = db_session.get(
                        Payment, existing_key_attempt.opportunity.payment_id
                    )
                    if (
                        existing_payment is not None
                        and existing_payment.customer_id != customer_id
                    ):
                        logger.warning(
                            "Tenant isolation violation: Idempotency key '%s' belongs to customer %s, attempted by customer %s",
                            derived_key,
                            existing_payment.customer_id,
                            customer_id,
                        )
                        raise HTTPException(
                            status_code=403,
                            detail="Idempotency key collision across tenant boundary.",
                        )

                # Check for existing completed / succeeded recovery attempt
                existing_successful_attempt = (
                    db_session.execute(
                        select(RecoveryAttempt)
                        .join(
                            RecoveryOpportunity,
                            RecoveryAttempt.opportunity_id == RecoveryOpportunity.id,
                        )
                        .where(
                            (RecoveryAttempt.idempotency_key == derived_key)
                            | (RecoveryOpportunity.payment_id == payment_id)
                        )
                        .where(
                            RecoveryAttempt.status.in_(
                                ["succeeded", "success", "simulated"]
                            )
                        )
                        .order_by(RecoveryAttempt.created_at.desc())
                    )
                    .scalars()
                    .first()
                )

                if existing_successful_attempt is not None:
                    logger.info(
                        "Execution replay detected for payment %s (key %s). Returning existing idempotent result.",
                        payment_id,
                        derived_key,
                    )
                    is_replay = True
                    ext_ref = existing_successful_attempt.external_reference
                    resource_url = (
                        f"https://rzp.io/i/{ext_ref.replace('plink_', '')}"
                        if ext_ref and not ext_ref.startswith("http")
                        else ext_ref
                    )
                    execution_dto = ActionExecutionResultDTO(
                        action=approved_action,
                        attempted=False,
                        status="already_executed",
                        success=True,
                        reference_id=ext_ref,
                        resource_url=resource_url,
                        message="Recovery action already successfully executed; returning existing idempotent record.",
                        persisted=True,
                        persistence_error=None,
                    )
                else:
                    # Check if currently in_progress by another active worker
                    existing_in_progress = (
                        db_session.execute(
                            select(RecoveryAttempt)
                            .where(
                                (RecoveryAttempt.idempotency_key == derived_key)
                                | (
                                    RecoveryAttempt.idempotency_key
                                    == f"rec_{payment_id.hex}"
                                )
                            )
                            .where(RecoveryAttempt.status == "in_progress")
                        )
                        .scalars()
                        .first()
                    )
                    if existing_in_progress is not None:
                        logger.info(
                            "Concurrent execution in progress for payment %s (key %s).",
                            payment_id,
                            derived_key,
                        )
                        is_replay = True
                        execution_dto = ActionExecutionResultDTO(
                            action=approved_action,
                            attempted=False,
                            status="in_progress",
                            success=False,
                            reference_id=None,
                            resource_url=None,
                            message="Recovery execution currently in progress for this payment.",
                            persisted=True,
                            persistence_error=None,
                        )

            # 5b. If not replaying, atomically reserve attempt and execute via Gateway
            if not is_replay:
                if db_session is not None:
                    try:
                        customer = db_session.get(Customer, customer_id)
                        if customer is None:
                            customer = Customer(
                                id=customer_id,
                                name=f"Customer {str(customer_id)[:8]}",
                                email=f"customer_{str(customer_id)[:8]}@revora.internal",
                                total_payments=0,
                                successful_payments=0,
                                failed_payments=0,
                            )
                            db_session.add(customer)
                            db_session.flush()

                        payment_rec = db_session.get(Payment, payment_id)
                        if payment_rec is None:
                            payment_rec = Payment(
                                id=payment_id,
                                customer_id=customer.id,
                                amount=request.amount,
                                currency=request.currency,
                                payment_method=request.payment_method,
                                status=request.payment_status or "failed",
                                failure_reason=request.failure_reason,
                                created_at=utc_now(),
                            )
                            db_session.add(payment_rec)
                            db_session.flush()

                        opp_rec = (
                            db_session.execute(
                                select(RecoveryOpportunity).where(
                                    RecoveryOpportunity.payment_id == payment_rec.id
                                )
                            )
                            .scalars()
                            .first()
                        )
                        if opp_rec is None:
                            opp_rec = RecoveryOpportunity(
                                id=opportunity_id,
                                payment_id=payment_rec.id,
                                status="in_progress",
                                revenue_at_risk=request.amount,
                                expected_recovery=0.0,
                                recommended_action=approved_action.value,
                                confidence=agent_result.recommendation.confidence,
                                created_at=utc_now(),
                            )
                            db_session.add(opp_rec)
                            db_session.flush()

                        # Check if an existing attempt with this key exists (retry of failed execution)
                        attempt_rec = (
                            db_session.execute(
                                select(RecoveryAttempt).where(
                                    RecoveryAttempt.idempotency_key == derived_key
                                )
                            )
                            .scalars()
                            .first()
                        )
                        if attempt_rec is not None:
                            attempt_rec.status = "in_progress"
                            attempt_rec.error_code = None
                            attempt_rec.action = approved_action.value
                        else:
                            attempt_rec = RecoveryAttempt(
                                id=uuid4(),
                                opportunity_id=opp_rec.id,
                                action=approved_action.value,
                                status="in_progress",
                                idempotency_key=derived_key,
                                amount_recovered=0.0,
                                created_at=utc_now(),
                            )
                            db_session.add(attempt_rec)

                        db_session.commit()
                    except Exception as reserve_err:  # noqa: BLE001
                        db_session.rollback()
                        logger.warning(
                            "Concurrent reservation error for key %s: %s",
                            derived_key,
                            reserve_err,
                        )
                        attempt_rec = None

                # 5c. Outbound Gateway Execution with deterministic reference_id
                logger.info(
                    "recovery_execution_started request_id=%s action=%s idempotency_key=%s",
                    req_id,
                    approved_action.value,
                    derived_key,
                )
                action_result = await self._action_executor.execute(
                    approved_action=approved_action,
                    policy_context=policy_context,
                    context=context,
                    reference_id=derived_key,
                )
                execution_dto = ActionExecutionResultDTO(
                    action=action_result.action,
                    attempted=action_result.attempted,
                    status=action_result.status,
                    success=action_result.success,
                    reference_id=action_result.reference_id,
                    resource_url=action_result.resource_url,
                    message=action_result.message,
                    error=action_result.error,
                )

                # 6. Closed-Loop Transactional Persistence to Database
                if db_session is not None:
                    try:
                        customer = db_session.get(Customer, customer_id)
                        if customer is None:
                            customer = Customer(
                                id=customer_id,
                                name=f"Customer {str(customer_id)[:8]}",
                                email=f"customer_{str(customer_id)[:8]}@revora.internal",
                                total_payments=0,
                                successful_payments=0,
                                failed_payments=0,
                            )
                            db_session.add(customer)
                            db_session.flush()

                        payment_rec = db_session.get(Payment, payment_id)
                        if payment_rec is None:
                            payment_rec = Payment(
                                id=payment_id,
                                customer_id=customer.id,
                                amount=request.amount,
                                currency=request.currency,
                                payment_method=request.payment_method,
                                status="succeeded"
                                if action_result.success
                                else "failed",
                                failure_reason=request.failure_reason,
                                created_at=utc_now(),
                            )
                            db_session.add(payment_rec)
                            db_session.flush()
                        else:
                            payment_rec.status = (
                                "succeeded" if action_result.success else "failed"
                            )

                        opp_rec = (
                            db_session.execute(
                                select(RecoveryOpportunity).where(
                                    RecoveryOpportunity.payment_id == payment_rec.id
                                )
                            )
                            .scalars()
                            .first()
                        )
                        prior_attempts_count = len(
                            context.current_payment_attempts or []
                        )
                        opp_status = (
                            "recovered"
                            if action_result.success
                            else (
                                "failed"
                                if (prior_attempts_count + 1 >= request.max_attempts)
                                else "open"
                            )
                        )
                        if opp_rec is None:
                            opp_rec = RecoveryOpportunity(
                                id=opportunity_id,
                                payment_id=payment_rec.id,
                                status=opp_status,
                                revenue_at_risk=request.amount,
                                expected_recovery=request.amount
                                if action_result.success
                                else 0.0,
                                recommended_action=approved_action.value,
                                confidence=agent_result.recommendation.confidence,
                                created_at=utc_now(),
                            )
                            db_session.add(opp_rec)
                            db_session.flush()
                        else:
                            opp_rec.status = opp_status
                            opp_rec.expected_recovery = (
                                request.amount if action_result.success else 0.0
                            )

                        amount_recov = request.amount if action_result.success else 0.0
                        if attempt_rec is None:
                            attempt_rec = (
                                db_session.execute(
                                    select(RecoveryAttempt).where(
                                        RecoveryAttempt.idempotency_key == derived_key
                                    )
                                )
                                .scalars()
                                .first()
                            )
                        if attempt_rec is None:
                            attempt_rec = RecoveryAttempt(
                                id=uuid4(),
                                opportunity_id=opp_rec.id,
                                action=approved_action.value,
                                status=action_result.status,
                                idempotency_key=derived_key,
                                amount_recovered=amount_recov,
                                external_reference=action_result.reference_id,
                                error_code=action_result.error,
                                created_at=utc_now(),
                                completed_at=utc_now(),
                            )
                            db_session.add(attempt_rec)
                        else:
                            attempt_rec.status = action_result.status
                            attempt_rec.amount_recovered = amount_recov
                            attempt_rec.external_reference = action_result.reference_id
                            attempt_rec.error_code = action_result.error
                            attempt_rec.completed_at = utc_now()

                        # Update Customer lifetime counters
                        customer.total_payments += 1
                        if action_result.success:
                            customer.successful_payments += 1
                        else:
                            customer.failed_payments += 1

                        # Persist AuditEvent record
                        audit = AuditEvent(
                            opportunity_id=opp_rec.id,
                            event_type="recovery_action_executed",
                            description=(
                                f"Action {approved_action.value} executed with status: "
                                f"{action_result.status} (success={action_result.success})."
                            ),
                            metadata_payload={
                                "action": approved_action.value,
                                "status": action_result.status,
                                "success": action_result.success,
                                "reference_id": action_result.reference_id,
                                "resource_url": action_result.resource_url,
                                "idempotency_key": derived_key,
                                "request_id": req_id,
                                "error": action_result.error,
                                "agent_used": agent_result.agent_used,
                                "policy_overridden": policy_overridden,
                                "is_fallback": agent_result.is_fallback,
                            },
                            created_at=utc_now(),
                        )
                        db_session.add(audit)
                        db_session.commit()

                        # 7. Adaptive Feedback: Dynamically Ingest Case into Runtime RAG Index
                        if (
                            self._vector_index is not None
                            and self._embedding_service is not None
                        ):
                            try:
                                success_rate = (
                                    customer.successful_payments
                                    / customer.total_payments
                                    if customer.total_payments > 0
                                    else 0.0
                                )
                                new_case = HistoricalCase(
                                    payment_id=payment_rec.id,
                                    customer_id=customer.id,
                                    amount=float(request.amount),
                                    currency=str(request.currency or "INR"),
                                    payment_method=str(
                                        request.payment_method or "card"
                                    ),
                                    failure_reason=str(request.failure_reason)
                                    if request.failure_reason
                                    else None,
                                    recovery_action=approved_action.value,
                                    recovery_status="recovered"
                                    if action_result.success
                                    else "failed",
                                    amount_recovered=float(amount_recov),
                                    was_recovered=bool(action_result.success),
                                    external_payment_id=f"case_live_{str(payment_id)[:8]}",
                                    created_at=payment_rec.created_at,
                                    completed_at=attempt_rec.completed_at,
                                    metadata={
                                        "case_id": f"case_live_{str(payment_id)[:8]}",
                                        "customer_success_rate": success_rate,
                                    },
                                )
                                doc = historical_case_to_document(new_case)
                                emb = self._embedding_service.embed(doc.canonical_text)
                                self._vector_index.add(doc, emb)
                                logger.info(
                                    "Dynamically ingested recovery outcome into runtime RAG index: %s",
                                    doc.document_id,
                                )
                            except Exception as rag_err:  # noqa: BLE001
                                logger.warning(
                                    "Failed to ingest completed recovery case into vector index: %s",
                                    rag_err,
                                )
                    except Exception:
                        logger.exception("Failed to persist executed recovery outcome.")
                        db_session.rollback()
                        try:
                            # Reset attempt status from in_progress to failed so retry is not blocked
                            if attempt_rec is not None:
                                attempt_rec.status = "failed"
                                attempt_rec.error_code = "DB_PERSISTENCE_FAILED"
                                db_session.commit()
                        except Exception:  # noqa: BLE001
                            db_session.rollback()
                        if execution_dto is not None:
                            execution_dto = execution_dto.model_copy(
                                update={
                                    "persisted": False,
                                    "persistence_error": "Recovery outcome could not be persisted; reconciliation required.",
                                }
                            )

                logger.info(
                    "recovery_execution_completed request_id=%s action=%s success=%s status=%s persisted=%s reference_id=%s",
                    req_id,
                    approved_action.value,
                    action_result.success,
                    action_result.status,
                    execution_dto.persisted if execution_dto else False,
                    action_result.reference_id,
                )

        # 8. Map AgentDecisionResult & Execution to API Response DTO
        policy_overridden = bool(agent_result.metadata.get("policy_overridden", False))

        resp_payment_id = payment_id
        resp_opportunity_id = opportunity_id
        resp_opportunity_status = (
            context.current_opportunity.status
            if context.current_opportunity
            else "open"
        )
        resp_attempts: list[RecoveryAttemptDTO] = []
        resp_attempt_count = 0
        resp_customer: CustomerProfileDTO | None = None

        if db_session is not None:
            active_opp = opp_rec
            if active_opp is None and payment_rec is not None:
                active_opp = payment_rec.recovery_opportunity
            if active_opp is None and request.payment_id is not None:
                active_pay = db_session.get(Payment, request.payment_id)
                if active_pay is not None:
                    active_opp = active_pay.recovery_opportunity

            if active_opp is not None:
                resp_opportunity_id = active_opp.id
                resp_opportunity_status = active_opp.status
                db_attempts = (
                    db_session.execute(
                        select(RecoveryAttempt)
                        .where(RecoveryAttempt.opportunity_id == active_opp.id)
                        .order_by(
                            RecoveryAttempt.created_at.asc(), RecoveryAttempt.id.asc()
                        )
                    )
                    .scalars()
                    .all()
                )
                resp_attempts = [
                    RecoveryAttemptDTO(
                        action=a.action,
                        status=a.status,
                        amount_recovered=float(a.amount_recovered),
                        error_code=a.error_code,
                    )
                    for a in db_attempts
                ]
                resp_attempt_count = len(resp_attempts)
            else:
                resp_attempts = [
                    RecoveryAttemptDTO(
                        action=a.action,
                        status=a.status,
                        amount_recovered=float(a.amount_recovered),
                        error_code=a.error_code,
                    )
                    for a in (context.current_payment_attempts or [])
                ]
                resp_attempt_count = len(resp_attempts)

            db_cust = customer or db_session.get(Customer, customer_id)
            if db_cust is not None:
                succ_rate = (
                    round(db_cust.successful_payments / db_cust.total_payments, 2)
                    if db_cust.total_payments > 0
                    else 0.0
                )
                resp_customer = CustomerProfileDTO(
                    customer_id=db_cust.id,
                    total_payments=db_cust.total_payments,
                    successful_payments=db_cust.successful_payments,
                    failed_payments=db_cust.failed_payments,
                    historical_success_rate=succ_rate,
                )
        else:
            resp_attempts = [
                RecoveryAttemptDTO(
                    action=a.action,
                    status=a.status,
                    amount_recovered=float(a.amount_recovered),
                    error_code=a.error_code,
                )
                for a in (context.current_payment_attempts or [])
            ]
            resp_attempt_count = len(resp_attempts)
            if context.customer:
                resp_customer = CustomerProfileDTO(
                    customer_id=context.customer.customer_id,
                    total_payments=context.customer.total_payments,
                    successful_payments=context.customer.successful_payments,
                    failed_payments=context.customer.failed_payments,
                    historical_success_rate=context.customer.historical_success_rate,
                )

        return RecoveryDecisionResponse(
            recommended_action=approved_action,
            confidence=agent_result.recommendation.confidence,
            reasoning=agent_result.recommendation.reasoning,
            key_factors=list(agent_result.recommendation.key_factors),
            referenced_case_ids=list(agent_result.recommendation.referenced_case_ids),
            agent_used=agent_result.agent_used,
            policy_overridden=policy_overridden,
            is_fallback=agent_result.is_fallback,
            fallback_reason=agent_result.fallback_reason,
            execution=execution_dto,
            request_id=req_id,
            payment_id=resp_payment_id,
            opportunity_id=resp_opportunity_id,
            opportunity_status=resp_opportunity_status,
            attempt_count=resp_attempt_count,
            previous_attempts=resp_attempts,
            customer=resp_customer,
        )


def get_recovery_decision_service(
    settings: Settings = Depends(get_settings),  # noqa: B008
    vector_index: VectorIndex | None = Depends(get_vector_index),  # noqa: B008
    embedding_service: EmbeddingService | None = Depends(get_embedding_service),  # noqa: B008
) -> RecoveryDecisionService:
    """
    FastAPI dependency that constructs RecoveryDecisionService with configured LLMProvider.
    Respects LLM_PROVIDER from settings, failing fast on invalid OpenAI configurations.
    """
    resolved_settings = settings if isinstance(settings, Settings) else get_settings()
    provider = create_llm_provider(resolved_settings)
    orchestrator = AgentOrchestrator(provider=provider)

    from app.razorpay_adapter import RazorpayAdapter

    adapter = RazorpayAdapter(
        key_id=resolved_settings.RAZORPAY_KEY_ID,
        key_secret=resolved_settings.RAZORPAY_KEY_SECRET,
        base_url=resolved_settings.RAZORPAY_BASE_URL,
        dry_run=resolved_settings.RAZORPAY_DRY_RUN,
    )
    action_executor = ActionExecutor(razorpay_adapter=adapter)

    resolved_index = (
        vector_index if isinstance(vector_index, VectorIndex) else get_vector_index()
    )
    resolved_embedding = (
        embedding_service
        if isinstance(embedding_service, EmbeddingService)
        else get_embedding_service()
    )

    return RecoveryDecisionService(
        agent_orchestrator=orchestrator,
        action_executor=action_executor,
        vector_index=resolved_index,
        embedding_service=resolved_embedding,
    )
