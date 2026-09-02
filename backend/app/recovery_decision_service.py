"""
Revora Recovery Decision Application Service.

Coordinates the end-to-end recovery decision pipeline without database coupling:
request mapping -> policy resolution -> RAG evidence synthesis -> AgentOrchestrator -> response DTO.
"""

import logging
from collections.abc import Sequence
from uuid import uuid4

from fastapi import Depends

from app.action_executor import ActionExecutor
from app.agent.factory import create_llm_provider
from app.agent.orchestrator import AgentOrchestrator
from app.agent.schemas import AgentDecisionResult
from app.config import Settings, get_settings
from app.context import (
    CustomerContext,
    CustomerRecoveryContext,
    PaymentContext,
    RecoveryAttemptContext,
    RecoveryOpportunityContext,
)
from app.embedding_service import EmbeddingService, get_embedding_service
from app.historical_retrieval import HistoricalCase
from app.hybrid_historical_retriever import HybridHistoricalRetriever
from app.policies.resolver import resolve_policy_context
from app.schemas.decision import (
    ActionExecutionResultDTO,
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
        self, request: RecoveryDecisionRequest
    ) -> RecoveryDecisionResponse:
        """
        Execute the adaptive recovery decision workflow for the incoming request payload.

        Args:
            request: Validated RecoveryDecisionRequest DTO.

        Returns:
            RecoveryDecisionResponse containing effective action, confidence, reasoning,
            policy status, and telemetry.
        """
        # 1. Translate DTO to Domain CustomerRecoveryContext
        customer_id = request.customer.customer_id or uuid4()
        payment_id = request.payment_id or uuid4()
        opportunity_id = uuid4()

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

        attempts_ctx = [
            RecoveryAttemptContext(
                action=attempt.action,
                status=attempt.status,
                amount_recovered=attempt.amount_recovered,
                error_code=attempt.error_code,
            )
            for attempt in request.previous_attempts
        ]

        context = CustomerRecoveryContext(
            customer=customer_ctx,
            current_payment=payment_ctx,
            current_opportunity=opportunity_ctx,
            current_payment_attempts=attempts_ctx,
        )

        # 2. Resolve Structured Recovery Policy Context
        policy_context = resolve_policy_context(
            context=context,
            provider="razorpay",
            max_attempts=request.max_attempts,
        )

        # 3. Retrieve Empirical Historical Evidence
        historical_cases = self._resolve_rag_cases(context)

        # 4. Invoke Agent Decision Pipeline
        agent_result: AgentDecisionResult = await self._orchestrator.decide(
            context=context,
            policy_context=policy_context,
            historical_cases=historical_cases,
        )

        # 5. Execute Action if Requested (Only after PolicyValidator approval)
        approved_action = agent_result.recommendation.recommended_action
        execution_dto: ActionExecutionResultDTO | None = None

        if request.execute_action:
            action_result = await self._action_executor.execute(
                approved_action=approved_action,
                policy_context=policy_context,
                context=context,
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

        # 6. Map AgentDecisionResult & Execution to API Response DTO
        policy_overridden = bool(agent_result.metadata.get("policy_overridden", False))

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
