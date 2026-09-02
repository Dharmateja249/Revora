"""
Revora Decision Evaluation Runner & Pipeline Adapters.

Evaluates decision engines and agent orchestrators against independent DecisionGroundTruth
oracles, producing immutable DecisionEvalResult records and aggregate DecisionBenchmarkReport summaries.
"""

import asyncio
import concurrent.futures
import inspect
import time
from collections.abc import Sequence
from typing import Any, Protocol, runtime_checkable

from app.agent.orchestrator import AgentOrchestrator
from app.agent.schemas import AgentDecisionResult
from app.context import CustomerRecoveryContext
from app.decision_engine import DecisionEngine, RecoveryAction, RecoveryDecision
from app.evaluation.decision_metrics import compute_aggregate_decision_metrics
from app.evaluation.schemas import (
    DecisionBenchmarkReport,
    DecisionEvalResult,
    DecisionGroundTruth,
    EvaluationCase,
)
from app.historical_retrieval import HistoricalCase
from app.policies.resolver import resolve_policy_context
from app.policies.schemas import RecoveryPolicyContext


def extract_historical_cases_from_context(
    context: CustomerRecoveryContext,
) -> list[HistoricalCase]:
    """Extract and materialize HistoricalCase domain models from CustomerRecoveryContext."""
    customer_id = context.customer.customer_id
    ext_cust_id = context.customer.external_customer_id
    cases: list[HistoricalCase] = []
    for hp in context.historical_payments:
        cases.append(
            HistoricalCase(
                payment_id=hp.payment_id,
                customer_id=customer_id,
                external_payment_id=hp.external_payment_id,
                external_customer_id=ext_cust_id,
                amount=hp.amount,
                currency=hp.currency,
                payment_method=hp.payment_method,
                failure_reason=hp.failure_reason,
                recovery_action=hp.recovery_action,
                recovery_status="recovered" if hp.was_recovered else "failed",
                amount_recovered=hp.amount if hp.was_recovered else 0.0,
                was_recovered=hp.was_recovered,
                created_at=hp.created_at,
            )
        )
    return cases


@runtime_checkable
class DecisionPipeline(Protocol):
    """Protocol for components capable of evaluating a recovery decision."""

    @property
    def name(self) -> str: ...

    def evaluate(
        self,
        context: CustomerRecoveryContext,
        historical_cases: Sequence[HistoricalCase] | None = None,
        policy_context: RecoveryPolicyContext | None = None,
    ) -> Any: ...


class DeterministicBaselinePipeline:
    """Evaluates DecisionEngine without historical RAG evidence."""

    def __init__(
        self,
        decision_engine: DecisionEngine | None = None,
        name: str = "deterministic_baseline",
    ):
        self._engine = decision_engine or DecisionEngine()
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    def evaluate(
        self,
        context: CustomerRecoveryContext,
        historical_cases: Sequence[HistoricalCase] | None = None,
        policy_context: RecoveryPolicyContext | None = None,
    ) -> RecoveryDecision:
        return self._engine.evaluate(
            context=context,
            historical_cases=None,
            policy_context=policy_context,
        )


class DeterministicRAGPipeline:
    """Evaluates DecisionEngine with empirical historical RAG evidence."""

    def __init__(
        self,
        decision_engine: DecisionEngine | None = None,
        name: str = "deterministic_rag",
    ):
        self._engine = decision_engine or DecisionEngine()
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    def evaluate(
        self,
        context: CustomerRecoveryContext,
        historical_cases: Sequence[HistoricalCase] | None = None,
        policy_context: RecoveryPolicyContext | None = None,
    ) -> RecoveryDecision:
        return self._engine.evaluate(
            context=context,
            historical_cases=historical_cases,
            policy_context=policy_context,
        )


class AgentRAGPipeline:
    """Evaluates AgentOrchestrator with RAG evidence and policy validation."""

    def __init__(
        self,
        agent_orchestrator: AgentOrchestrator,
        name: str = "agent_rag",
    ):
        if not isinstance(agent_orchestrator, AgentOrchestrator):
            raise TypeError(
                f"Expected AgentOrchestrator, got {type(agent_orchestrator).__name__}"
            )
        self._orchestrator = agent_orchestrator
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    async def evaluate_async(
        self,
        context: CustomerRecoveryContext,
        historical_cases: Sequence[HistoricalCase] | None = None,
        policy_context: RecoveryPolicyContext | None = None,
    ) -> AgentDecisionResult:
        if policy_context is None:
            policy_context = resolve_policy_context(context=context)
        return await self._orchestrator.decide(
            context=context,
            policy_context=policy_context,
            historical_cases=historical_cases,
        )

    def evaluate(
        self,
        context: CustomerRecoveryContext,
        historical_cases: Sequence[HistoricalCase] | None = None,
        policy_context: RecoveryPolicyContext | None = None,
    ) -> AgentDecisionResult:
        return _run_async_or_sync(
            self.evaluate_async(
                context=context,
                historical_cases=historical_cases,
                policy_context=policy_context,
            )
        )


def _run_async_or_sync(coro: Any) -> Any:
    """Helper to execute an async coroutine from synchronous caller safely."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        # Running inside an existing event loop: execute in a dedicated worker thread
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(asyncio.run, coro)
            return future.result()
    return asyncio.run(coro)


def _create_error_eval_result(
    case: EvaluationCase,
    pipeline_name: str,
    elapsed_ms: float,
    error_msg: str,
) -> DecisionEvalResult:
    """Construct an error-tagged DecisionEvalResult adhering to failure-isolation contract."""
    dgt: DecisionGroundTruth = case.decision_ground_truth  # type: ignore[assignment]
    return DecisionEvalResult(
        query_id=case.query_id,
        pipeline_name=pipeline_name,
        predicted_action=RecoveryAction.NO_ACTION,
        expected_action=dgt.expected_action,
        acceptable_actions=dgt.acceptable_actions,
        prohibited_actions=dgt.prohibited_actions,
        expected_policy_ids=dgt.expected_policy_ids,
        is_exact_match=False,
        is_acceptable_match=False,
        confidence=0.0,
        policy_overridden=False,
        is_fallback=True,
        fallback_reason="Pipeline execution error",
        applied_policy_ids=(),
        violated_policy_ids=(),
        referenced_case_ids=(),
        key_factors=(),
        latency_ms=elapsed_ms,
        error=error_msg,
        metadata={"status": "error"},
    )


class DecisionEvaluator:
    """
    Decoupled benchmark runner for evaluating recovery decision pipelines against golden ground truth.
    """

    def __init__(
        self,
        evaluation_cases: Sequence[EvaluationCase],
        dataset_name: str = "retrieval_golden_dataset_v1",
    ):
        if not isinstance(evaluation_cases, (list, tuple)):
            raise TypeError(
                f"evaluation_cases must be a sequence of EvaluationCase instances, got {type(evaluation_cases).__name__}"
            )
        if not evaluation_cases:
            raise ValueError("evaluation_cases sequence cannot be empty.")

        for idx, case in enumerate(evaluation_cases):
            if not isinstance(case, EvaluationCase):
                raise TypeError(
                    f"Item at index {idx} in evaluation_cases must be EvaluationCase, got {type(case).__name__}"
                )
            if case.decision_ground_truth is None:
                raise ValueError(
                    f"EvaluationCase at index {idx} (query_id={case.query_id}) lacks decision_ground_truth."
                )

        if not isinstance(dataset_name, str) or not dataset_name.strip():
            raise ValueError(
                f"dataset_name must be a non-empty string, got: {dataset_name!r}"
            )

        self.evaluation_cases: tuple[EvaluationCase, ...] = tuple(evaluation_cases)
        self.dataset_name: str = dataset_name.strip()

    def evaluate_case(
        self,
        case: EvaluationCase,
        pipeline: Any,
        policy_context: RecoveryPolicyContext | None = None,
        historical_cases: Sequence[HistoricalCase] | None = None,
    ) -> DecisionEvalResult:
        """
        Evaluate a single EvaluationCase against a decision pipeline.

        Catches unexpected pipeline errors and malformed outputs, producing an error-tagged
        DecisionEvalResult without terminating full dataset evaluation.
        """
        if case.decision_ground_truth is None:
            raise ValueError(
                f"EvaluationCase {case.query_id} lacks decision_ground_truth."
            )

        dgt: DecisionGroundTruth = case.decision_ground_truth
        pipeline_name = (
            getattr(pipeline, "name", None)
            or getattr(pipeline, "pipeline_name", None)
            or type(pipeline).__name__
        )

        resolved_policy = policy_context or resolve_policy_context(case.context)
        resolved_hist_cases = (
            historical_cases
            if historical_cases is not None
            else extract_historical_cases_from_context(case.context)
        )

        start_time = time.perf_counter()

        try:
            # 1. Pipeline Execution
            raw_output: Any = None
            if hasattr(pipeline, "evaluate_async") and callable(
                pipeline.evaluate_async
            ):
                raw_output = _run_async_or_sync(
                    pipeline.evaluate_async(
                        context=case.context,
                        historical_cases=resolved_hist_cases,
                        policy_context=resolved_policy,
                    )
                )
            elif hasattr(pipeline, "evaluate") and callable(pipeline.evaluate):
                res = pipeline.evaluate(
                    context=case.context,
                    historical_cases=resolved_hist_cases,
                    policy_context=resolved_policy,
                )
                if inspect.iscoroutine(res):
                    raw_output = _run_async_or_sync(res)
                else:
                    raw_output = res
            elif hasattr(pipeline, "decide") and callable(pipeline.decide):
                res = pipeline.decide(
                    context=case.context,
                    policy_context=resolved_policy,
                    historical_cases=resolved_hist_cases,
                )
                if inspect.iscoroutine(res):
                    raw_output = _run_async_or_sync(res)
                else:
                    raw_output = res
            elif callable(pipeline):
                res = pipeline(
                    context=case.context,
                    historical_cases=resolved_hist_cases,
                    policy_context=resolved_policy,
                )
                if inspect.iscoroutine(res):
                    raw_output = _run_async_or_sync(res)
                else:
                    raw_output = res
            else:
                raise TypeError(
                    f"Pipeline of type '{type(pipeline).__name__}' does not implement 'evaluate', 'decide', or __call__."
                )

            # 2. Output Parsing & Normalization
            predicted_action: RecoveryAction
            confidence: float
            policy_overridden: bool = False
            is_fallback: bool = False
            fallback_reason: str | None = None
            applied_policy_ids: tuple[str, ...] = ()
            violated_policy_ids: tuple[str, ...] = ()
            referenced_case_ids: tuple[str, ...] = ()
            key_factors: tuple[str, ...] = ()

            if isinstance(raw_output, RecoveryDecision):
                predicted_action = raw_output.recommended_action
                confidence = raw_output.confidence
                basis = dict(raw_output.decision_basis)
                policy_overridden = bool(basis.get("policy_overridden", False))
                applied_policy_ids = tuple(
                    str(x) for x in basis.get("applied_policy_ids", ())
                )
                violated_policy_ids = tuple(
                    str(x) for x in basis.get("violated_policy_ids", ())
                )
                is_fallback = bool(basis.get("is_fallback", False))
                fallback_reason = basis.get("fallback_reason")
            elif isinstance(raw_output, AgentDecisionResult):
                predicted_action = raw_output.recommendation.recommended_action
                confidence = raw_output.recommendation.confidence
                meta = dict(raw_output.metadata)
                policy_overridden = bool(meta.get("policy_overridden", False))
                applied_policy_ids = tuple(
                    str(x) for x in meta.get("applied_policy_ids", ())
                )
                violated_policy_ids = tuple(
                    str(x) for x in meta.get("violated_policy_ids", ())
                )
                is_fallback = raw_output.is_fallback
                fallback_reason = raw_output.fallback_reason
                referenced_case_ids = tuple(
                    str(x) for x in raw_output.recommendation.referenced_case_ids
                )
                key_factors = tuple(
                    str(x) for x in raw_output.recommendation.key_factors
                )
            elif hasattr(raw_output, "recommended_action"):
                predicted_action = raw_output.recommended_action
                confidence = getattr(raw_output, "confidence", 1.0)
            else:
                raise TypeError(
                    f"Unsupported output type '{type(raw_output).__name__}' returned by pipeline {pipeline_name}."
                )

            # Validate predicted_action
            if not isinstance(predicted_action, RecoveryAction):
                try:
                    predicted_action = RecoveryAction(predicted_action)
                except Exception as act_exc:
                    raise ValueError(
                        f"Invalid predicted action '{predicted_action}': {act_exc}"
                    ) from act_exc

            # Validate confidence
            try:
                confidence = float(confidence)
            except (TypeError, ValueError) as conf_exc:
                raise ValueError(
                    f"Confidence must be a numeric float, got: {confidence!r}"
                ) from conf_exc

            if confidence < 0.0 or confidence > 1.0:
                raise ValueError(
                    f"Confidence {confidence} is outside valid range [0.0, 1.0]."
                )

            is_exact = bool(predicted_action == dgt.expected_action)
            is_acceptable = bool(predicted_action in dgt.acceptable_actions)
            elapsed_ms = max(0.0, (time.perf_counter() - start_time) * 1000.0)

            return DecisionEvalResult(
                query_id=case.query_id,
                pipeline_name=pipeline_name,
                predicted_action=predicted_action,
                expected_action=dgt.expected_action,
                acceptable_actions=dgt.acceptable_actions,
                prohibited_actions=dgt.prohibited_actions,
                expected_policy_ids=dgt.expected_policy_ids,
                is_exact_match=is_exact,
                is_acceptable_match=is_acceptable,
                confidence=confidence,
                policy_overridden=policy_overridden,
                is_fallback=is_fallback,
                fallback_reason=fallback_reason,
                applied_policy_ids=applied_policy_ids,
                violated_policy_ids=violated_policy_ids,
                referenced_case_ids=referenced_case_ids,
                key_factors=key_factors,
                latency_ms=elapsed_ms,
                error=None,
                metadata={"status": "success"},
            )
        except Exception as exc:  # noqa: BLE001
            elapsed_ms = max(0.0, (time.perf_counter() - start_time) * 1000.0)
            return _create_error_eval_result(
                case=case,
                pipeline_name=pipeline_name,
                elapsed_ms=elapsed_ms,
                error_msg=f"{type(exc).__name__}: {exc}",
            )

    def evaluate(
        self,
        pipeline: Any,
        pipeline_name: str | None = None,
        policy_context: RecoveryPolicyContext | None = None,
    ) -> DecisionBenchmarkReport:
        """
        Execute evaluation of the provided pipeline across all configured EvaluationCase scenarios.

        Args:
            pipeline: DecisionPipeline, DecisionEngine, AgentOrchestrator, or callable.
            pipeline_name: Optional override for pipeline name in reporting.
            policy_context: Optional explicit RecoveryPolicyContext.

        Returns:
            Immutable DecisionBenchmarkReport.
        """
        name = (
            pipeline_name
            or getattr(pipeline, "name", None)
            or getattr(pipeline, "pipeline_name", None)
            or type(pipeline).__name__
        )

        results: list[DecisionEvalResult] = []
        for case in self.evaluation_cases:
            res = self.evaluate_case(
                case=case,
                pipeline=pipeline,
                policy_context=policy_context,
            )
            results.append(res)

        aggregate_metrics = compute_aggregate_decision_metrics(results)

        return DecisionBenchmarkReport(
            pipeline_name=name,
            dataset_name=self.dataset_name,
            num_queries=len(results),
            results=tuple(results),
            aggregate_metrics=aggregate_metrics,
            evaluation_version="1.0",
        )
