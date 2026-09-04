"""
Revora Decision Evaluator Tests.

Comprehensive testing of DecisionEvaluator, pipeline adapters, match semantics,
error resilience, report generation, and full 50-case golden dataset benchmarking.
"""

from datetime import datetime, timezone
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest
from app.agent.orchestrator import AgentOrchestrator
from app.agent.schemas import (
    AgentDecisionResult,
    LLMRecoveryRecommendation,
)
from app.context import (
    CustomerContext,
    CustomerRecoveryContext,
    CustomerRecoveryStatsContext,
    PaymentContext,
    RecoveryOpportunityContext,
)
from app.decision_engine import DecisionEngine, RecoveryAction, RecoveryDecision
from app.evaluation.decision_evaluator import (
    AgentRAGPipeline,
    DecisionEvaluator,
    DeterministicBaselinePipeline,
    DeterministicRAGPipeline,
    extract_historical_cases_from_context,
)
from app.evaluation.schemas import (
    DecisionBenchmarkReport,
    DecisionGroundTruth,
    EvaluationCase,
)
from pydantic import ValidationError

from tests.fixtures.retrieval_golden_dataset import (
    get_golden_evaluation_cases,
)


def _make_dummy_case(
    expected_action: RecoveryAction = RecoveryAction.RETRY_PAYMENT,
    acceptable_actions: tuple[RecoveryAction, ...] = (RecoveryAction.RETRY_PAYMENT,),
    prohibited_actions: tuple[RecoveryAction, ...] = (),
    expected_policy_ids: tuple[str, ...] = (),
    failure_reason: str = "bank_timeout",
    amount: float = 500.0,
) -> EvaluationCase:
    cust_id = uuid4()
    pay_id = uuid4()
    t_now = datetime.now(timezone.utc)

    context = CustomerRecoveryContext(
        customer=CustomerContext(
            customer_id=cust_id,
            external_customer_id="cust_ext_001",
            risk_tier="low",
            created_at=t_now,
        ),
        current_payment=PaymentContext(
            payment_id=pay_id,
            external_payment_id="pay_ext_001",
            amount=amount,
            currency="INR",
            payment_method="upi",
            status="failed",
            failure_reason=failure_reason,
            created_at=t_now,
        ),
        current_opportunity=RecoveryOpportunityContext(
            opportunity_id=uuid4(),
            status="failed",
            revenue_at_risk=amount,
        ),
        current_payment_attempts=[],
        historical_payments=[],
        recovery_statistics=CustomerRecoveryStatsContext(
            total_recovery_opportunities=0,
            recovered_opportunities=0,
            failed_opportunities=0,
            recovery_rate=0.0,
            previously_successful_actions=[],
            previously_failed_actions=[],
            total_amount_recovered=0.0,
        ),
        retrieved_at=t_now,
    )

    decision_gt = DecisionGroundTruth(
        expected_action=expected_action,
        acceptable_actions=acceptable_actions,
        prohibited_actions=prohibited_actions,
        expected_policy_ids=expected_policy_ids,
        rationale="Test ground truth",
    )

    return EvaluationCase(
        query_id=uuid4(),
        context=context,
        ground_truth=(),
        decision_ground_truth=decision_gt,
        description="Test case",
    )


# =============================================================================
# Match Semantics & Case Evaluation Tests
# =============================================================================


def test_evaluator_exact_match():
    case = _make_dummy_case(
        expected_action=RecoveryAction.RETRY_PAYMENT,
        acceptable_actions=(RecoveryAction.RETRY_PAYMENT,),
    )

    class MockPipeline:
        name = "mock_exact"

        def evaluate(self, context, historical_cases=None, policy_context=None):
            return RecoveryDecision(
                recommended_action=RecoveryAction.RETRY_PAYMENT,
                reason="Precedent retry succeeded",
                confidence=0.95,
                decision_basis={"reason": "precedent"},
            )

    evaluator = DecisionEvaluator([case])
    result = evaluator.evaluate_case(case, MockPipeline())

    assert result.predicted_action == RecoveryAction.RETRY_PAYMENT
    assert result.is_exact_match is True
    assert result.is_acceptable_match is True
    assert result.confidence == 0.95
    assert result.error is None
    assert result.latency_ms >= 0.0


def test_evaluator_acceptable_alternative_match():
    case = _make_dummy_case(
        expected_action=RecoveryAction.PAYMENT_LINK,
        acceptable_actions=(
            RecoveryAction.PAYMENT_LINK,
            RecoveryAction.CHANGE_PAYMENT_METHOD,
        ),
        prohibited_actions=(RecoveryAction.RETRY_PAYMENT,),
    )

    class MockPipeline:
        name = "mock_alt"

        def evaluate(self, context, historical_cases=None, policy_context=None):
            return RecoveryDecision(
                recommended_action=RecoveryAction.CHANGE_PAYMENT_METHOD,
                reason="Change payment method acceptable",
                confidence=0.85,
                decision_basis={},
            )

    evaluator = DecisionEvaluator([case])
    result = evaluator.evaluate_case(case, MockPipeline())

    assert result.predicted_action == RecoveryAction.CHANGE_PAYMENT_METHOD
    assert result.is_exact_match is False
    assert result.is_acceptable_match is True
    assert result.error is None


def test_evaluator_prohibited_action_safety_violation():
    case = _make_dummy_case(
        expected_action=RecoveryAction.PAYMENT_LINK,
        acceptable_actions=(RecoveryAction.PAYMENT_LINK,),
        prohibited_actions=(RecoveryAction.RETRY_PAYMENT,),
    )

    class MockPipeline:
        name = "mock_prohibited"

        def evaluate(self, context, historical_cases=None, policy_context=None):
            return RecoveryDecision(
                recommended_action=RecoveryAction.RETRY_PAYMENT,
                reason="Unsafe retry attempt",
                confidence=0.99,
                decision_basis={},
            )

    evaluator = DecisionEvaluator([case])
    result = evaluator.evaluate_case(case, MockPipeline())

    assert result.predicted_action == RecoveryAction.RETRY_PAYMENT
    assert result.is_exact_match is False
    assert result.is_acceptable_match is False
    assert RecoveryAction.RETRY_PAYMENT in result.prohibited_actions


def test_evaluator_wrong_non_prohibited_action():
    case = _make_dummy_case(
        expected_action=RecoveryAction.RETRY_PAYMENT,
        acceptable_actions=(RecoveryAction.RETRY_PAYMENT,),
        prohibited_actions=(),
    )

    class MockPipeline:
        name = "mock_wrong"

        def evaluate(self, context, historical_cases=None, policy_context=None):
            return RecoveryDecision(
                recommended_action=RecoveryAction.NO_ACTION,
                reason="No action chosen",
                confidence=0.5,
                decision_basis={},
            )

    evaluator = DecisionEvaluator([case])
    result = evaluator.evaluate_case(case, MockPipeline())

    assert result.predicted_action == RecoveryAction.NO_ACTION
    assert result.is_exact_match is False
    assert result.is_acceptable_match is False


# =============================================================================
# Telemetry, Fallback & Policy Tests
# =============================================================================


def test_evaluator_agent_decision_result_telemetry():
    case = _make_dummy_case(
        expected_action=RecoveryAction.PAYMENT_LINK,
        expected_policy_ids=("RZP_CUSTOMER_AUTH_2FA_REQUIRED",),
    )

    class MockAgentPipeline:
        name = "mock_agent"

        def evaluate(self, context, historical_cases=None, policy_context=None):
            return AgentDecisionResult(
                recommendation=LLMRecoveryRecommendation(
                    recommended_action=RecoveryAction.PAYMENT_LINK,
                    confidence=0.92,
                    reasoning="2FA failure precedent",
                    key_factors=("otp_expired", "2fa_required"),
                    referenced_case_ids=("case_123", "case_456"),
                ),
                agent_used=True,
                provider="mock",
                model_name="mock-model",
                is_fallback=False,
                fallback_reason=None,
                latency_ms=120.0,
                metadata={
                    "applied_policy_ids": ["RZP_CUSTOMER_AUTH_2FA_REQUIRED"],
                    "violated_policy_ids": [],
                    "policy_overridden": False,
                },
            )

    evaluator = DecisionEvaluator([case])
    result = evaluator.evaluate_case(case, MockAgentPipeline())

    assert result.predicted_action == RecoveryAction.PAYMENT_LINK
    assert result.is_exact_match is True
    assert result.confidence == 0.92
    assert result.is_fallback is False
    assert result.applied_policy_ids == ("RZP_CUSTOMER_AUTH_2FA_REQUIRED",)
    assert result.referenced_case_ids == ("case_123", "case_456")
    assert result.key_factors == ("otp_expired", "2fa_required")


def test_evaluator_agent_fallback_capture():
    case = _make_dummy_case(expected_action=RecoveryAction.RETRY_PAYMENT)

    class MockFallbackPipeline:
        name = "mock_fallback"

        def evaluate(self, context, historical_cases=None, policy_context=None):
            return AgentDecisionResult(
                recommendation=LLMRecoveryRecommendation(
                    recommended_action=RecoveryAction.RETRY_PAYMENT,
                    confidence=0.7,
                    reasoning="Deterministic fallback applied",
                    key_factors=(),
                    referenced_case_ids=(),
                ),
                agent_used=False,
                provider="mock",
                model_name="mock-model",
                is_fallback=True,
                fallback_reason="LLM timeout",
                latency_ms=500.0,
                metadata={},
            )

    evaluator = DecisionEvaluator([case])
    result = evaluator.evaluate_case(case, MockFallbackPipeline())

    assert result.is_fallback is True
    assert result.fallback_reason == "LLM timeout"


# =============================================================================
# Failure Isolation & Dataset Evaluation Tests
# =============================================================================


def test_evaluator_failure_isolation():
    case1 = _make_dummy_case()
    case2 = _make_dummy_case()

    call_count = 0

    class BuggyPipeline:
        name = "buggy"

        def evaluate(self, context, historical_cases=None, policy_context=None):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("Network outage during evaluation")
            return RecoveryDecision(
                recommended_action=RecoveryAction.RETRY_PAYMENT,
                reason="Retry succeeded",
                confidence=0.9,
                decision_basis={},
            )

    evaluator = DecisionEvaluator([case1, case2])
    report = evaluator.evaluate(BuggyPipeline())

    assert report.num_queries == 2
    assert len(report.results) == 2

    # Case 1 failed safely with error captured
    res1 = report.results[0]
    assert res1.error is not None
    assert "RuntimeError: Network outage" in res1.error
    assert res1.is_exact_match is False
    assert res1.is_acceptable_match is False

    # Case 2 succeeded normally
    res2 = report.results[1]
    assert res2.error is None
    assert res2.is_exact_match is True


def test_decision_benchmark_report_immutability_and_serialization():
    case = _make_dummy_case()
    pipeline = DeterministicBaselinePipeline()
    evaluator = DecisionEvaluator([case], dataset_name="unit_test_dataset")

    report = evaluator.evaluate(pipeline)

    assert isinstance(report, DecisionBenchmarkReport)
    assert report.pipeline_name == "deterministic_baseline"
    assert report.dataset_name == "unit_test_dataset"
    assert report.num_queries == 1
    assert "exact_match_rate" in report.aggregate_metrics

    # Immutability
    with pytest.raises(ValidationError):
        report.pipeline_name = "modified_name"

    # Serialization round-trip
    dumped = report.model_dump(mode="json")
    restored = DecisionBenchmarkReport.model_validate(dumped)
    assert restored.pipeline_name == report.pipeline_name
    assert restored.num_queries == report.num_queries
    assert restored.aggregate_metrics == report.aggregate_metrics


def test_evaluator_requires_non_empty_cases_with_ground_truth():
    with pytest.raises(ValueError, match="cannot be empty"):
        DecisionEvaluator([])

    context = _make_dummy_case().context
    case_no_gt = EvaluationCase(
        query_id=uuid4(),
        context=context,
        ground_truth=(),
        decision_ground_truth=None,
    )
    with pytest.raises(ValueError, match="lacks decision_ground_truth"):
        DecisionEvaluator([case_no_gt])


def test_agent_rag_pipeline_execution():
    mock_orchestrator = AsyncMock(spec=AgentOrchestrator)
    mock_orchestrator.decide.return_value = AgentDecisionResult(
        recommendation=LLMRecoveryRecommendation(
            recommended_action=RecoveryAction.RETRY_PAYMENT,
            confidence=0.91,
            reasoning="Precedent retry match",
            key_factors=("bank_timeout",),
            referenced_case_ids=("case_1",),
        ),
        agent_used=True,
        provider="mock",
        model_name="mock-model",
        is_fallback=False,
        latency_ms=85.0,
        metadata={},
    )

    pipeline = AgentRAGPipeline(agent_orchestrator=mock_orchestrator)
    assert pipeline.name == "agent_rag"

    case = _make_dummy_case(expected_action=RecoveryAction.RETRY_PAYMENT)
    evaluator = DecisionEvaluator([case])
    report = evaluator.evaluate(pipeline)

    assert report.num_queries == 1
    assert report.results[0].predicted_action == RecoveryAction.RETRY_PAYMENT
    assert report.results[0].is_exact_match is True


# =============================================================================
# Full 50-Case Golden Dataset Evaluation Test
# =============================================================================


def test_deterministic_baseline_evaluates_all_50_golden_cases():
    cases = get_golden_evaluation_cases()
    assert len(cases) == 50

    evaluator = DecisionEvaluator(cases, dataset_name="retrieval_golden_dataset_50")
    engine = DecisionEngine()
    pipeline = DeterministicBaselinePipeline(decision_engine=engine)

    report = evaluator.evaluate(pipeline)

    assert report.num_queries == 50
    assert len(report.results) == 50
    assert report.pipeline_name == "deterministic_baseline"

    # Structural correctness checks across all 50 evaluation results
    for res in report.results:
        assert res.error is None
        assert isinstance(res.predicted_action, RecoveryAction)
        assert 0.0 <= res.confidence <= 1.0
        assert res.latency_ms >= 0.0
        assert isinstance(res.is_exact_match, bool)
        assert isinstance(res.is_acceptable_match, bool)

    # Aggregate metrics existence
    metrics = report.aggregate_metrics
    assert 0.0 <= metrics["exact_match_rate"] <= 1.0
    assert 0.0 <= metrics["acceptable_match_rate"] <= 1.0
    assert 0.0 <= metrics["safety_violation_rate"] <= 1.0
    assert 0.0 <= metrics["mean_confidence"] <= 1.0
    assert metrics["mean_latency_ms"] >= 0.0


def test_deterministic_rag_evaluates_all_50_golden_cases():
    cases = get_golden_evaluation_cases()
    evaluator = DecisionEvaluator(cases, dataset_name="retrieval_golden_dataset_50")
    pipeline = DeterministicRAGPipeline()

    report = evaluator.evaluate(pipeline)

    assert report.num_queries == 50
    assert len(report.results) == 50
    assert report.pipeline_name == "deterministic_rag"
    assert all(r.error is None for r in report.results)


def test_extract_historical_cases_from_context():
    case = get_golden_evaluation_cases()[0]
    hist_cases = extract_historical_cases_from_context(case.context)
    assert isinstance(hist_cases, list)
    if case.context.historical_payments:
        assert len(hist_cases) == len(case.context.historical_payments)
        assert (
            hist_cases[0].payment_id == case.context.historical_payments[0].payment_id
        )
