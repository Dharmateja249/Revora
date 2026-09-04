"""
Revora Decision Evaluation Metrics Tests.

Comprehensive testing of mathematical decision metrics:
- exact_match_rate
- acceptable_match_rate
- safety_violation_rate
- mean_confidence
- mean_latency_ms
- fallback_rate
- policy_match_rate
- policy_violation_rate
- policy_override_rate
- compute_aggregate_decision_metrics
- Edge cases, empty inputs, error handling, and type safety
"""

from uuid import uuid4

import pytest
from app.decision_engine import RecoveryAction
from app.evaluation.decision_metrics import (
    acceptable_match_rate,
    compute_aggregate_decision_metrics,
    exact_match_rate,
    fallback_rate,
    mean_confidence,
    mean_latency_ms,
    policy_match_rate,
    policy_override_rate,
    policy_violation_rate,
    safety_violation_rate,
)
from app.evaluation.schemas import DecisionEvalResult


def _make_result(
    predicted_action: RecoveryAction = RecoveryAction.RETRY_PAYMENT,
    expected_action: RecoveryAction = RecoveryAction.RETRY_PAYMENT,
    acceptable_actions: tuple[RecoveryAction, ...] = (RecoveryAction.RETRY_PAYMENT,),
    prohibited_actions: tuple[RecoveryAction, ...] = (),
    expected_policy_ids: tuple[str, ...] = (),
    applied_policy_ids: tuple[str, ...] = (),
    violated_policy_ids: tuple[str, ...] = (),
    is_exact_match: bool = True,
    is_acceptable_match: bool = True,
    confidence: float = 0.9,
    policy_overridden: bool = False,
    is_fallback: bool = False,
    latency_ms: float = 50.0,
    error: str | None = None,
) -> DecisionEvalResult:
    return DecisionEvalResult(
        query_id=uuid4(),
        pipeline_name="test_pipeline",
        predicted_action=predicted_action,
        expected_action=expected_action,
        acceptable_actions=acceptable_actions,
        prohibited_actions=prohibited_actions,
        expected_policy_ids=expected_policy_ids,
        applied_policy_ids=applied_policy_ids,
        violated_policy_ids=violated_policy_ids,
        is_exact_match=is_exact_match,
        is_acceptable_match=is_acceptable_match,
        confidence=confidence,
        policy_overridden=policy_overridden,
        is_fallback=is_fallback,
        latency_ms=latency_ms,
        error=error,
    )


# =============================================================================
# Match & Accuracy Metrics Tests
# =============================================================================


def test_exact_match_rate_all_correct():
    results = [
        _make_result(is_exact_match=True),
        _make_result(is_exact_match=True),
        _make_result(is_exact_match=True),
    ]
    assert exact_match_rate(results) == 1.0


def test_exact_match_rate_partial_and_errors():
    results = [
        _make_result(is_exact_match=True),
        _make_result(is_exact_match=False),
        _make_result(is_exact_match=True, error="Execution error"),
        _make_result(is_exact_match=False),
    ]
    # 1 valid exact match out of 4 queries = 0.25
    assert exact_match_rate(results) == 0.25


def test_acceptable_match_rate_includes_alternatives():
    results = [
        _make_result(is_exact_match=True, is_acceptable_match=True),
        _make_result(is_exact_match=False, is_acceptable_match=True),
        _make_result(is_exact_match=False, is_acceptable_match=False),
        _make_result(
            is_exact_match=False,
            is_acceptable_match=True,
            error="Connection timeout",
        ),
    ]
    # 2 valid acceptable matches out of 4 queries = 0.5
    assert acceptable_match_rate(results) == 0.5


def test_safety_violation_rate():
    results = [
        _make_result(
            predicted_action=RecoveryAction.RETRY_PAYMENT,
            prohibited_actions=(RecoveryAction.RETRY_PAYMENT,),
        ),
        _make_result(
            predicted_action=RecoveryAction.PAYMENT_LINK,
            prohibited_actions=(RecoveryAction.RETRY_PAYMENT,),
        ),
        _make_result(
            predicted_action=RecoveryAction.CHANGE_PAYMENT_METHOD,
            prohibited_actions=(),
        ),
        _make_result(
            predicted_action=RecoveryAction.WAIT_AND_RETRY,
            prohibited_actions=(RecoveryAction.WAIT_AND_RETRY,),
        ),
    ]
    # 2 violations out of 4 queries = 0.5
    assert safety_violation_rate(results) == 0.5


# =============================================================================
# Confidence & Latency Metrics Tests
# =============================================================================


def test_mean_confidence():
    results = [
        _make_result(confidence=0.8),
        _make_result(confidence=1.0),
        _make_result(confidence=0.6),
        _make_result(confidence=0.9, error="Ignored due to error"),
    ]
    # mean of [0.8, 1.0, 0.6] = 2.4 / 3 = 0.8
    assert pytest.approx(mean_confidence(results), rel=1e-5) == 0.8


def test_mean_latency_ms():
    results = [
        _make_result(latency_ms=10.0),
        _make_result(latency_ms=20.0),
        _make_result(latency_ms=30.0),
    ]
    assert pytest.approx(mean_latency_ms(results), rel=1e-5) == 20.0


# =============================================================================
# Policy & Fallback Metrics Tests
# =============================================================================


def test_fallback_rate():
    results = [
        _make_result(is_fallback=True),
        _make_result(is_fallback=False),
        _make_result(is_fallback=True),
        _make_result(is_fallback=False),
    ]
    assert fallback_rate(results) == 0.5


def test_policy_match_rate():
    results = [
        # 1. Expected policy applied -> match
        _make_result(
            expected_policy_ids=("POL_A",),
            applied_policy_ids=("POL_A", "POL_B"),
            violated_policy_ids=(),
        ),
        # 2. Expected policy violated -> mismatch
        _make_result(
            expected_policy_ids=("POL_A",),
            applied_policy_ids=("POL_A",),
            violated_policy_ids=("POL_A",),
        ),
        # 3. Expected policy missing -> mismatch
        _make_result(
            expected_policy_ids=("POL_A",),
            applied_policy_ids=(),
            violated_policy_ids=(),
        ),
        # 4. No expected policy and no violations -> match
        _make_result(
            expected_policy_ids=(),
            applied_policy_ids=(),
            violated_policy_ids=(),
        ),
        # 5. No expected policy but unprompted violation -> mismatch
        _make_result(
            expected_policy_ids=(),
            applied_policy_ids=(),
            violated_policy_ids=("POL_UNWANTED",),
        ),
    ]
    # Matches: #1 and #4 (2 out of 5 = 0.4)
    assert pytest.approx(policy_match_rate(results), rel=1e-5) == 0.4


def test_policy_violation_and_override_rate():
    results = [
        _make_result(policy_overridden=True, violated_policy_ids=("POL_1",)),
        _make_result(policy_overridden=False, violated_policy_ids=()),
        _make_result(policy_overridden=False, violated_policy_ids=("POL_2",)),
        _make_result(policy_overridden=False, violated_policy_ids=()),
    ]
    assert policy_override_rate(results) == 0.25
    assert policy_violation_rate(results) == 0.5


# =============================================================================
# Aggregate Metrics & Empty Input Tests
# =============================================================================


def test_compute_aggregate_decision_metrics():
    results = [
        _make_result(
            is_exact_match=True,
            is_acceptable_match=True,
            confidence=0.9,
            latency_ms=100.0,
            is_fallback=False,
        ),
        _make_result(
            is_exact_match=False,
            is_acceptable_match=True,
            confidence=0.7,
            latency_ms=50.0,
            is_fallback=True,
            policy_overridden=True,
        ),
    ]
    metrics = compute_aggregate_decision_metrics(results)

    assert metrics["exact_match_rate"] == 0.5
    assert metrics["acceptable_match_rate"] == 1.0
    assert metrics["safety_violation_rate"] == 0.0
    assert pytest.approx(metrics["mean_confidence"]) == 0.8
    assert pytest.approx(metrics["mean_latency_ms"]) == 75.0
    assert metrics["fallback_rate"] == 0.5
    assert metrics["policy_override_rate"] == 0.5


def test_empty_results_safe_defaults():
    empty: list[DecisionEvalResult] = []

    assert exact_match_rate(empty) == 0.0
    assert acceptable_match_rate(empty) == 0.0
    assert safety_violation_rate(empty) == 0.0
    assert mean_confidence(empty) == 0.0
    assert mean_latency_ms(empty) == 0.0
    assert fallback_rate(empty) == 0.0
    assert policy_match_rate(empty) == 0.0
    assert policy_violation_rate(empty) == 0.0
    assert policy_override_rate(empty) == 0.0

    agg = compute_aggregate_decision_metrics(empty)
    assert all(val == 0.0 for val in agg.values())


def test_invalid_results_input_type_rejected():
    with pytest.raises(TypeError, match="results must be a sequence"):
        exact_match_rate("not_a_sequence")  # type: ignore[arg-type]

    with pytest.raises(TypeError, match="must be DecisionEvalResult"):
        exact_match_rate([1, 2, 3])  # type: ignore[list-item]
