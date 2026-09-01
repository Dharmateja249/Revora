"""
Revora Decision Evaluation Regression & Quality Gate Tests.

Comprehensive testing of decision quality thresholds, safety invariant enforcement,
comparative run regression detection, and CLI quality gate integration.
"""

from uuid import uuid4

import pytest

from app.decision_engine import RecoveryAction
from app.evaluation.decision_benchmark import run_decision_cli
from app.evaluation.decision_regression import (
    DecisionQualityGateResult,
    DecisionQualityThresholds,
    assert_decision_quality_gate,
    compare_decision_runs,
    evaluate_decision_quality_gate,
    format_quality_gate_terminal_summary,
)
from app.evaluation.schemas import (
    DecisionBenchmarkReport,
    DecisionEvalResult,
    EvaluationRegressionError,
)
from tests.fixtures.retrieval_golden_dataset import get_golden_evaluation_cases


def _make_benchmark_report(
    pipeline_name="deterministic_rag",
    metrics=None,
) -> DecisionBenchmarkReport:
    m = metrics or {
        "exact_match_rate": 0.88,
        "acceptable_match_rate": 0.96,
        "safety_violation_rate": 0.0,
        "policy_match_rate": 0.92,
        "policy_violation_rate": 0.02,
        "policy_override_rate": 0.02,
        "fallback_rate": 0.05,
        "mean_confidence": 0.85,
        "mean_latency_ms": 50.0,
    }
    dummy_result = DecisionEvalResult(
        query_id=uuid4(),
        pipeline_name=pipeline_name,
        predicted_action=RecoveryAction.RETRY_PAYMENT,
        expected_action=RecoveryAction.RETRY_PAYMENT,
        acceptable_actions=(RecoveryAction.RETRY_PAYMENT,),
        is_exact_match=True,
        is_acceptable_match=True,
        confidence=0.9,
    )
    return DecisionBenchmarkReport(
        pipeline_name=pipeline_name,
        dataset_name="golden_dataset_50",
        num_queries=1,
        results=(dummy_result,),
        aggregate_metrics=m,
    )


# =============================================================================
# Quality Gate Evaluation Tests
# =============================================================================


def test_quality_gate_all_thresholds_pass():
    report = _make_benchmark_report()
    thresholds = DecisionQualityThresholds(
        min_exact_match_rate=0.80,
        min_acceptable_match_rate=0.90,
        max_safety_violation_rate=0.0,
        max_policy_violation_rate=0.05,
        max_fallback_rate=0.10,
        max_mean_latency_ms=100.0,
    )

    res = evaluate_decision_quality_gate(report, thresholds=thresholds)

    assert isinstance(res, DecisionQualityGateResult)
    assert res.passed is True
    assert len(res.failed_checks) == 0
    assert "QUALITY GATE PASS" in res.summary_message


def test_quality_gate_exact_match_failure():
    report = _make_benchmark_report(metrics={"exact_match_rate": 0.75})
    thresholds = DecisionQualityThresholds(min_exact_match_rate=0.85)

    res = evaluate_decision_quality_gate(report, thresholds=thresholds)

    assert res.passed is False
    assert len(res.failed_checks) == 1
    failed = res.failed_checks[0]
    assert failed.metric_name == "exact_match_rate"
    assert failed.passed is False
    assert failed.actual_value == 0.75
    assert failed.threshold_value == 0.85


def test_quality_gate_acceptable_match_failure():
    report = _make_benchmark_report(metrics={"acceptable_match_rate": 0.80})
    thresholds = DecisionQualityThresholds(min_acceptable_match_rate=0.95)

    res = evaluate_decision_quality_gate(report, thresholds=thresholds)

    assert res.passed is False
    assert any(c.metric_name == "acceptable_match_rate" for c in res.failed_checks)


def test_quality_gate_safety_violation_failure():
    report = _make_benchmark_report(metrics={"safety_violation_rate": 0.04})
    thresholds = DecisionQualityThresholds(max_safety_violation_rate=0.0)

    res = evaluate_decision_quality_gate(report, thresholds=thresholds)

    assert res.passed is False
    assert any(c.metric_name == "safety_violation_rate" for c in res.failed_checks)
    v_check = next(
        c for c in res.failed_checks if c.metric_name == "safety_violation_rate"
    )
    assert "CRITICAL" in v_check.message


def test_quality_gate_policy_violation_failure():
    report = _make_benchmark_report(metrics={"policy_violation_rate": 0.12})
    thresholds = DecisionQualityThresholds(max_policy_violation_rate=0.05)

    res = evaluate_decision_quality_gate(report, thresholds=thresholds)

    assert res.passed is False
    assert any(c.metric_name == "policy_violation_rate" for c in res.failed_checks)


def test_quality_gate_fallback_rate_failure():
    report = _make_benchmark_report(metrics={"fallback_rate": 0.25})
    thresholds = DecisionQualityThresholds(max_fallback_rate=0.10)

    res = evaluate_decision_quality_gate(report, thresholds=thresholds)

    assert res.passed is False
    assert any(c.metric_name == "fallback_rate" for c in res.failed_checks)


def test_quality_gate_latency_failure():
    report = _make_benchmark_report(metrics={"mean_latency_ms": 250.0})
    thresholds = DecisionQualityThresholds(max_mean_latency_ms=100.0)

    res = evaluate_decision_quality_gate(report, thresholds=thresholds)

    assert res.passed is False
    assert any(c.metric_name == "mean_latency_ms" for c in res.failed_checks)


def test_quality_gate_multiple_failures():
    report = _make_benchmark_report(
        metrics={
            "exact_match_rate": 0.60,
            "safety_violation_rate": 0.08,
            "mean_latency_ms": 400.0,
        }
    )
    thresholds = DecisionQualityThresholds(
        min_exact_match_rate=0.85,
        max_safety_violation_rate=0.0,
        max_mean_latency_ms=100.0,
    )

    res = evaluate_decision_quality_gate(report, thresholds=thresholds)

    assert res.passed is False
    assert len(res.failed_checks) == 3
    failed_names = {c.metric_name for c in res.failed_checks}
    assert failed_names == {
        "exact_match_rate",
        "safety_violation_rate",
        "mean_latency_ms",
    }


def test_quality_gate_invalid_report_type_raises_type_error():
    with pytest.raises(TypeError, match="Expected DecisionBenchmarkReport"):
        evaluate_decision_quality_gate("invalid_report")  # type: ignore[arg-type]


# =============================================================================
# Comparative Regression Tests
# =============================================================================


def test_compare_decision_runs_no_regression():
    base = _make_benchmark_report(
        metrics={
            "exact_match_rate": 0.85,
            "safety_violation_rate": 0.0,
            "mean_latency_ms": 50.0,
        }
    )
    curr = _make_benchmark_report(
        metrics={
            "exact_match_rate": 0.88,
            "safety_violation_rate": 0.0,
            "mean_latency_ms": 45.0,
        }
    )

    comp = compare_decision_runs(current_report=curr, baseline_report=base)

    assert comp.passed is True
    assert len(comp.violations) == 0
    assert "exact_match_rate" in comp.improved_metrics
    assert "mean_latency_ms" in comp.improved_metrics


def test_compare_decision_runs_safety_violation_regression():
    base = _make_benchmark_report(metrics={"safety_violation_rate": 0.0})
    curr = _make_benchmark_report(metrics={"safety_violation_rate": 0.02})

    comp = compare_decision_runs(current_report=curr, baseline_report=base)

    assert comp.passed is False
    assert len(comp.violations) >= 1
    assert any("safety_violation_rate" in v for v in comp.violations)


def test_compare_decision_runs_quality_drop_exceeding_tolerance():
    base = _make_benchmark_report(metrics={"exact_match_rate": 0.90})
    curr = _make_benchmark_report(metrics={"exact_match_rate": 0.80})
    thresholds = DecisionQualityThresholds(max_allowed_quality_drop=0.05)

    comp = compare_decision_runs(
        current_report=curr,
        baseline_report=base,
        thresholds=thresholds,
    )

    assert comp.passed is False
    assert any("exact_match_rate regressed" in v for v in comp.violations)


# =============================================================================
# Assertion & Terminal Formatting Tests
# =============================================================================


def test_assert_decision_quality_gate_raises_on_failure():
    report = _make_benchmark_report(metrics={"exact_match_rate": 0.50})
    thresholds = DecisionQualityThresholds(min_exact_match_rate=0.85)

    with pytest.raises(EvaluationRegressionError, match="QUALITY GATE FAIL"):
        assert_decision_quality_gate(report, thresholds=thresholds)


def test_assert_decision_quality_gate_passes_cleanly():
    report = _make_benchmark_report(metrics={"exact_match_rate": 0.90})
    thresholds = DecisionQualityThresholds(min_exact_match_rate=0.85)
    assert_decision_quality_gate(report, thresholds=thresholds)


def test_format_quality_gate_terminal_summary():
    report = _make_benchmark_report()
    thresholds = DecisionQualityThresholds(
        min_exact_match_rate=0.80,
        max_safety_violation_rate=0.0,
    )
    res = evaluate_decision_quality_gate(report, thresholds=thresholds)
    text = format_quality_gate_terminal_summary(res)

    assert "Benchmark: deterministic_rag" in text
    assert "exact_match_rate" in text
    assert "[PASS]" in text
    assert "QUALITY GATE: PASS" in text


# =============================================================================
# CLI Quality Gate Tests
# =============================================================================


def test_run_decision_cli_quality_gate_success(capsys):
    cases = get_golden_evaluation_cases()[:3]
    exit_code = run_decision_cli(
        args=[
            "-p",
            "deterministic_baseline",
            "--assert-quality-gate",
            "--max-safety-violation",
            "0.0",
            "--no-save",
        ],
        evaluation_cases=cases,
    )

    assert exit_code == 0
    captured = capsys.readouterr()
    assert "QUALITY GATE: PASS" in captured.out


def test_run_decision_cli_quality_gate_failure(capsys):
    cases = get_golden_evaluation_cases()[:3]
    # Unachievable 100% exact match requirement for test failure validation
    exit_code = run_decision_cli(
        args=[
            "-p",
            "deterministic_baseline",
            "--min-exact-match",
            "0.999",
            "--no-save",
        ],
        evaluation_cases=cases,
    )

    assert exit_code == 1
    captured = capsys.readouterr()
    assert "QUALITY GATE: FAIL" in captured.out
