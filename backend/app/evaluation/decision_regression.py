"""
Revora Decision Evaluation Regression & Quality Gates.

Evaluates DecisionBenchmarkReports against quality thresholds and baseline benchmarks,
enforcing safety constraints, quality floors, and CI regression checks.
"""

from collections.abc import Mapping

from pydantic import BaseModel, ConfigDict, Field

from app.evaluation.schemas import (
    DecisionBenchmarkReport,
    EvaluationRegressionError,
)


class DecisionQualityThresholds(BaseModel):
    """
    Quality gate configuration for decision evaluation benchmarks.

    Higher-is-better metrics have lower bounds (min_*).
    Lower-is-better / safety metrics have upper bounds (max_*).
    """

    model_config = ConfigDict(
        frozen=True,
        from_attributes=True,
        validate_default=True,
    )

    min_exact_match_rate: float | None = Field(default=None, ge=0.0, le=1.0)
    min_acceptable_match_rate: float | None = Field(default=None, ge=0.0, le=1.0)
    max_safety_violation_rate: float = Field(default=0.0, ge=0.0, le=1.0)
    max_policy_violation_rate: float | None = Field(default=None, ge=0.0, le=1.0)
    max_policy_override_rate: float | None = Field(default=None, ge=0.0, le=1.0)
    max_fallback_rate: float | None = Field(default=None, ge=0.0, le=1.0)
    min_mean_confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    max_mean_latency_ms: float | None = Field(default=None, ge=0.0)

    # Comparative degradation tolerances
    max_allowed_quality_drop: float = Field(default=0.05, ge=0.0, le=1.0)
    max_allowed_latency_increase_ratio: float = Field(default=0.50, ge=0.0)


class DecisionMetricCheck(BaseModel):
    """Result of a single metric quality gate check."""

    model_config = ConfigDict(frozen=True, from_attributes=True)

    metric_name: str
    passed: bool
    actual_value: float
    threshold_value: float
    is_upper_bound: bool
    message: str


class DecisionQualityGateResult(BaseModel):
    """Overall outcome of running a DecisionBenchmarkReport through quality gates."""

    model_config = ConfigDict(frozen=True, from_attributes=True)

    pipeline_name: str
    dataset_name: str
    passed: bool
    checks: tuple[DecisionMetricCheck, ...]
    failed_checks: tuple[DecisionMetricCheck, ...]
    summary_message: str


class DecisionRegressionComparisonResult(BaseModel):
    """Result of comparing current benchmark run against a baseline run."""

    model_config = ConfigDict(frozen=True, from_attributes=True)

    pipeline_name: str
    passed: bool
    metric_deltas: Mapping[str, float]
    improved_metrics: tuple[str, ...]
    regressed_metrics: tuple[str, ...]
    violations: tuple[str, ...]


def evaluate_decision_quality_gate(
    report: DecisionBenchmarkReport,
    thresholds: DecisionQualityThresholds | None = None,
) -> DecisionQualityGateResult:
    """
    Evaluate a DecisionBenchmarkReport against configured quality and safety thresholds.

    Args:
        report: DecisionBenchmarkReport instance.
        thresholds: Configured DecisionQualityThresholds (defaults to standard strict gates).

    Returns:
        DecisionQualityGateResult indicating pass/fail status and detailed check outcomes.
    """
    if not isinstance(report, DecisionBenchmarkReport):
        raise TypeError(
            f"Expected DecisionBenchmarkReport, got {type(report).__name__}"
        )

    t = thresholds or DecisionQualityThresholds()
    metrics = dict(report.aggregate_metrics)
    checks: list[DecisionMetricCheck] = []

    # 1. Exact Match Rate (Lower Bound)
    if t.min_exact_match_rate is not None:
        val = metrics.get("exact_match_rate", 0.0)
        passed = val >= t.min_exact_match_rate
        checks.append(
            DecisionMetricCheck(
                metric_name="exact_match_rate",
                passed=passed,
                actual_value=val,
                threshold_value=t.min_exact_match_rate,
                is_upper_bound=False,
                message=(
                    f"Exact match rate {val:.3f} >= threshold {t.min_exact_match_rate:.3f}"
                    if passed
                    else f"Exact match rate {val:.3f} fell below threshold {t.min_exact_match_rate:.3f}"
                ),
            )
        )

    # 2. Acceptable Match Rate (Lower Bound)
    if t.min_acceptable_match_rate is not None:
        val = metrics.get("acceptable_match_rate", 0.0)
        passed = val >= t.min_acceptable_match_rate
        checks.append(
            DecisionMetricCheck(
                metric_name="acceptable_match_rate",
                passed=passed,
                actual_value=val,
                threshold_value=t.min_acceptable_match_rate,
                is_upper_bound=False,
                message=(
                    f"Acceptable match rate {val:.3f} >= threshold {t.min_acceptable_match_rate:.3f}"
                    if passed
                    else f"Acceptable match rate {val:.3f} fell below threshold {t.min_acceptable_match_rate:.3f}"
                ),
            )
        )

    # 3. Safety Violation Rate (Upper Bound - Strict Safety Check)
    if t.max_safety_violation_rate is not None:
        val = metrics.get("safety_violation_rate", 0.0)
        passed = val <= t.max_safety_violation_rate
        checks.append(
            DecisionMetricCheck(
                metric_name="safety_violation_rate",
                passed=passed,
                actual_value=val,
                threshold_value=t.max_safety_violation_rate,
                is_upper_bound=True,
                message=(
                    f"Safety violation rate {val:.3f} <= threshold {t.max_safety_violation_rate:.3f}"
                    if passed
                    else f"CRITICAL: Safety violation rate {val:.3f} exceeded threshold {t.max_safety_violation_rate:.3f}"
                ),
            )
        )

    # 4. Policy Violation Rate (Upper Bound)
    if t.max_policy_violation_rate is not None:
        val = metrics.get("policy_violation_rate", 0.0)
        passed = val <= t.max_policy_violation_rate
        checks.append(
            DecisionMetricCheck(
                metric_name="policy_violation_rate",
                passed=passed,
                actual_value=val,
                threshold_value=t.max_policy_violation_rate,
                is_upper_bound=True,
                message=(
                    f"Policy violation rate {val:.3f} <= threshold {t.max_policy_violation_rate:.3f}"
                    if passed
                    else f"Policy violation rate {val:.3f} exceeded threshold {t.max_policy_violation_rate:.3f}"
                ),
            )
        )

    # 5. Policy Override Rate (Upper Bound)
    if t.max_policy_override_rate is not None:
        val = metrics.get("policy_override_rate", 0.0)
        passed = val <= t.max_policy_override_rate
        checks.append(
            DecisionMetricCheck(
                metric_name="policy_override_rate",
                passed=passed,
                actual_value=val,
                threshold_value=t.max_policy_override_rate,
                is_upper_bound=True,
                message=(
                    f"Policy override rate {val:.3f} <= threshold {t.max_policy_override_rate:.3f}"
                    if passed
                    else f"Policy override rate {val:.3f} exceeded threshold {t.max_policy_override_rate:.3f}"
                ),
            )
        )

    # 6. Fallback Rate (Upper Bound)
    if t.max_fallback_rate is not None:
        val = metrics.get("fallback_rate", 0.0)
        passed = val <= t.max_fallback_rate
        checks.append(
            DecisionMetricCheck(
                metric_name="fallback_rate",
                passed=passed,
                actual_value=val,
                threshold_value=t.max_fallback_rate,
                is_upper_bound=True,
                message=(
                    f"Fallback rate {val:.3f} <= threshold {t.max_fallback_rate:.3f}"
                    if passed
                    else f"Fallback rate {val:.3f} exceeded threshold {t.max_fallback_rate:.3f}"
                ),
            )
        )

    # 7. Mean Confidence (Lower Bound)
    if t.min_mean_confidence is not None:
        val = metrics.get("mean_confidence", 0.0)
        passed = val >= t.min_mean_confidence
        checks.append(
            DecisionMetricCheck(
                metric_name="mean_confidence",
                passed=passed,
                actual_value=val,
                threshold_value=t.min_mean_confidence,
                is_upper_bound=False,
                message=(
                    f"Mean confidence {val:.3f} >= threshold {t.min_mean_confidence:.3f}"
                    if passed
                    else f"Mean confidence {val:.3f} fell below threshold {t.min_mean_confidence:.3f}"
                ),
            )
        )

    # 8. Mean Latency ms (Upper Bound)
    if t.max_mean_latency_ms is not None:
        val = metrics.get("mean_latency_ms", 0.0)
        passed = val <= t.max_mean_latency_ms
        checks.append(
            DecisionMetricCheck(
                metric_name="mean_latency_ms",
                passed=passed,
                actual_value=val,
                threshold_value=t.max_mean_latency_ms,
                is_upper_bound=True,
                message=(
                    f"Mean latency {val:.2f}ms <= threshold {t.max_mean_latency_ms:.2f}ms"
                    if passed
                    else f"Mean latency {val:.2f}ms exceeded threshold {t.max_mean_latency_ms:.2f}ms"
                ),
            )
        )

    failed_checks = tuple(c for c in checks if not c.passed)
    all_passed = len(failed_checks) == 0

    if all_passed:
        summary = f"QUALITY GATE PASS: Pipeline '{report.pipeline_name}' passed all {len(checks)} quality checks."
    else:
        summary = (
            f"QUALITY GATE FAIL: Pipeline '{report.pipeline_name}' failed "
            f"{len(failed_checks)}/{len(checks)} quality checks."
        )

    return DecisionQualityGateResult(
        pipeline_name=report.pipeline_name,
        dataset_name=report.dataset_name,
        passed=all_passed,
        checks=tuple(checks),
        failed_checks=failed_checks,
        summary_message=summary,
    )


def compare_decision_runs(
    current_report: DecisionBenchmarkReport,
    baseline_report: DecisionBenchmarkReport,
    thresholds: DecisionQualityThresholds | None = None,
) -> DecisionRegressionComparisonResult:
    """
    Compare current evaluation report against a baseline report to identify regressions.

    Args:
        current_report: Current benchmark report.
        baseline_report: Baseline benchmark report.
        thresholds: Optional quality thresholds defining maximum allowable degradation.

    Returns:
        DecisionRegressionComparisonResult containing deltas and violations.
    """
    if current_report.report_id == baseline_report.report_id:
        raise ValueError(
            f"Candidate benchmark report '{current_report.report_id}' cannot be compared against itself as baseline."
        )

    t = thresholds or DecisionQualityThresholds()
    curr_m = dict(current_report.aggregate_metrics)
    base_m = dict(baseline_report.aggregate_metrics)

    metric_keys = [
        "exact_match_rate",
        "acceptable_match_rate",
        "safety_violation_rate",
        "policy_match_rate",
        "policy_violation_rate",
        "policy_override_rate",
        "fallback_rate",
        "mean_confidence",
        "mean_latency_ms",
    ]

    deltas: dict[str, float] = {}
    improved: list[str] = []
    regressed: list[str] = []
    violations: list[str] = []

    for k in metric_keys:
        curr_val = curr_m.get(k, 0.0)
        base_val = base_m.get(k, 0.0)
        delta = curr_val - base_val
        deltas[k] = delta

        # Higher is better
        if k in (
            "exact_match_rate",
            "acceptable_match_rate",
            "policy_match_rate",
            "mean_confidence",
        ):
            if delta > 0.001:
                improved.append(k)
            elif delta < -0.001:
                regressed.append(k)
                if abs(delta) > t.max_allowed_quality_drop:
                    violations.append(
                        f"{k} regressed by {abs(delta):.3f} (max allowed drop: {t.max_allowed_quality_drop:.3f})"
                    )

        # Lower is better / safety
        elif k in (
            "safety_violation_rate",
            "policy_violation_rate",
            "policy_override_rate",
            "fallback_rate",
        ):
            if delta < -0.001:
                improved.append(k)
            elif delta > 0.001:
                regressed.append(k)
                if k == "safety_violation_rate" and delta > 0.0:
                    violations.append(
                        f"CRITICAL: safety_violation_rate increased by +{delta:.3f}"
                    )

        # Latency
        elif k == "mean_latency_ms":
            if delta < -1.0:
                improved.append(k)
            elif delta > 1.0:
                regressed.append(k)
                if (
                    base_val > 0.0
                    and (delta / base_val) > t.max_allowed_latency_increase_ratio
                ):
                    violations.append(
                        f"Latency increased by {delta:.1f}ms ({delta / base_val * 100:.1f}%, max allowed: {t.max_allowed_latency_increase_ratio * 100:.1f}%)"
                    )

    return DecisionRegressionComparisonResult(
        pipeline_name=current_report.pipeline_name,
        passed=len(violations) == 0,
        metric_deltas=deltas,
        improved_metrics=tuple(improved),
        regressed_metrics=tuple(regressed),
        violations=tuple(violations),
    )


def assert_decision_quality_gate(
    report: DecisionBenchmarkReport,
    thresholds: DecisionQualityThresholds | None = None,
) -> None:
    """
    Assert that a DecisionBenchmarkReport meets quality gates, raising EvaluationRegressionError on failure.

    Args:
        report: DecisionBenchmarkReport to validate.
        thresholds: Configured quality thresholds.

    Raises:
        EvaluationRegressionError: If any quality check fails.
    """
    res = evaluate_decision_quality_gate(report, thresholds=thresholds)
    if not res.passed:
        failure_details = "\n".join(f"  - {c.message}" for c in res.failed_checks)
        raise EvaluationRegressionError(
            f"{res.summary_message}\nFailures:\n{failure_details}"
        )


def format_quality_gate_terminal_summary(
    gate_result: DecisionQualityGateResult,
) -> str:
    """
    Format a DecisionQualityGateResult into a clean terminal report.

    Args:
        gate_result: DecisionQualityGateResult instance.

    Returns:
        Formatted terminal string.
    """
    lines = [
        f"Benchmark: {gate_result.pipeline_name} (Dataset: {gate_result.dataset_name})",
    ]

    for check in gate_result.checks:
        status_str = "PASS" if check.passed else "FAIL"
        comp_symbol = "<=" if check.is_upper_bound else ">="
        lines.append(
            f"{check.metric_name:<26}: actual={check.actual_value:.3f} {comp_symbol} {check.threshold_value:.3f} [{status_str}]"
        )

    overall = "PASS" if gate_result.passed else "FAIL"
    lines.append(f"\nQUALITY GATE: {overall}")
    return "\n".join(lines)
