"""
Revora Recovery Quality Gates & Regression Detection.

Evaluates RecoveryBenchmarkReport results against strict financial, operational,
and compliance thresholds to catch degradation in revenue recovery or safety violations.
"""

from pydantic import BaseModel, ConfigDict, Field

from app.evaluation.recovery_comparison import (
    RecoveryStrategyUplift,
    compute_recovery_strategy_uplift,
)
from app.evaluation.recovery_schemas import RecoveryBenchmarkReport


class RecoveryQualityThresholds(BaseModel):
    """
    Configurable quality and safety gates for synthetic recovery outcome benchmarks.
    """

    model_config = ConfigDict(frozen=True, from_attributes=True, validate_default=True)

    min_recovery_rate: float | None = Field(default=None, ge=0.0, le=1.0)
    min_net_recovered_amount: float | None = Field(default=None, ge=0.0)
    max_policy_violation_rate: float = Field(default=0.0, ge=0.0, le=1.0)
    max_stopping_rule_violation_rate: float = Field(default=0.0, ge=0.0, le=1.0)
    max_unnecessary_intervention_rate: float | None = Field(
        default=None, ge=0.0, le=1.0
    )
    max_total_intervention_cost: float | None = Field(default=None, ge=0.0)
    max_cost_per_recovered_dollar: float | None = Field(default=None, ge=0.0)


class RecoveryMetricCheck(BaseModel):
    """
    Individual metric evaluation result within a recovery quality gate.
    """

    model_config = ConfigDict(frozen=True, from_attributes=True)

    metric_name: str
    actual_value: float
    threshold_value: float
    passed: bool
    is_upper_bound: bool
    message: str


class RecoveryQualityGateResult(BaseModel):
    """
    Complete recovery quality gate evaluation result.
    """

    model_config = ConfigDict(frozen=True, from_attributes=True)

    pipeline_name: str
    passed: bool
    thresholds: RecoveryQualityThresholds
    checks: list[RecoveryMetricCheck] = Field(default_factory=list)
    violations: list[str] = Field(default_factory=list)


class RecoveryRegressionComparisonResult(BaseModel):
    """
    Comparative regression result between a candidate run and a baseline run.
    """

    model_config = ConfigDict(frozen=True, from_attributes=True)

    candidate_pipeline: str
    baseline_pipeline: str
    passed: bool
    violations: list[str] = Field(default_factory=list)
    uplift: RecoveryStrategyUplift


def evaluate_recovery_quality_gate(
    report: RecoveryBenchmarkReport,
    thresholds: RecoveryQualityThresholds | None = None,
) -> RecoveryQualityGateResult:
    """
    Evaluate a RecoveryBenchmarkReport against recovery quality and safety thresholds.

    Args:
        report: RecoveryBenchmarkReport instance.
        thresholds: Configurable thresholds (defaults to strict zero-violation standards).

    Returns:
        RecoveryQualityGateResult.
    """
    if not isinstance(report, RecoveryBenchmarkReport):
        raise TypeError(
            f"report must be RecoveryBenchmarkReport, got {type(report).__name__}"
        )

    t = thresholds or RecoveryQualityThresholds()
    checks: list[RecoveryMetricCheck] = []
    violations: list[str] = []

    # 1. Recovery Rate (Lower bound)
    if t.min_recovery_rate is not None:
        passed = report.recovery_rate >= t.min_recovery_rate
        msg = f"Recovery Rate {report.recovery_rate:.1%} >= {t.min_recovery_rate:.1%}"
        if not passed:
            violations.append(
                f"Recovery Rate failure: actual {report.recovery_rate:.1%} < minimum required {t.min_recovery_rate:.1%}"
            )
        checks.append(
            RecoveryMetricCheck(
                metric_name="recovery_rate",
                actual_value=report.recovery_rate,
                threshold_value=t.min_recovery_rate,
                passed=passed,
                is_upper_bound=False,
                message=msg,
            )
        )

    # 2. Net Recovered Amount (Lower bound)
    if t.min_net_recovered_amount is not None:
        passed = report.net_recovered_amount >= t.min_net_recovered_amount
        msg = f"Net Recovered ₹{report.net_recovered_amount:,.2f} >= ₹{t.min_net_recovered_amount:,.2f}"
        if not passed:
            violations.append(
                f"Net Recovered Revenue failure: actual ₹{report.net_recovered_amount:,.2f} < minimum required ₹{t.min_net_recovered_amount:,.2f}"
            )
        checks.append(
            RecoveryMetricCheck(
                metric_name="net_recovered_amount",
                actual_value=report.net_recovered_amount,
                threshold_value=t.min_net_recovered_amount,
                passed=passed,
                is_upper_bound=False,
                message=msg,
            )
        )

    # 3. Policy Violation Rate (Strict Upper bound)
    passed = report.policy_violation_rate <= t.max_policy_violation_rate
    msg = f"Policy Violation Rate {report.policy_violation_rate:.1%} <= {t.max_policy_violation_rate:.1%}"
    if not passed:
        violations.append(
            f"Policy Violation failure: actual {report.policy_violation_rate:.1%} > maximum allowed {t.max_policy_violation_rate:.1%}"
        )
    checks.append(
        RecoveryMetricCheck(
            metric_name="policy_violation_rate",
            actual_value=report.policy_violation_rate,
            threshold_value=t.max_policy_violation_rate,
            passed=passed,
            is_upper_bound=True,
            message=msg,
        )
    )

    # 4. Stopping Rule Violation Rate (Strict Upper bound)
    passed = report.stopping_rule_violation_rate <= t.max_stopping_rule_violation_rate
    msg = f"Stopping Rule Violation Rate {report.stopping_rule_violation_rate:.1%} <= {t.max_stopping_rule_violation_rate:.1%}"
    if not passed:
        violations.append(
            f"Stopping Rule Violation failure: actual {report.stopping_rule_violation_rate:.1%} > maximum allowed {t.max_stopping_rule_violation_rate:.1%}"
        )
    checks.append(
        RecoveryMetricCheck(
            metric_name="stopping_rule_violation_rate",
            actual_value=report.stopping_rule_violation_rate,
            threshold_value=t.max_stopping_rule_violation_rate,
            passed=passed,
            is_upper_bound=True,
            message=msg,
        )
    )

    # 5. Unnecessary Intervention Rate (Upper bound)
    if t.max_unnecessary_intervention_rate is not None:
        passed = (
            report.unnecessary_intervention_rate <= t.max_unnecessary_intervention_rate
        )
        msg = f"Unnecessary Intervention Rate {report.unnecessary_intervention_rate:.1%} <= {t.max_unnecessary_intervention_rate:.1%}"
        if not passed:
            violations.append(
                f"Unnecessary Intervention failure: actual {report.unnecessary_intervention_rate:.1%} > maximum allowed {t.max_unnecessary_intervention_rate:.1%}"
            )
        checks.append(
            RecoveryMetricCheck(
                metric_name="unnecessary_intervention_rate",
                actual_value=report.unnecessary_intervention_rate,
                threshold_value=t.max_unnecessary_intervention_rate,
                passed=passed,
                is_upper_bound=True,
                message=msg,
            )
        )

    # 6. Total Intervention Cost (Upper bound)
    if t.max_total_intervention_cost is not None:
        passed = report.total_intervention_cost <= t.max_total_intervention_cost
        msg = f"Total Intervention Cost ₹{report.total_intervention_cost:,.2f} <= ₹{t.max_total_intervention_cost:,.2f}"
        if not passed:
            violations.append(
                f"Intervention Cost failure: actual ₹{report.total_intervention_cost:,.2f} > maximum allowed ₹{t.max_total_intervention_cost:,.2f}"
            )
        checks.append(
            RecoveryMetricCheck(
                metric_name="total_intervention_cost",
                actual_value=report.total_intervention_cost,
                threshold_value=t.max_total_intervention_cost,
                passed=passed,
                is_upper_bound=True,
                message=msg,
            )
        )

    all_passed = len(violations) == 0
    return RecoveryQualityGateResult(
        pipeline_name=report.pipeline_name,
        passed=all_passed,
        thresholds=t,
        checks=checks,
        violations=violations,
    )


def compare_recovery_runs(
    current_report: RecoveryBenchmarkReport,
    baseline_report: RecoveryBenchmarkReport,
    thresholds: RecoveryQualityThresholds | None = None,
) -> RecoveryRegressionComparisonResult:
    """
    Compare a current recovery benchmark run against a baseline run and determine whether regressions occurred.

    Args:
        current_report: Candidate benchmark report.
        baseline_report: Stored baseline benchmark report.
        thresholds: Optional quality gate thresholds.

    Returns:
        RecoveryRegressionComparisonResult.
    """
    uplift = compute_recovery_strategy_uplift(
        candidate_report=current_report,
        baseline_report=baseline_report,
    )

    violations: list[str] = []

    # 1. Regressions in net revenue recovered
    if uplift.net_recovery_uplift < -0.01:
        violations.append(
            f"Net Recovery Revenue regressed: ₹{uplift.net_recovery_uplift:,.2f} lower than baseline ({uplift.net_recovery_uplift_pct:.1%})"
        )

    # 2. Safety / Policy violation regressions
    if current_report.policy_violation_rate > baseline_report.policy_violation_rate:
        violations.append(
            f"Policy Violation Rate increased from {baseline_report.policy_violation_rate:.1%} to {current_report.policy_violation_rate:.1%}"
        )

    if (
        current_report.stopping_rule_violation_rate
        > baseline_report.stopping_rule_violation_rate
    ):
        violations.append(
            f"Stopping Rule Violation Rate increased from {baseline_report.stopping_rule_violation_rate:.1%} to {current_report.stopping_rule_violation_rate:.1%}"
        )

    # 3. Check standalone quality gate
    gate_res = evaluate_recovery_quality_gate(current_report, thresholds=thresholds)
    violations.extend(gate_res.violations)

    # Deduplicate violations while preserving order
    unique_violations: list[str] = []
    for v in violations:
        if v not in unique_violations:
            unique_violations.append(v)

    return RecoveryRegressionComparisonResult(
        candidate_pipeline=current_report.pipeline_name,
        baseline_pipeline=baseline_report.pipeline_name,
        passed=len(unique_violations) == 0,
        violations=unique_violations,
        uplift=uplift,
    )


def assert_recovery_quality_gate(
    report: RecoveryBenchmarkReport,
    thresholds: RecoveryQualityThresholds | None = None,
) -> None:
    """
    Assert that a RecoveryBenchmarkReport passes all quality and safety gates, raising AssertionError on failure.
    """
    result = evaluate_recovery_quality_gate(report, thresholds=thresholds)
    if not result.passed:
        formatted_violations = "\n".join(f"  - {v}" for v in result.violations)
        raise AssertionError(
            f"Recovery quality gate failed for pipeline '{report.pipeline_name}':\n{formatted_violations}"
        )


def format_recovery_quality_gate_terminal_summary(
    result: RecoveryQualityGateResult,
) -> str:
    """
    Format a terminal-friendly summary of a RecoveryQualityGateResult.
    """
    status_str = "PASSED" if result.passed else "FAILED"
    lines = [
        f"Recovery Quality Gate [{result.pipeline_name}]: {status_str}",
        "-" * 70,
    ]
    for chk in result.checks:
        symbol = "[PASS]" if chk.passed else "[FAIL]"
        val_str = (
            f"{chk.actual_value:.1%}"
            if "rate" in chk.metric_name
            else (
                f"₹{chk.actual_value:,.2f}"
                if "cost" in chk.metric_name or "amount" in chk.metric_name
                else f"{chk.actual_value:.2f}"
            )
        )
        thresh_str = (
            f"{chk.threshold_value:.1%}"
            if "rate" in chk.metric_name
            else (
                f"₹{chk.threshold_value:,.2f}"
                if "cost" in chk.metric_name or "amount" in chk.metric_name
                else f"{chk.threshold_value:.2f}"
            )
        )
        op = "<=" if chk.is_upper_bound else ">="
        lines.append(
            f"  {symbol:<6} {chk.metric_name:<32} {val_str:<10} {op} {thresh_str}"
        )

    if result.violations:
        lines.append("\nViolations:")
        for v in result.violations:
            lines.append(f"  [!] {v}")

    return "\n".join(lines)
