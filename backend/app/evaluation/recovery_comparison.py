"""
Revora Recovery Strategy Comparison & Financial Uplift Analysis.

Provides deterministic calculation of business-level uplift metrics (gross/net uplift,
incremental revenue, ROI, cost-per-recovered-dollar, per-case delta distribution),
category-level cross-strategy breakdown, and multi-pipeline leaderboard rankings.
"""

from collections import defaultdict
from collections.abc import Mapping, Sequence

from pydantic import BaseModel, ConfigDict, Field

from app.evaluation.recovery_schemas import (
    RecoveryBenchmarkReport,
    SimulatedRecoveryOutcome,
)


class RecoveryStrategyUplift(BaseModel):
    """
    Financial and operational uplift of a candidate recovery strategy against a baseline strategy.
    """

    model_config = ConfigDict(frozen=True, from_attributes=True, validate_default=True)

    candidate_pipeline: str
    baseline_pipeline: str
    gross_recovery_uplift: float
    gross_recovery_uplift_pct: float
    net_recovery_uplift: float
    net_recovery_uplift_pct: float
    recovery_rate_uplift: float
    incremental_revenue_recovered: float
    incremental_intervention_cost: float
    candidate_roi: float
    baseline_roi: float
    candidate_cost_per_recovered_dollar: float
    baseline_cost_per_recovered_dollar: float
    improved_cases_count: int = Field(ge=0)
    improved_cases_pct: float = Field(ge=0.0, le=1.0)
    worsened_cases_count: int = Field(ge=0)
    worsened_cases_pct: float = Field(ge=0.0, le=1.0)
    identical_cases_count: int = Field(ge=0)
    identical_cases_pct: float = Field(ge=0.0, le=1.0)


class RecoveryLeaderboardEntry(BaseModel):
    """
    Ranked leaderboard entry for a recovery pipeline.
    """

    model_config = ConfigDict(frozen=True, from_attributes=True, validate_default=True)

    rank: int = Field(ge=1)
    pipeline_name: str
    gross_recovered: float = Field(ge=0.0)
    intervention_cost: float = Field(ge=0.0)
    net_recovered: float
    recovery_rate: float = Field(ge=0.0, le=1.0)
    roi: float
    cost_per_recovered_dollar: float
    policy_violation_rate: float = Field(ge=0.0, le=1.0)
    stopping_rule_violation_rate: float = Field(ge=0.0, le=1.0)
    is_compliant: bool


def calculate_recovery_roi(
    gross_recovered: float,
    intervention_cost: float,
) -> float:
    """
    Calculate Return on Investment (ROI) for a recovery strategy.
    Formula: (Gross Recovered - Intervention Cost) / Intervention Cost
    If Intervention Cost is 0: returns float('inf') if gross is positive, else 0.0.
    """
    if intervention_cost <= 0.0:
        return float("inf") if gross_recovered > 0.0 else 0.0
    return round((gross_recovered - intervention_cost) / intervention_cost, 4)


def calculate_cost_per_recovered_dollar(
    gross_recovered: float,
    intervention_cost: float,
) -> float:
    """
    Calculate the operational cost spent per dollar of gross recovered revenue.
    Formula: Intervention Cost / Gross Recovered
    If Gross Recovered is 0: returns 0.0 if cost is 0, else 1.0.
    """
    if gross_recovered <= 0.0:
        return 1.0 if intervention_cost > 0.0 else 0.0
    return round(intervention_cost / gross_recovered, 4)


def compute_recovery_strategy_uplift(
    candidate_report: RecoveryBenchmarkReport,
    baseline_report: RecoveryBenchmarkReport,
) -> RecoveryStrategyUplift:
    """
    Calculate deterministic financial and per-case uplift of a candidate report against a baseline.

    Args:
        candidate_report: Evaluated candidate strategy benchmark.
        baseline_report: Baseline strategy benchmark.

    Returns:
        RecoveryStrategyUplift instance.
    """
    c_gross = candidate_report.gross_recovered_amount
    b_gross = baseline_report.gross_recovered_amount

    c_cost = candidate_report.total_intervention_cost
    b_cost = baseline_report.total_intervention_cost

    c_net = candidate_report.net_recovered_amount
    b_net = baseline_report.net_recovered_amount

    gross_uplift = round(c_gross - b_gross, 2)
    gross_uplift_pct = round((c_gross - b_gross) / b_gross, 4) if b_gross > 0.0 else 0.0

    net_uplift = round(c_net - b_net, 2)
    net_uplift_pct = round((c_net - b_net) / abs(b_net), 4) if abs(b_net) > 0.0 else 0.0

    rec_rate_uplift = round(
        candidate_report.recovery_rate - baseline_report.recovery_rate, 4
    )
    inc_revenue = gross_uplift
    inc_cost = round(c_cost - b_cost, 2)

    c_roi = calculate_recovery_roi(c_gross, c_cost)
    b_roi = calculate_recovery_roi(b_gross, b_cost)

    c_cprd = calculate_cost_per_recovered_dollar(c_gross, c_cost)
    b_cprd = calculate_cost_per_recovered_dollar(b_gross, b_cost)

    # Per-case delta analysis
    baseline_case_map: dict[str, SimulatedRecoveryOutcome] = {
        o.scenario_id: o for o in baseline_report.outcomes
    }

    improved = 0
    worsened = 0
    identical = 0

    for c_out in candidate_report.outcomes:
        b_out = baseline_case_map.get(c_out.scenario_id)
        if b_out is None:
            continue

        # Better net recovery, or resolved violation without losing net
        net_diff = c_out.net_recovered - b_out.net_recovered
        viol_diff = int(
            b_out.is_policy_violation or b_out.is_stopping_rule_violation
        ) - int(c_out.is_policy_violation or c_out.is_stopping_rule_violation)

        if net_diff > 0.01 or (net_diff >= -0.01 and viol_diff > 0):
            improved += 1
        elif net_diff < -0.01 or viol_diff < 0:
            worsened += 1
        else:
            identical += 1

    n = max(len(candidate_report.outcomes), 1)

    return RecoveryStrategyUplift(
        candidate_pipeline=candidate_report.pipeline_name,
        baseline_pipeline=baseline_report.pipeline_name,
        gross_recovery_uplift=gross_uplift,
        gross_recovery_uplift_pct=gross_uplift_pct,
        net_recovery_uplift=net_uplift,
        net_recovery_uplift_pct=net_uplift_pct,
        recovery_rate_uplift=rec_rate_uplift,
        incremental_revenue_recovered=inc_revenue,
        incremental_intervention_cost=inc_cost,
        candidate_roi=c_roi,
        baseline_roi=b_roi,
        candidate_cost_per_recovered_dollar=c_cprd,
        baseline_cost_per_recovered_dollar=b_cprd,
        improved_cases_count=improved,
        improved_cases_pct=round(improved / n, 4),
        worsened_cases_count=worsened,
        worsened_cases_pct=round(worsened / n, 4),
        identical_cases_count=identical,
        identical_cases_pct=round(identical / n, 4),
    )


def generate_recovery_leaderboard(
    reports: Sequence[RecoveryBenchmarkReport] | Mapping[str, RecoveryBenchmarkReport],
) -> list[RecoveryLeaderboardEntry]:
    """
    Rank evaluated pipelines primarily by net recovered revenue and compliance.

    Args:
        reports: Sequence or Mapping of RecoveryBenchmarkReport instances.

    Returns:
        Ordered list of RecoveryLeaderboardEntry instances.
    """
    if isinstance(reports, Mapping):
        rep_list = list(reports.values())
    else:
        rep_list = list(reports)

    # Sort key: 1) Fully compliant first, 2) Net recovered revenue desc, 3) Recovery rate desc
    def _sort_key(r: RecoveryBenchmarkReport) -> tuple[int, float, float]:
        is_safe = (
            1
            if (
                r.policy_violation_rate == 0.0 and r.stopping_rule_violation_rate == 0.0
            )
            else 0
        )
        return (is_safe, r.net_recovered_amount, r.recovery_rate)

    sorted_reports = sorted(rep_list, key=_sort_key, reverse=True)

    leaderboard: list[RecoveryLeaderboardEntry] = []
    for idx, rep in enumerate(sorted_reports, start=1):
        is_compliant = (
            rep.policy_violation_rate == 0.0 and rep.stopping_rule_violation_rate == 0.0
        )
        roi = calculate_recovery_roi(
            rep.gross_recovered_amount,
            rep.total_intervention_cost,
        )
        cprd = calculate_cost_per_recovered_dollar(
            rep.gross_recovered_amount,
            rep.total_intervention_cost,
        )

        leaderboard.append(
            RecoveryLeaderboardEntry(
                rank=idx,
                pipeline_name=rep.pipeline_name,
                gross_recovered=rep.gross_recovered_amount,
                intervention_cost=rep.total_intervention_cost,
                net_recovered=rep.net_recovered_amount,
                recovery_rate=rep.recovery_rate,
                roi=roi,
                cost_per_recovered_dollar=cprd,
                policy_violation_rate=rep.policy_violation_rate,
                stopping_rule_violation_rate=rep.stopping_rule_violation_rate,
                is_compliant=is_compliant,
            )
        )

    return leaderboard


def compare_category_recovery_performance(
    reports: Sequence[RecoveryBenchmarkReport] | Mapping[str, RecoveryBenchmarkReport],
) -> dict[str, dict[str, dict[str, float]]]:
    """
    Produce a cross-pipeline comparison dictionary grouped by failure category.

    Structure:
    {
        "category_name": {
            "pipeline_name": {
                "recovered": float,
                "net": float,
                "cost": float,
                "recovery_rate": float,
            }
        }
    }
    """
    if isinstance(reports, Mapping):
        rep_list = list(reports.values())
    else:
        rep_list = list(reports)

    category_matrix: dict[str, dict[str, dict[str, float]]] = defaultdict(dict)

    for rep in rep_list:
        for cat, stats in rep.category_breakdown.items():
            category_matrix[cat][rep.pipeline_name] = {
                "attempted": stats.get("attempted", 0.0),
                "recoverable": stats.get("recoverable", 0.0),
                "recovered": stats.get("recovered", 0.0),
                "cost": stats.get("cost", 0.0),
                "net": stats.get("net", 0.0),
                "recovery_rate": stats.get("recovery_rate", 0.0),
            }

    return dict(category_matrix)
