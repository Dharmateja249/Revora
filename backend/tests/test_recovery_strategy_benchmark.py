"""
Revora Recovery Strategy Benchmark & Outcome Comparison Tests (Stage 7.2).

Comprehensive testing of strategy uplift calculations, ROI, cost-per-recovered-dollar,
leaderboard ranking, category breakdowns, quality gates, regression checks, and CLI options.
"""

from pathlib import Path

import pytest
from app.decision_engine import RecoveryAction
from app.evaluation.recovery_benchmark import run_recovery_cli
from app.evaluation.recovery_comparison import (
    calculate_cost_per_recovered_dollar,
    calculate_recovery_roi,
    compare_category_recovery_performance,
    compute_recovery_strategy_uplift,
    generate_recovery_leaderboard,
)
from app.evaluation.recovery_regression import (
    RecoveryQualityThresholds,
    assert_recovery_quality_gate,
    compare_recovery_runs,
    evaluate_recovery_quality_gate,
    format_recovery_quality_gate_terminal_summary,
)
from app.evaluation.recovery_reporting import generate_recovery_comparison_markdown
from app.evaluation.recovery_schemas import (
    RecoveryBenchmarkReport,
    SimulatedRecoveryOutcome,
)

from tests.fixtures.synthetic_recovery_dataset import get_synthetic_recovery_dataset


def _make_recovery_outcome(
    scenario_id: str,
    pipeline_name: str,
    amount_attempted: float,
    amount_recovered: float,
    intervention_cost: float,
    is_policy_violation: bool = False,
    is_stopping_rule_violation: bool = False,
) -> SimulatedRecoveryOutcome:
    return SimulatedRecoveryOutcome(
        scenario_id=scenario_id,
        pipeline_name=pipeline_name,
        predicted_action=RecoveryAction.RETRY_PAYMENT
        if amount_recovered > 0
        else RecoveryAction.NO_ACTION,
        was_recovered=amount_recovered > 0,
        amount_attempted=amount_attempted,
        amount_recovered=amount_recovered,
        intervention_cost=intervention_cost,
        net_recovered=round(amount_recovered - intervention_cost, 2),
        is_policy_violation=is_policy_violation,
        is_stopping_rule_violation=is_stopping_rule_violation,
    )


def _make_benchmark_report(
    pipeline_name: str,
    gross: float = 1000.0,
    cost: float = 50.0,
    rec_rate: float = 0.80,
    policy_viol_rate: float = 0.0,
    stopping_viol_rate: float = 0.0,
    outcomes: list[SimulatedRecoveryOutcome] | None = None,
) -> RecoveryBenchmarkReport:
    outs = outcomes or [
        _make_recovery_outcome(f"SC-{i}", pipeline_name, 100.0, 80.0, 4.0)
        for i in range(10)
    ]
    return RecoveryBenchmarkReport(
        pipeline_name=pipeline_name,
        dataset_name="synthetic_recovery_100",
        num_scenarios=len(outs),
        total_attempted_revenue=1200.0,
        total_recoverable_revenue=1000.0,
        total_recovered_revenue=gross,
        recovery_rate=rec_rate,
        gross_recovered_amount=gross,
        total_intervention_cost=cost,
        net_recovered_amount=round(gross - cost, 2),
        average_recovered_per_case=round(gross / len(outs), 2),
        average_net_per_case=round((gross - cost) / len(outs), 2),
        policy_violation_rate=policy_viol_rate,
        stopping_rule_violation_rate=stopping_viol_rate,
        unnecessary_intervention_rate=0.0,
        duplicate_action_rate=0.0,
        category_breakdown={
            "network_timeout": {
                "count": 5.0,
                "attempted": 500.0,
                "recoverable": 500.0,
                "recovered": gross * 0.5,
                "cost": cost * 0.5,
                "net": (gross - cost) * 0.5,
                "recovery_rate": rec_rate,
            }
        },
        outcomes=tuple(outs),
    )


# =============================================================================
# ROI & CPRD Unit Tests
# =============================================================================


def test_calculate_recovery_roi_and_cprd():
    # $1000 recovered at $100 cost -> Net $900 -> ROI 9.0x (900%), CPRD $0.10
    roi = calculate_recovery_roi(1000.0, 100.0)
    assert roi == 9.0

    cprd = calculate_cost_per_recovered_dollar(1000.0, 100.0)
    assert cprd == 0.10

    # High ROI case: $54,592.50 recovered at $28.25 cost -> (54592.5 - 28.25) / 28.25 = 1931.4779
    high_roi = calculate_recovery_roi(54592.50, 28.25)
    assert high_roi == 1931.4779
    assert round(high_roi, 1) == 1931.5


def test_calculate_recovery_roi_and_cprd_edge_cases():
    import math

    # Zero cost, positive gross -> infinite return
    zero_cost_roi = calculate_recovery_roi(500.0, 0.0)
    assert math.isinf(zero_cost_roi)
    assert zero_cost_roi > 0
    assert calculate_cost_per_recovered_dollar(500.0, 0.0) == 0.0

    # Zero gross, positive cost -> -1.0 (-100% ROI)
    assert calculate_recovery_roi(0.0, 50.0) == -1.0
    assert calculate_cost_per_recovered_dollar(0.0, 50.0) == 1.0

    # Both zero -> 0.0
    assert calculate_recovery_roi(0.0, 0.0) == 0.0
    assert calculate_cost_per_recovered_dollar(0.0, 0.0) == 0.0


# =============================================================================
# Strategy Uplift Calculation Tests
# =============================================================================


def test_compute_recovery_strategy_uplift():
    b_outs = [
        _make_recovery_outcome("SC-1", "baseline", 100.0, 50.0, 5.0),
        _make_recovery_outcome("SC-2", "baseline", 100.0, 0.0, 5.0),
        _make_recovery_outcome("SC-3", "baseline", 100.0, 80.0, 5.0),
    ]
    c_outs = [
        _make_recovery_outcome("SC-1", "rag", 100.0, 80.0, 5.0),  # Improved
        _make_recovery_outcome(
            "SC-2", "rag", 100.0, 0.0, 0.0
        ),  # Improved (no cost spent)
        _make_recovery_outcome("SC-3", "rag", 100.0, 80.0, 5.0),  # Identical
    ]

    base_rep = _make_benchmark_report(
        "baseline", gross=130.0, cost=15.0, rec_rate=0.4333, outcomes=b_outs
    )
    cand_rep = _make_benchmark_report(
        "rag", gross=160.0, cost=10.0, rec_rate=0.5333, outcomes=c_outs
    )

    uplift = compute_recovery_strategy_uplift(cand_rep, base_rep)

    assert uplift.gross_recovery_uplift == 30.0
    assert uplift.net_recovery_uplift == 35.0
    assert uplift.recovery_rate_uplift == 0.10
    assert uplift.incremental_revenue_recovered == 30.0
    assert uplift.incremental_intervention_cost == -5.0
    assert uplift.improved_cases_count == 2
    assert uplift.worsened_cases_count == 0
    assert uplift.identical_cases_count == 1
    assert uplift.improved_cases_pct == 0.6667


# =============================================================================
# Leaderboard & Category Performance Tests
# =============================================================================


def test_generate_recovery_leaderboard_ordering():
    r_baseline = _make_benchmark_report(
        "baseline", gross=800.0, cost=50.0, rec_rate=0.60
    )
    r_rag = _make_benchmark_report("rag", gross=1000.0, cost=60.0, rec_rate=0.80)
    r_unsafe = _make_benchmark_report(
        "unsafe", gross=1200.0, cost=40.0, rec_rate=0.90, policy_viol_rate=0.05
    )

    board = generate_recovery_leaderboard([r_baseline, r_rag, r_unsafe])

    assert len(board) == 3
    # Rank 1: rag (compliant, highest net)
    assert board[0].pipeline_name == "rag"
    assert board[0].rank == 1
    assert board[0].is_compliant is True

    # Rank 2: baseline (compliant, lower net)
    assert board[1].pipeline_name == "baseline"
    assert board[1].rank == 2

    # Rank 3: unsafe (non-compliant, despite higher gross/net)
    assert board[2].pipeline_name == "unsafe"
    assert board[2].rank == 3
    assert board[2].is_compliant is False


def test_compare_category_recovery_performance():
    r1 = _make_benchmark_report("p1")
    r2 = _make_benchmark_report("p2")

    cats = compare_category_recovery_performance([r1, r2])
    assert "network_timeout" in cats
    assert "p1" in cats["network_timeout"]
    assert "p2" in cats["network_timeout"]


# =============================================================================
# Quality Gate Evaluation Tests
# =============================================================================


def test_evaluate_recovery_quality_gate_all_pass():
    report = _make_benchmark_report(
        "p1",
        gross=1000.0,
        cost=40.0,
        rec_rate=0.85,
        policy_viol_rate=0.0,
        stopping_viol_rate=0.0,
    )
    thresholds = RecoveryQualityThresholds(
        min_recovery_rate=0.80,
        min_net_recovered_amount=900.0,
        max_policy_violation_rate=0.0,
        max_stopping_rule_violation_rate=0.0,
        max_total_intervention_cost=50.0,
    )

    res = evaluate_recovery_quality_gate(report, thresholds=thresholds)
    assert res.passed is True
    assert len(res.violations) == 0


def test_evaluate_recovery_quality_gate_failures():
    report = _make_benchmark_report(
        "p1",
        gross=500.0,
        cost=80.0,
        rec_rate=0.40,
        policy_viol_rate=0.02,
        stopping_viol_rate=0.01,
    )
    thresholds = RecoveryQualityThresholds(
        min_recovery_rate=0.70,
        min_net_recovered_amount=600.0,
        max_policy_violation_rate=0.0,
        max_stopping_rule_violation_rate=0.0,
        max_total_intervention_cost=50.0,
    )

    res = evaluate_recovery_quality_gate(report, thresholds=thresholds)
    assert res.passed is False
    assert (
        len(res.violations) == 5
    )  # recovery rate, net amount, policy viol, stopping viol, cost


def test_assert_recovery_quality_gate_raises_assertion_error():
    report = _make_benchmark_report("bad_pipe", policy_viol_rate=0.05)
    with pytest.raises(AssertionError, match="Recovery quality gate failed"):
        assert_recovery_quality_gate(report)


def test_compare_recovery_runs_regression():
    base = _make_benchmark_report("p", gross=1000.0, cost=50.0, rec_rate=0.80)
    current_worse = _make_benchmark_report("p", gross=800.0, cost=50.0, rec_rate=0.60)

    res = compare_recovery_runs(current_worse, base)
    assert res.passed is False
    assert any("Net Recovery Revenue regressed" in v for v in res.violations)


def test_format_recovery_quality_gate_terminal_summary():
    report = _make_benchmark_report("p1")
    res = evaluate_recovery_quality_gate(report)
    summary = format_recovery_quality_gate_terminal_summary(res)
    assert "Recovery Quality Gate [p1]: PASSED" in summary
    assert "[PASS] policy_violation_rate" in summary


# =============================================================================
# Reporting & Markdown Tests
# =============================================================================


def test_generate_recovery_comparison_markdown_structure():
    r1 = _make_benchmark_report("deterministic_baseline", gross=800.0, cost=40.0)
    r2 = _make_benchmark_report("deterministic_rag", gross=950.0, cost=45.0)

    md = generate_recovery_comparison_markdown(
        reports=[r1, r2],
        baseline_pipeline="deterministic_baseline",
    )

    assert "# Revora Recovery Strategy Benchmark & Outcome Comparison" in md
    assert "## 1. Executive Summary" in md
    assert "## 2. Recovery Strategy Leaderboard" in md
    assert "## 3. Financial Comparison & Efficiency" in md
    assert "## 4. Failure Category Performance Breakdown" in md
    assert "## 5. Safety & Operational Compliance" in md
    assert "## 6. Strategy Uplift vs Baseline (`deterministic_baseline`)" in md
    assert "Gross Recovery Uplift" in md
    assert "Cost / ₹ Recovered" in md
    assert "₹" in md
    assert "$" not in md
    assert "USD" not in md


# =============================================================================
# Agent Resolution & 3-Pipeline End-to-End Tests
# =============================================================================


def test_agent_rag_resolves_and_runs_offline(tmp_path: Path):
    from app.evaluation.decision_benchmark import resolve_decision_pipeline

    pipe = resolve_decision_pipeline("agent_rag")
    assert pipe.name == "agent_rag"

    scenarios = get_synthetic_recovery_dataset()[:5]
    from app.evaluation.recovery_simulator import RecoverySimulator

    sim = RecoverySimulator()
    rep = sim.simulate_batch(scenarios=scenarios, pipeline=pipe)

    assert rep.pipeline_name == "agent_rag"
    assert rep.num_scenarios == 5
    assert rep.policy_violation_rate == 0.0
    assert rep.gross_recovered_amount > 0.0


def test_recovery_benchmark_executes_all_three_pipelines(tmp_path: Path):
    from app.evaluation.recovery_benchmark import run_recovery_benchmark

    scenarios = get_synthetic_recovery_dataset()[:10]
    reports = run_recovery_benchmark(
        scenarios=scenarios,
        pipelines=["deterministic_baseline", "deterministic_rag", "agent_rag"],
        output_dir=tmp_path,
        save_artifacts=True,
    )

    assert len(reports) == 3
    assert "deterministic_baseline" in reports
    assert "deterministic_rag" in reports
    assert "agent_rag" in reports

    for rep in reports.values():
        assert rep.num_scenarios == 10
        assert rep.gross_recovered_amount >= 0.0


def test_run_recovery_cli_with_all_three_pipelines(tmp_path: Path, capsys):
    scenarios = get_synthetic_recovery_dataset()[:5]

    exit_code = run_recovery_cli(
        args=[
            "-p",
            "deterministic_baseline",
            "-p",
            "deterministic_rag",
            "-p",
            "agent_rag",
            "--baseline",
            "deterministic_baseline",
            "--output-dir",
            str(tmp_path),
        ],
        scenarios=scenarios,
    )

    assert exit_code == 0
    captured = capsys.readouterr()
    assert "REVORA RECOVERY OUTCOME & FINANCIAL LEADERBOARD SUMMARY" in captured.out
    assert "deterministic_baseline" in captured.out
    assert "deterministic_rag" in captured.out
    assert "agent_rag" in captured.out
    assert "₹" in captured.out
    assert "$" not in captured.out
    assert "USD" not in captured.out


def test_run_recovery_cli_with_llm_provider_mock(capsys):
    """Verify recovery CLI runs with --llm-provider mock completely offline."""
    scenarios = get_synthetic_recovery_dataset()[:2]

    exit_code = run_recovery_cli(
        args=[
            "-p",
            "agent_rag",
            "--llm-provider",
            "mock",
            "--no-save",
            "--quiet",
        ],
        scenarios=scenarios,
    )

    assert exit_code == 0


def test_run_recovery_cli_with_llm_provider_openai_missing_key_fails(
    monkeypatch, capsys
):
    """Verify recovery CLI with --llm-provider openai fails clearly when key is missing."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    from app.config import get_settings

    settings = get_settings()
    monkeypatch.setattr(settings, "OPENAI_API_KEY", None)

    scenarios = get_synthetic_recovery_dataset()[:1]

    exit_code = run_recovery_cli(
        args=[
            "-p",
            "agent_rag",
            "--llm-provider",
            "openai",
            "--no-save",
        ],
        scenarios=scenarios,
    )

    assert exit_code == 1
    captured = capsys.readouterr()
    assert "OpenAI API key must be provided" in captured.err
