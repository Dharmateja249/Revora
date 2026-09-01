"""
Revora Recovery Outcome Evaluation & Synthetic Batch Simulation Tests.

Comprehensive testing of the 100-scenario synthetic recovery dataset, deterministic
recovery simulation, intervention costs, stopping rules, policy constraint enforcement,
financial metric calculations, pipeline comparisons, persistence, and CLI execution.
"""

import json
from pathlib import Path

import pytest

from app.decision_engine import DecisionEngine, RecoveryAction
from app.evaluation.decision_evaluator import (
    DeterministicBaselinePipeline,
    DeterministicRAGPipeline,
)
from app.evaluation.recovery_benchmark import (
    format_recovery_benchmark_terminal_summary,
    run_recovery_benchmark,
    run_recovery_cli,
)
from app.evaluation.recovery_persistence import (
    list_recovery_reports,
    load_latest_recovery_report,
    load_recovery_report,
    save_recovery_report,
)
from app.evaluation.recovery_reporting import (
    compare_recovery_pipelines,
    generate_recovery_comparison_markdown,
    generate_recovery_json_report,
    generate_recovery_markdown_report,
    save_recovery_benchmark_artifacts,
)
from app.evaluation.recovery_schemas import (
    DEFAULT_ACTION_COSTS,
    RecoveryBenchmarkReport,
    RecoveryScenario,
    SimulatedRecoveryOutcome,
)
from app.evaluation.recovery_simulator import RecoverySimulator
from tests.fixtures.synthetic_recovery_dataset import get_synthetic_recovery_dataset

# =============================================================================
# Dataset Contract & Validity Tests
# =============================================================================


def test_synthetic_recovery_dataset_cardinality_and_uniqueness():
    scenarios = get_synthetic_recovery_dataset()
    assert len(scenarios) == 100

    ids = [s.scenario_id for s in scenarios]
    assert len(ids) == len(set(ids))

    for sc in scenarios:
        assert isinstance(sc, RecoveryScenario)
        assert sc.payment_amount > 0.0
        assert sc.current_attempt_count >= 1
        assert sc.max_allowed_attempts >= 1
        assert sc.failure_category
        assert sc.context.current_payment is not None
        assert sc.context.current_payment.amount == sc.payment_amount


def test_synthetic_recovery_dataset_category_distribution():
    scenarios = get_synthetic_recovery_dataset()
    categories = {s.failure_category for s in scenarios}

    expected_categories = {
        "network_timeout",
        "insufficient_funds",
        "expired_card",
        "fraud_hard_decline",
        "gateway_routing",
        "max_attempts_exhausted",
        "high_value_vip",
    }
    assert expected_categories.issubset(categories)


def test_synthetic_recovery_dataset_fraud_non_recoverable_invariants():
    scenarios = get_synthetic_recovery_dataset()
    fraud_cases = [s for s in scenarios if s.failure_category == "fraud_hard_decline"]
    assert len(fraud_cases) == 15

    for fc in fraud_cases:
        assert fc.is_recoverable is False
        assert fc.expected_recoverable_amount == 0.0
        assert RecoveryAction.RETRY_PAYMENT in fc.prohibited_actions


# =============================================================================
# Simulator Behavior & Financial Outcome Tests
# =============================================================================


def test_simulator_successful_recovery_and_cost_deduction():
    scenarios = get_synthetic_recovery_dataset()
    net_timeout_case = scenarios[0]  # REC-NET-001 ($32.50)
    assert net_timeout_case.failure_category == "network_timeout"

    pipeline = DeterministicBaselinePipeline()
    simulator = RecoverySimulator()

    outcome = simulator.simulate_scenario(pipeline, net_timeout_case)

    assert isinstance(outcome, SimulatedRecoveryOutcome)
    assert outcome.was_recovered is True
    assert outcome.amount_recovered == net_timeout_case.payment_amount
    assert (
        outcome.intervention_cost == DEFAULT_ACTION_COSTS[RecoveryAction.RETRY_PAYMENT]
    )
    assert outcome.net_recovered == round(
        net_timeout_case.payment_amount
        - DEFAULT_ACTION_COSTS[RecoveryAction.RETRY_PAYMENT],
        2,
    )
    assert outcome.is_policy_violation is False
    assert outcome.is_stopping_rule_violation is False


def test_simulator_stopping_rule_violation_and_zero_recovery():
    scenarios = get_synthetic_recovery_dataset()
    exhausted_case = next(
        s for s in scenarios if s.failure_category == "max_attempts_exhausted"
    )
    assert exhausted_case.current_attempt_count == 4
    assert exhausted_case.max_allowed_attempts == 4

    # If a pipeline recommends RETRY_PAYMENT on an exhausted attempt, it must fail
    class BadRetryPipeline:
        name = "bad_retry_pipeline"

        def evaluate(self, context):
            return RecoveryAction.RETRY_PAYMENT

    simulator = RecoverySimulator()
    outcome = simulator.simulate_scenario(BadRetryPipeline(), exhausted_case)

    assert outcome.is_stopping_rule_violation is True
    assert outcome.was_recovered is False
    assert outcome.amount_recovered == 0.0
    assert outcome.net_recovered == -DEFAULT_ACTION_COSTS[RecoveryAction.RETRY_PAYMENT]


def test_simulator_policy_violation_detection():
    scenarios = get_synthetic_recovery_dataset()
    expired_case = next(s for s in scenarios if s.failure_category == "expired_card")
    assert RecoveryAction.RETRY_PAYMENT in expired_case.prohibited_actions

    class ProhibitedRetryPipeline:
        name = "prohibited_retry_pipeline"

        def evaluate(self, context):
            return RecoveryAction.RETRY_PAYMENT

    simulator = RecoverySimulator()
    outcome = simulator.simulate_scenario(ProhibitedRetryPipeline(), expired_case)

    assert outcome.is_policy_violation is True
    assert outcome.was_recovered is False
    assert outcome.amount_recovered == 0.0


def test_simulator_unnecessary_intervention_on_fraud():
    scenarios = get_synthetic_recovery_dataset()
    fraud_case = next(
        s for s in scenarios if s.failure_category == "fraud_hard_decline"
    )

    class UnnecessaryInterventionPipeline:
        name = "unnecessary_intervention_pipeline"

        def evaluate(self, context):
            return RecoveryAction.RETRY_PAYMENT

    simulator = RecoverySimulator()
    outcome = simulator.simulate_scenario(UnnecessaryInterventionPipeline(), fraud_case)

    assert outcome.is_unnecessary_intervention is True
    assert outcome.is_policy_violation is True
    assert outcome.was_recovered is False


def test_simulator_duplicate_action_detection():
    scenarios = get_synthetic_recovery_dataset()
    case_with_last_retry = next(
        s
        for s in scenarios
        if s.context.current_payment_attempts
        and s.context.current_payment_attempts[-1].action == "retry_payment"
    )

    class RetryPipeline:
        name = "retry_pipeline"

        def evaluate(self, context):
            return RecoveryAction.RETRY_PAYMENT

    simulator = RecoverySimulator()
    outcome = simulator.simulate_scenario(RetryPipeline(), case_with_last_retry)

    assert outcome.is_duplicate_action is True


# =============================================================================
# Full Batch Simulation & Aggregate Metrics Tests
# =============================================================================


def test_simulate_recovery_batch_deterministic_baseline():
    scenarios = get_synthetic_recovery_dataset()
    pipeline = DeterministicBaselinePipeline(decision_engine=DecisionEngine())
    simulator = RecoverySimulator()

    report = simulator.simulate_batch(
        pipeline=pipeline,
        scenarios=scenarios,
        dataset_name="synthetic_recovery_100",
    )

    assert isinstance(report, RecoveryBenchmarkReport)
    assert report.num_scenarios == 100
    assert report.total_attempted_revenue > 0.0
    assert report.total_recoverable_revenue > 0.0
    assert report.gross_recovered_amount > 0.0
    assert report.total_intervention_cost > 0.0
    assert report.net_recovered_amount == round(
        report.gross_recovered_amount - report.total_intervention_cost, 2
    )
    assert 0.0 <= report.recovery_rate <= 1.0
    assert len(report.category_breakdown) > 0


def test_simulate_recovery_batch_empty_raises_value_error():
    simulator = RecoverySimulator()
    pipeline = DeterministicBaselinePipeline()
    with pytest.raises(ValueError, match="scenarios cannot be empty"):
        simulator.simulate_batch(pipeline=pipeline, scenarios=[])


# =============================================================================
# Pipeline Comparison & Reporting Tests
# =============================================================================


def test_compare_recovery_pipelines_matrix_and_markdown():
    scenarios = get_synthetic_recovery_dataset()[:20]
    p1 = DeterministicBaselinePipeline()
    p2 = DeterministicRAGPipeline()

    simulator = RecoverySimulator()
    rep1 = simulator.simulate_batch(p1, scenarios)
    rep2 = simulator.simulate_batch(p2, scenarios)

    comp = compare_recovery_pipelines([rep1, rep2])
    assert comp["num_pipelines"] == 2
    assert "deterministic_baseline" in comp["pipelines"]
    assert "deterministic_rag" in comp["pipelines"]
    assert "gross_recovered_amount" in comp["metrics"]

    md = generate_recovery_comparison_markdown([rep1, rep2])
    assert "# Revora Recovery Strategy Benchmark & Outcome Comparison" in md
    assert "**`deterministic_baseline`**" in md
    assert "**`deterministic_rag`**" in md
    assert "Net Recovered Revenue" in md


def test_generate_recovery_markdown_report_structure():
    scenarios = get_synthetic_recovery_dataset()[:10]
    pipeline = DeterministicBaselinePipeline()
    simulator = RecoverySimulator()
    report = simulator.simulate_batch(pipeline, scenarios)

    md = generate_recovery_markdown_report(report, include_categories=True)
    assert f"# Recovery Outcome Benchmark Report: `{pipeline.name}`" in md
    assert "## Financial Summary" in md
    assert "## Policy & Operational Compliance" in md
    assert "## Failure Category Breakdown" in md


def test_generate_recovery_json_report_serialization():
    scenarios = get_synthetic_recovery_dataset()[:5]
    pipeline = DeterministicBaselinePipeline()
    simulator = RecoverySimulator()
    report = simulator.simulate_batch(pipeline, scenarios)

    json_str = generate_recovery_json_report(report)
    parsed = json.loads(json_str)
    assert parsed["pipeline_name"] == pipeline.name
    assert parsed["num_scenarios"] == 5
    assert "gross_recovered_amount" in parsed


# =============================================================================
# Persistence Tests
# =============================================================================


def test_save_and_load_recovery_report(tmp_path: Path):
    scenarios = get_synthetic_recovery_dataset()[:5]
    pipeline = DeterministicBaselinePipeline()
    simulator = RecoverySimulator()
    report = simulator.simulate_batch(pipeline, scenarios)

    saved_file = save_recovery_report(
        report=report,
        directory=tmp_path,
        report_id="rec_run_test",
    )
    assert saved_file.exists()

    loaded = load_recovery_report("rec_run_test", directory=tmp_path)
    assert isinstance(loaded, RecoveryBenchmarkReport)
    assert loaded.pipeline_name == report.pipeline_name
    assert loaded.gross_recovered_amount == report.gross_recovered_amount

    latest = load_latest_recovery_report(
        pipeline_name=pipeline.name, directory=tmp_path
    )
    assert latest is not None
    assert latest.pipeline_name == pipeline.name

    report_ids = list_recovery_reports(directory=tmp_path)
    assert "rec_run_test" in report_ids


def test_save_recovery_benchmark_artifacts(tmp_path: Path):
    scenarios = get_synthetic_recovery_dataset()[:5]
    pipeline = DeterministicBaselinePipeline()
    simulator = RecoverySimulator()
    report = simulator.simulate_batch(pipeline, scenarios)

    json_path, md_path = save_recovery_benchmark_artifacts(
        report=report,
        output_dir=tmp_path,
        base_filename="custom_recovery_run",
    )

    assert json_path.exists()
    assert md_path.exists()
    assert json_path.name == "custom_recovery_run.json"
    assert md_path.name == "custom_recovery_run.md"


# =============================================================================
# Benchmark Runner & CLI Tests
# =============================================================================


def test_run_recovery_benchmark_multiple_pipelines(tmp_path: Path):
    scenarios = get_synthetic_recovery_dataset()[:5]
    reports = run_recovery_benchmark(
        scenarios=scenarios,
        pipelines=["deterministic_baseline", "deterministic_rag"],
        output_dir=tmp_path,
        save_artifacts=True,
    )

    assert len(reports) == 2
    assert "deterministic_baseline" in reports
    assert "deterministic_rag" in reports

    # Check comparative artifacts created
    assert (tmp_path / "recovery_pipeline_comparison.json").exists()
    assert (tmp_path / "recovery_pipeline_comparison.md").exists()


def test_format_recovery_benchmark_terminal_summary():
    scenarios = get_synthetic_recovery_dataset()[:3]
    reports = run_recovery_benchmark(
        scenarios=scenarios,
        pipelines=["deterministic_baseline"],
        save_artifacts=False,
    )

    summary = format_recovery_benchmark_terminal_summary(reports)
    assert "REVORA RECOVERY OUTCOME & FINANCIAL LEADERBOARD SUMMARY" in summary
    assert "deterministic_baseline" in summary
    assert "Gross Rec (₹)" in summary
    assert "Net Rec (₹)" in summary
    assert "$" not in summary
    assert "USD" not in summary


def test_synthetic_recovery_dataset_currency_inr():
    scenarios = get_synthetic_recovery_dataset()
    assert len(scenarios) == 100
    for s in scenarios:
        assert s.context.current_payment.currency == "INR"
        assert "$" not in s.description
        assert "₹" in s.description


def test_run_recovery_cli_standard_execution(tmp_path: Path, capsys):
    scenarios = get_synthetic_recovery_dataset()[:3]
    exit_code = run_recovery_cli(
        args=[
            "-p",
            "deterministic_baseline",
            "--output-dir",
            str(tmp_path),
        ],
        scenarios=scenarios,
    )

    assert exit_code == 0
    captured = capsys.readouterr()
    assert "REVORA RECOVERY OUTCOME & FINANCIAL LEADERBOARD SUMMARY" in captured.out


def test_run_recovery_cli_json_flag(capsys):
    scenarios = get_synthetic_recovery_dataset()[:2]
    exit_code = run_recovery_cli(
        args=["-p", "deterministic_baseline", "--json", "--no-save"],
        scenarios=scenarios,
    )

    assert exit_code == 0
    captured = capsys.readouterr()
    parsed = json.loads(captured.out)
    assert "deterministic_baseline" in parsed
