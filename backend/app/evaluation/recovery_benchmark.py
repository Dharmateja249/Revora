"""
Revora Recovery Outcome Benchmark Runner & CLI.

Orchestrates batch recovery simulations across synthetic scenarios, evaluating financial
capture, operational costs, policy adherence, and stopping rule compliance.
"""

import argparse
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from app.agent.orchestrator import AgentOrchestrator
from app.evaluation.decision_benchmark import resolve_decision_pipeline
from app.evaluation.decision_evaluator import DecisionPipeline
from app.evaluation.recovery_persistence import get_recovery_evaluation_directory
from app.evaluation.recovery_reporting import (
    generate_recovery_comparison_markdown,
    generate_recovery_json_report,
    generate_recovery_markdown_report,
    save_recovery_benchmark_artifacts,
)
from app.evaluation.recovery_schemas import (
    RecoveryBenchmarkReport,
    RecoveryScenario,
)
from app.evaluation.recovery_simulator import RecoverySimulator


def run_recovery_benchmark(
    scenarios: Sequence[RecoveryScenario] | None = None,
    pipelines: Sequence[str | DecisionPipeline | Any] | None = None,
    dataset_name: str = "synthetic_recovery_100",
    output_dir: Path | str | None = None,
    save_artifacts: bool = True,
    agent_orchestrator: AgentOrchestrator | None = None,
    llm_provider: str = "mock",
) -> dict[str, RecoveryBenchmarkReport]:
    """
    Execute synthetic batch recovery outcome simulations across one or more pipelines.

    Args:
        scenarios: Optional pre-loaded sequence of RecoveryScenarios (defaults to 100-scenario dataset).
        pipelines: Optional sequence of pipeline names or instances.
        dataset_name: Dataset identifier.
        output_dir: Output directory for benchmark artifacts.
        save_artifacts: Whether to write JSON/Markdown reports to disk.
        agent_orchestrator: Optional AgentOrchestrator instance for agent pipelines.
        llm_provider: LLM provider identifier for agent pipelines ('mock' or 'openai'). Defaults to 'mock'.

    Returns:
        Dictionary mapping pipeline_name -> RecoveryBenchmarkReport.
    """
    case_list = scenarios
    if case_list is None:
        from tests.fixtures.synthetic_recovery_dataset import (
            get_synthetic_recovery_dataset,
        )

        case_list = get_synthetic_recovery_dataset()

    pipe_specs = (
        pipelines
        if pipelines is not None
        else ["deterministic_baseline", "deterministic_rag"]
    )

    resolved_pipelines = [
        resolve_decision_pipeline(
            p,
            agent_orchestrator=agent_orchestrator,
            llm_provider=llm_provider,
        )
        for p in pipe_specs
    ]

    simulator = RecoverySimulator()
    reports: dict[str, RecoveryBenchmarkReport] = {}

    for pipe in resolved_pipelines:
        rep = simulator.simulate_batch(
            pipeline=pipe,
            scenarios=case_list,
            dataset_name=dataset_name,
        )
        reports[rep.pipeline_name] = rep

    if save_artifacts:
        target_dir = (
            Path(output_dir)
            if output_dir is not None
            else get_recovery_evaluation_directory()
        )
        target_dir.mkdir(parents=True, exist_ok=True)

        for rep in reports.values():
            save_recovery_benchmark_artifacts(
                report=rep,
                output_dir=target_dir,
            )

        if len(reports) > 1:
            save_recovery_benchmark_artifacts(
                report=reports,
                output_dir=target_dir,
                base_filename="recovery_pipeline_comparison",
            )

    return reports


def format_recovery_benchmark_terminal_summary(
    reports: Mapping[str, RecoveryBenchmarkReport] | Sequence[RecoveryBenchmarkReport],
) -> str:
    """
    Format a clean ASCII summary table and leaderboard for terminal recovery outcome benchmark reporting.
    """
    from app.evaluation.recovery_comparison import generate_recovery_leaderboard

    norm: dict[str, RecoveryBenchmarkReport] = {}
    if isinstance(reports, Mapping):
        norm = dict(reports)
    elif isinstance(reports, (list, tuple)):
        for r in reports:
            norm[r.pipeline_name] = r

    if not norm:
        return "No recovery benchmark reports to display."

    leaderboard = generate_recovery_leaderboard(norm)

    import math

    lines = [
        "=" * 120,
        " REVORA RECOVERY OUTCOME & FINANCIAL LEADERBOARD SUMMARY",
        "=" * 120,
        f"{'Rank':<5} | {'Pipeline':<24} | {'Net Rec (₹)':<13} | {'Gross Rec (₹)':<15} | {'Cost (₹)':<11} | {'Rec Rate':<9} | {'ROI':<11} | {'Status':<12}",
        "-" * 120,
    ]

    for entry in leaderboard:
        status = "COMPLIANT" if entry.is_compliant else "VIOLATIONS"
        if math.isinf(entry.roi):
            roi_str = "INF"
        else:
            roi_str = f"{entry.roi:,.1f}x"
        lines.append(
            f"#{entry.rank:<4} | {entry.pipeline_name:<24} | ₹{entry.net_recovered:<12,.2f} | ₹{entry.gross_recovered:<14,.2f} | ₹{entry.intervention_cost:<10,.2f} | {entry.recovery_rate * 100.0:<8.1f}% | {roi_str:<11} | {status:<12}"
        )

    lines.append("=" * 120)
    return "\n".join(lines)


def run_recovery_cli(
    args: list[str] | None = None,
    scenarios: Sequence[RecoveryScenario] | None = None,
) -> int:
    """
    Command-line interface entrypoint for running synthetic recovery outcome benchmarks.
    """
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except (AttributeError, ValueError, OSError):
                continue

    parser = argparse.ArgumentParser(
        prog="revora-recovery-benchmark",
        description="Revora Synthetic Recovery Outcome & Financial Benchmark CLI",
    )
    parser.add_argument(
        "--pipeline",
        "-p",
        action="append",
        dest="pipelines",
        help="Specify pipelines to evaluate (deterministic_baseline, deterministic_rag, agent_rag).",
    )
    parser.add_argument(
        "--llm-provider",
        type=str,
        default="mock",
        choices=["mock", "openai"],
        help="LLM provider for agent pipelines ('mock' for offline evaluation, 'openai' for live OpenAI). Defaults to 'mock'.",
    )
    parser.add_argument(
        "--baseline",
        "-b",
        type=str,
        default="deterministic_baseline",
        help="Baseline pipeline name for calculating comparative uplift (defaults to deterministic_baseline).",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        default=None,
        help="Directory to write output JSON and Markdown benchmark reports.",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Disable automatic artifact file persistence.",
    )
    format_group = parser.add_mutually_exclusive_group()
    format_group.add_argument(
        "--json",
        action="store_true",
        help="Print machine-readable JSON report to stdout.",
    )
    format_group.add_argument(
        "--markdown",
        action="store_true",
        help="Print human-readable Markdown report to stdout.",
    )
    parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Suppress terminal ASCII summary table.",
    )
    parser.add_argument(
        "--assert-quality-gate",
        action="store_true",
        help="Assert quality and safety gates for each evaluated pipeline, exiting non-zero on failure.",
    )
    parser.add_argument(
        "--min-recovery-rate",
        type=float,
        default=None,
        help="Minimum required recovery rate in [0.0, 1.0].",
    )
    parser.add_argument(
        "--min-net-recovered",
        type=float,
        default=None,
        help="Minimum required net recovered amount in INR.",
    )
    parser.add_argument(
        "--max-policy-violation",
        type=float,
        default=None,
        help="Maximum allowed policy violation rate (defaults to 0.0).",
    )
    parser.add_argument(
        "--max-stopping-violation",
        type=float,
        default=None,
        help="Maximum allowed stopping rule violation rate (defaults to 0.0).",
    )
    parser.add_argument(
        "--max-unnecessary-intervention",
        type=float,
        default=None,
        help="Maximum allowed unnecessary intervention rate on non-recoverable debt.",
    )
    parser.add_argument(
        "--max-cost",
        type=float,
        default=None,
        help="Maximum allowed total intervention cost in INR.",
    )
    parser.add_argument(
        "--max-cost-per-recovered-dollar",
        type=float,
        default=None,
        help="Maximum allowed cost spent per recovered currency unit.",
    )
    parser.add_argument(
        "--compare-baseline",
        type=str,
        default=None,
        help="Path or ID of a baseline recovery report on disk to compare against (e.g. 'latest').",
    )
    parser.add_argument(
        "--assert-no-regressions",
        action="store_true",
        help="Exit non-zero if comparative financial regression or increased violations are detected against the disk baseline.",
    )

    parsed = parser.parse_args(args)

    if parsed.assert_no_regressions and not parsed.compare_baseline:
        print(
            "Error: --assert-no-regressions requires --compare-baseline <baseline_id_or_path>.",
            file=sys.stderr,
        )
        return 1

    try:
        # 1. Resolve / read previous baseline first if requested
        pre_resolved_baselines: dict[str, RecoveryBenchmarkReport] = {}
        if parsed.compare_baseline:
            from app.evaluation.recovery_persistence import (
                load_latest_recovery_report,
                load_recovery_report,
            )

            pipe_names = (
                parsed.pipelines
                if parsed.pipelines
                else ["deterministic_baseline", "deterministic_rag", "agent_rag"]
            )
            for p_name in pipe_names:
                if parsed.compare_baseline == "latest":
                    b_rep = load_latest_recovery_report(
                        pipeline_name=p_name,
                        directory=parsed.output_dir,
                    )
                else:
                    b_rep = load_recovery_report(
                        parsed.compare_baseline,
                        directory=parsed.output_dir,
                    )
                if b_rep is not None:
                    pre_resolved_baselines[p_name] = b_rep

        # 2. Run / create candidate benchmark without persisting prematurely
        reports = run_recovery_benchmark(
            scenarios=scenarios,
            pipelines=parsed.pipelines,
            output_dir=parsed.output_dir,
            save_artifacts=False,
            llm_provider=parsed.llm_provider,
        )

        if not parsed.quiet and not parsed.json and not parsed.markdown:
            print(format_recovery_benchmark_terminal_summary(reports))

        if parsed.json:
            print(generate_recovery_json_report(reports))

        if parsed.markdown:
            if len(reports) == 1:
                single_rep = next(iter(reports.values()))
                print(generate_recovery_markdown_report(single_rep))
            else:
                print(
                    generate_recovery_comparison_markdown(
                        reports, baseline_pipeline=parsed.baseline
                    )
                )

        # 3. Compare candidate against retained baseline
        has_regression = False
        if parsed.compare_baseline:
            from app.evaluation.recovery_persistence import (
                load_latest_recovery_report,
                load_recovery_report,
            )
            from app.evaluation.recovery_regression import compare_recovery_runs

            for rep in reports.values():
                b_rep = pre_resolved_baselines.get(rep.pipeline_name)
                if b_rep is None:
                    if parsed.compare_baseline == "latest":
                        b_rep = load_latest_recovery_report(
                            pipeline_name=rep.pipeline_name,
                            directory=parsed.output_dir,
                            exclude_report_id=rep.report_id,
                        )
                    else:
                        b_rep = load_recovery_report(
                            parsed.compare_baseline,
                            directory=parsed.output_dir,
                        )
                if b_rep is None:
                    raise FileNotFoundError(
                        f"No latest recovery baseline report found on disk for '{rep.pipeline_name}'."
                    )

                comp_res = compare_recovery_runs(
                    current_report=rep,
                    baseline_report=b_rep,
                )
                if not parsed.quiet:
                    status_str = "PASS" if comp_res.passed else "REGRESSION DETECTED"
                    print(
                        f"\nBaseline Comparison for '{rep.pipeline_name}': {status_str}"
                    )
                    if comp_res.violations:
                        for v in comp_res.violations:
                            print(f"  [!] {v}")

                if not comp_res.passed:
                    has_regression = True

        # 4. Check quality gates if requested
        quality_gate_failed = False
        quality_gate_requested = bool(
            parsed.assert_quality_gate
            or parsed.min_recovery_rate is not None
            or parsed.min_net_recovered is not None
            or parsed.max_policy_violation is not None
            or parsed.max_stopping_violation is not None
            or parsed.max_unnecessary_intervention is not None
            or parsed.max_cost is not None
            or parsed.max_cost_per_recovered_dollar is not None
        )

        if quality_gate_requested:
            from app.evaluation.recovery_regression import (
                RecoveryQualityThresholds,
                evaluate_recovery_quality_gate,
                format_recovery_quality_gate_terminal_summary,
            )

            thresholds = RecoveryQualityThresholds(
                min_recovery_rate=parsed.min_recovery_rate,
                min_net_recovered_amount=parsed.min_net_recovered,
                max_policy_violation_rate=(
                    parsed.max_policy_violation
                    if parsed.max_policy_violation is not None
                    else 0.0
                ),
                max_stopping_rule_violation_rate=(
                    parsed.max_stopping_violation
                    if parsed.max_stopping_violation is not None
                    else 0.0
                ),
                max_unnecessary_intervention_rate=parsed.max_unnecessary_intervention,
                max_total_intervention_cost=parsed.max_cost,
                max_cost_per_recovered_dollar=parsed.max_cost_per_recovered_dollar,
            )

            for rep in reports.values():
                gate_res = evaluate_recovery_quality_gate(rep, thresholds=thresholds)
                if not parsed.quiet:
                    print(
                        "\n" + format_recovery_quality_gate_terminal_summary(gate_res)
                    )
                if not gate_res.passed:
                    quality_gate_failed = True

        # 5. Persist / publish candidate and latest appropriately
        if not parsed.no_save:
            from app.evaluation.recovery_persistence import (
                get_recovery_evaluation_directory,
            )
            from app.evaluation.recovery_reporting import (
                save_recovery_benchmark_artifacts,
            )

            target_dir = get_recovery_evaluation_directory(parsed.output_dir)
            for rep in reports.values():
                save_recovery_benchmark_artifacts(
                    report=rep,
                    output_dir=target_dir,
                )
            if len(reports) > 1:
                save_recovery_benchmark_artifacts(
                    report=reports,
                    output_dir=target_dir,
                    base_filename="recovery_strategy_comparison",
                )

        if parsed.assert_no_regressions and has_regression:
            return 1

        if quality_gate_failed:
            return 1

        return 0
    except Exception as exc:  # noqa: BLE001
        print(f"Error running recovery benchmark: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(run_recovery_cli())
