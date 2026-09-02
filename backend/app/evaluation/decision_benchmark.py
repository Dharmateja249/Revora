"""
Revora Reproducible Decision Benchmark Runner & CLI.

Orchestrates end-to-end benchmark execution across deterministic and agent recovery pipelines
using the golden evaluation dataset, producing structured JSON and Markdown artifacts.
"""

import argparse
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from app.agent.orchestrator import AgentOrchestrator
from app.evaluation.decision_evaluator import (
    AgentRAGPipeline,
    DecisionEvaluator,
    DecisionPipeline,
    DeterministicBaselinePipeline,
    DeterministicRAGPipeline,
)
from app.evaluation.decision_reporting import (
    generate_decision_comparison_markdown,
    generate_decision_json_report,
    generate_decision_markdown_report,
    save_decision_benchmark_artifacts,
)
from app.evaluation.persistence import get_evaluation_directory
from app.evaluation.schemas import DecisionBenchmarkReport, EvaluationCase


def resolve_decision_pipeline(
    name_or_instance: str | DecisionPipeline | Any,
    agent_orchestrator: AgentOrchestrator | None = None,
    allow_default_evaluation_orchestrator: bool = True,
    llm_provider: str = "mock",
) -> Any:
    """
    Resolve a pipeline name or instance into an executable DecisionPipeline adapter.

    Args:
        name_or_instance: String identifier or pipeline object.
        agent_orchestrator: Optional AgentOrchestrator instance for agent pipelines.
        allow_default_evaluation_orchestrator: Whether to auto-construct an offline evaluation
            orchestrator if agent_orchestrator is not provided.
        llm_provider: LLM provider identifier for agent pipelines ('mock' or 'openai'). Defaults to 'mock'.

    Returns:
        Executable pipeline adapter.
    """
    if not isinstance(name_or_instance, str):
        return name_or_instance

    norm_name = name_or_instance.strip().lower()
    if norm_name in ("deterministic_baseline", "baseline", "default"):
        return DeterministicBaselinePipeline()
    if norm_name in ("deterministic_rag", "rag"):
        return DeterministicRAGPipeline()
    if norm_name in ("agent_rag", "agent", "orchestrator", "openai_agent_rag"):
        pipeline_name = (
            "openai_agent_rag" if norm_name == "openai_agent_rag" else "agent_rag"
        )
        effective_provider = (
            "openai"
            if norm_name == "openai_agent_rag"
            else llm_provider.strip().lower()
        )
        if effective_provider not in {"mock", "openai"}:
            raise ValueError(
                f"Unsupported LLM provider: '{effective_provider}'. Supported providers are: 'mock', 'openai'."
            )
        if agent_orchestrator is None:
            if not allow_default_evaluation_orchestrator:
                raise ValueError(
                    "AgentRAGPipeline requires an active AgentOrchestrator instance."
                )
            if effective_provider == "openai":
                from app.agent.factory import create_llm_provider

                provider = create_llm_provider(provider="openai")
                agent_orchestrator = AgentOrchestrator(provider=provider)
            else:
                from app.evaluation.agent_evaluation_provider import (
                    create_evaluation_agent_orchestrator,
                )

                agent_orchestrator = create_evaluation_agent_orchestrator()
        return AgentRAGPipeline(
            agent_orchestrator=agent_orchestrator, name=pipeline_name
        )

    raise ValueError(f"Unknown decision pipeline identifier: '{name_or_instance}'")


def run_decision_benchmark(
    evaluation_cases: Sequence[EvaluationCase] | None = None,
    pipelines: Sequence[str | DecisionPipeline | Any] | None = None,
    dataset_name: str = "retrieval_golden_dataset_50",
    output_dir: Path | str | None = None,
    save_artifacts: bool = True,
    agent_orchestrator: AgentOrchestrator | None = None,
    llm_provider: str = "mock",
) -> dict[str, DecisionBenchmarkReport]:
    """
    Execute reproducible decision evaluation across one or more recovery pipelines.

    Args:
        evaluation_cases: Optional pre-loaded sequence of EvaluationCase objects with ground truth.
        pipelines: Optional sequence of pipeline names or instances (defaults to baseline + RAG).
        dataset_name: Dataset identifier for metadata reporting.
        output_dir: Target directory for persisted artifacts (defaults to evaluation_results/decisions).
        save_artifacts: Whether to write JSON and Markdown benchmark files to disk.
        agent_orchestrator: Optional orchestrator instance for agent pipelines.
        llm_provider: LLM provider identifier for agent pipelines ('mock' or 'openai'). Defaults to 'mock'.

    Returns:
        Dictionary mapping pipeline_name -> DecisionBenchmarkReport.
    """
    cases = evaluation_cases
    if cases is None:
        from tests.fixtures.retrieval_golden_dataset import (
            get_golden_evaluation_cases,
        )

        cases = get_golden_evaluation_cases()

    evaluator = DecisionEvaluator(
        evaluation_cases=cases,
        dataset_name=dataset_name,
    )

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

    reports: dict[str, DecisionBenchmarkReport] = {}
    for pipe in resolved_pipelines:
        rep = evaluator.evaluate(pipe)
        reports[rep.pipeline_name] = rep

    if save_artifacts:
        target_dir = (
            Path(output_dir)
            if output_dir is not None
            else get_evaluation_directory() / "decisions"
        )
        target_dir.mkdir(parents=True, exist_ok=True)

        for rep in reports.values():
            save_decision_benchmark_artifacts(
                report=rep,
                output_dir=target_dir,
            )

        if len(reports) > 1:
            save_decision_benchmark_artifacts(
                report=reports,
                output_dir=target_dir,
                base_filename="decision_pipeline_comparison",
            )

    return reports


def format_decision_benchmark_terminal_summary(
    reports: Mapping[str, DecisionBenchmarkReport] | Sequence[DecisionBenchmarkReport],
) -> str:
    """
    Generate a clean, ASCII terminal summary table for interactive benchmark inspection.

    Args:
        reports: Mapping or Sequence of DecisionBenchmarkReport instances.

    Returns:
        Formatted ASCII summary string.
    """
    report_dict: dict[str, DecisionBenchmarkReport] = {}
    if isinstance(reports, Mapping):
        report_dict = dict(reports)
    elif isinstance(reports, (list, tuple)):
        for r in reports:
            report_dict[r.pipeline_name] = r

    if not report_dict:
        return "No decision benchmark reports to display."

    lines = [
        "=" * 100,
        " REVORA DECISION BENCHMARK SUMMARY",
        "=" * 100,
        f"{'Pipeline':<24} | {'Queries':<7} | {'Exact Match':<11} | {'Acceptable':<10} | {'Safety Viol':<11} | {'Fallback':<8} | {'Latency (ms)':<12}",
        "-" * 100,
    ]

    for name, rep in report_dict.items():
        m = rep.aggregate_metrics
        exact = f"{m.get('exact_match_rate', 0.0) * 100.0:.1f}%"
        acc = f"{m.get('acceptable_match_rate', 0.0) * 100.0:.1f}%"
        safe = f"{m.get('safety_violation_rate', 0.0) * 100.0:.1f}%"
        fb = f"{m.get('fallback_rate', 0.0) * 100.0:.1f}%"
        lat = f"{m.get('mean_latency_ms', 0.0):.2f}"
        q_cnt = str(rep.num_queries)

        lines.append(
            f"{name:<24} | {q_cnt:<7} | {exact:<11} | {acc:<10} | {safe:<11} | {fb:<8} | {lat:<12}"
        )

    lines.append("=" * 100)
    return "\n".join(lines)


def run_decision_cli(
    args: list[str] | None = None,
    evaluation_cases: Sequence[EvaluationCase] | None = None,
) -> int:
    """
    Command-line interface entrypoint for running reproducible decision benchmarks.

    Args:
        args: Optional list of CLI argument strings.
        evaluation_cases: Optional pre-loaded evaluation cases for test injection.

    Returns:
        Exit code (0 for success, 1 for error).
    """
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except (AttributeError, ValueError, OSError):
                continue

    parser = argparse.ArgumentParser(
        prog="revora-decision-benchmark",
        description="Revora Decision Benchmark & Pipeline Quality Evaluation CLI",
    )
    parser.add_argument(
        "--pipeline",
        "-p",
        action="append",
        dest="pipelines",
        help="Specify one or more pipelines to evaluate (deterministic_baseline, deterministic_rag, agent_rag).",
    )
    parser.add_argument(
        "--llm-provider",
        type=str,
        default="mock",
        choices=["mock", "openai"],
        help="LLM provider for agent pipelines ('mock' for offline evaluation, 'openai' for live OpenAI). Defaults to 'mock'.",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        default=None,
        help="Directory where JSON and Markdown benchmark reports will be written.",
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
        help="Print machine-readable JSON benchmark report to stdout.",
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
        "--min-exact-match",
        type=float,
        default=None,
        help="Minimum required exact match rate in [0.0, 1.0].",
    )
    parser.add_argument(
        "--min-acceptable-match",
        type=float,
        default=None,
        help="Minimum required acceptable match rate in [0.0, 1.0].",
    )
    parser.add_argument(
        "--max-safety-violation",
        type=float,
        default=None,
        help="Maximum allowed safety violation rate (defaults to 0.0).",
    )
    parser.add_argument(
        "--max-policy-violation",
        type=float,
        default=None,
        help="Maximum allowed policy violation rate.",
    )
    parser.add_argument(
        "--max-fallback",
        type=float,
        default=None,
        help="Maximum allowed fallback rate.",
    )
    parser.add_argument(
        "--max-latency",
        type=float,
        default=None,
        help="Maximum allowed mean latency in milliseconds.",
    )
    parser.add_argument(
        "--compare-baseline",
        type=str,
        default=None,
        help="Path or ID of a baseline decision report to compare against (e.g. 'latest').",
    )
    parser.add_argument(
        "--assert-no-regressions",
        action="store_true",
        help="Exit non-zero if comparative regression is detected against the baseline.",
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
        pre_resolved_baselines: dict[str, DecisionBenchmarkReport] = {}
        if parsed.compare_baseline:
            from app.evaluation.decision_persistence import (
                load_decision_report,
                load_latest_decision_report,
            )

            pipe_names = (
                parsed.pipelines
                if parsed.pipelines
                else ["deterministic_baseline", "deterministic_rag"]
            )
            for p_name in pipe_names:
                if parsed.compare_baseline == "latest":
                    b_rep = load_latest_decision_report(
                        pipeline_name=p_name,
                        directory=parsed.output_dir,
                    )
                else:
                    b_rep = load_decision_report(
                        parsed.compare_baseline,
                        directory=parsed.output_dir,
                    )
                if b_rep is not None:
                    pre_resolved_baselines[p_name] = b_rep

        # 2. Run / create candidate benchmark without persisting prematurely
        reports = run_decision_benchmark(
            evaluation_cases=evaluation_cases,
            pipelines=parsed.pipelines,
            output_dir=parsed.output_dir,
            save_artifacts=False,
            llm_provider=parsed.llm_provider,
        )

        if not parsed.quiet and not parsed.json and not parsed.markdown:
            print(format_decision_benchmark_terminal_summary(reports))

        if parsed.json:
            print(generate_decision_json_report(reports))

        if parsed.markdown:
            if len(reports) == 1:
                single_rep = next(iter(reports.values()))
                print(generate_decision_markdown_report(single_rep))
            else:
                print(generate_decision_comparison_markdown(reports))

        # 3. Compare candidate against retained baseline
        has_regression = False
        if parsed.compare_baseline:
            from app.evaluation.decision_persistence import (
                load_decision_report,
                load_latest_decision_report,
            )
            from app.evaluation.decision_regression import compare_decision_runs

            for rep in reports.values():
                b_rep = pre_resolved_baselines.get(rep.pipeline_name)
                if b_rep is None:
                    if parsed.compare_baseline == "latest":
                        b_rep = load_latest_decision_report(
                            pipeline_name=rep.pipeline_name,
                            directory=parsed.output_dir,
                            exclude_report_id=rep.report_id,
                        )
                    else:
                        b_rep = load_decision_report(
                            parsed.compare_baseline,
                            directory=parsed.output_dir,
                        )
                if b_rep is None:
                    raise FileNotFoundError(
                        f"No baseline report found on disk for pipeline '{rep.pipeline_name}'."
                    )

                comp_res = compare_decision_runs(
                    current_report=rep,
                    baseline_report=b_rep,
                )
                if not parsed.quiet:
                    print(
                        f"\nBaseline Comparison for '{rep.pipeline_name}': "
                        f"{'PASS' if comp_res.passed else 'REGRESSION DETECTED'}"
                    )
                    if comp_res.violations:
                        for v in comp_res.violations:
                            print(f"  - {v}")

                if not comp_res.passed:
                    has_regression = True

        # 4. Evaluate quality gate assertions
        quality_gate_failed = False
        quality_gate_requested = bool(
            parsed.assert_quality_gate
            or parsed.min_exact_match is not None
            or parsed.min_acceptable_match is not None
            or parsed.max_safety_violation is not None
            or parsed.max_policy_violation is not None
            or parsed.max_fallback is not None
            or parsed.max_latency is not None
        )

        if quality_gate_requested:
            from app.evaluation.decision_regression import (
                DecisionQualityThresholds,
                evaluate_decision_quality_gate,
                format_quality_gate_terminal_summary,
            )

            thresholds = DecisionQualityThresholds(
                min_exact_match_rate=parsed.min_exact_match,
                min_acceptable_match_rate=parsed.min_acceptable_match,
                max_safety_violation_rate=(
                    parsed.max_safety_violation
                    if parsed.max_safety_violation is not None
                    else 0.0
                ),
                max_policy_violation_rate=parsed.max_policy_violation,
                max_fallback_rate=parsed.max_fallback,
                max_mean_latency_ms=parsed.max_latency,
            )

            for rep in reports.values():
                gate_res = evaluate_decision_quality_gate(rep, thresholds=thresholds)
                if not parsed.quiet:
                    print("\n" + format_quality_gate_terminal_summary(gate_res))
                if not gate_res.passed:
                    quality_gate_failed = True

        # 5. Persist / publish candidate and latest appropriately
        if not parsed.no_save:
            from app.evaluation.decision_persistence import (
                get_decision_evaluation_directory,
            )
            from app.evaluation.decision_reporting import (
                save_decision_benchmark_artifacts,
            )

            target_dir = get_decision_evaluation_directory(parsed.output_dir)
            for rep in reports.values():
                save_decision_benchmark_artifacts(
                    report=rep,
                    output_dir=target_dir,
                )

            if len(reports) > 1:
                save_decision_benchmark_artifacts(
                    report=reports,
                    output_dir=target_dir,
                    base_filename="decision_pipeline_comparison",
                )

        if parsed.assert_no_regressions and has_regression:
            return 1

        if quality_gate_failed:
            return 1

        return 0
    except Exception as exc:  # noqa: BLE001
        print(f"Error running decision benchmark: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(run_decision_cli())
