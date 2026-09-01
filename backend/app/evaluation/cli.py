"""
Revora Retrieval Evaluation CLI.

Command-line entrypoint for running golden benchmarks, persisting evaluation artifacts,
comparing benchmark runs, and asserting regression safety in CI.
"""

import argparse
import sys
from collections.abc import Sequence
from typing import Any

from app.evaluation.benchmark import run_benchmark
from app.evaluation.persistence import (
    get_evaluation_directory,
    load_report,
    save_report,
)
from app.evaluation.regression import (
    assert_no_regressions,
    compare_reports,
)
from app.evaluation.reporting import (
    create_evaluation_report,
    generate_json_report,
    generate_markdown_report,
    save_benchmark_artifacts,
)
from app.evaluation.schemas import EvaluationRegressionError


def run_cli(
    args: list[str] | None = None, evaluation_cases: Sequence[Any] | None = None
) -> int:
    parser = argparse.ArgumentParser(
        description="Revora Retrieval Evaluation & Regression Detection CLI",
    )
    parser.add_argument(
        "--run",
        action="store_true",
        help="Run benchmark across all production retrievers.",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Persist evaluation report and markdown artifacts.",
    )
    parser.add_argument(
        "--compare-baseline",
        type=str,
        default=None,
        help="Baseline report_id or path to compare against.",
    )
    parser.add_argument(
        "--assert-no-regressions",
        action="store_true",
        help="Exit non-zero if regressions are detected.",
    )
    parser.add_argument(
        "--json", action="store_true", help="Print JSON report to stdout."
    )
    parser.add_argument(
        "--markdown", action="store_true", help="Print Markdown report to stdout."
    )

    parsed = parser.parse_args(args)

    if parsed.assert_no_regressions and not parsed.compare_baseline:
        print(
            "Error: --assert-no-regressions requires --compare-baseline <baseline_id_or_path>.",
            file=sys.stderr,
        )
        return 1

    if not (
        parsed.run
        or parsed.compare_baseline
        or parsed.save
        or parsed.json
        or parsed.markdown
    ):
        parsed.run = True
        parsed.markdown = True

    cases = evaluation_cases
    if cases is None:
        try:
            # Lazy import for test fixture dataset when running CLI
            from tests.fixtures.retrieval_golden_dataset import (
                get_golden_evaluation_cases,
            )

            cases = get_golden_evaluation_cases()
        except ImportError:
            print(
                "Error: No evaluation_cases provided and golden dataset fixture could not be imported.",
                file=sys.stderr,
            )
            return 1

    reports = run_benchmark(evaluation_cases=cases)
    eval_report = create_evaluation_report(reports)

    reg_report = None
    if parsed.compare_baseline:
        try:
            baseline_report = load_report(parsed.compare_baseline)
            reg_report = compare_reports(
                baseline=baseline_report, candidate=eval_report
            )
        except (FileNotFoundError, ValueError, OSError) as e:
            print(f"Error loading/comparing baseline: {e}", file=sys.stderr)
            return 1

    if parsed.save:
        eval_dir = get_evaluation_directory()
        save_report(eval_report, directory=eval_dir, overwrite=True)
        save_benchmark_artifacts(reports=reports, output_dir=eval_dir, regressions=None)
        print(f"Saved evaluation artifacts to {eval_dir}", file=sys.stderr)

    if parsed.json:
        print(generate_json_report(eval_report, regressions=reg_report))
    elif parsed.markdown or (not parsed.json and not parsed.save):
        print(generate_markdown_report(eval_report, regressions=reg_report))

    if parsed.assert_no_regressions:
        if reg_report is None:
            print(
                "Error: No regression analysis available to assert against.",
                file=sys.stderr,
            )
            return 1
        try:
            assert_no_regressions(reg_report)
        except EvaluationRegressionError as e:
            print(str(e), file=sys.stderr)
            return 1

    return 0


if __name__ == "__main__":
    sys.exit(run_cli())
