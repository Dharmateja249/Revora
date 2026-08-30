"""
Revora Retrieval Evaluation CLI.

Command-line entrypoint for running golden benchmarks, persisting evaluation artifacts,
comparing benchmark runs, and asserting regression safety in CI.
"""

import argparse
import sys
from typing import Optional

from app.evaluation.benchmark import run_benchmark
from app.evaluation.persistence import (
    get_evaluation_directory,
    load_latest_report,
    load_report,
    save_report,
)
from app.evaluation.regression import (
    RegressionThresholds,
    assert_no_regressions,
    compare_reports,
)
from app.evaluation.reporting import (
    create_evaluation_report,
    generate_json_report,
    generate_markdown_report,
    save_benchmark_artifacts,
)


def run_cli(args: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Revora Retrieval Evaluation & Regression Detection CLI",
    )
    parser.add_argument("--run", action="store_true", help="Run benchmark across all production retrievers.")
    parser.add_argument("--save", action="store_true", help="Persist evaluation report and markdown artifacts.")
    parser.add_argument("--compare-baseline", type=str, default=None, help="Baseline report_id or path to compare against.")
    parser.add_argument("--assert-no-regressions", action="store_true", help="Exit non-zero if regressions are detected.")
    parser.add_argument("--json", action="store_true", help="Print JSON report to stdout.")
    parser.add_argument("--markdown", action="store_true", help="Print Markdown report to stdout.")

    parsed = parser.parse_args(args)

    if not (parsed.run or parsed.compare_baseline or parsed.save or parsed.json or parsed.markdown):
        parsed.run = True
        parsed.markdown = True

    reports = run_benchmark()
    eval_report = create_evaluation_report(reports)

    reg_report = None
    if parsed.compare_baseline:
        try:
            baseline_report = load_report(parsed.compare_baseline)
            reg_report = compare_reports(baseline=baseline_report, candidate=eval_report)
        except Exception as e:
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

    if parsed.assert_no_regressions and reg_report:
        try:
            assert_no_regressions(reg_report)
        except Exception as e:
            print(str(e), file=sys.stderr)
            return 1

    return 0


if __name__ == "__main__":
    sys.exit(run_cli())
