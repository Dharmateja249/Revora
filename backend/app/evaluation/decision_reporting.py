"""
Revora Decision Evaluation Reporting.

Generates human-readable Markdown reports, structured failure diagnostics,
machine-readable JSON reports, and side-by-side pipeline comparisons
from immutable DecisionBenchmarkReport artifacts.
"""

import json
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.decision_engine import RecoveryAction
from app.evaluation.schemas import DecisionBenchmarkReport


def analyze_decision_failures(
    report: DecisionBenchmarkReport,
) -> list[dict[str, Any]]:
    """
    Extract comprehensive failure diagnostics for all imperfect or anomalous queries.

    Identifies cases where:
    - predicted_action did not exactly match expected_action
    - predicted_action was not acceptable
    - predicted_action violated safety constraints (in prohibited_actions)
    - a policy was violated or overridden
    - deterministic fallback was triggered
    - an execution error occurred

    Args:
        report: DecisionBenchmarkReport instance.

    Returns:
        List of structured failure diagnostic dictionaries.
    """
    failures: list[dict[str, Any]] = []

    for r in report.results:
        is_safety_violation = bool(
            r.prohibited_actions and r.predicted_action in r.prohibited_actions
        )
        is_imperfect = (
            not r.is_exact_match
            or not r.is_acceptable_match
            or is_safety_violation
            or bool(r.violated_policy_ids)
            or r.policy_overridden
            or r.is_fallback
            or r.error is not None
        )

        if is_imperfect:
            failures.append(
                {
                    "query_id": str(r.query_id),
                    "pipeline_name": r.pipeline_name,
                    "expected_action": (
                        r.expected_action.value
                        if isinstance(r.expected_action, RecoveryAction)
                        else str(r.expected_action)
                    ),
                    "predicted_action": (
                        r.predicted_action.value
                        if isinstance(r.predicted_action, RecoveryAction)
                        else str(r.predicted_action)
                    ),
                    "is_exact_match": r.is_exact_match,
                    "is_acceptable_match": r.is_acceptable_match,
                    "is_safety_violation": is_safety_violation,
                    "acceptable_actions": [
                        a.value if isinstance(a, RecoveryAction) else str(a)
                        for a in r.acceptable_actions
                    ],
                    "prohibited_actions": [
                        a.value if isinstance(a, RecoveryAction) else str(a)
                        for a in r.prohibited_actions
                    ],
                    "expected_policy_ids": list(r.expected_policy_ids),
                    "applied_policy_ids": list(r.applied_policy_ids),
                    "violated_policy_ids": list(r.violated_policy_ids),
                    "policy_overridden": r.policy_overridden,
                    "is_fallback": r.is_fallback,
                    "fallback_reason": r.fallback_reason,
                    "confidence": r.confidence,
                    "latency_ms": r.latency_ms,
                    "error": r.error,
                    "key_factors": list(r.key_factors),
                    "referenced_case_ids": list(r.referenced_case_ids),
                }
            )

    return failures


def generate_decision_markdown_report(
    report: DecisionBenchmarkReport,
    include_failures: bool = True,
    max_failure_details: int = 50,
) -> str:
    """
    Format a DecisionBenchmarkReport into a clean, human-readable Markdown document.

    Args:
        report: DecisionBenchmarkReport instance.
        include_failures: Whether to include individual failure diagnostic tables.
        max_failure_details: Maximum number of failure cases to detail individually.

    Returns:
        Formatted Markdown string.
    """
    metrics = dict(report.aggregate_metrics)
    exact_pct = metrics.get("exact_match_rate", 0.0) * 100.0
    acc_pct = metrics.get("acceptable_match_rate", 0.0) * 100.0
    safe_pct = metrics.get("safety_violation_rate", 0.0) * 100.0
    pol_match_pct = metrics.get("policy_match_rate", 0.0) * 100.0
    pol_viol_pct = metrics.get("policy_violation_rate", 0.0) * 100.0
    pol_over_pct = metrics.get("policy_override_rate", 0.0) * 100.0
    fb_pct = metrics.get("fallback_rate", 0.0) * 100.0
    mean_conf = metrics.get("mean_confidence", 0.0)
    mean_lat = metrics.get("mean_latency_ms", 0.0)

    lines: list[str] = [
        f"# Decision Benchmark Report: `{report.pipeline_name}`",
        "",
        "## Executive Summary",
        "",
        "| Metric | Value | Description |",
        "| :--- | :--- | :--- |",
        f"| **Exact Match Rate** | `{exact_pct:.1f}%` | Percentage of decisions exactly matching expected oracle action |",
        f"| **Acceptable Match Rate** | `{acc_pct:.1f}%` | Percentage of decisions within acceptable recovery actions |",
        f"| **Safety Violation Rate** | `{safe_pct:.1f}%` | Percentage of decisions predicting explicitly prohibited actions |",
        f"| **Policy Match Rate** | `{pol_match_pct:.1f}%` | Compliance rate with expected safety and business policies |",
        f"| **Policy Violation Rate** | `{pol_viol_pct:.1f}%` | Rate of policy constraint violations / overrides |",
        f"| **Policy Override Rate** | `{pol_over_pct:.1f}%` | Rate where candidate action was modified by policy rules |",
        f"| **Fallback Rate** | `{fb_pct:.1f}%` | Rate of fallback to deterministic recovery |",
        f"| **Mean Confidence** | `{mean_conf:.3f}` | Average confidence score across evaluated decisions |",
        f"| **Mean Latency** | `{mean_lat:.2f} ms` | Average end-to-end evaluation latency |",
        "",
        "## Benchmark Metadata",
        "",
        f"- **Pipeline:** `{report.pipeline_name}`",
        f"- **Dataset:** `{report.dataset_name}`",
        f"- **Evaluated Queries:** `{report.num_queries}`",
        f"- **Evaluation Version:** `{report.evaluation_version}`",
        f"- **Evaluated At (UTC):** `{report.evaluated_at.isoformat()}`",
        "",
    ]

    if include_failures:
        failures = analyze_decision_failures(report)
        lines.append("## Failure & Safety Diagnostics")
        lines.append("")
        if not failures:
            lines.append(
                "✅ **Zero decision anomalies, safety violations, or policy errors detected.**"
            )
            lines.append("")
        else:
            safety_violations = [f for f in failures if f["is_safety_violation"]]
            fallback_cases = [f for f in failures if f["is_fallback"]]
            error_cases = [f for f in failures if f["error"] is not None]

            lines.append(
                f"- **Total Anomalies / Suboptimal Decisions:** `{len(failures)}` / `{report.num_queries}`"
            )
            lines.append(
                f"- **Safety Violations (Prohibited Actions):** `{len(safety_violations)}`"
            )
            lines.append(f"- **Fallbacks Triggered:** `{len(fallback_cases)}`")
            lines.append(f"- **Execution Errors:** `{len(error_cases)}`")
            lines.append("")
            lines.append("### Detailed Case Diagnostics")
            lines.append("")
            lines.append(
                "| Query ID | Expected | Predicted | Acceptable | Safety Violation | Fallback | Error |"
            )
            lines.append("| :--- | :--- | :--- | :--- | :--- | :--- | :--- |")

            for f in failures[:max_failure_details]:
                q_short = f["query_id"][:8] + "..."
                exp = f"`{f['expected_action']}`"
                pred = f"`{f['predicted_action']}`"
                acc = "✅ Yes" if f["is_acceptable_match"] else "❌ No"
                safe = "🚨 **YES**" if f["is_safety_violation"] else "✅ No"
                fb = (
                    f"⚠️ Yes ({f['fallback_reason'] or 'unspecified'})"
                    if f["is_fallback"]
                    else "No"
                )
                err = f"`{f['error']}`" if f["error"] else "None"

                lines.append(
                    f"| `{q_short}` | {exp} | {pred} | {acc} | {safe} | {fb} | {err} |"
                )

            if len(failures) > max_failure_details:
                lines.append(
                    f"\n*...and {len(failures) - max_failure_details} more failure cases.*"
                )
            lines.append("")

    return "\n".join(lines)


def _normalize_reports_input(
    reports: Sequence[DecisionBenchmarkReport] | Mapping[str, DecisionBenchmarkReport],
) -> dict[str, DecisionBenchmarkReport]:
    """Normalize input reports sequence or mapping into a dictionary keyed by pipeline name."""
    if isinstance(reports, Mapping):
        return dict(reports)
    if isinstance(reports, (list, tuple)):
        result_dict: dict[str, DecisionBenchmarkReport] = {}
        for r in reports:
            if not isinstance(r, DecisionBenchmarkReport):
                raise TypeError(
                    f"Expected DecisionBenchmarkReport in sequence, got {type(r).__name__}"
                )
            result_dict[r.pipeline_name] = r
        return result_dict
    raise TypeError(
        f"reports must be a Sequence or Mapping of DecisionBenchmarkReport, got {type(reports).__name__}"
    )


def compare_decision_pipelines(
    reports: Sequence[DecisionBenchmarkReport] | Mapping[str, DecisionBenchmarkReport],
) -> dict[str, Any]:
    """
    Produce a side-by-side comparison dictionary across multiple DecisionBenchmarkReports.

    Args:
        reports: Sequence or Mapping of DecisionBenchmarkReport instances.

    Returns:
        Structured comparison dictionary containing metrics matrix, deltas, and diagnostic totals.
    """
    norm_reports = _normalize_reports_input(reports)
    if not norm_reports:
        return {
            "num_pipelines": 0,
            "pipelines": [],
            "metrics": {},
            "summary": {},
        }

    pipeline_names = list(norm_reports.keys())
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

    metrics_matrix: dict[str, dict[str, float]] = {k: {} for k in metric_keys}
    summary: dict[str, dict[str, Any]] = {}

    for name, rep in norm_reports.items():
        rep_metrics = dict(rep.aggregate_metrics)
        for k in metric_keys:
            metrics_matrix[k][name] = rep_metrics.get(k, 0.0)

        failures = analyze_decision_failures(rep)
        safety_viol_count = sum(1 for f in failures if f["is_safety_violation"])
        fallback_count = sum(1 for f in failures if f["is_fallback"])
        error_count = sum(1 for f in failures if f["error"] is not None)

        summary[name] = {
            "num_queries": rep.num_queries,
            "total_anomalies": len(failures),
            "safety_violations": safety_viol_count,
            "fallbacks": fallback_count,
            "errors": error_count,
            "dataset_name": rep.dataset_name,
            "evaluated_at": rep.evaluated_at.isoformat(),
        }

    return {
        "num_pipelines": len(pipeline_names),
        "pipelines": pipeline_names,
        "metrics": metrics_matrix,
        "summary": summary,
    }


def generate_decision_comparison_markdown(
    reports: Sequence[DecisionBenchmarkReport] | Mapping[str, DecisionBenchmarkReport],
) -> str:
    """
    Generate a side-by-side Markdown comparison report across multiple decision pipelines.

    Args:
        reports: Sequence or Mapping of DecisionBenchmarkReport instances.

    Returns:
        Formatted Markdown comparison string.
    """
    norm_reports = _normalize_reports_input(reports)
    if not norm_reports:
        return "# Decision Pipeline Comparison\n\n*No benchmark reports provided.*"

    comparison = compare_decision_pipelines(norm_reports)
    pipelines = comparison["pipelines"]
    matrix = comparison["metrics"]
    summary = comparison["summary"]

    lines: list[str] = [
        "# Decision Pipeline Comparison",
        "",
        "## Performance & Accuracy Comparison",
        "",
    ]

    # Header
    pipe_headers = " | ".join(f"**`{p}`**" for p in pipelines)
    align_headers = " | ".join(":---" for _ in pipelines)
    lines.append(f"| Metric | {pipe_headers} |")
    lines.append(f"| :--- | {align_headers} |")

    display_metrics = [
        ("Exact Match Rate", "exact_match_rate", "{:.1f}%", 100.0),
        ("Acceptable Match Rate", "acceptable_match_rate", "{:.1f}%", 100.0),
        ("Safety Violation Rate", "safety_violation_rate", "{:.1f}%", 100.0),
        ("Policy Match Rate", "policy_match_rate", "{:.1f}%", 100.0),
        ("Policy Violation Rate", "policy_violation_rate", "{:.1f}%", 100.0),
        ("Policy Override Rate", "policy_override_rate", "{:.1f}%", 100.0),
        ("Fallback Rate", "fallback_rate", "{:.1f}%", 100.0),
        ("Mean Confidence", "mean_confidence", "{:.3f}", 1.0),
        ("Mean Latency", "mean_latency_ms", "{:.2f} ms", 1.0),
    ]

    for label, key, fmt, multiplier in display_metrics:
        row_vals = []
        for p in pipelines:
            raw_val = matrix.get(key, {}).get(p, 0.0) * multiplier
            formatted = fmt.format(raw_val)
            row_vals.append(f"`{formatted}`")
        lines.append(f"| **{label}** | {' | '.join(row_vals)} |")

    lines.append("")
    lines.append("## Diagnostic Totals")
    lines.append("")
    lines.append(f"| Diagnostic | {pipe_headers} |")
    lines.append(f"| :--- | {align_headers} |")

    diag_rows = [
        ("Evaluated Queries", "num_queries"),
        ("Total Suboptimal Cases", "total_anomalies"),
        ("Safety Violations", "safety_violations"),
        ("Fallbacks Triggered", "fallbacks"),
        ("Execution Errors", "errors"),
    ]

    for label, key in diag_rows:
        row_vals = [f"`{summary[p][key]}`" for p in pipelines]
        lines.append(f"| **{label}** | {' | '.join(row_vals)} |")

    lines.append("")
    return "\n".join(lines)


def generate_decision_json_report(
    reports: DecisionBenchmarkReport
    | Mapping[str, DecisionBenchmarkReport]
    | Sequence[DecisionBenchmarkReport],
    indent: int = 2,
) -> str:
    """
    Serialize single or multiple DecisionBenchmarkReports into a deterministic JSON string.

    Args:
        reports: Single DecisionBenchmarkReport, Sequence, or Mapping.
        indent: JSON indentation spaces.

    Returns:
        Serialized JSON string.
    """
    if isinstance(reports, DecisionBenchmarkReport):
        data = reports.model_dump(mode="json")
    else:
        norm_reports = _normalize_reports_input(reports)
        data = {name: rep.model_dump(mode="json") for name, rep in norm_reports.items()}

    return json.dumps(data, indent=indent, sort_keys=True)


def save_decision_benchmark_artifacts(
    report: DecisionBenchmarkReport
    | Sequence[DecisionBenchmarkReport]
    | Mapping[str, DecisionBenchmarkReport],
    output_dir: Path | str,
    base_filename: str | None = None,
) -> tuple[Path, Path]:
    """
    Persist Markdown and JSON evaluation artifacts atomically to disk.

    Args:
        report: Single DecisionBenchmarkReport, Sequence, or Mapping.
        output_dir: Target directory Path or string.
        base_filename: Optional base filename prefix (without extension).

    Returns:
        Tuple of (json_path, markdown_path).
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    timestamp_str = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    if isinstance(report, DecisionBenchmarkReport):
        prefix = (
            base_filename
            or f"decision_benchmark_{report.pipeline_name}_{timestamp_str}"
        )
        md_content = generate_decision_markdown_report(report)
        json_content = generate_decision_json_report(report)
    else:
        prefix = base_filename or f"decision_comparison_{timestamp_str}"
        md_content = generate_decision_comparison_markdown(report)
        json_content = generate_decision_json_report(report)

    json_file = out_path / f"{prefix}.json"
    md_file = out_path / f"{prefix}.md"

    # Write atomically
    temp_json = out_path / f"{prefix}.json.tmp"
    temp_md = out_path / f"{prefix}.md.tmp"

    temp_json.write_text(json_content, encoding="utf-8")
    temp_md.write_text(md_content, encoding="utf-8")

    temp_json.replace(json_file)
    temp_md.replace(md_file)

    return json_file, md_file
