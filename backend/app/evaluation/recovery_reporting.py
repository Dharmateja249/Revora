"""
Revora Recovery Outcome Reporting.

Generates human-readable Markdown reports, category breakdowns, machine-readable JSON reports,
and side-by-side pipeline financial comparisons for recovery benchmark results.
"""

import json
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.evaluation.recovery_schemas import RecoveryBenchmarkReport


def generate_recovery_markdown_report(
    report: RecoveryBenchmarkReport,
    include_categories: bool = True,
) -> str:
    """
    Generate a formatted Markdown report from a RecoveryBenchmarkReport.

    Args:
        report: RecoveryBenchmarkReport instance.
        include_categories: Whether to include failure category breakdown table.

    Returns:
        Formatted Markdown string.
    """
    rec_pct = report.recovery_rate * 100.0
    pol_pct = report.policy_violation_rate * 100.0
    stp_pct = report.stopping_rule_violation_rate * 100.0
    unnec_pct = report.unnecessary_intervention_rate * 100.0
    dup_pct = report.duplicate_action_rate * 100.0

    lines: list[str] = [
        f"# Recovery Outcome Benchmark Report: `{report.pipeline_name}`",
        "",
        "## Financial Summary",
        "",
        "| Metric | Value | Description |",
        "| :--- | :--- | :--- |",
        f"| **Total Attempted Revenue** | `₹{report.total_attempted_revenue:,.2f}` | Total principal value of all failed payments |",
        f"| **Total Recoverable Revenue** | `₹{report.total_recoverable_revenue:,.2f}` | Theoretical maximum recoverable principal |",
        f"| **Gross Recovered Revenue** | `₹{report.gross_recovered_amount:,.2f}` | Total principal successfully recovered |",
        f"| **Intervention Cost** | `₹{report.total_intervention_cost:,.2f}` | Total operational and gateway intervention costs |",
        f"| **Net Recovered Revenue** | `₹{report.net_recovered_amount:,.2f}` | Gross recovered revenue minus intervention costs |",
        f"| **Recovery Rate** | `{rec_pct:.1f}%` | Fraction of recoverable revenue successfully captured |",
        f"| **Average Recovered / Case** | `₹{report.average_recovered_per_case:,.2f}` | Mean recovered revenue per payment scenario |",
        f"| **Average Net / Case** | `₹{report.average_net_per_case:,.2f}` | Mean net financial gain per payment scenario |",
        "",
        "## Policy & Operational Compliance",
        "",
        "| Compliance Metric | Rate | Status |",
        "| :--- | :--- | :--- |",
        f"| **Policy Violation Rate** | `{pol_pct:.1f}%` | {'✅ COMPLIANT' if pol_pct == 0.0 else '🚨 VIOLATIONS DETECTED'} |",
        f"| **Stopping Rule Violation Rate** | `{stp_pct:.1f}%` | {'✅ COMPLIANT' if stp_pct == 0.0 else '⚠️ VIOLATIONS DETECTED'} |",
        f"| **Unnecessary Intervention Rate**| `{unnec_pct:.1f}%` | Interventions on permanently non-recoverable debts |",
        f"| **Duplicate Action Rate** | `{dup_pct:.1f}%` | Repeated identical interventions without delay |",
        "",
        "## Benchmark Metadata",
        "",
        f"- **Pipeline:** `{report.pipeline_name}`",
        f"- **Dataset:** `{report.dataset_name}`",
        f"- **Scenarios Evaluated:** `{report.num_scenarios}`",
        f"- **Evaluated At (UTC):** `{report.evaluated_at.isoformat()}`",
        "",
    ]

    if include_categories and report.category_breakdown:
        lines.append("## Failure Category Breakdown")
        lines.append("")
        lines.append(
            "| Category | Scenarios | Attempted | Recoverable | Recovered | Cost | Net | Recovery Rate |"
        )
        lines.append("| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |")

        for cat, stats in sorted(report.category_breakdown.items()):
            cnt = int(stats.get("count", 0))
            att = stats.get("attempted", 0.0)
            recable = stats.get("recoverable", 0.0)
            rec = stats.get("recovered", 0.0)
            cst = stats.get("cost", 0.0)
            net = stats.get("net", 0.0)
            rate = stats.get("recovery_rate", 0.0) * 100.0

            lines.append(
                f"| `{cat}` | {cnt} | `₹{att:,.2f}` | `₹{recable:,.2f}` | `₹{rec:,.2f}` | `₹{cst:,.2f}` | `₹{net:,.2f}` | `{rate:.1f}%` |"
            )
        lines.append("")

    return "\n".join(lines)


def _normalize_recovery_reports(
    reports: Sequence[RecoveryBenchmarkReport] | Mapping[str, RecoveryBenchmarkReport],
) -> dict[str, RecoveryBenchmarkReport]:
    if isinstance(reports, Mapping):
        return dict(reports)
    if isinstance(reports, (list, tuple)):
        result: dict[str, RecoveryBenchmarkReport] = {}
        for r in reports:
            if not isinstance(r, RecoveryBenchmarkReport):
                raise TypeError(
                    f"Expected RecoveryBenchmarkReport, got {type(r).__name__}"
                )
            result[r.pipeline_name] = r
        return result
    raise TypeError(
        f"reports must be Sequence or Mapping of RecoveryBenchmarkReport, got {type(reports).__name__}"
    )


def compare_recovery_pipelines(
    reports: Sequence[RecoveryBenchmarkReport] | Mapping[str, RecoveryBenchmarkReport],
) -> dict[str, Any]:
    """
    Produce a side-by-side financial and operational comparison dictionary across multiple RecoveryBenchmarkReports.

    Args:
        reports: Sequence or Mapping of RecoveryBenchmarkReport instances.

    Returns:
        Structured comparison matrix dictionary.
    """
    norm = _normalize_recovery_reports(reports)
    if not norm:
        return {"num_pipelines": 0, "pipelines": [], "metrics": {}}

    pipelines = list(norm.keys())
    metric_keys = [
        "total_attempted_revenue",
        "total_recoverable_revenue",
        "gross_recovered_amount",
        "total_intervention_cost",
        "net_recovered_amount",
        "recovery_rate",
        "average_recovered_per_case",
        "average_net_per_case",
        "policy_violation_rate",
        "stopping_rule_violation_rate",
        "unnecessary_intervention_rate",
        "duplicate_action_rate",
    ]

    matrix: dict[str, dict[str, float]] = {k: {} for k in metric_keys}

    for name, rep in norm.items():
        for k in metric_keys:
            matrix[k][name] = getattr(rep, k, 0.0)

    return {
        "num_pipelines": len(pipelines),
        "pipelines": pipelines,
        "metrics": matrix,
    }


def generate_recovery_comparison_markdown(
    reports: Sequence[RecoveryBenchmarkReport] | Mapping[str, RecoveryBenchmarkReport],
    baseline_pipeline: str = "deterministic_baseline",
) -> str:
    """
    Generate a comprehensive Markdown strategy comparison and outcome report.

    Sections:
    1. Executive Summary
    2. Recovery Leaderboard
    3. Financial Comparison (Gross, Cost, Net, ROI, CPRD)
    4. Recovery-Rate & Operational Metrics
    5. Failure-Category Cross-Strategy Breakdown
    6. Safety & Compliance Telemetry
    7. Baseline Uplift Analysis
    """
    from app.evaluation.recovery_comparison import (
        compare_category_recovery_performance,
        compute_recovery_strategy_uplift,
        generate_recovery_leaderboard,
    )

    norm = _normalize_recovery_reports(reports)
    if not norm:
        return "# Recovery Pipeline Comparison\n\n*No benchmark reports provided.*"

    leaderboard = generate_recovery_leaderboard(norm)
    categories = compare_category_recovery_performance(norm)
    pipelines = list(norm.keys())

    lines: list[str] = [
        "# Revora Recovery Strategy Benchmark & Outcome Comparison",
        "",
        "## 1. Executive Summary",
        "",
        f"Evaluated **{len(pipelines)}** recovery pipelines across **{next(iter(norm.values())).num_scenarios}** synthetic failed payment scenarios.",
        f"Top-performing compliant strategy: **`{leaderboard[0].pipeline_name}`** with **₹{leaderboard[0].net_recovered:,.2f}** net recovered revenue (`{leaderboard[0].recovery_rate:.1%}` capture rate).",
        "",
        "## 2. Recovery Strategy Leaderboard",
        "",
        "| Rank | Pipeline | Net Recovered | Gross Recovered | Intervention Cost | Recovery Rate | ROI | Cost / ₹ Recovered | Status |",
        "| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
    ]

    import math

    for entry in leaderboard:
        status_badge = "✅ Compliant" if entry.is_compliant else "🚨 Violations"
        roi_str = "∞" if math.isinf(entry.roi) else f"{entry.roi:,.1f}x"
        lines.append(
            f"| **#{entry.rank}** | `{entry.pipeline_name}` | **₹{entry.net_recovered:,.2f}** | ₹{entry.gross_recovered:,.2f} | ₹{entry.intervention_cost:,.2f} | `{entry.recovery_rate:.1%}` | `{roi_str}` | `₹{entry.cost_per_recovered_dollar:.3f}` | {status_badge} |"
        )
    lines.append("")

    # 3. Financial Comparison
    lines.extend(
        [
            "## 3. Financial Comparison & Efficiency",
            "",
            f"| Metric | {' | '.join(f'**`{p}`**' for p in pipelines)} |",
            f"| :--- | {' | '.join(':---' for _ in pipelines)} |",
        ]
    )

    financial_metrics = [
        ("Total Attempted Revenue", "total_attempted_revenue", "₹{:,.2f}"),
        ("Total Recoverable Revenue", "total_recoverable_revenue", "₹{:,.2f}"),
        ("Gross Recovered Revenue", "gross_recovered_amount", "₹{:,.2f}"),
        ("Total Intervention Cost", "total_intervention_cost", "₹{:,.2f}"),
        ("Net Recovered Revenue", "net_recovered_amount", "₹{:,.2f}"),
        ("Average Net / Case", "average_net_per_case", "₹{:,.2f}"),
    ]
    for label, attr, fmt in financial_metrics:
        row_vals = [f"`{fmt.format(getattr(norm[p], attr, 0.0))}`" for p in pipelines]
        lines.append(f"| **{label}** | {' | '.join(row_vals)} |")
    lines.append("")

    # 4. Failure Category Breakdown
    if categories:
        lines.extend(
            [
                "## 4. Failure Category Performance Breakdown",
                "",
                "Net revenue recovered by category across strategies:",
                "",
                f"| Failure Category | {' | '.join(f'**`{p}` (Net / Rate)**' for p in pipelines)} |",
                f"| :--- | {' | '.join(':---' for _ in pipelines)} |",
            ]
        )

        for cat, pipe_data in sorted(categories.items()):
            row_vals = []
            for p in pipelines:
                st = pipe_data.get(p, {})
                net_val = st.get("net", 0.0)
                rate_val = st.get("recovery_rate", 0.0) * 100.0
                row_vals.append(f"`₹{net_val:,.2f}` ({rate_val:.0f}%)")
            lines.append(f"| `{cat}` | {' | '.join(row_vals)} |")
        lines.append("")

    # 5. Safety & Compliance Metrics
    lines.extend(
        [
            "## 5. Safety & Operational Compliance",
            "",
            f"| Compliance Metric | {' | '.join(f'**`{p}`**' for p in pipelines)} |",
            f"| :--- | {' | '.join(':---' for _ in pipelines)} |",
        ]
    )

    compliance_metrics = [
        ("Policy Violation Rate", "policy_violation_rate", "{:.1%}"),
        ("Stopping Rule Violations", "stopping_rule_violation_rate", "{:.1%}"),
        ("Unnecessary Interventions", "unnecessary_intervention_rate", "{:.1%}"),
        ("Duplicate Action Rate", "duplicate_action_rate", "{:.1%}"),
    ]
    for label, attr, fmt in compliance_metrics:
        row_vals = [f"`{fmt.format(getattr(norm[p], attr, 0.0))}`" for p in pipelines]
        lines.append(f"| **{label}** | {' | '.join(row_vals)} |")
    lines.append("")

    # 6. Baseline Uplift
    baseline_rep = norm.get(baseline_pipeline)
    if baseline_rep is not None and len(norm) > 1:
        candidates = [p for p in pipelines if p != baseline_pipeline]
        if candidates:
            lines.extend(
                [
                    f"## 6. Strategy Uplift vs Baseline (`{baseline_pipeline}`)",
                    "",
                    f"| Uplift Metric | {' | '.join(f'**`{p}`**' for p in candidates)} |",
                    f"| :--- | {' | '.join(':---' for _ in candidates)} |",
                ]
            )

            uplift_rows = [
                ("Gross Recovery Uplift", "gross_recovery_uplift", "₹{:,.2f}"),
                ("Gross Uplift (%)", "gross_recovery_uplift_pct", "{:+.1%}"),
                ("Net Recovery Uplift", "net_recovery_uplift", "₹{:,.2f}"),
                ("Net Uplift (%)", "net_recovery_uplift_pct", "{:+.1%}"),
                ("Recovery Rate Uplift", "recovery_rate_uplift", "{:+.1%}"),
                ("Incremental Revenue", "incremental_revenue_recovered", "₹{:,.2f}"),
                ("Incremental Cost", "incremental_intervention_cost", "₹{:,.2f}"),
                ("Cases Improved", "improved_cases_pct", "{:.1%}"),
                ("Cases Worsened", "worsened_cases_pct", "{:.1%}"),
            ]

            uplifts = {
                p: compute_recovery_strategy_uplift(norm[p], baseline_rep)
                for p in candidates
            }

            for label, attr, fmt in uplift_rows:
                row_vals = [
                    f"`{fmt.format(getattr(uplifts[p], attr, 0.0))}`"
                    for p in candidates
                ]
                lines.append(f"| **{label}** | {' | '.join(row_vals)} |")
            lines.append("")

    return "\n".join(lines)


def generate_recovery_json_report(
    reports: RecoveryBenchmarkReport
    | Mapping[str, RecoveryBenchmarkReport]
    | Sequence[RecoveryBenchmarkReport],
    indent: int = 2,
) -> str:
    """
    Serialize single or multiple RecoveryBenchmarkReports into a deterministic JSON string.

    Args:
        reports: Single RecoveryBenchmarkReport, Sequence, or Mapping.
        indent: JSON indentation spaces.

    Returns:
        Serialized JSON string.
    """
    if isinstance(reports, RecoveryBenchmarkReport):
        data = reports.model_dump(mode="json")
    else:
        norm = _normalize_recovery_reports(reports)
        data = {name: rep.model_dump(mode="json") for name, rep in norm.items()}

    return json.dumps(data, indent=indent, sort_keys=True)


def save_recovery_benchmark_artifacts(
    report: RecoveryBenchmarkReport
    | Sequence[RecoveryBenchmarkReport]
    | Mapping[str, RecoveryBenchmarkReport],
    output_dir: Path | str,
    base_filename: str | None = None,
) -> tuple[Path, Path]:
    """
    Persist Markdown and JSON recovery benchmark artifacts atomically to disk.

    Args:
        report: Single RecoveryBenchmarkReport, Sequence, or Mapping.
        output_dir: Target directory path.
        base_filename: Optional base filename prefix.

    Returns:
        Tuple of (json_file_path, markdown_file_path).
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    timestamp_str = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    if isinstance(report, RecoveryBenchmarkReport):
        prefix = (
            base_filename
            or f"recovery_benchmark_{report.pipeline_name}_{timestamp_str}"
        )
        md_content = generate_recovery_markdown_report(report)
        json_content = generate_recovery_json_report(report)
    else:
        prefix = base_filename or f"recovery_comparison_{timestamp_str}"
        md_content = generate_recovery_comparison_markdown(report)
        json_content = generate_recovery_json_report(report)

    json_file = out_path / f"{prefix}.json"
    md_file = out_path / f"{prefix}.md"

    temp_json = out_path / f"{prefix}.json.tmp"
    temp_md = out_path / f"{prefix}.md.tmp"

    temp_json.write_text(json_content, encoding="utf-8")
    temp_md.write_text(md_content, encoding="utf-8")

    temp_json.replace(json_file)
    temp_md.replace(md_file)

    return json_file, md_file
