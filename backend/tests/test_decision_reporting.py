"""
Revora Decision Evaluation Reporting Tests.

Comprehensive testing of decision benchmark reporting, failure diagnostics,
Markdown report generation, JSON serialization, and multi-pipeline comparison.
"""

import json
from pathlib import Path
from uuid import uuid4

import pytest

from app.decision_engine import RecoveryAction
from app.evaluation.decision_reporting import (
    analyze_decision_failures,
    compare_decision_pipelines,
    generate_decision_comparison_markdown,
    generate_decision_json_report,
    generate_decision_markdown_report,
    save_decision_benchmark_artifacts,
)
from app.evaluation.schemas import DecisionBenchmarkReport, DecisionEvalResult


def _make_eval_result(
    query_id=None,
    pipeline_name="deterministic_baseline",
    predicted_action=RecoveryAction.RETRY_PAYMENT,
    expected_action=RecoveryAction.RETRY_PAYMENT,
    acceptable_actions=(RecoveryAction.RETRY_PAYMENT,),
    prohibited_actions=(),
    expected_policy_ids=(),
    applied_policy_ids=(),
    violated_policy_ids=(),
    is_exact_match=True,
    is_acceptable_match=True,
    confidence=0.95,
    policy_overridden=False,
    is_fallback=False,
    fallback_reason=None,
    referenced_case_ids=(),
    key_factors=(),
    latency_ms=15.0,
    error=None,
) -> DecisionEvalResult:
    return DecisionEvalResult(
        query_id=query_id or uuid4(),
        pipeline_name=pipeline_name,
        predicted_action=predicted_action,
        expected_action=expected_action,
        acceptable_actions=acceptable_actions,
        prohibited_actions=prohibited_actions,
        expected_policy_ids=expected_policy_ids,
        applied_policy_ids=applied_policy_ids,
        violated_policy_ids=violated_policy_ids,
        is_exact_match=is_exact_match,
        is_acceptable_match=is_acceptable_match,
        confidence=confidence,
        policy_overridden=policy_overridden,
        is_fallback=is_fallback,
        fallback_reason=fallback_reason,
        referenced_case_ids=referenced_case_ids,
        key_factors=key_factors,
        latency_ms=latency_ms,
        error=error,
    )


def _make_benchmark_report(
    pipeline_name="deterministic_baseline",
    results=None,
    metrics=None,
) -> DecisionBenchmarkReport:
    res = tuple(results) if results is not None else ()
    m = metrics or {
        "exact_match_rate": 0.85,
        "acceptable_match_rate": 0.95,
        "safety_violation_rate": 0.02,
        "policy_match_rate": 0.90,
        "policy_violation_rate": 0.05,
        "policy_override_rate": 0.04,
        "fallback_rate": 0.10,
        "mean_confidence": 0.88,
        "mean_latency_ms": 42.5,
    }
    return DecisionBenchmarkReport(
        pipeline_name=pipeline_name,
        dataset_name="golden_dataset_v1",
        num_queries=len(res),
        results=res,
        aggregate_metrics=m,
        evaluation_version="1.0",
    )


# =============================================================================
# Failure Diagnostics Tests
# =============================================================================


def test_analyze_decision_failures_detects_mismatches_and_violations():
    q1 = uuid4()
    q2 = uuid4()
    q3 = uuid4()
    q4 = uuid4()

    results = [
        # 1. Perfect match (no anomaly)
        _make_eval_result(
            query_id=q1,
            is_exact_match=True,
            is_acceptable_match=True,
        ),
        # 2. Safety violation (prohibited action)
        _make_eval_result(
            query_id=q2,
            predicted_action=RecoveryAction.RETRY_PAYMENT,
            expected_action=RecoveryAction.PAYMENT_LINK,
            acceptable_actions=(RecoveryAction.PAYMENT_LINK,),
            prohibited_actions=(RecoveryAction.RETRY_PAYMENT,),
            is_exact_match=False,
            is_acceptable_match=False,
        ),
        # 3. Fallback trigger
        _make_eval_result(
            query_id=q3,
            is_fallback=True,
            fallback_reason="LLM timeout",
        ),
        # 4. Execution error
        _make_eval_result(
            query_id=q4,
            is_exact_match=False,
            is_acceptable_match=False,
            error="ConnectionRefusedError: DB unavailable",
        ),
    ]

    report = _make_benchmark_report(results=results)
    failures = analyze_decision_failures(report)

    assert len(failures) == 3
    q_ids = {f["query_id"] for f in failures}
    assert str(q1) not in q_ids
    assert str(q2) in q_ids
    assert str(q3) in q_ids
    assert str(q4) in q_ids

    # Check detailed safety violation fields
    v_item = next(f for f in failures if f["query_id"] == str(q2))
    assert v_item["is_safety_violation"] is True
    assert v_item["predicted_action"] == "retry_payment"
    assert v_item["expected_action"] == "payment_link"

    # Check fallback fields
    fb_item = next(f for f in failures if f["query_id"] == str(q3))
    assert fb_item["is_fallback"] is True
    assert fb_item["fallback_reason"] == "LLM timeout"

    # Check error fields
    err_item = next(f for f in failures if f["query_id"] == str(q4))
    assert "ConnectionRefusedError" in err_item["error"]


def test_analyze_decision_failures_empty_when_all_perfect():
    results = [
        _make_eval_result(is_exact_match=True, is_acceptable_match=True),
        _make_eval_result(is_exact_match=True, is_acceptable_match=True),
    ]
    report = _make_benchmark_report(results=results)
    assert analyze_decision_failures(report) == []


# =============================================================================
# Markdown Report Generation Tests
# =============================================================================


def test_generate_decision_markdown_report_structure():
    results = [
        _make_eval_result(is_exact_match=True),
        _make_eval_result(
            predicted_action=RecoveryAction.RETRY_PAYMENT,
            prohibited_actions=(RecoveryAction.RETRY_PAYMENT,),
            is_exact_match=False,
            is_acceptable_match=False,
        ),
    ]
    report = _make_benchmark_report(pipeline_name="agent_rag_pipeline", results=results)
    md = generate_decision_markdown_report(report, include_failures=True)

    assert "# Decision Benchmark Report: `agent_rag_pipeline`" in md
    assert "## Executive Summary" in md
    assert "**Exact Match Rate**" in md
    assert "**Safety Violation Rate**" in md
    assert "**Mean Latency**" in md
    assert "## Failure & Safety Diagnostics" in md
    assert "Detailed Case Diagnostics" in md
    assert "🚨 **YES**" in md


def test_generate_decision_markdown_report_zero_failures():
    results = [_make_eval_result(is_exact_match=True, is_acceptable_match=True)]
    report = _make_benchmark_report(results=results)
    md = generate_decision_markdown_report(report, include_failures=True)

    assert (
        "Zero decision anomalies, safety violations, or policy errors detected." in md
    )


def test_generate_decision_markdown_report_exclude_failures():
    results = [_make_eval_result(is_exact_match=False)]
    report = _make_benchmark_report(results=results)
    md = generate_decision_markdown_report(report, include_failures=False)

    assert "## Executive Summary" in md
    assert "## Failure & Safety Diagnostics" not in md


# =============================================================================
# Multi-Pipeline Comparison Tests
# =============================================================================


def test_compare_decision_pipelines_matrix():
    r1 = _make_benchmark_report(
        pipeline_name="deterministic_baseline",
        metrics={"exact_match_rate": 0.80, "mean_latency_ms": 5.0},
    )
    r2 = _make_benchmark_report(
        pipeline_name="agent_rag",
        metrics={"exact_match_rate": 0.92, "mean_latency_ms": 150.0},
    )

    comp = compare_decision_pipelines([r1, r2])

    assert comp["num_pipelines"] == 2
    assert "deterministic_baseline" in comp["pipelines"]
    assert "agent_rag" in comp["pipelines"]
    assert comp["metrics"]["exact_match_rate"]["deterministic_baseline"] == 0.80
    assert comp["metrics"]["exact_match_rate"]["agent_rag"] == 0.92
    assert comp["metrics"]["mean_latency_ms"]["agent_rag"] == 150.0


def test_generate_decision_comparison_markdown():
    r1 = _make_benchmark_report(
        pipeline_name="deterministic_baseline",
        metrics={
            "exact_match_rate": 0.80,
            "acceptable_match_rate": 0.90,
            "safety_violation_rate": 0.0,
            "mean_latency_ms": 10.0,
        },
    )
    r2 = _make_benchmark_report(
        pipeline_name="agent_rag",
        metrics={
            "exact_match_rate": 0.95,
            "acceptable_match_rate": 0.98,
            "safety_violation_rate": 0.0,
            "mean_latency_ms": 180.0,
        },
    )

    md = generate_decision_comparison_markdown([r1, r2])

    assert "# Decision Pipeline Comparison" in md
    assert "**`deterministic_baseline`**" in md
    assert "**`agent_rag`**" in md
    assert "Exact Match Rate" in md
    assert "`80.0%`" in md
    assert "`95.0%`" in md
    assert "Diagnostic Totals" in md


def test_compare_decision_pipelines_empty():
    comp = compare_decision_pipelines([])
    assert comp["num_pipelines"] == 0
    assert comp["pipelines"] == []

    md = generate_decision_comparison_markdown([])
    assert "No benchmark reports provided" in md


def test_compare_decision_pipelines_invalid_type_rejected():
    with pytest.raises(TypeError, match="Expected DecisionBenchmarkReport"):
        compare_decision_pipelines(["invalid_report"])  # type: ignore[list-item]


# =============================================================================
# JSON Serialization & Persistence Tests
# =============================================================================


def test_generate_decision_json_report_single():
    report = _make_benchmark_report()
    json_str = generate_decision_json_report(report)
    parsed = json.loads(json_str)

    assert parsed["pipeline_name"] == "deterministic_baseline"
    assert parsed["dataset_name"] == "golden_dataset_v1"
    assert "aggregate_metrics" in parsed


def test_generate_decision_json_report_multiple():
    r1 = _make_benchmark_report(pipeline_name="p1")
    r2 = _make_benchmark_report(pipeline_name="p2")

    json_str = generate_decision_json_report([r1, r2])
    parsed = json.loads(json_str)

    assert "p1" in parsed
    assert "p2" in parsed
    assert parsed["p1"]["pipeline_name"] == "p1"
    assert parsed["p2"]["pipeline_name"] == "p2"


def test_save_decision_benchmark_artifacts(tmp_path: Path):
    report = _make_benchmark_report(pipeline_name="test_pipeline")
    json_path, md_path = save_decision_benchmark_artifacts(
        report=report,
        output_dir=tmp_path,
        base_filename="custom_report",
    )

    assert json_path.exists()
    assert md_path.exists()
    assert json_path.name == "custom_report.json"
    assert md_path.name == "custom_report.md"

    # Verify JSON content
    parsed = json.loads(json_path.read_text(encoding="utf-8"))
    assert parsed["pipeline_name"] == "test_pipeline"

    # Verify Markdown content
    md_content = md_path.read_text(encoding="utf-8")
    assert "# Decision Benchmark Report: `test_pipeline`" in md_content
