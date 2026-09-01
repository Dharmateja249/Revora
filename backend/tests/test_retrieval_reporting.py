"""
Revora Retrieval Evaluation Reporting Tests.

Validates MetricSnapshot, RetrieverEvaluationSummary, EvaluationReport, JSON serialization,
and Markdown report generation.
"""

import json

import pytest
from app.evaluation.benchmark import run_benchmark
from app.evaluation.cli import run_cli
from app.evaluation.regression import (
    RegressionAnalysis,
    RegressionFinding,
    RegressionSeverity,
)
from app.evaluation.reporting import (
    create_evaluation_report,
    generate_json_report,
    generate_markdown_report,
    save_benchmark_artifacts,
)
from app.evaluation.schemas import (
    EvaluationReport,
    MetricSnapshot,
    RetrieverEvaluationSummary,
)
from pydantic import ValidationError

from tests.fixtures.retrieval_golden_dataset import get_golden_evaluation_cases


def _make_sample_summary(
    name: str = "HistoricalRetriever", mrr: float = 0.98, lat: float = 0.05
) -> RetrieverEvaluationSummary:
    return RetrieverEvaluationSummary(
        retriever_name=name,
        query_count=50,
        mrr=mrr,
        mean_latency_ms=lat,
        precision_at_k={1: 0.98, 3: 0.88, 5: 0.54},
        recall_at_k={1: 0.37, 3: 0.97, 5: 0.98},
        ndcg_at_k={1: 0.98, 3: 0.97, 5: 0.97},
    )


def test_metric_snapshot_contract():
    snap = MetricSnapshot(metric_name="mrr", value=0.98)
    assert snap.metric_name == "mrr"
    assert snap.value == 0.98
    with pytest.raises(ValidationError):
        snap.value = 1.0  # type: ignore


def test_retriever_evaluation_summary_immutability_and_validation():
    summary = _make_sample_summary()
    assert summary.retriever_name == "HistoricalRetriever"
    assert summary.precision_at_k[1] == 0.98

    # Reject empty name
    with pytest.raises(ValidationError):
        RetrieverEvaluationSummary(
            retriever_name="",
            query_count=10,
            mrr=0.9,
            mean_latency_ms=1.0,
        )

    # Reject invalid K
    with pytest.raises(ValidationError):
        RetrieverEvaluationSummary(
            retriever_name="Test",
            query_count=10,
            mrr=0.9,
            mean_latency_ms=1.0,
            precision_at_k={-1: 0.5},
        )

    # Immutability
    with pytest.raises(ValidationError):
        summary.mrr = 0.5  # type: ignore


def test_evaluation_report_construction_and_immutability():
    summary_det = _make_sample_summary("HistoricalRetriever", mrr=0.98, lat=0.05)
    summary_sem = _make_sample_summary("SemanticRetriever", mrr=0.95, lat=1.00)

    report = EvaluationReport(
        report_id="eval_run_001",
        dataset_name="retrieval_golden_dataset",
        dataset_version="v1",
        query_count=50,
        configured_k_values=(1, 3, 5),
        retriever_summaries={
            "HistoricalRetriever": summary_det,
            "SemanticRetriever": summary_sem,
        },
    )

    assert report.report_id == "eval_run_001"
    assert report.query_count == 50
    assert len(report.retriever_summaries) == 2
    assert report.created_at.tzinfo is not None

    with pytest.raises(ValidationError):
        report.query_count = 100  # type: ignore


def test_create_evaluation_report_from_benchmark_results():
    cases = get_golden_evaluation_cases()[:5]
    reports = run_benchmark(evaluation_cases=cases, k_values=(1, 3))
    eval_rep = create_evaluation_report(
        reports, report_id="test_run_bench", dataset_name="golden_v1"
    )

    assert eval_rep.report_id == "test_run_bench"
    assert eval_rep.dataset_name == "golden_v1"
    assert (
        0.0
        <= eval_rep.retriever_summaries["DeterministicHistoricalRetriever"].mrr
        <= 1.0
    )


def test_json_report_serialization_deterministic():
    summary = _make_sample_summary("HistoricalRetriever")
    report = EvaluationReport(
        report_id="eval_det_test",
        dataset_name="golden_v1",
        dataset_version="v1",
        query_count=50,
        configured_k_values=(1, 3, 5),
        retriever_summaries={"HistoricalRetriever": summary},
    )

    json_1 = generate_json_report(report)
    json_2 = generate_json_report(report)
    assert json_1 == json_2

    parsed = json.loads(json_1)
    assert parsed["report_id"] == "eval_det_test"
    assert "HistoricalRetriever" in parsed["retriever_summaries"]


def test_json_report_serialization_with_regression_analysis_mapping():
    summary = _make_sample_summary("HistoricalRetriever")
    report = EvaluationReport(
        report_id="eval_reg_json_test",
        dataset_name="golden_v1",
        dataset_version="v1",
        query_count=50,
        configured_k_values=(1, 3),
        retriever_summaries={"HistoricalRetriever": summary},
    )
    reg_analysis = RegressionAnalysis(
        retriever_name="HistoricalRetriever",
        status="WARN",
        findings=(
            RegressionFinding(
                retriever_name="HistoricalRetriever",
                metric="mrr",
                baseline_value=0.98,
                candidate_value=0.92,
                absolute_delta=-0.06,
                relative_delta_percent=-6.12,
                severity=RegressionSeverity.WARNING,
                message="MRR dropped by 0.06",
            ),
        ),
    )

    json_out = generate_json_report(
        report, regressions={"HistoricalRetriever": reg_analysis}
    )
    parsed = json.loads(json_out)
    assert "regression_analysis" in parsed
    assert "HistoricalRetriever" in parsed["regression_analysis"]
    assert parsed["regression_analysis"]["HistoricalRetriever"]["status"] == "WARN"


def test_markdown_report_from_evaluation_report():
    summary_det = _make_sample_summary(
        "DeterministicHistoricalRetriever", mrr=0.98, lat=0.05
    )
    summary_sem = _make_sample_summary(
        "SemanticHistoricalRetriever", mrr=0.95, lat=1.00
    )
    summary_hyb = _make_sample_summary("HybridHistoricalRetriever", mrr=0.98, lat=1.20)

    report = EvaluationReport(
        report_id="eval_md_test",
        dataset_name="retrieval_golden_dataset",
        dataset_version="v1",
        query_count=50,
        configured_k_values=(1, 3, 5),
        retriever_summaries={
            "DeterministicHistoricalRetriever": summary_det,
            "SemanticHistoricalRetriever": summary_sem,
            "HybridHistoricalRetriever": summary_hyb,
        },
    )

    md = generate_markdown_report(report)
    assert "# Revora Retrieval Evaluation Report" in md
    assert "## 1. Benchmark Overview" in md
    assert "## 2. Executive Summary" in md
    assert "## 4. Comprehensive Metrics Comparison" in md
    assert "DeterministicHistoricalRetriever" in md
    assert "SemanticHistoricalRetriever" in md
    assert "HybridHistoricalRetriever" in md


def test_markdown_report_dynamic_judgments_count():
    summary = _make_sample_summary("HistoricalRetriever")
    report = EvaluationReport(
        report_id="eval_md_test",
        dataset_name="retrieval_golden_dataset",
        dataset_version="v1",
        query_count=50,
        configured_k_values=(1, 3),
        retriever_summaries={"HistoricalRetriever": summary},
    )

    # 1. Without total_judgments: ground-truth judgments row is omitted (not hardcoded to 230)
    md_omitted = generate_markdown_report(report)
    assert "Ground-Truth Judgments" not in md_omitted

    # 2. With explicit total_judgments: rendered accurately
    md_explicit = generate_markdown_report(report, total_judgments=175)
    assert "| **Ground-Truth Judgments** | `175` |" in md_explicit


def test_cli_assert_no_regressions_without_baseline_fails(capsys):
    ret_code = run_cli(["--assert-no-regressions"])
    assert ret_code == 1
    captured = capsys.readouterr()
    assert "Error: --assert-no-regressions requires --compare-baseline" in captured.err


def test_cli_assert_no_regressions_with_baseline_success(tmp_path):
    cases = get_golden_evaluation_cases()[:3]
    reports = run_benchmark(evaluation_cases=cases, k_values=(1, 3, 5, 10))
    eval_rep = create_evaluation_report(
        reports, report_id="cli_base_pass", dataset_name="retrieval_golden_dataset_v1"
    )
    # Buffer latency in baseline to avoid CPU timing jitter in fast unit tests
    buffered_summaries = {
        name: RetrieverEvaluationSummary(
            retriever_name=s.retriever_name,
            query_count=s.query_count,
            mrr=s.mrr,
            mean_latency_ms=max(s.mean_latency_ms, 50.0),
            precision_at_k=s.precision_at_k,
            recall_at_k=s.recall_at_k,
            ndcg_at_k=s.ndcg_at_k,
        )
        for name, s in eval_rep.retriever_summaries.items()
    }
    base_report = EvaluationReport(
        report_id="cli_base_pass",
        dataset_name="retrieval_golden_dataset_v1",
        dataset_version="v1",
        query_count=3,
        configured_k_values=(1, 3, 5, 10),
        retriever_summaries=buffered_summaries,
    )
    from app.evaluation.persistence import save_report

    base_file = save_report(base_report, directory=tmp_path)

    ret_code = run_cli(
        ["--compare-baseline", str(base_file), "--assert-no-regressions"],
        evaluation_cases=cases,
    )
    assert ret_code == 0


def test_cli_assert_no_regressions_with_regression_fails(tmp_path, capsys):
    cases = get_golden_evaluation_cases()[:3]
    reports = run_benchmark(evaluation_cases=cases, k_values=(1, 3, 5, 10))
    # Make baseline report artificially high
    high_summaries = {}
    for name in create_evaluation_report(reports).retriever_summaries:
        high_summaries[name] = RetrieverEvaluationSummary(
            retriever_name=name,
            query_count=3,
            mrr=1.0,
            mean_latency_ms=0.01,
            precision_at_k={1: 1.0, 3: 1.0, 5: 1.0, 10: 1.0},
            recall_at_k={1: 1.0, 3: 1.0, 5: 1.0, 10: 1.0},
            ndcg_at_k={1: 1.0, 3: 1.0, 5: 1.0, 10: 1.0},
        )
    base_report = EvaluationReport(
        report_id="cli_base_high",
        dataset_name="retrieval_golden_dataset_v1",
        dataset_version="v1",
        query_count=3,
        configured_k_values=(1, 3, 5, 10),
        retriever_summaries=high_summaries,
    )
    from app.evaluation.persistence import save_report

    base_file = save_report(base_report, directory=tmp_path)

    ret_code = run_cli(
        ["--compare-baseline", str(base_file), "--assert-no-regressions"],
        evaluation_cases=cases,
    )
    assert ret_code == 1
    captured = capsys.readouterr()
    assert "Evaluation Regression Detected" in captured.err


def test_save_benchmark_artifacts_from_evaluation_report(tmp_path):
    summary = _make_sample_summary("HistoricalRetriever")
    report = EvaluationReport(
        report_id="save_test_01",
        dataset_name="golden_v1",
        dataset_version="v1",
        query_count=50,
        configured_k_values=(1, 3),
        retriever_summaries={"HistoricalRetriever": summary},
    )

    artifacts = save_benchmark_artifacts(report, output_dir=tmp_path)
    assert artifacts["json"].exists()
    assert artifacts["markdown"].exists()
