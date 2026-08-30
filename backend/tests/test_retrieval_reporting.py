"""
Revora Retrieval Evaluation Reporting Tests.

Validates MetricSnapshot, RetrieverEvaluationSummary, EvaluationReport, JSON serialization,
and Markdown report generation.
"""

from datetime import datetime, timezone
import json
from uuid import uuid4
import pytest
from pydantic import ValidationError

from app.evaluation.benchmark import run_benchmark
from app.evaluation.reporting import (
    create_evaluation_report,
    generate_json_report,
    generate_markdown_report,
    save_benchmark_artifacts,
)
from app.evaluation.schemas import (
    EvaluationReport,
    MetricSnapshot,
    RetrievalEvalResult,
    RetrieverBenchmarkReport,
    RetrieverEvaluationSummary,
)


def _make_sample_summary(name: str = "HistoricalRetriever", mrr: float = 0.98, lat: float = 0.05) -> RetrieverEvaluationSummary:
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
    reports = run_benchmark(k_values=(1, 3))
    eval_rep = create_evaluation_report(reports, report_id="test_run_bench", dataset_name="golden_v1")

    assert eval_rep.report_id == "test_run_bench"
    assert eval_rep.dataset_name == "golden_v1"
    assert len(eval_rep.retriever_summaries) == 3
    assert "DeterministicHistoricalRetriever" in eval_rep.retriever_summaries
    assert eval_rep.retriever_summaries["DeterministicHistoricalRetriever"].mrr == pytest.approx(0.98, rel=1e-2)


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


def test_markdown_report_from_evaluation_report():
    summary_det = _make_sample_summary("DeterministicHistoricalRetriever", mrr=0.98, lat=0.05)
    summary_sem = _make_sample_summary("SemanticHistoricalRetriever", mrr=0.95, lat=1.00)
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
