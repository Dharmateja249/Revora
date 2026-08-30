"""
Revora Retrieval Benchmark & Comparative Reporting Tests.

Validates end-to-end benchmark execution across Deterministic, Semantic, and Hybrid
retrievers over the golden evaluation dataset.
"""

import pytest

from app.evaluation.benchmark import (
    compare_benchmarks,
    format_comparison_markdown,
    populate_benchmark_vector_index,
    run_benchmark,
)
from app.evaluation.schemas import RetrieverBenchmarkReport
from tests.fixtures.retrieval_golden_dataset import get_golden_evaluation_cases


def test_populate_benchmark_vector_index():
    cases = get_golden_evaluation_cases()
    vector_index = populate_benchmark_vector_index(cases)

    # Verify vector index size is populated with all valid candidate payments
    assert vector_index.size > 0
    assert vector_index.dimension == 64


def test_run_benchmark_end_to_end():
    """Run benchmark over subset or full golden dataset and verify report structures."""
    cases = get_golden_evaluation_cases()[:5]  # Test subset for quick execution in unit test
    reports = run_benchmark(evaluation_cases=cases, k_values=(1, 3, 5))

    assert "DeterministicHistoricalRetriever" in reports
    assert "SemanticHistoricalRetriever" in reports
    assert "HybridHistoricalRetriever" in reports

    for name, report in reports.items():
        assert isinstance(report, RetrieverBenchmarkReport)
        assert report.retriever_name == name
        assert report.num_queries == 5
        assert report.k_values == (1, 3, 5)
        assert len(report.results) == 15  # 5 queries * 3 K values
        assert "mrr" in report.aggregate_metrics
        assert "mean_precision_at_1" in report.aggregate_metrics
        assert "mean_recall_at_1" in report.aggregate_metrics
        assert "mean_ndcg_at_1" in report.aggregate_metrics
        assert "mean_latency_ms" in report.aggregate_metrics


def test_compare_benchmarks_and_markdown_formatting():
    cases = get_golden_evaluation_cases()[:3]
    reports = run_benchmark(evaluation_cases=cases, k_values=(1, 3))

    summary = compare_benchmarks(reports)
    assert "retrievers" in summary
    assert len(summary["retrievers"]) == 3
    assert "metrics_comparison" in summary
    assert "mrr" in summary["metrics_comparison"]

    md = format_comparison_markdown(reports)
    assert isinstance(md, str)
    assert "DeterministicHistoricalRetriever" in md
    assert "SemanticHistoricalRetriever" in md
    assert "HybridHistoricalRetriever" in md
    assert "MRR" in md
    assert "Precision@1" in md
    assert "Precision@3" in md


def test_full_50_case_golden_benchmark():
    """Execute complete benchmark over all 50 golden evaluation cases across all 3 retrievers."""
    reports = run_benchmark(k_values=(1, 3, 5, 10))

    assert len(reports) == 3
    for name in ("DeterministicHistoricalRetriever", "SemanticHistoricalRetriever", "HybridHistoricalRetriever"):
        assert name in reports
        rep = reports[name]
        assert rep.num_queries == 50
        assert rep.k_values == (1, 3, 5, 10)
        assert len(rep.results) == 200  # 50 queries * 4 K values
        assert 0.0 <= rep.aggregate_metrics["mrr"] <= 1.0
        assert rep.aggregate_metrics["mean_latency_ms"] >= 0.0

    summary_md = format_comparison_markdown(reports)
    assert "Precision@10" in summary_md
    assert "Recall@10" in summary_md
    assert "NDCG@10" in summary_md
