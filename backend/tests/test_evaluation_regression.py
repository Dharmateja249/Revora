"""
Revora Retrieval Evaluation Regression Detection Unit Tests.

Validates regression detection, threshold enforcement, severity mapping, per-query
diagnostics, and immutability.
"""

from typing import Mapping
from uuid import uuid4
import pytest
from pydantic import ValidationError

from app.evaluation.regression import (
    EvaluationThresholds,
    QueryRegressionDiagnostic,
    RegressionAnalysis,
    RegressionFinding,
    RegressionSeverity,
    compare_benchmark_runs,
    detect_regressions,
)
from app.evaluation.schemas import (
    RetrievalEvalResult,
    RetrieverBenchmarkReport,
)


def _make_report_with_metrics(
    retriever_name: str = "TestRetriever",
    metrics: dict[str, float] | None = None,
    results: list[RetrievalEvalResult] | None = None,
) -> RetrieverBenchmarkReport:
    m = {
        "mrr": 0.95,
        "mean_latency_ms": 1.0,
        "mean_precision_at_1": 0.95,
        "mean_recall_at_1": 0.40,
        "mean_ndcg_at_1": 0.95,
        "mean_precision_at_3": 0.85,
        "mean_recall_at_3": 0.95,
        "mean_ndcg_at_3": 0.90,
    }
    if metrics:
        m.update(metrics)

    res_list = results or [
        RetrievalEvalResult(
            query_id=uuid4(),
            retriever_name=retriever_name,
            k=1,
            retrieved_payment_ids=(uuid4(),),
            precision_at_k=m.get("mean_precision_at_1", 0.95),
            recall_at_k=m.get("mean_recall_at_1", 0.40),
            reciprocal_rank=m.get("mrr", 0.95),
            ndcg_at_k=m.get("mean_ndcg_at_1", 0.95),
            latency_ms=m.get("mean_latency_ms", 1.0),
        )
    ]

    return RetrieverBenchmarkReport(
        retriever_name=retriever_name,
        dataset_name="golden_test_v1",
        num_queries=len(res_list),
        k_values=(1, 3),
        results=tuple(res_list),
        aggregate_metrics=m,
    )


# ============================================================================
# 1. Quality & Latency Regression Tests
# ============================================================================


def test_detect_no_regression():
    base = _make_report_with_metrics(metrics={"mrr": 0.95, "mean_latency_ms": 1.0})
    cand = _make_report_with_metrics(metrics={"mrr": 0.95, "mean_latency_ms": 1.0})

    analysis = detect_regressions(base, cand)
    assert analysis.status == "PASS"
    assert analysis.has_critical_regression is False
    assert len(analysis.findings) == 0


def test_detect_positive_improvement():
    base = _make_report_with_metrics(metrics={"mrr": 0.90, "mean_latency_ms": 2.0})
    cand = _make_report_with_metrics(metrics={"mrr": 0.98, "mean_latency_ms": 1.0})

    analysis = detect_regressions(base, cand)
    assert analysis.status == "PASS"
    assert analysis.has_critical_regression is False


def test_detect_mrr_quality_floor_regression_critical():
    base = _make_report_with_metrics(metrics={"mrr": 0.95})
    cand = _make_report_with_metrics(metrics={"mrr": 0.85})  # Below 0.90 floor

    thresholds = EvaluationThresholds(mrr_min=0.90)
    analysis = detect_regressions(base, cand, thresholds=thresholds)

    assert analysis.status == "FAIL"
    assert analysis.has_critical_regression is True
    assert any(f.metric == "mrr" and f.severity == RegressionSeverity.CRITICAL for f in analysis.findings)


def test_detect_relative_quality_drop_warning():
    base = _make_report_with_metrics(metrics={"mean_recall_at_3": 0.95})
    # 8% relative drop (0.95 -> 0.874): exceeds 5% max_relative_quality_drop, below 15% critical
    cand = _make_report_with_metrics(metrics={"mean_recall_at_3": 0.874})

    thresholds = EvaluationThresholds(max_quality_drop=0.10, max_relative_quality_drop=0.05, critical_relative_quality_drop=0.15)
    analysis = detect_regressions(base, cand, thresholds=thresholds)
    assert analysis.status == "WARN"
    assert analysis.has_critical_regression is False
    finding = next(f for f in analysis.findings if f.metric == "mean_recall_at_3")
    assert finding.severity == RegressionSeverity.WARNING
    assert finding.relative_delta_percent == pytest.approx(-8.0, rel=1e-2)


def test_detect_latency_regression_threshold():
    base = _make_report_with_metrics(metrics={"mean_latency_ms": 1.5})
    cand = _make_report_with_metrics(metrics={"mean_latency_ms": 6.0})

    thresholds = EvaluationThresholds(max_latency_ms=5.0)
    analysis = detect_regressions(base, cand, thresholds=thresholds)

    assert analysis.status == "FAIL"
    assert analysis.has_critical_regression is True
    finding = next(f for f in analysis.findings if f.metric == "mean_latency_ms")
    assert finding.severity == RegressionSeverity.CRITICAL


def test_compare_benchmark_runs_multi_retriever():
    base_det = _make_report_with_metrics("DeterministicHistoricalRetriever", {"mrr": 0.98})
    base_sem = _make_report_with_metrics("SemanticHistoricalRetriever", {"mrr": 0.92})

    cand_det = _make_report_with_metrics("DeterministicHistoricalRetriever", {"mrr": 0.98})
    cand_sem = _make_report_with_metrics("SemanticHistoricalRetriever", {"mrr": 0.70})  # Significant drop

    analyses = compare_benchmark_runs(
        baseline_reports={"DeterministicHistoricalRetriever": base_det, "SemanticHistoricalRetriever": base_sem},
        candidate_reports={"DeterministicHistoricalRetriever": cand_det, "SemanticHistoricalRetriever": cand_sem},
    )

    assert analyses["DeterministicHistoricalRetriever"].status == "PASS"
    assert analyses["SemanticHistoricalRetriever"].status == "FAIL"


# ============================================================================
# 2. Per-Query Diagnostics Tests
# ============================================================================


def test_per_query_regression_diagnostics():
    qid = uuid4()
    b_res = RetrievalEvalResult(
        query_id=qid,
        retriever_name="RetrieverA",
        k=3,
        retrieved_payment_ids=(uuid4(),),
        precision_at_k=1.0,
        recall_at_k=1.0,
        reciprocal_rank=1.0,
        ndcg_at_k=1.0,
        metadata={"case_description": "UPI Timeout Scenario"},
    )
    c_res = RetrievalEvalResult(
        query_id=qid,
        retriever_name="RetrieverA",
        k=3,
        retrieved_payment_ids=(uuid4(),),
        precision_at_k=0.33,
        recall_at_k=0.33,
        reciprocal_rank=0.33,
        ndcg_at_k=0.50,  # 0.50 drop from 1.0 (exceeds 0.20 diagnostic threshold)
        metadata={"case_description": "UPI Timeout Scenario"},
    )

    base = _make_report_with_metrics("RetrieverA", results=[b_res])
    cand = _make_report_with_metrics("RetrieverA", results=[c_res])

    analysis = detect_regressions(base, cand)
    assert len(analysis.query_diagnostics) > 0
    diag = analysis.query_diagnostics[0]
    assert diag.query_id == qid
    assert diag.metric_name == "ndcg_at_3"
    assert diag.delta == pytest.approx(-0.50)
    assert diag.description == "UPI Timeout Scenario"


# ============================================================================
# 3. Threshold Configuration & Bounds Tests
# ============================================================================


def test_evaluation_thresholds_validation():
    # Valid construction
    t = EvaluationThresholds(
        mrr_min=0.90,
        precision_at_k_min={1: 0.90, 3: 0.80},
        recall_at_k_min={3: 0.90},
        ndcg_at_k_min={3: 0.85},
        max_latency_ms=5.0,
    )
    assert t.mrr_min == 0.90
    assert t.precision_at_k_min[1] == 0.90

    # Invalid MRR (> 1.0 or < 0.0)
    with pytest.raises(ValidationError):
        EvaluationThresholds(mrr_min=1.5)

    with pytest.raises(ValidationError):
        EvaluationThresholds(mrr_min=-0.1)

    # Invalid K (bool or <= 0)
    with pytest.raises(ValidationError):
        EvaluationThresholds(precision_at_k_min={0: 0.5})

    with pytest.raises(ValidationError):
        EvaluationThresholds(precision_at_k_min={True: 0.5})  # type: ignore

    # Invalid latency (< 0.0)
    with pytest.raises(ValidationError):
        EvaluationThresholds(max_latency_ms=-1.0)


# ============================================================================
# 4. Immutability Tests
# ============================================================================


def test_regression_entities_immutability():
    finding = RegressionFinding(
        retriever_name="RetrieverA",
        metric="mrr",
        baseline_value=0.98,
        candidate_value=0.91,
        absolute_delta=-0.07,
        relative_delta_percent=-7.14,
        severity=RegressionSeverity.WARNING,
        message="MRR dropped",
    )
    with pytest.raises(ValidationError):
        finding.severity = RegressionSeverity.CRITICAL  # type: ignore

    diag = QueryRegressionDiagnostic(
        query_id=uuid4(),
        retriever_name="RetrieverA",
        metric_name="ndcg_at_3",
        baseline_value=1.0,
        candidate_value=0.5,
        delta=-0.5,
    )
    with pytest.raises(ValidationError):
        diag.delta = 0.0  # type: ignore

    analysis = RegressionAnalysis(
        retriever_name="RetrieverA",
        status="PASS",
    )
    with pytest.raises(ValidationError):
        analysis.status = "FAIL"  # type: ignore
