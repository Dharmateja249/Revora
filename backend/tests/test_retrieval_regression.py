"""
Revora Retrieval Evaluation Regression Detection Tests.

Validates report comparison, threshold enforcement, per-query diagnostics,
dataset compatibility checks, and CI assertions.
"""

from uuid import uuid4
import pytest
from pydantic import ValidationError

from app.evaluation.regression import (
    RegressionSeverity,
    RegressionThresholds,
    assert_no_regressions,
    compare_reports,
    find_worst_query_regressions,
)
from app.evaluation.schemas import (
    EvaluationReport,
    EvaluationRegressionError,
    RegressionCheck,
    RegressionReport,
    RetrievalEvalResult,
    RetrieverEvaluationSummary,
)


def _make_report(
    report_id: str,
    mrr: float = 0.98,
    lat: float = 0.05,
    p1: float = 0.98,
    r3: float = 0.97,
    n3: float = 0.97,
    dataset_name: str = "golden_v1",
    dataset_version: str = "v1",
    query_count: int = 50,
    k_vals=(1, 3),
) -> EvaluationReport:
    summary = RetrieverEvaluationSummary(
        retriever_name="HistoricalRetriever",
        query_count=query_count,
        mrr=mrr,
        mean_latency_ms=lat,
        precision_at_k={1: p1, 3: 0.88},
        recall_at_k={1: 0.37, 3: r3},
        ndcg_at_k={1: 0.98, 3: n3},
    )
    return EvaluationReport(
        report_id=report_id,
        dataset_name=dataset_name,
        dataset_version=dataset_version,
        query_count=query_count,
        configured_k_values=k_vals,
        retriever_summaries={"HistoricalRetriever": summary},
    )


def test_compare_reports_no_regression():
    base = _make_report("base_01", mrr=0.98, lat=0.05)
    cand = _make_report("cand_01", mrr=0.98, lat=0.05)

    reg_report = compare_reports(base, cand)
    assert reg_report.overall_status == "PASS"
    assert len(reg_report.checks) > 0
    assert all(c.status == "PASS" for c in reg_report.checks)

    # assert_no_regressions should not raise
    assert_no_regressions(reg_report)


def test_compare_reports_quality_regression_failure():
    base = _make_report("base_01", mrr=0.98, r3=0.97)
    cand = _make_report("cand_01", mrr=0.90, r3=0.80)  # > 0.02 drop

    thresholds = RegressionThresholds(max_quality_drop=0.02)
    reg_report = compare_reports(base, cand, thresholds=thresholds)

    assert reg_report.overall_status == "FAIL"
    failed_checks = [c for c in reg_report.checks if c.status == "FAIL"]
    assert len(failed_checks) > 0

    with pytest.raises(EvaluationRegressionError, match="Evaluation Regression Detected"):
        assert_no_regressions(reg_report)


def test_compare_reports_latency_regression_failure():
    base = _make_report("base_01", lat=1.0)
    cand = _make_report("cand_01", lat=1.5)  # 50% increase exceeds 10% threshold

    thresholds = RegressionThresholds(max_latency_increase_ratio=0.10)
    reg_report = compare_reports(base, cand, thresholds=thresholds)

    assert reg_report.overall_status == "FAIL"
    lat_check = next(c for c in reg_report.checks if c.metric_name == "mean_latency_ms")
    assert lat_check.status == "FAIL"


def test_compare_reports_dataset_incompatibility():
    base = _make_report("base_01", dataset_name="dataset_A", dataset_version="v1")
    cand = _make_report("cand_01", dataset_name="dataset_B", dataset_version="v1")

    with pytest.raises(ValueError, match="Dataset name mismatch"):
        compare_reports(base, cand)

    base_v1 = _make_report("base_v1", dataset_name="golden_v1", dataset_version="v1")
    cand_v2 = _make_report("cand_02", dataset_name="golden_v1", dataset_version="v2")
    with pytest.raises(ValueError, match="Dataset version mismatch"):
        compare_reports(base_v1, cand_v2)

    cand_q40 = _make_report("cand_03", dataset_name="golden_v1", dataset_version="v1", query_count=40)
    with pytest.raises(ValueError, match="Query count mismatch"):
        compare_reports(base_v1, cand_q40)

    cand_k = _make_report("cand_04", dataset_name="golden_v1", dataset_version="v1", k_vals=(1, 5))
    with pytest.raises(ValueError, match="K values mismatch"):
        compare_reports(base_v1, cand_k)


def test_find_worst_query_regressions():
    qid1, qid2, qid3 = uuid4(), uuid4(), uuid4()

    base_results = [
        RetrievalEvalResult(
            query_id=qid1,
            retriever_name="RetrieverA",
            k=3,
            retrieved_payment_ids=(uuid4(),),
            precision_at_k=1.0,
            recall_at_k=1.0,
            reciprocal_rank=1.0,
            ndcg_at_k=1.0,
        ),
        RetrievalEvalResult(
            query_id=qid2,
            retriever_name="RetrieverA",
            k=3,
            retrieved_payment_ids=(uuid4(),),
            precision_at_k=1.0,
            recall_at_k=1.0,
            reciprocal_rank=1.0,
            ndcg_at_k=1.0,
        ),
    ]

    cand_results = [
        RetrievalEvalResult(
            query_id=qid1,
            retriever_name="RetrieverA",
            k=3,
            retrieved_payment_ids=(uuid4(),),
            precision_at_k=0.5,
            recall_at_k=0.5,
            reciprocal_rank=0.5,
            ndcg_at_k=0.4,  # drop = -0.6
        ),
        RetrievalEvalResult(
            query_id=qid2,
            retriever_name="RetrieverA",
            k=3,
            retrieved_payment_ids=(uuid4(),),
            precision_at_k=0.8,
            recall_at_k=0.8,
            reciprocal_rank=0.8,
            ndcg_at_k=0.8,  # drop = -0.2
        ),
    ]

    worst = find_worst_query_regressions(base_results, cand_results, metric="ndcg_at_3", limit=5)
    assert len(worst) == 2
    # Worst regression (delta = -0.6) should be first
    assert worst[0].query_id == qid1
    assert worst[0].delta == pytest.approx(-0.6)
    assert worst[1].query_id == qid2
    assert worst[1].delta == pytest.approx(-0.2)


def test_find_worst_query_regressions_mrr_no_duplicates():
    """Verify that MRR regression diagnostics produce exactly one entry per degraded query despite multiple K rows."""
    qid = uuid4()
    # Baseline results for K=1, 3, 5
    base_results = [
        RetrievalEvalResult(
            query_id=qid,
            retriever_name="RetrieverA",
            k=k,
            retrieved_payment_ids=(uuid4(),),
            precision_at_k=1.0,
            recall_at_k=1.0,
            reciprocal_rank=1.0,
            ndcg_at_k=1.0,
        )
        for k in (1, 3, 5)
    ]
    # Candidate results for K=1, 3, 5 with degraded MRR (1.0 -> 0.33)
    cand_results = [
        RetrievalEvalResult(
            query_id=qid,
            retriever_name="RetrieverA",
            k=k,
            retrieved_payment_ids=(uuid4(),),
            precision_at_k=0.33,
            recall_at_k=0.33,
            reciprocal_rank=0.33,
            ndcg_at_k=0.33,
        )
        for k in (1, 3, 5)
    ]

    worst_mrr = find_worst_query_regressions(base_results, cand_results, metric="mrr", limit=10)
    assert len(worst_mrr) == 1, f"Expected exactly 1 diagnostic for query MRR regression, got {len(worst_mrr)}"
    assert worst_mrr[0].query_id == qid
    assert worst_mrr[0].metric_name == "mrr"
    assert worst_mrr[0].delta == pytest.approx(-0.67, rel=1e-2)


def test_regression_report_immutability():
    check = RegressionCheck(
        metric_name="mrr",
        retriever_name="HistoricalRetriever",
        baseline_value=0.98,
        candidate_value=0.98,
        delta=0.0,
        relative_change=0.0,
        threshold=0.02,
        status="PASS",
    )
    with pytest.raises(ValidationError):
        check.status = "FAIL"  # type: ignore

    rep = RegressionReport(
        baseline_report_id="base",
        candidate_report_id="cand",
        checks=(check,),
        overall_status="PASS",
    )
    with pytest.raises(ValidationError):
        rep.overall_status = "FAIL"  # type: ignore
