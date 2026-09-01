"""
Revora Retrieval Evaluation Schemas Unit Tests.

Tests validation rules, bounds, duplicate detection, and immutability for
EvaluationCase, GroundTruthJudgment, RetrievalEvalResult, and RetrieverBenchmarkReport.
"""

import uuid
from datetime import datetime

import pytest
from app.context import (
    CustomerContext,
    CustomerRecoveryContext,
    PaymentContext,
)
from app.evaluation.schemas import (
    EvaluationCase,
    GroundTruthJudgment,
    RetrievalEvalResult,
    RetrieverBenchmarkReport,
)
from pydantic import ValidationError


def _make_dummy_context(
    customer_id: uuid.UUID | None = None,
) -> CustomerRecoveryContext:
    cid = customer_id or uuid.uuid4()
    return CustomerRecoveryContext(
        customer=CustomerContext(customer_id=cid),
        current_payment=PaymentContext(
            payment_id=uuid.uuid4(),
            amount=500.0,
            currency="INR",
            payment_method="upi",
            status="failed",
            failure_reason="timeout",
        ),
    )


# ============================================================================
# 1. GroundTruthJudgment Tests
# ============================================================================


def test_ground_truth_judgment_valid_construction():
    pid = uuid.uuid4()
    for grade in (0, 1, 2, 3):
        judgment = GroundTruthJudgment(
            payment_id=pid,
            relevance_grade=grade,
            rationale=f"Grade {grade} rationale",
        )
        assert judgment.payment_id == pid
        assert judgment.relevance_grade == grade
        assert judgment.rationale == f"Grade {grade} rationale"


def test_ground_truth_judgment_relevance_grade_bounds():
    pid = uuid.uuid4()

    # Negative grade rejected
    with pytest.raises(ValidationError) as exc:
        GroundTruthJudgment(payment_id=pid, relevance_grade=-1)
    assert "relevance_grade" in str(exc.value)

    # Grade > 3 rejected
    with pytest.raises(ValidationError) as exc:
        GroundTruthJudgment(payment_id=pid, relevance_grade=4)
    assert "relevance_grade" in str(exc.value)

    # Float/string/bool rejected
    with pytest.raises(ValidationError):
        GroundTruthJudgment(payment_id=pid, relevance_grade=True)  # type: ignore

    with pytest.raises(ValidationError):
        GroundTruthJudgment(payment_id=pid, relevance_grade=2.5)  # type: ignore


def test_ground_truth_judgment_invalid_uuid():
    with pytest.raises(ValidationError):
        GroundTruthJudgment(payment_id="not-a-uuid", relevance_grade=3)  # type: ignore


def test_ground_truth_judgment_immutability():
    judgment = GroundTruthJudgment(payment_id=uuid.uuid4(), relevance_grade=2)
    with pytest.raises(ValidationError):
        judgment.relevance_grade = 3  # type: ignore


# ============================================================================
# 2. EvaluationCase Tests
# ============================================================================


def test_evaluation_case_valid_construction():
    qid = uuid.uuid4()
    context = _make_dummy_context()
    pid1 = uuid.uuid4()
    pid2 = uuid.uuid4()

    gt = (
        GroundTruthJudgment(
            payment_id=pid1, relevance_grade=3, rationale="Exact match"
        ),
        GroundTruthJudgment(payment_id=pid2, relevance_grade=1, rationale="Weak match"),
    )

    case = EvaluationCase(
        query_id=qid,
        context=context,
        ground_truth=gt,
        description="Scenario 1",
        metadata={"category": "upi_timeout"},
    )

    assert case.query_id == qid
    assert case.context == context
    assert len(case.ground_truth) == 2
    assert case.ground_truth[0].payment_id == pid1
    assert case.description == "Scenario 1"
    assert case.metadata["category"] == "upi_timeout"
    assert isinstance(case.created_at, datetime)


def test_evaluation_case_empty_ground_truth():
    qid = uuid.uuid4()
    context = _make_dummy_context()

    case = EvaluationCase(
        query_id=qid,
        context=context,
        ground_truth=(),
    )
    assert case.ground_truth == ()


def test_evaluation_case_duplicate_ground_truth_payment_id_rejected():
    qid = uuid.uuid4()
    context = _make_dummy_context()
    duplicate_pid = uuid.uuid4()

    gt = (
        GroundTruthJudgment(payment_id=duplicate_pid, relevance_grade=3),
        GroundTruthJudgment(payment_id=duplicate_pid, relevance_grade=1),
    )

    with pytest.raises(ValidationError) as exc:
        EvaluationCase(
            query_id=qid,
            context=context,
            ground_truth=gt,
        )
    assert "Duplicate payment_id" in str(exc.value)


def test_evaluation_case_invalid_query_id_or_context():
    # Invalid UUID
    with pytest.raises(ValidationError):
        EvaluationCase(
            query_id="invalid-uuid",  # type: ignore
            context=_make_dummy_context(),
        )

    # Invalid context
    with pytest.raises(ValidationError):
        EvaluationCase(
            query_id=uuid.uuid4(),
            context="invalid-context",  # type: ignore
        )


def test_evaluation_case_immutability():
    case = EvaluationCase(
        query_id=uuid.uuid4(),
        context=_make_dummy_context(),
    )
    with pytest.raises(ValidationError):
        case.query_id = uuid.uuid4()  # type: ignore


# ============================================================================
# 3. RetrievalEvalResult Tests
# ============================================================================


def test_retrieval_eval_result_valid_construction():
    qid = uuid.uuid4()
    pids = (uuid.uuid4(), uuid.uuid4(), uuid.uuid4())

    res = RetrievalEvalResult(
        query_id=qid,
        retriever_name="DeterministicHistoricalRetriever",
        k=3,
        retrieved_payment_ids=pids,
        precision_at_k=0.6667,
        recall_at_k=1.0,
        reciprocal_rank=1.0,
        ndcg_at_k=0.9234,
        latency_ms=12.5,
        metadata={"fetch_k": 10},
    )

    assert res.query_id == qid
    assert res.retriever_name == "DeterministicHistoricalRetriever"
    assert res.k == 3
    assert res.retrieved_payment_ids == pids
    assert res.precision_at_k == 0.6667
    assert res.recall_at_k == 1.0
    assert res.reciprocal_rank == 1.0
    assert res.ndcg_at_k == 0.9234
    assert res.latency_ms == 12.5
    assert res.metadata["fetch_k"] == 10


def test_retrieval_eval_result_invalid_k():
    qid = uuid.uuid4()

    # k = 0 rejected
    with pytest.raises(ValidationError):
        RetrievalEvalResult(
            query_id=qid,
            retriever_name="Deterministic",
            k=0,
            precision_at_k=1.0,
            recall_at_k=1.0,
            reciprocal_rank=1.0,
            ndcg_at_k=1.0,
        )

    # k < 0 rejected
    with pytest.raises(ValidationError):
        RetrievalEvalResult(
            query_id=qid,
            retriever_name="Deterministic",
            k=-5,
            precision_at_k=1.0,
            recall_at_k=1.0,
            reciprocal_rank=1.0,
            ndcg_at_k=1.0,
        )

    # bool k rejected
    with pytest.raises(ValidationError):
        RetrievalEvalResult(
            query_id=qid,
            retriever_name="Deterministic",
            k=True,  # type: ignore
            precision_at_k=1.0,
            recall_at_k=1.0,
            reciprocal_rank=1.0,
            ndcg_at_k=1.0,
        )


def test_retrieval_eval_result_metric_bounds():
    qid = uuid.uuid4()

    # Metric > 1.0 rejected
    with pytest.raises(ValidationError):
        RetrievalEvalResult(
            query_id=qid,
            retriever_name="Deterministic",
            k=5,
            precision_at_k=1.5,
            recall_at_k=1.0,
            reciprocal_rank=1.0,
            ndcg_at_k=1.0,
        )

    # Metric < 0.0 rejected
    with pytest.raises(ValidationError):
        RetrievalEvalResult(
            query_id=qid,
            retriever_name="Deterministic",
            k=5,
            precision_at_k=-0.1,
            recall_at_k=1.0,
            reciprocal_rank=1.0,
            ndcg_at_k=1.0,
        )


def test_retrieval_eval_result_negative_latency_rejected():
    qid = uuid.uuid4()
    with pytest.raises(ValidationError):
        RetrievalEvalResult(
            query_id=qid,
            retriever_name="Deterministic",
            k=5,
            precision_at_k=1.0,
            recall_at_k=1.0,
            reciprocal_rank=1.0,
            ndcg_at_k=1.0,
            latency_ms=-1.0,
        )


def test_retrieval_eval_result_invalid_retriever_name():
    qid = uuid.uuid4()
    with pytest.raises(ValidationError):
        RetrievalEvalResult(
            query_id=qid,
            retriever_name="",  # empty string rejected
            k=5,
            precision_at_k=1.0,
            recall_at_k=1.0,
            reciprocal_rank=1.0,
            ndcg_at_k=1.0,
        )


def test_retrieval_eval_result_immutability():
    res = RetrievalEvalResult(
        query_id=uuid.uuid4(),
        retriever_name="Deterministic",
        k=5,
        precision_at_k=1.0,
        recall_at_k=1.0,
        reciprocal_rank=1.0,
        ndcg_at_k=1.0,
    )
    with pytest.raises(ValidationError):
        res.precision_at_k = 0.5  # type: ignore


# ============================================================================
# 4. RetrieverBenchmarkReport Tests
# ============================================================================


def test_retriever_benchmark_report_valid_construction():
    res = RetrievalEvalResult(
        query_id=uuid.uuid4(),
        retriever_name="HybridHistoricalRetriever",
        k=5,
        retrieved_payment_ids=(uuid.uuid4(),),
        precision_at_k=1.0,
        recall_at_k=1.0,
        reciprocal_rank=1.0,
        ndcg_at_k=1.0,
        latency_ms=8.2,
    )

    report = RetrieverBenchmarkReport(
        retriever_name="HybridHistoricalRetriever",
        dataset_name="golden_retrieval_v1",
        num_queries=1,
        k_values=(1, 3, 5, 10),
        results=(res,),
        aggregate_metrics={
            "mean_precision_at_5": 1.0,
            "mean_recall_at_5": 1.0,
            "mrr": 1.0,
            "mean_ndcg_at_5": 1.0,
            "mean_latency_ms": 8.2,
        },
        metadata={"rrf_k": 60},
    )

    assert report.retriever_name == "HybridHistoricalRetriever"
    assert report.dataset_name == "golden_retrieval_v1"
    assert report.num_queries == 1
    assert report.k_values == (1, 3, 5, 10)
    assert len(report.results) == 1
    assert report.aggregate_metrics["mean_precision_at_5"] == 1.0
    assert report.metadata["rrf_k"] == 60
    assert isinstance(report.evaluated_at, datetime)


def test_retriever_benchmark_report_invalid_names():
    with pytest.raises(ValidationError):
        RetrieverBenchmarkReport(
            retriever_name="",  # empty
            dataset_name="golden",
            num_queries=1,
        )

    with pytest.raises(ValidationError):
        RetrieverBenchmarkReport(
            retriever_name="Hybrid",
            dataset_name="",  # empty
            num_queries=1,
        )


def test_retriever_benchmark_report_invalid_num_queries():
    with pytest.raises(ValidationError):
        RetrieverBenchmarkReport(
            retriever_name="Hybrid",
            dataset_name="golden",
            num_queries=-1,  # negative
        )


def test_retriever_benchmark_report_invalid_k_values():
    with pytest.raises(ValidationError):
        RetrieverBenchmarkReport(
            retriever_name="Hybrid",
            dataset_name="golden",
            num_queries=1,
            k_values=(1, 0, 5),  # 0 is invalid
        )

    with pytest.raises(ValidationError):
        RetrieverBenchmarkReport(
            retriever_name="Hybrid",
            dataset_name="golden",
            num_queries=1,
            k_values=(1, -3, 5),  # negative invalid
        )


def test_retriever_benchmark_report_immutability():
    report = RetrieverBenchmarkReport(
        retriever_name="Hybrid",
        dataset_name="golden",
        num_queries=0,
    )
    with pytest.raises(ValidationError):
        report.dataset_name = "other"  # type: ignore
