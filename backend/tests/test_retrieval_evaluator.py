"""
Revora Retrieval Evaluator Unit and Integration Tests.

Tests execution flow, metric computation, aggregation, error propagation, K validation,
determinism, and protocol compatibility of RetrievalEvaluator.
"""

from datetime import datetime, timezone
import inspect
from typing import List, Sequence
from uuid import UUID, uuid4
import pytest

from app.context import (
    CustomerContext,
    CustomerRecoveryContext,
    PaymentContext,
)
from app.evaluation import (
    EvaluationCase,
    GroundTruthJudgment,
    RetrievalEvalResult,
    RetrievalEvaluator,
    RetrieverBenchmarkReport,
    RetrieverProtocol,
)
from app.historical_retrieval import HistoricalCase


def _make_dummy_context(customer_id: UUID | None = None) -> CustomerRecoveryContext:
    cid = customer_id or uuid4()
    return CustomerRecoveryContext(
        customer=CustomerContext(customer_id=cid),
        current_payment=PaymentContext(
            payment_id=uuid4(),
            amount=500.0,
            currency="INR",
            payment_method="upi",
            status="failed",
            failure_reason="timeout",
        ),
    )


def _make_dummy_historical_case(payment_id: UUID, customer_id: UUID | None = None) -> HistoricalCase:
    return HistoricalCase(
        payment_id=payment_id,
        customer_id=customer_id or uuid4(),
        amount=500.0,
        currency="INR",
        payment_method="upi",
        failure_reason="timeout",
        recovery_action="retry_payment",
        recovery_status="succeeded",
        was_recovered=True,
    )


class StubRetriever:
    """Configurable test stub for historical case retrieval."""

    def __init__(self, mapping: dict[UUID, Sequence[HistoricalCase]] | None = None):
        self.mapping = mapping or {}
        self.invocations: List[tuple[CustomerRecoveryContext, int]] = []

    def retrieve_relevant_cases(
        self,
        context: CustomerRecoveryContext,
        top_k: int,
    ) -> Sequence[HistoricalCase]:
        self.invocations.append((context, top_k))
        curr_id = context.current_payment.payment_id if context.current_payment else None
        cases = self.mapping.get(curr_id, [])
        return cases[:top_k]


# ============================================================================
# 1. Basic Execution & Flow Tests
# ============================================================================


def test_evaluator_single_evaluation_case():
    qid = uuid4()
    context = _make_dummy_context()
    p1, p2 = uuid4(), uuid4()

    case = EvaluationCase(
        query_id=qid,
        context=context,
        ground_truth=(
            GroundTruthJudgment(payment_id=p1, relevance_grade=3),
            GroundTruthJudgment(payment_id=p2, relevance_grade=0),
        ),
    )

    stub = StubRetriever(mapping={context.current_payment.payment_id: [_make_dummy_historical_case(p1)]})
    evaluator = RetrievalEvaluator(evaluation_cases=[case], dataset_name="test_dataset")

    report = evaluator.evaluate(retriever=stub, retriever_name="StubRetriever", k_values=(1, 3))

    assert isinstance(report, RetrieverBenchmarkReport)
    assert report.retriever_name == "StubRetriever"
    assert report.dataset_name == "test_dataset"
    assert report.num_queries == 1
    assert report.k_values == (1, 3)
    assert len(report.results) == 2  # 1 query * 2 K values

    assert len(stub.invocations) == 1
    passed_context, passed_top_k = stub.invocations[0]
    assert passed_context == context
    assert passed_top_k == 3  # max_k


def test_evaluator_multiple_evaluation_cases():
    cases = []
    mapping = {}
    for _ in range(3):
        context = _make_dummy_context()
        p1 = uuid4()
        cases.append(
            EvaluationCase(
                query_id=uuid4(),
                context=context,
                ground_truth=(GroundTruthJudgment(payment_id=p1, relevance_grade=3),),
            )
        )
        mapping[context.current_payment.payment_id] = [_make_dummy_historical_case(p1)]

    stub = StubRetriever(mapping=mapping)
    evaluator = RetrievalEvaluator(evaluation_cases=cases)
    report = evaluator.evaluate(retriever=stub, retriever_name="StubRetriever", k_values=(1, 5))

    assert report.num_queries == 3
    assert len(report.results) == 6  # 3 queries * 2 K values
    assert len(stub.invocations) == 3


def test_evaluator_alternate_retrieve_method_support():
    """Verify retriever implementing .retrieve(context, top_k) is supported."""
    context = _make_dummy_context()
    p1 = uuid4()
    case = EvaluationCase(
        query_id=uuid4(),
        context=context,
        ground_truth=(GroundTruthJudgment(payment_id=p1, relevance_grade=3),),
    )

    class SemanticStyleRetriever:
        def retrieve(self, context: CustomerRecoveryContext, top_k: int) -> Sequence[HistoricalCase]:
            return [_make_dummy_historical_case(p1)]

    evaluator = RetrievalEvaluator(evaluation_cases=[case])
    report = evaluator.evaluate(retriever=SemanticStyleRetriever(), retriever_name="SemanticStyle")
    assert report.num_queries == 1
    assert report.aggregate_metrics["mean_precision_at_1"] == 1.0


# ============================================================================
# 2. K Values Configuration & Validation Tests
# ============================================================================


def test_evaluator_k_values_handling():
    case = EvaluationCase(
        query_id=uuid4(),
        context=_make_dummy_context(),
        ground_truth=(),
    )
    evaluator = RetrievalEvaluator(evaluation_cases=[case])
    stub = StubRetriever()

    # Valid custom K values
    report = evaluator.evaluate(stub, "Stub", k_values=[1, 3, 5, 10])
    assert report.k_values == (1, 3, 5, 10)

    # Invalid K <= 0
    with pytest.raises(ValueError):
        evaluator.evaluate(stub, "Stub", k_values=[1, 0, 5])

    with pytest.raises(ValueError):
        evaluator.evaluate(stub, "Stub", k_values=[1, -2, 5])

    # Bool K rejected
    with pytest.raises(TypeError):
        evaluator.evaluate(stub, "Stub", k_values=[1, True, 5])  # type: ignore

    # Duplicate K rejected
    with pytest.raises(ValueError):
        evaluator.evaluate(stub, "Stub", k_values=[1, 3, 3, 5])

    # Empty K sequence rejected
    with pytest.raises(ValueError):
        evaluator.evaluate(stub, "Stub", k_values=[])


# ============================================================================
# 3. Metrics Calculation & Aggregation Tests
# ============================================================================


def test_evaluator_perfect_retrieval():
    p1, p2, p3 = uuid4(), uuid4(), uuid4()
    context = _make_dummy_context()
    case = EvaluationCase(
        query_id=uuid4(),
        context=context,
        ground_truth=(
            GroundTruthJudgment(payment_id=p1, relevance_grade=3),
            GroundTruthJudgment(payment_id=p2, relevance_grade=2),
            GroundTruthJudgment(payment_id=p3, relevance_grade=1),
        ),
    )

    stub = StubRetriever(
        mapping={
            context.current_payment.payment_id: [
                _make_dummy_historical_case(p1),
                _make_dummy_historical_case(p2),
                _make_dummy_historical_case(p3),
            ]
        }
    )

    evaluator = RetrievalEvaluator(evaluation_cases=[case])
    report = evaluator.evaluate(stub, "PerfectStub", k_values=(1, 3))

    assert report.aggregate_metrics["mean_precision_at_1"] == pytest.approx(1.0)
    assert report.aggregate_metrics["mean_precision_at_3"] == pytest.approx(1.0)
    assert report.aggregate_metrics["mean_recall_at_1"] == pytest.approx(1 / 3)
    assert report.aggregate_metrics["mean_recall_at_3"] == pytest.approx(1.0)
    assert report.aggregate_metrics["mrr"] == pytest.approx(1.0)
    assert report.aggregate_metrics["mean_ndcg_at_1"] == pytest.approx(1.0)
    assert report.aggregate_metrics["mean_ndcg_at_3"] == pytest.approx(1.0)


def test_evaluator_zero_relevance_retrieval():
    p1, p_irrel = uuid4(), uuid4()
    context = _make_dummy_context()
    case = EvaluationCase(
        query_id=uuid4(),
        context=context,
        ground_truth=(
            GroundTruthJudgment(payment_id=p1, relevance_grade=3),
            GroundTruthJudgment(payment_id=p_irrel, relevance_grade=0),
        ),
    )

    stub = StubRetriever(
        mapping={
            context.current_payment.payment_id: [
                _make_dummy_historical_case(p_irrel),
            ]
        }
    )

    evaluator = RetrievalEvaluator(evaluation_cases=[case])
    report = evaluator.evaluate(stub, "ZeroStub", k_values=(1, 3))

    assert report.aggregate_metrics["mean_precision_at_1"] == pytest.approx(0.0)
    assert report.aggregate_metrics["mean_recall_at_1"] == pytest.approx(0.0)
    assert report.aggregate_metrics["mrr"] == pytest.approx(0.0)
    assert report.aggregate_metrics["mean_ndcg_at_1"] == pytest.approx(0.0)


def test_evaluator_multi_query_mean_averaging():
    # Query 1: Perfect (precision@1 = 1.0)
    p1 = uuid4()
    c1 = _make_dummy_context()
    case1 = EvaluationCase(
        query_id=uuid4(),
        context=c1,
        ground_truth=(GroundTruthJudgment(payment_id=p1, relevance_grade=3),),
    )

    # Query 2: Zero relevant (precision@1 = 0.0)
    p2, p_unrelated = uuid4(), uuid4()
    c2 = _make_dummy_context()
    case2 = EvaluationCase(
        query_id=uuid4(),
        context=c2,
        ground_truth=(GroundTruthJudgment(payment_id=p2, relevance_grade=3),),
    )

    stub = StubRetriever(
        mapping={
            c1.current_payment.payment_id: [_make_dummy_historical_case(p1)],
            c2.current_payment.payment_id: [_make_dummy_historical_case(p_unrelated)],
        }
    )

    evaluator = RetrievalEvaluator(evaluation_cases=[case1, case2])
    report = evaluator.evaluate(stub, "AvgStub", k_values=(1,))

    assert report.aggregate_metrics["mean_precision_at_1"] == pytest.approx(0.5)


def test_evaluator_empty_dataset():
    evaluator = RetrievalEvaluator(evaluation_cases=[], dataset_name="empty_dataset")
    stub = StubRetriever()
    report = evaluator.evaluate(stub, "EmptyStub", k_values=(1, 5))

    assert report.num_queries == 0
    assert len(report.results) == 0
    assert report.aggregate_metrics["mean_precision_at_1"] == 0.0
    assert report.aggregate_metrics["mrr"] == 0.0


# ============================================================================
# 4. Error Propagation & Edge Cases
# ============================================================================


def test_evaluator_retriever_exception_propagates():
    context = _make_dummy_context()
    case = EvaluationCase(query_id=uuid4(), context=context)

    class FailingRetriever:
        def retrieve_relevant_cases(self, context, top_k):
            raise ConnectionError("Vector database unavailable")

    evaluator = RetrievalEvaluator(evaluation_cases=[case])

    with pytest.raises(ConnectionError) as exc:
        evaluator.evaluate(FailingRetriever(), "FailingRetriever")
    assert "Vector database unavailable" in str(exc.value)


def test_evaluator_duplicate_retrieved_ids_fail_in_metrics():
    context = _make_dummy_context()
    dup_pid = uuid4()
    case = EvaluationCase(
        query_id=uuid4(),
        context=context,
        ground_truth=(GroundTruthJudgment(payment_id=dup_pid, relevance_grade=3),),
    )

    class DuplicateYieldingRetriever:
        def retrieve_relevant_cases(self, context, top_k):
            return [
                _make_dummy_historical_case(dup_pid),
                _make_dummy_historical_case(dup_pid),
            ]

    evaluator = RetrievalEvaluator(evaluation_cases=[case])

    with pytest.raises(ValueError) as exc:
        evaluator.evaluate(DuplicateYieldingRetriever(), "DuplicateRetriever")
    assert "Duplicate payment_id" in str(exc.value)


def test_evaluator_malformed_retriever_result():
    context = _make_dummy_context()
    case = EvaluationCase(query_id=uuid4(), context=context)

    class MalformedRetriever:
        def retrieve_relevant_cases(self, context, top_k):
            return ["not-a-historical-case"]  # type: ignore

    evaluator = RetrievalEvaluator(evaluation_cases=[case])

    with pytest.raises(TypeError):
        evaluator.evaluate(MalformedRetriever(), "Malformed")


# ============================================================================
# 5. Determinism & Isolation Tests
# ============================================================================


def test_evaluator_determinism():
    p1 = uuid4()
    context = _make_dummy_context()
    case = EvaluationCase(
        query_id=uuid4(),
        context=context,
        ground_truth=(GroundTruthJudgment(payment_id=p1, relevance_grade=3),),
    )

    stub = StubRetriever(mapping={context.current_payment.payment_id: [_make_dummy_historical_case(p1)]})
    evaluator = RetrievalEvaluator(evaluation_cases=[case])

    report1 = evaluator.evaluate(stub, "DetStub", k_values=(1, 3))
    report2 = evaluator.evaluate(stub, "DetStub", k_values=(1, 3))

    # Compare all deterministic metrics
    for metric_key in report1.aggregate_metrics:
        if metric_key != "mean_latency_ms":
            assert report1.aggregate_metrics[metric_key] == report2.aggregate_metrics[metric_key]

    assert len(report1.results) == len(report2.results)
    for r1, r2 in zip(report1.results, report2.results):
        assert r1.precision_at_k == r2.precision_at_k
        assert r1.recall_at_k == r2.recall_at_k
        assert r1.ndcg_at_k == r2.ndcg_at_k
        assert r1.reciprocal_rank == r2.reciprocal_rank


def test_evaluator_module_isolation():
    """Verify app/evaluation/evaluator.py does not import concrete production retrievers."""
    import app.evaluation.evaluator as eval_mod

    source = inspect.getsource(eval_mod)
    assert "from app.historical_retriever import HistoricalRetriever" not in source
    assert "from app.semantic_historical_retriever import SemanticHistoricalRetriever" not in source
    assert "from app.hybrid_historical_retriever import HybridHistoricalRetriever" not in source


def test_real_retrievers_signature_compatibility():
    """
    Verify concrete retriever classes satisfy the evaluator's expected method signatures.
    """
    from app.historical_retriever import HistoricalRetriever
    from app.hybrid_historical_retriever import HybridHistoricalRetriever
    from app.semantic_historical_retriever import SemanticHistoricalRetriever

    # 1. HistoricalRetriever has retrieve_relevant_cases(self, context, top_k)
    assert hasattr(HistoricalRetriever, "retrieve_relevant_cases")
    sig_det = inspect.signature(HistoricalRetriever.retrieve_relevant_cases)
    assert "context" in sig_det.parameters
    assert "top_k" in sig_det.parameters

    # 2. HybridHistoricalRetriever has retrieve_relevant_cases(self, context, top_k)
    assert hasattr(HybridHistoricalRetriever, "retrieve_relevant_cases")
    sig_hyb = inspect.signature(HybridHistoricalRetriever.retrieve_relevant_cases)
    assert "context" in sig_hyb.parameters
    assert "top_k" in sig_hyb.parameters

    # 3. SemanticHistoricalRetriever has retrieve(self, context, top_k)
    assert hasattr(SemanticHistoricalRetriever, "retrieve")
    sig_sem = inspect.signature(SemanticHistoricalRetriever.retrieve)
    assert "context" in sig_sem.parameters
    assert "top_k" in sig_sem.parameters
