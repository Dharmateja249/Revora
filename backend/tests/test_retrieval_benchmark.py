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
    cases = get_golden_evaluation_cases()
    reports = run_benchmark(evaluation_cases=cases, k_values=(1, 3, 5, 10))

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


def test_run_benchmark_requires_explicit_evaluation_cases():
    with pytest.raises(ValueError, match="evaluation_cases must be explicitly provided"):
        run_benchmark(evaluation_cases=None)  # type: ignore


def test_cross_customer_negative_candidates_and_tenant_isolation():
    """
    Verify that foreign-customer negative cases are materialized in the shared vector index
    with their real foreign customer identity, and that production retrievers enforce
    tenant isolation while an unsafe retriever leaks them and is penalized by metrics.
    """
    from uuid import UUID
    from app.embedding_service import get_embedding_service
    from app.evaluation.evaluator import RetrievalEvaluator
    from app.historical_retrieval import HistoricalCase
    from app.semantic_historical_retriever import SemanticHistoricalRetriever

    cases = get_golden_evaluation_cases()
    svc = get_embedding_service()
    vector_index = populate_benchmark_vector_index(cases, svc)

    # 1. Verify foreign-customer candidates are present in the shared vector index with their real customer_id
    foreign_cust_id = UUID("00000000-0000-0014-0000-000000000063")  # customer 99 in hex (0x63)
    foreign_docs = [
        doc for _, doc in vector_index._entries.values()
        if doc.metadata.get("customer_id") == str(foreign_cust_id)
    ]
    assert len(foreign_docs) > 0, "Cross-customer negative cases must be materialized in the index."

    # 2. Test Scenario 1 (Customer 1):
    case1 = cases[0]
    cust1_id = case1.context.customer.customer_id
    assert cust1_id != foreign_cust_id

    # 3. Production SemanticHistoricalRetriever strictly filters on Customer 1
    prod_retriever = SemanticHistoricalRetriever(vector_index=vector_index, embedding_service=svc)
    retrieved_prod = prod_retriever.retrieve(case1.context, top_k=10)
    for c in retrieved_prod:
        assert c.customer_id == cust1_id, "Production retriever must NEVER leak foreign customer cases."

    # 4. Intentionally UNSAFE retriever (bypasses tenant isolation filter)
    class UnsafeLeakyRetriever:
        def __init__(self, v_idx, emb_svc):
            self.v_idx = v_idx
            self.emb_svc = emb_svc

        def retrieve_relevant_cases(self, context, top_k=10):
            q_text = f"{context.current_payment.payment_method} {context.current_payment.failure_reason or ''}"
            q_vec = self.emb_svc.embed(q_text)
            matches = self.v_idx.search(q_vec, top_k=top_k)
            # Returns raw documents from any customer without filtering
            results = []
            for match in matches:
                doc = match.document
                meta = doc.metadata
                results.append(
                    HistoricalCase(
                        payment_id=UUID(meta["payment_id"]),
                        customer_id=UUID(meta["customer_id"]),
                        amount=float(meta.get("amount", 0.0)),
                        currency=meta.get("currency", "INR"),
                        payment_method=meta.get("payment_method", "upi"),
                        recovery_status=meta.get("recovery_status", "recovered" if meta.get("was_recovered") else "failed"),
                        status="succeeded" if meta.get("was_recovered") else "failed",
                        was_recovered=bool(meta.get("was_recovered", False)),
                    )
                )
            return results

    unsafe_retriever = UnsafeLeakyRetriever(vector_index, svc)
    retrieved_unsafe = unsafe_retriever.retrieve_relevant_cases(case1.context, top_k=50)

    # Verify unsafe retriever leaks foreign customer case
    leaked_foreign = [c for c in retrieved_unsafe if c.customer_id == foreign_cust_id]
    assert len(leaked_foreign) > 0, "Unsafe retriever should have matched the foreign customer candidate."

    # 5. Verify that the evaluation runner penalizes the leaky retriever on Scenario 1
    safe_eval = RetrievalEvaluator(evaluation_cases=[case1]).evaluate(retriever=prod_retriever, retriever_name="ProdSemantic", k_values=(3,)).results[0]
    unsafe_eval = RetrievalEvaluator(evaluation_cases=[case1]).evaluate(retriever=unsafe_retriever, retriever_name="UnsafeLeaky", k_values=(3,)).results[0]

    # The unsafe retriever should score lower or equal Precision/NDCG because foreign candidate has relevance Grade 0
    assert safe_eval.precision_at_k >= unsafe_eval.precision_at_k

