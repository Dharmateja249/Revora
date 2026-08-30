"""
Revora Retrieval Benchmarking & Comparative Reporting.

Executes reproducible evaluation of HistoricalRetriever (Deterministic),
SemanticHistoricalRetriever (Semantic), and HybridHistoricalRetriever (Hybrid RRF)
across the golden evaluation dataset, producing per-retriever benchmark reports
and comparative performance summaries.
"""

from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple
from uuid import UUID

from app.embedding_service import EmbeddingService, get_embedding_service
from app.evaluation.evaluator import RetrievalEvaluator
from app.evaluation.schemas import EvaluationCase, RetrieverBenchmarkReport
from app.historical_retrieval import HistoricalCase
from app.historical_retriever import HistoricalRetriever
from app.hybrid_historical_retriever import HybridHistoricalRetriever
from app.retrieval_document import historical_case_to_document
from app.semantic_historical_retriever import SemanticHistoricalRetriever
from app.vector_index import VectorIndex
from tests.fixtures.retrieval_golden_dataset import get_golden_evaluation_cases


def populate_benchmark_vector_index(
    evaluation_cases: Sequence[EvaluationCase],
    embedding_service: Optional[EmbeddingService] = None,
) -> VectorIndex:
    """
    Populate a VectorIndex with all historical payment documents from the evaluation cases.

    Ensures that semantic and hybrid retrievers have access to the exact candidate corpus
    represented across the benchmark.
    """
    svc = embedding_service or get_embedding_service()
    vector_index = VectorIndex(dimension=svc.dimension)

    indexed_payment_ids: set[UUID] = set()

    for case in evaluation_cases:
        customer_id = case.context.customer.customer_id
        ext_cust_id = case.context.customer.external_customer_id

        for hp in case.context.historical_payments:
            if hp.payment_id in indexed_payment_ids:
                continue

            hist_case = HistoricalCase(
                payment_id=hp.payment_id,
                customer_id=customer_id,
                external_payment_id=hp.external_payment_id,
                external_customer_id=ext_cust_id,
                amount=hp.amount,
                currency=hp.currency,
                payment_method=hp.payment_method,
                failure_reason=hp.failure_reason,
                recovery_action=hp.recovery_action,
                recovery_status="recovered" if hp.was_recovered else "failed",
                amount_recovered=hp.amount if hp.was_recovered else 0.0,
                was_recovered=hp.was_recovered,
                created_at=hp.created_at,
            )

            doc = historical_case_to_document(hist_case)
            vec = svc.embed(doc.text)
            vector_index.add(doc, vec)
            indexed_payment_ids.add(hp.payment_id)

    return vector_index


def run_benchmark(
    evaluation_cases: Optional[Sequence[EvaluationCase]] = None,
    k_values: Sequence[int] = (1, 3, 5, 10),
    embedding_service: Optional[EmbeddingService] = None,
) -> Dict[str, RetrieverBenchmarkReport]:
    """
    Execute benchmark evaluation across all three Revora historical retrievers.

    Args:
        evaluation_cases: Sequence of EvaluationCase instances (defaults to golden dataset).
        k_values: Ranking depths to evaluate (defaults to (1, 3, 5, 10)).
        embedding_service: EmbeddingService instance for semantic components.

    Returns:
        Dictionary mapping retriever name to its RetrieverBenchmarkReport.
    """
    cases = tuple(evaluation_cases) if evaluation_cases is not None else get_golden_evaluation_cases()
    svc = embedding_service or get_embedding_service()

    # 1. Populate shared vector index for semantic evaluation
    vector_index = populate_benchmark_vector_index(cases, svc)

    # 2. Instantiate all 3 production retrievers
    deterministic_retriever = HistoricalRetriever(db_session=None)
    semantic_retriever = SemanticHistoricalRetriever(
        vector_index=vector_index,
        embedding_service=svc,
    )
    hybrid_retriever = HybridHistoricalRetriever(
        deterministic_retriever=deterministic_retriever,
        semantic_retriever=semantic_retriever,
    )

    evaluator = RetrievalEvaluator(evaluation_cases=cases)

    # 3. Evaluate each retriever against identical queries and K values
    reports: Dict[str, RetrieverBenchmarkReport] = {
        "DeterministicHistoricalRetriever": evaluator.evaluate(
            retriever=deterministic_retriever,
            retriever_name="DeterministicHistoricalRetriever",
            k_values=k_values,
        ),
        "SemanticHistoricalRetriever": evaluator.evaluate(
            retriever=semantic_retriever,
            retriever_name="SemanticHistoricalRetriever",
            k_values=k_values,
        ),
        "HybridHistoricalRetriever": evaluator.evaluate(
            retriever=hybrid_retriever,
            retriever_name="HybridHistoricalRetriever",
            k_values=k_values,
        ),
    }

    return reports


def compare_benchmarks(
    reports: Mapping[str, RetrieverBenchmarkReport],
) -> Dict[str, Any]:
    """
    Generate a structured comparative analysis of retriever benchmark reports.

    Args:
        reports: Mapping of retriever name to RetrieverBenchmarkReport.

    Returns:
        Structured dictionary comparing key metrics and latency across retrievers.
    """
    summary: Dict[str, Any] = {
        "retrievers": list(reports.keys()),
        "metrics_comparison": {},
    }

    if not reports:
        return summary

    # Discover evaluated K values from the first report
    first_report = next(iter(reports.values()))
    k_values = first_report.k_values

    # Metric keys to compare
    metric_keys = ["mrr", "mean_latency_ms"]
    for k in k_values:
        metric_keys.extend([
            f"mean_precision_at_{k}",
            f"mean_recall_at_{k}",
            f"mean_ndcg_at_{k}",
        ])

    for m_key in metric_keys:
        summary["metrics_comparison"][m_key] = {
            name: rep.aggregate_metrics.get(m_key, 0.0)
            for name, rep in reports.items()
        }

    return summary


def format_comparison_markdown(
    reports: Mapping[str, RetrieverBenchmarkReport],
) -> str:
    """
    Format a clean GitHub-Flavored Markdown comparison table from benchmark reports.
    """
    if not reports:
        return "*No benchmark reports available.*"

    retriever_names = list(reports.keys())
    first_report = next(iter(reports.values()))
    k_values = first_report.k_values

    lines: List[str] = [
        "| Metric | " + " | ".join(retriever_names) + " |",
        "| :--- | " + " | ".join([":---:" for _ in retriever_names]) + " |",
    ]

    def _fmt(val: Any) -> str:
        if isinstance(val, float):
            return f"{val:.4f}"
        return str(val)

    # Core Summary Metrics
    lines.append(
        "| **MRR** | "
        + " | ".join([_fmt(reports[n].aggregate_metrics.get("mrr", 0.0)) for n in retriever_names])
        + " |"
    )
    lines.append(
        "| **Latency (ms)** | "
        + " | ".join([f"{reports[n].aggregate_metrics.get('mean_latency_ms', 0.0):.2f} ms" for n in retriever_names])
        + " |"
    )

    for k in k_values:
        lines.append(
            f"| **Precision@{k}** | "
            + " | ".join([_fmt(reports[n].aggregate_metrics.get(f"mean_precision_at_{k}", 0.0)) for n in retriever_names])
            + " |"
        )
        lines.append(
            f"| **Recall@{k}** | "
            + " | ".join([_fmt(reports[n].aggregate_metrics.get(f"mean_recall_at_{k}", 0.0)) for n in retriever_names])
            + " |"
        )
        lines.append(
            f"| **NDCG@{k}** | "
            + " | ".join([_fmt(reports[n].aggregate_metrics.get(f"mean_ndcg_at_{k}", 0.0)) for n in retriever_names])
            + " |"
        )

    return "\n".join(lines)
