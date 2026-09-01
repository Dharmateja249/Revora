"""
Revora Retrieval Evaluation Runner.

Orchestrates evaluation of historical case retrievers against EvaluationCase datasets,
computing precision@K, recall@K, MRR, NDCG@K, and latency, and compiling aggregate
benchmark reports.
"""

import time
from collections.abc import Sequence
from datetime import datetime, timezone
from typing import (
    Any,
    Protocol,
    runtime_checkable,
)
from uuid import UUID

from app.context import CustomerRecoveryContext
from app.evaluation.metrics import (
    mean_reciprocal_rank,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)
from app.evaluation.schemas import (
    EvaluationCase,
    RetrievalEvalResult,
    RetrieverBenchmarkReport,
)
from app.historical_retrieval import HistoricalCase


@runtime_checkable
class RetrieverProtocol(Protocol):
    """
    Protocol defining the retrieval interface expected by RetrievalEvaluator.
    Supports either retrieve_relevant_cases or retrieve methods.
    """

    def retrieve_relevant_cases(
        self,
        context: CustomerRecoveryContext,
        top_k: int,
    ) -> Sequence[HistoricalCase]: ...


def _validate_k_values(k_values: Sequence[int]) -> tuple[int, ...]:
    """Validate and normalize benchmark depth values (k_values)."""
    if not isinstance(k_values, (list, tuple, set)):
        raise TypeError(
            f"k_values must be a sequence of positive integers, got {type(k_values).__name__}"
        )
    if not k_values:
        raise ValueError("k_values sequence cannot be empty.")

    normalized: list[int] = []
    seen: set[int] = set()
    for idx, item in enumerate(k_values):
        if isinstance(item, bool) or not isinstance(item, int):
            raise TypeError(
                f"Item at index {idx} in k_values must be an integer, got {type(item).__name__}"
            )
        if item <= 0:
            raise ValueError(
                f"Item at index {idx} in k_values must be positive (> 0), got {item}"
            )
        if item in seen:
            raise ValueError(f"Duplicate K value {item} found in k_values.")
        seen.add(item)
        normalized.append(item)

    return tuple(normalized)


class RetrievalEvaluator:
    """
    Generic, decoupled evaluation engine for benchmarking historical case retrievers.
    """

    def __init__(
        self,
        evaluation_cases: Sequence[EvaluationCase],
        dataset_name: str = "retrieval_golden_dataset_v1",
    ) -> None:
        if not isinstance(evaluation_cases, (list, tuple)):
            raise TypeError(
                f"evaluation_cases must be a sequence of EvaluationCase instances, got {type(evaluation_cases).__name__}"
            )
        for idx, case in enumerate(evaluation_cases):
            if not isinstance(case, EvaluationCase):
                raise TypeError(
                    f"Item at index {idx} in evaluation_cases must be EvaluationCase, got {type(case).__name__}"
                )

        if not isinstance(dataset_name, str) or not dataset_name.strip():
            raise ValueError(
                f"dataset_name must be a non-empty string, got: {dataset_name!r}"
            )

        self.evaluation_cases: tuple[EvaluationCase, ...] = tuple(evaluation_cases)
        self.dataset_name: str = dataset_name.strip()

    def _invoke_retriever(
        self,
        retriever: Any,
        context: CustomerRecoveryContext,
        top_k: int,
    ) -> Sequence[HistoricalCase]:
        """
        Invoke the retriever using its supported interface (retrieve_relevant_cases, retrieve, or __call__).
        Preserves original exceptions without suppressing them.
        """
        if hasattr(retriever, "retrieve_relevant_cases") and callable(
            retriever.retrieve_relevant_cases
        ):
            results = retriever.retrieve_relevant_cases(context=context, top_k=top_k)
        elif hasattr(retriever, "retrieve") and callable(retriever.retrieve):
            results = retriever.retrieve(context=context, top_k=top_k)
        elif callable(retriever):
            results = retriever(context=context, top_k=top_k)
        else:
            raise TypeError(
                f"Retriever of type '{type(retriever).__name__}' does not implement 'retrieve_relevant_cases', "
                f"'retrieve', or __call__."
            )

        if not isinstance(results, (list, tuple)):
            raise TypeError(
                f"Retriever returned invalid result type {type(results).__name__}, expected Sequence[HistoricalCase]"
            )

        for idx, item in enumerate(results):
            if not hasattr(item, "payment_id") or not isinstance(item.payment_id, UUID):
                raise TypeError(
                    f"Item at index {idx} returned by retriever lacks a valid payment_id UUID attribute: {item!r}"
                )

        return results

    def evaluate(
        self,
        retriever: Any,
        retriever_name: str,
        k_values: Sequence[int] = (1, 3, 5, 10),
    ) -> RetrieverBenchmarkReport:
        """
        Execute evaluation of the provided retriever over all configured EvaluationCase queries.

        Args:
            retriever: Any compatible retriever instance or callable.
            retriever_name: Non-empty descriptive identifier for the retriever.
            k_values: Sequence of ranking depths to evaluate (default: (1, 3, 5, 10)).

        Returns:
            RetrieverBenchmarkReport aggregating query-level and summary metrics.
        """
        if not isinstance(retriever_name, str) or not retriever_name.strip():
            raise ValueError(
                f"retriever_name must be a non-empty string, got: {retriever_name!r}"
            )
        clean_retriever_name = retriever_name.strip()

        valid_k_values = _validate_k_values(k_values)
        max_k = max(valid_k_values)

        all_results: list[RetrievalEvalResult] = []
        per_query_latencies: list[float] = []
        per_query_mrrs: list[float] = []
        metrics_by_k: dict[int, dict[str, list[float]]] = {
            k: {"precision": [], "recall": [], "ndcg": []} for k in valid_k_values
        }

        # Deterministic sequential processing over all evaluation cases
        for case in self.evaluation_cases:
            # 1. Monotonic elapsed latency measurement around retriever execution
            t_start = time.perf_counter()
            retrieved_cases = self._invoke_retriever(
                retriever=retriever,
                context=case.context,
                top_k=max_k,
            )
            t_elapsed_ms = (time.perf_counter() - t_start) * 1000.0
            per_query_latencies.append(t_elapsed_ms)

            # 2. Extract ranked payment UUIDs exactly without deduplication
            retrieved_pids = tuple(c.payment_id for c in retrieved_cases)

            # 3. Compute Query-level Reciprocal Rank (over full retrieved list)
            mrr = mean_reciprocal_rank(
                retrieved_payment_ids=retrieved_pids,
                ground_truth=case.ground_truth,
            )
            per_query_mrrs.append(mrr)

            # 4. Compute Metrics for each K
            for k in valid_k_values:
                prec = precision_at_k(
                    retrieved_payment_ids=retrieved_pids,
                    ground_truth=case.ground_truth,
                    k=k,
                )
                rec = recall_at_k(
                    retrieved_payment_ids=retrieved_pids,
                    ground_truth=case.ground_truth,
                    k=k,
                )
                ndcg = ndcg_at_k(
                    retrieved_payment_ids=retrieved_pids,
                    ground_truth=case.ground_truth,
                    k=k,
                )

                metrics_by_k[k]["precision"].append(prec)
                metrics_by_k[k]["recall"].append(rec)
                metrics_by_k[k]["ndcg"].append(ndcg)

                eval_result = RetrievalEvalResult(
                    query_id=case.query_id,
                    retriever_name=clean_retriever_name,
                    k=k,
                    retrieved_payment_ids=retrieved_pids[:k],
                    precision_at_k=prec,
                    recall_at_k=rec,
                    reciprocal_rank=mrr,
                    ndcg_at_k=ndcg,
                    latency_ms=t_elapsed_ms,
                    metadata={
                        "case_description": case.description,
                        "total_retrieved": len(retrieved_pids),
                    },
                )
                all_results.append(eval_result)

        # 5. Compute Aggregate Summary Metrics
        aggregate_metrics: dict[str, float] = {}
        num_queries = len(self.evaluation_cases)

        if num_queries > 0:
            aggregate_metrics["mrr"] = float(sum(per_query_mrrs) / num_queries)
            aggregate_metrics["mean_latency_ms"] = float(
                sum(per_query_latencies) / num_queries
            )

            for k in valid_k_values:
                prec_list = metrics_by_k[k]["precision"]
                rec_list = metrics_by_k[k]["recall"]
                ndcg_list = metrics_by_k[k]["ndcg"]

                aggregate_metrics[f"mean_precision_at_{k}"] = float(
                    sum(prec_list) / num_queries
                )
                aggregate_metrics[f"mean_recall_at_{k}"] = float(
                    sum(rec_list) / num_queries
                )
                aggregate_metrics[f"mean_ndcg_at_{k}"] = float(
                    sum(ndcg_list) / num_queries
                )
        else:
            aggregate_metrics["mrr"] = 0.0
            aggregate_metrics["mean_latency_ms"] = 0.0
            for k in valid_k_values:
                aggregate_metrics[f"mean_precision_at_{k}"] = 0.0
                aggregate_metrics[f"mean_recall_at_{k}"] = 0.0
                aggregate_metrics[f"mean_ndcg_at_{k}"] = 0.0

        return RetrieverBenchmarkReport(
            retriever_name=clean_retriever_name,
            dataset_name=self.dataset_name,
            num_queries=num_queries,
            k_values=valid_k_values,
            results=tuple(all_results),
            aggregate_metrics=aggregate_metrics,
            evaluated_at=datetime.now(timezone.utc),
            metadata={
                "k_values": list(valid_k_values),
                "num_evaluation_cases": num_queries,
            },
        )
