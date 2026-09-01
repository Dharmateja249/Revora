"""
Revora Retrieval Evaluation Metrics.

Pure, deterministic mathematical calculation of ranking and retrieval metrics
(Precision@K, Recall@K, Mean Reciprocal Rank, and NDCG@K) using binary and
graded relevance judgments.
"""

import math
from collections.abc import Sequence
from uuid import UUID

from app.evaluation.schemas import GroundTruthJudgment


def _validate_k(k: int) -> int:
    """Validate that k is a positive integer."""
    if isinstance(k, bool) or not isinstance(k, int):
        raise TypeError(f"k must be an integer, got {type(k).__name__}")
    if k <= 0:
        raise ValueError(f"k must be a positive integer (> 0), got {k}")
    return k


def _validate_and_index_ground_truth(
    ground_truth: Sequence[GroundTruthJudgment],
) -> dict[UUID, int]:
    """
    Validate ground truth sequence and return mapping of payment_id -> relevance_grade.

    Raises:
        TypeError: If ground_truth is not a sequence or contains invalid elements.
        ValueError: If duplicate payment_ids exist in ground_truth.
    """
    if not isinstance(ground_truth, (list, tuple)):
        raise TypeError(
            f"ground_truth must be a sequence of GroundTruthJudgment instances, got {type(ground_truth).__name__}"
        )

    gt_map: dict[UUID, int] = {}
    for idx, j in enumerate(ground_truth):
        if not isinstance(j, GroundTruthJudgment):
            raise TypeError(
                f"Item at index {idx} in ground_truth must be GroundTruthJudgment, got {type(j).__name__}"
            )
        if j.payment_id in gt_map:
            raise ValueError(
                f"Duplicate payment_id '{j.payment_id}' found in ground_truth judgments."
            )
        gt_map[j.payment_id] = j.relevance_grade

    return gt_map


def _validate_retrieved_payment_ids(
    retrieved_payment_ids: Sequence[UUID],
) -> Sequence[UUID]:
    """
    Validate retrieved payment IDs sequence and ensure uniqueness.

    Raises:
        TypeError: If retrieved_payment_ids is not a sequence or contains non-UUID items.
        ValueError: If duplicate payment_ids exist in retrieved_payment_ids.
    """
    if not isinstance(retrieved_payment_ids, (list, tuple)):
        raise TypeError(
            f"retrieved_payment_ids must be a sequence of UUIDs, got {type(retrieved_payment_ids).__name__}"
        )

    seen: set[UUID] = set()
    for idx, pid in enumerate(retrieved_payment_ids):
        if not isinstance(pid, UUID):
            raise TypeError(
                f"Item at index {idx} in retrieved_payment_ids must be a UUID instance, got {type(pid).__name__}"
            )
        if pid in seen:
            raise ValueError(
                f"Duplicate payment_id '{pid}' found in retrieved_payment_ids at index {idx}."
            )
        seen.add(pid)

    return retrieved_payment_ids


def precision_at_k(
    retrieved_payment_ids: Sequence[UUID],
    ground_truth: Sequence[GroundTruthJudgment],
    k: int,
) -> float:
    """
    Compute Precision@K for a single query.

    Formula:
        Precision@K = (number of relevant retrieved items in top K) / K

    Relevance semantics:
        Binary relevance: relevance_grade > 0 is relevant, relevance_grade == 0 is irrelevant.

    Args:
        retrieved_payment_ids: Ranked sequence of retrieved payment UUIDs.
        ground_truth: Sequence of GroundTruthJudgment objects for the query.
        k: Ranking cutoff depth (must be a positive integer).

    Returns:
        Precision score in [0.0, 1.0]. Returns 0.0 if retrieved list or ground truth is empty.
    """
    _validate_k(k)
    _validate_retrieved_payment_ids(retrieved_payment_ids)
    gt_map = _validate_and_index_ground_truth(ground_truth)

    if not retrieved_payment_ids or not gt_map:
        return 0.0

    retrieved_k = retrieved_payment_ids[:k]
    relevant_retrieved_count = sum(1 for pid in retrieved_k if gt_map.get(pid, 0) > 0)

    return float(relevant_retrieved_count / k)


def recall_at_k(
    retrieved_payment_ids: Sequence[UUID],
    ground_truth: Sequence[GroundTruthJudgment],
    k: int,
) -> float:
    """
    Compute Recall@K for a single query.

    Formula:
        Recall@K = (number of relevant items retrieved in top K) / (total relevant items in ground truth)

    Relevance semantics:
        Binary relevance: relevance_grade > 0 is relevant.
        Grade 0 judgments are excluded from the denominator.

    Args:
        retrieved_payment_ids: Ranked sequence of retrieved payment UUIDs.
        ground_truth: Sequence of GroundTruthJudgment objects for the query.
        k: Ranking cutoff depth (must be a positive integer).

    Returns:
        Recall score in [0.0, 1.0]. Returns 0.0 if there are zero relevant items in ground truth.
    """
    _validate_k(k)
    _validate_retrieved_payment_ids(retrieved_payment_ids)
    gt_map = _validate_and_index_ground_truth(ground_truth)

    total_relevant_in_gt = sum(1 for grade in gt_map.values() if grade > 0)
    if total_relevant_in_gt == 0:
        return 0.0

    if not retrieved_payment_ids:
        return 0.0

    retrieved_k = retrieved_payment_ids[:k]
    relevant_retrieved_count = sum(1 for pid in retrieved_k if gt_map.get(pid, 0) > 0)

    return float(relevant_retrieved_count / total_relevant_in_gt)


def mean_reciprocal_rank(
    retrieved_payment_ids: Sequence[UUID],
    ground_truth: Sequence[GroundTruthJudgment],
) -> float:
    """
    Compute Reciprocal Rank (RR) for a single query.

    Formula:
        RR = 1.0 / rank_of_first_relevant_result (1-based)

    Relevance semantics:
        Binary relevance: relevance_grade > 0 is relevant.

    Args:
        retrieved_payment_ids: Ranked sequence of retrieved payment UUIDs.
        ground_truth: Sequence of GroundTruthJudgment objects for the query.

    Returns:
        Reciprocal rank score in [0.0, 1.0]. Returns 0.0 if no relevant result is retrieved.
    """
    _validate_retrieved_payment_ids(retrieved_payment_ids)
    gt_map = _validate_and_index_ground_truth(ground_truth)

    if not retrieved_payment_ids or not gt_map:
        return 0.0

    for rank_1_based, pid in enumerate(retrieved_payment_ids, start=1):
        if gt_map.get(pid, 0) > 0:
            return float(1.0 / rank_1_based)

    return 0.0


def ndcg_at_k(
    retrieved_payment_ids: Sequence[UUID],
    ground_truth: Sequence[GroundTruthJudgment],
    k: int,
) -> float:
    """
    Compute Normalized Discounted Cumulative Gain (NDCG@K) for a single query.

    Formula:
        DCG@K = sum_{i=1}^{min(|retrieved|, K)} (2^{rel_i} - 1) / log_2(i + 1)
        IDCG@K = sum_{i=1}^{min(|ideal|, K)} (2^{ideal_rel_i} - 1) / log_2(i + 1)
        NDCG@K = DCG@K / IDCG@K

    Relevance semantics:
        Graded relevance: uses relevance_grade (0, 1, 2, 3) directly.
        Higher grades contribute exponentially more gain (2^g - 1).

    Args:
        retrieved_payment_ids: Ranked sequence of retrieved payment UUIDs.
        ground_truth: Sequence of GroundTruthJudgment objects for the query.
        k: Ranking cutoff depth (must be a positive integer).

    Returns:
        NDCG score in [0.0, 1.0]. Returns 0.0 if IDCG@K is 0.0.
    """
    _validate_k(k)
    _validate_retrieved_payment_ids(retrieved_payment_ids)
    gt_map = _validate_and_index_ground_truth(ground_truth)

    # Calculate Ideal DCG (IDCG@K)
    sorted_ideal_grades = sorted(gt_map.values(), reverse=True)[:k]
    idcg = 0.0
    for rank_1_based, grade in enumerate(sorted_ideal_grades, start=1):
        if grade > 0:
            gain = (2.0**grade) - 1.0
            discount = math.log2(rank_1_based + 1.0)
            idcg += gain / discount

    if idcg <= 1e-12:
        return 0.0

    # Calculate Actual DCG (DCG@K)
    retrieved_k = retrieved_payment_ids[:k]
    dcg = 0.0
    for rank_1_based, pid in enumerate(retrieved_k, start=1):
        grade = gt_map.get(pid, 0)
        if grade > 0:
            gain = (2.0**grade) - 1.0
            discount = math.log2(rank_1_based + 1.0)
            dcg += gain / discount

    ndcg = dcg / idcg
    return float(max(0.0, min(1.0, ndcg)))
