"""
Revora Retrieval Metrics Unit Tests.

Comprehensive tests for pure metric functions:
precision_at_k, recall_at_k, mean_reciprocal_rank, and ndcg_at_k.
"""

import math
import uuid

import pytest
from app.evaluation.metrics import (
    mean_reciprocal_rank,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)
from app.evaluation.schemas import GroundTruthJudgment


def _uuid() -> uuid.UUID:
    return uuid.uuid4()


# ============================================================================
# 1. Precision@K Tests
# ============================================================================


def test_precision_all_relevant():
    p1, p2, p3 = _uuid(), _uuid(), _uuid()
    gt = [
        GroundTruthJudgment(payment_id=p1, relevance_grade=3),
        GroundTruthJudgment(payment_id=p2, relevance_grade=2),
        GroundTruthJudgment(payment_id=p3, relevance_grade=1),
    ]
    retrieved = [p1, p2, p3]

    score = precision_at_k(retrieved, gt, k=3)
    assert score == pytest.approx(1.0)


def test_precision_none_relevant():
    p1, p2, p3 = _uuid(), _uuid(), _uuid()
    gt = [
        GroundTruthJudgment(payment_id=p1, relevance_grade=0),
        GroundTruthJudgment(payment_id=p2, relevance_grade=0),
    ]
    retrieved = [p1, p2, p3]

    score = precision_at_k(retrieved, gt, k=3)
    assert score == pytest.approx(0.0)


def test_precision_mixed_relevance():
    p1, p2, p3, p4 = _uuid(), _uuid(), _uuid(), _uuid()
    gt = [
        GroundTruthJudgment(payment_id=p1, relevance_grade=3),
        GroundTruthJudgment(payment_id=p2, relevance_grade=0),
        GroundTruthJudgment(payment_id=p3, relevance_grade=1),
        GroundTruthJudgment(payment_id=p4, relevance_grade=0),
    ]
    retrieved = [p1, p2, p3, p4]

    # At K=4: 2 relevant (p1, p3) out of 4 -> 0.5
    assert precision_at_k(retrieved, gt, k=4) == pytest.approx(0.5)
    # At K=2: 1 relevant (p1) out of 2 -> 0.5
    assert precision_at_k(retrieved, gt, k=2) == pytest.approx(0.5)
    # At K=1: 1 relevant (p1) out of 1 -> 1.0
    assert precision_at_k(retrieved, gt, k=1) == pytest.approx(1.0)


def test_precision_grade_0_excluded():
    p1, p2 = _uuid(), _uuid()
    gt = [
        GroundTruthJudgment(payment_id=p1, relevance_grade=0),
        GroundTruthJudgment(payment_id=p2, relevance_grade=0),
    ]
    retrieved = [p1, p2]
    assert precision_at_k(retrieved, gt, k=2) == pytest.approx(0.0)


def test_precision_k_smaller_than_result_list():
    p1, p2, p3, p4 = _uuid(), _uuid(), _uuid(), _uuid()
    gt = [
        GroundTruthJudgment(payment_id=p1, relevance_grade=3),
        GroundTruthJudgment(payment_id=p2, relevance_grade=0),
        GroundTruthJudgment(payment_id=p3, relevance_grade=3),
        GroundTruthJudgment(payment_id=p4, relevance_grade=3),
    ]
    retrieved = [p1, p2, p3, p4]

    # At K=2: only [p1, p2] evaluated. p1 is relevant -> 1/2 = 0.5
    assert precision_at_k(retrieved, gt, k=2) == pytest.approx(0.5)


def test_precision_empty_retrieved_list():
    gt = [GroundTruthJudgment(payment_id=_uuid(), relevance_grade=3)]
    assert precision_at_k([], gt, k=5) == pytest.approx(0.0)


def test_precision_empty_ground_truth():
    retrieved = [_uuid(), _uuid()]
    assert precision_at_k(retrieved, [], k=2) == pytest.approx(0.0)


def test_precision_invalid_k():
    gt = [GroundTruthJudgment(payment_id=_uuid(), relevance_grade=3)]
    retrieved = [_uuid()]

    with pytest.raises(ValueError):
        precision_at_k(retrieved, gt, k=0)

    with pytest.raises(ValueError):
        precision_at_k(retrieved, gt, k=-3)

    with pytest.raises(TypeError):
        precision_at_k(retrieved, gt, k=True)  # type: ignore

    with pytest.raises(TypeError):
        precision_at_k(retrieved, gt, k="5")  # type: ignore


def test_precision_duplicate_retrieved_ids_rejected():
    dup_id = _uuid()
    gt = [GroundTruthJudgment(payment_id=dup_id, relevance_grade=3)]
    retrieved = [dup_id, dup_id]

    with pytest.raises(ValueError) as exc:
        precision_at_k(retrieved, gt, k=2)
    assert "Duplicate payment_id" in str(exc.value)


# ============================================================================
# 2. Recall@K Tests
# ============================================================================


def test_recall_all_relevant_retrieved():
    p1, p2, p3 = _uuid(), _uuid(), _uuid()
    gt = [
        GroundTruthJudgment(payment_id=p1, relevance_grade=3),
        GroundTruthJudgment(payment_id=p2, relevance_grade=2),
        GroundTruthJudgment(payment_id=p3, relevance_grade=1),
    ]
    retrieved = [p1, p2, p3]

    assert recall_at_k(retrieved, gt, k=3) == pytest.approx(1.0)


def test_recall_partial_recall():
    p1, p2, p3, p4 = _uuid(), _uuid(), _uuid(), _uuid()
    gt = [
        GroundTruthJudgment(payment_id=p1, relevance_grade=3),
        GroundTruthJudgment(payment_id=p2, relevance_grade=2),
        GroundTruthJudgment(payment_id=p3, relevance_grade=1),
        GroundTruthJudgment(payment_id=p4, relevance_grade=1),
    ]
    retrieved = [p1, p2]  # Retrieved 2 of 4 relevant

    assert recall_at_k(retrieved, gt, k=2) == pytest.approx(2 / 4)


def test_recall_zero_recall():
    p1, p2, p3 = _uuid(), _uuid(), _uuid()
    gt = [
        GroundTruthJudgment(payment_id=p1, relevance_grade=3),
    ]
    retrieved = [p2, p3]  # Disjoint

    assert recall_at_k(retrieved, gt, k=2) == pytest.approx(0.0)


def test_recall_grade_0_excluded_from_denominator():
    p1, p2, p3 = _uuid(), _uuid(), _uuid()
    gt = [
        GroundTruthJudgment(payment_id=p1, relevance_grade=3),
        GroundTruthJudgment(payment_id=p2, relevance_grade=0),  # Irrelevant
        GroundTruthJudgment(payment_id=p3, relevance_grade=0),  # Irrelevant
    ]
    retrieved = [p1]

    # Total relevant in GT = 1 (p1). Retrieved p1 -> Recall = 1/1 = 1.0 (not 1/3)
    assert recall_at_k(retrieved, gt, k=1) == pytest.approx(1.0)


def test_recall_k_truncation():
    p1, p2, p3 = _uuid(), _uuid(), _uuid()
    gt = [
        GroundTruthJudgment(payment_id=p1, relevance_grade=3),
        GroundTruthJudgment(payment_id=p2, relevance_grade=3),
        GroundTruthJudgment(payment_id=p3, relevance_grade=3),
    ]
    retrieved = [p1, p2, p3]

    # At K=1, only p1 counts -> 1/3
    assert recall_at_k(retrieved, gt, k=1) == pytest.approx(1 / 3)
    # At K=2, [p1, p2] count -> 2/3
    assert recall_at_k(retrieved, gt, k=2) == pytest.approx(2 / 3)


def test_recall_empty_ground_truth_and_no_relevant_gt():
    retrieved = [_uuid()]
    assert recall_at_k(retrieved, [], k=5) == pytest.approx(0.0)

    # GT has only grade 0
    gt_zero = [GroundTruthJudgment(payment_id=_uuid(), relevance_grade=0)]
    assert recall_at_k(retrieved, gt_zero, k=5) == pytest.approx(0.0)


def test_recall_invalid_k():
    gt = [GroundTruthJudgment(payment_id=_uuid(), relevance_grade=3)]
    retrieved = [_uuid()]

    with pytest.raises(ValueError):
        recall_at_k(retrieved, gt, k=0)

    with pytest.raises(ValueError):
        recall_at_k(retrieved, gt, k=-1)

    with pytest.raises(TypeError):
        recall_at_k(retrieved, gt, k=False)  # type: ignore


def test_recall_duplicate_retrieved_ids_rejected():
    dup_id = _uuid()
    gt = [GroundTruthJudgment(payment_id=dup_id, relevance_grade=3)]
    retrieved = [dup_id, dup_id]

    with pytest.raises(ValueError):
        recall_at_k(retrieved, gt, k=2)


# ============================================================================
# 3. Mean Reciprocal Rank (MRR) Tests
# ============================================================================


def test_mrr_first_result_relevant():
    p1, p2 = _uuid(), _uuid()
    gt = [
        GroundTruthJudgment(payment_id=p1, relevance_grade=3),
        GroundTruthJudgment(payment_id=p2, relevance_grade=2),
    ]
    retrieved = [p1, p2]

    # First relevant at rank 1 -> 1/1 = 1.0
    assert mean_reciprocal_rank(retrieved, gt) == pytest.approx(1.0)


def test_mrr_relevant_at_rank_2():
    p1, p2, p3 = _uuid(), _uuid(), _uuid()
    gt = [
        GroundTruthJudgment(payment_id=p1, relevance_grade=0),
        GroundTruthJudgment(payment_id=p2, relevance_grade=2),
    ]
    retrieved = [p1, p2, p3]

    # First relevant at rank 2 -> 1/2 = 0.5
    assert mean_reciprocal_rank(retrieved, gt) == pytest.approx(0.5)


def test_mrr_relevant_at_rank_4():
    p1, p2, p3, p4 = _uuid(), _uuid(), _uuid(), _uuid()
    gt = [
        GroundTruthJudgment(payment_id=p1, relevance_grade=0),
        GroundTruthJudgment(payment_id=p2, relevance_grade=0),
        GroundTruthJudgment(payment_id=p3, relevance_grade=0),
        GroundTruthJudgment(payment_id=p4, relevance_grade=1),
    ]
    retrieved = [p1, p2, p3, p4]

    # First relevant at rank 4 -> 1/4 = 0.25
    assert mean_reciprocal_rank(retrieved, gt) == pytest.approx(0.25)


def test_mrr_no_relevant_result():
    p1, p2 = _uuid(), _uuid()
    gt = [
        GroundTruthJudgment(payment_id=p1, relevance_grade=0),
    ]
    retrieved = [p1, p2]

    assert mean_reciprocal_rank(retrieved, gt) == pytest.approx(0.0)


def test_mrr_empty_inputs():
    assert mean_reciprocal_rank([], []) == pytest.approx(0.0)
    assert mean_reciprocal_rank([_uuid()], []) == pytest.approx(0.0)
    assert mean_reciprocal_rank(
        [], [GroundTruthJudgment(payment_id=_uuid(), relevance_grade=3)]
    ) == pytest.approx(0.0)


def test_mrr_duplicate_retrieved_ids_rejected():
    dup_id = _uuid()
    gt = [GroundTruthJudgment(payment_id=dup_id, relevance_grade=3)]
    retrieved = [dup_id, dup_id]

    with pytest.raises(ValueError):
        mean_reciprocal_rank(retrieved, gt)


# ============================================================================
# 4. NDCG@K Tests
# ============================================================================


def test_ndcg_perfect_ranking():
    p1, p2, p3 = _uuid(), _uuid(), _uuid()
    gt = [
        GroundTruthJudgment(payment_id=p1, relevance_grade=3),
        GroundTruthJudgment(payment_id=p2, relevance_grade=2),
        GroundTruthJudgment(payment_id=p3, relevance_grade=1),
    ]
    retrieved = [p1, p2, p3]

    # Perfect ordering matches ideal DCG -> NDCG = 1.0
    assert ndcg_at_k(retrieved, gt, k=3) == pytest.approx(1.0)


def test_ndcg_reversed_ranking_produces_lower_score():
    p1, p2, p3 = _uuid(), _uuid(), _uuid()
    gt = [
        GroundTruthJudgment(payment_id=p1, relevance_grade=3),
        GroundTruthJudgment(payment_id=p2, relevance_grade=2),
        GroundTruthJudgment(payment_id=p3, relevance_grade=1),
    ]
    # Ideal: [p1 (3), p2 (2), p3 (1)]
    # Reversed: [p3 (1), p2 (2), p1 (3)]
    reversed_retrieved = [p3, p2, p1]

    score = ndcg_at_k(reversed_retrieved, gt, k=3)
    assert score < 1.0
    assert score > 0.0

    # Explicit manual verification:
    # Gains: g(3) = 2^3 - 1 = 7, g(2) = 2^2 - 1 = 3, g(1) = 2^1 - 1 = 1
    # IDCG = 7/log2(2) + 3/log2(3) + 1/log2(4) = 7/1 + 3/1.58496 + 1/2 = 7 + 1.89279 + 0.5 = 9.39279
    # DCG = 1/log2(2) + 3/log2(3) + 7/log2(4) = 1/1 + 3/1.58496 + 7/2 = 1 + 1.89279 + 3.5 = 6.39279
    # Expected NDCG = 6.39279 / 9.39279 ≈ 0.6806
    expected_idcg = (7.0 / 1.0) + (3.0 / math.log2(3)) + (1.0 / 2.0)
    expected_dcg = (1.0 / 1.0) + (3.0 / math.log2(3)) + (7.0 / 2.0)
    assert score == pytest.approx(expected_dcg / expected_idcg, rel=1e-4)


def test_ndcg_grade_3_outranks_grade_2():
    p3, p2 = _uuid(), _uuid()
    gt = [
        GroundTruthJudgment(payment_id=p3, relevance_grade=3),
        GroundTruthJudgment(payment_id=p2, relevance_grade=2),
    ]

    score_p3_first = ndcg_at_k([p3, p2], gt, k=2)
    score_p2_first = ndcg_at_k([p2, p3], gt, k=2)

    assert score_p3_first == pytest.approx(1.0)
    assert score_p3_first > score_p2_first


def test_ndcg_grade_2_outranks_grade_1():
    p2, p1 = _uuid(), _uuid()
    gt = [
        GroundTruthJudgment(payment_id=p2, relevance_grade=2),
        GroundTruthJudgment(payment_id=p1, relevance_grade=1),
    ]

    score_p2_first = ndcg_at_k([p2, p1], gt, k=2)
    score_p1_first = ndcg_at_k([p1, p2], gt, k=2)

    assert score_p2_first == pytest.approx(1.0)
    assert score_p2_first > score_p1_first


def test_ndcg_all_grade_0_returns_zero():
    p1, p2 = _uuid(), _uuid()
    gt = [
        GroundTruthJudgment(payment_id=p1, relevance_grade=0),
        GroundTruthJudgment(payment_id=p2, relevance_grade=0),
    ]
    retrieved = [p1, p2]

    # IDCG is 0.0 -> NDCG = 0.0
    assert ndcg_at_k(retrieved, gt, k=2) == pytest.approx(0.0)


def test_ndcg_empty_ground_truth_and_retrieved():
    assert ndcg_at_k([], [], k=5) == pytest.approx(0.0)
    assert ndcg_at_k([_uuid()], [], k=5) == pytest.approx(0.0)
    assert ndcg_at_k(
        [], [GroundTruthJudgment(payment_id=_uuid(), relevance_grade=3)], k=5
    ) == pytest.approx(0.0)


def test_ndcg_k_truncation():
    p1, p2, p3 = _uuid(), _uuid(), _uuid()
    gt = [
        GroundTruthJudgment(payment_id=p1, relevance_grade=3),
        GroundTruthJudgment(payment_id=p2, relevance_grade=2),
        GroundTruthJudgment(payment_id=p3, relevance_grade=1),
    ]
    retrieved = [p1, p2, p3]

    # At K=1: only rank 1 evaluated. Retrieved p1 (grade 3), ideal is grade 3 -> 1.0
    assert ndcg_at_k(retrieved, gt, k=1) == pytest.approx(1.0)


def test_ndcg_missing_retrieved_documents():
    p1, p2, p3 = _uuid(), _uuid(), _uuid()
    unindexed = _uuid()
    gt = [
        GroundTruthJudgment(payment_id=p1, relevance_grade=3),
        GroundTruthJudgment(payment_id=p2, relevance_grade=2),
        GroundTruthJudgment(payment_id=p3, relevance_grade=1),
    ]
    # Retrieved unindexed document (grade 0 in GT) at rank 1, then p1 (grade 3)
    retrieved = [unindexed, p1]

    score = ndcg_at_k(retrieved, gt, k=2)
    assert score > 0.0
    assert score < 1.0


def test_ndcg_invalid_k():
    gt = [GroundTruthJudgment(payment_id=_uuid(), relevance_grade=3)]
    retrieved = [_uuid()]

    with pytest.raises(ValueError):
        ndcg_at_k(retrieved, gt, k=0)

    with pytest.raises(ValueError):
        ndcg_at_k(retrieved, gt, k=-2)

    with pytest.raises(TypeError):
        ndcg_at_k(retrieved, gt, k=True)  # type: ignore


def test_ndcg_duplicate_retrieved_ids_rejected():
    dup_id = _uuid()
    gt = [GroundTruthJudgment(payment_id=dup_id, relevance_grade=3)]
    retrieved = [dup_id, dup_id]

    with pytest.raises(ValueError):
        ndcg_at_k(retrieved, gt, k=2)


# ============================================================================
# 5. Property Invariants Tests
# ============================================================================


def test_metric_property_invariants():
    """Verify mathematical bounds [0, 1] across diverse generated permutations."""
    p_ids = [_uuid() for _ in range(5)]
    gt = [
        GroundTruthJudgment(payment_id=p_ids[0], relevance_grade=3),
        GroundTruthJudgment(payment_id=p_ids[1], relevance_grade=2),
        GroundTruthJudgment(payment_id=p_ids[2], relevance_grade=1),
        GroundTruthJudgment(payment_id=p_ids[3], relevance_grade=0),
        GroundTruthJudgment(payment_id=p_ids[4], relevance_grade=0),
    ]

    permutations = [
        [p_ids[0], p_ids[1], p_ids[2]],
        [p_ids[2], p_ids[1], p_ids[0]],
        [p_ids[3], p_ids[4]],
        [p_ids[0]],
        [p_ids[4], p_ids[0]],
    ]

    for ret in permutations:
        for k in (1, 2, 3, 5):
            p = precision_at_k(ret, gt, k=k)
            r = recall_at_k(ret, gt, k=k)
            nd = ndcg_at_k(ret, gt, k=k)

            assert 0.0 <= p <= 1.0
            assert 0.0 <= r <= 1.0
            assert 0.0 <= nd <= 1.0

        mrr = mean_reciprocal_rank(ret, gt)
        assert 0.0 <= mrr <= 1.0
