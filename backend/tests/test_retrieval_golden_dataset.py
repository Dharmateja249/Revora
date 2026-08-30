"""
Revora Retrieval Golden Dataset Validation Tests.

Validates integrity, uniqueness, determinism, grade distributions, temporal isolation,
tenant isolation, and schema validity for the 50 golden evaluation cases.
"""

from collections import Counter
from datetime import timezone
import inspect
import sys
from typing import Set
from uuid import UUID
import pytest

from app.context import CustomerRecoveryContext
from app.evaluation.schemas import EvaluationCase, GroundTruthJudgment
from tests.fixtures.retrieval_golden_dataset import (
    GOLDEN_EVALUATION_CASES,
    get_golden_evaluation_cases,
)


def test_golden_dataset_size_and_contract():
    """Verify dataset contains at least 50 valid EvaluationCase instances."""
    cases = get_golden_evaluation_cases()
    assert isinstance(cases, tuple)
    assert len(cases) >= 50
    for case in cases:
        assert isinstance(case, EvaluationCase)
        assert isinstance(case.query_id, UUID)
        assert isinstance(case.context, CustomerRecoveryContext)


def test_golden_dataset_unique_query_ids():
    """Verify all 50 query_ids are strictly unique."""
    cases = get_golden_evaluation_cases()
    query_ids = [c.query_id for c in cases]
    assert len(query_ids) == len(set(query_ids))


def test_golden_dataset_ground_truth_payment_id_uniqueness_within_query():
    """Verify ground truth judgments within each query contain no duplicate payment_ids."""
    cases = get_golden_evaluation_cases()
    for case in cases:
        judgment_ids = [j.payment_id for j in case.ground_truth]
        assert len(judgment_ids) == len(set(judgment_ids)), f"Duplicate judgment in query {case.query_id}"


def test_golden_dataset_relevance_grade_bounds_and_distribution():
    """Verify all judgments use grades 0..3 and have a healthy distribution across all grades."""
    cases = get_golden_evaluation_cases()
    grade_counts = Counter()

    for case in cases:
        assert len(case.ground_truth) > 0, f"Query {case.query_id} has empty ground truth"
        for judgment in case.ground_truth:
            assert isinstance(judgment, GroundTruthJudgment)
            assert judgment.relevance_grade in (0, 1, 2, 3)
            grade_counts[judgment.relevance_grade] += 1

    # Verify every grade 0, 1, 2, 3 has meaningful representation
    assert grade_counts[3] >= 40, f"Insufficient Grade 3 judgments: {grade_counts[3]}"
    assert grade_counts[2] >= 30, f"Insufficient Grade 2 judgments: {grade_counts[2]}"
    assert grade_counts[1] >= 20, f"Insufficient Grade 1 judgments: {grade_counts[1]}"
    assert grade_counts[0] >= 50, f"Insufficient Grade 0 judgments: {grade_counts[0]}"


def test_golden_dataset_payment_methods_and_failure_categories_distribution():
    """Verify coverage across UPI, card, netbanking, wallet and diverse failure categories."""
    cases = get_golden_evaluation_cases()
    method_counts = Counter()
    failure_counts = Counter()

    for case in cases:
        pm = case.context.current_payment.payment_method
        fail = case.context.current_payment.failure_reason
        method_counts[pm] += 1
        failure_counts[fail] += 1

    # Payment rails
    assert method_counts["upi"] >= 10
    assert method_counts["card"] >= 15
    assert method_counts["netbanking"] >= 5
    assert method_counts["wallet"] >= 3

    # Failure categories
    assert failure_counts["bank_timeout"] >= 5
    assert failure_counts["insufficient_funds"] >= 5
    assert failure_counts["otp_expired"] >= 1
    assert failure_counts["card_expired"] >= 1
    assert failure_counts["system_error"] >= 2


def test_golden_dataset_determinism():
    """Verify dataset construction is 100% deterministic and reproducible."""
    cases_1 = get_golden_evaluation_cases()
    cases_2 = get_golden_evaluation_cases()

    assert len(cases_1) == len(cases_2)
    for c1, c2 in zip(cases_1, cases_2):
        assert c1.query_id == c2.query_id
        assert c1.context.customer.customer_id == c2.context.customer.customer_id
        assert c1.context.current_payment.payment_id == c2.context.current_payment.payment_id
        assert len(c1.ground_truth) == len(c2.ground_truth)
        for j1, j2 in zip(c1.ground_truth, c2.ground_truth):
            assert j1.payment_id == j2.payment_id
            assert j1.relevance_grade == j2.relevance_grade


def test_golden_dataset_timezone_awareness():
    """Verify all query timestamps and candidate timestamps are timezone-aware UTC."""
    cases = get_golden_evaluation_cases()
    for case in cases:
        assert case.created_at.tzinfo is not None
        assert case.created_at.tzinfo == timezone.utc
        assert case.context.current_payment.created_at.tzinfo == timezone.utc

        for hp in case.context.historical_payments:
            if hp.created_at is not None:
                assert hp.created_at.tzinfo == timezone.utc


def test_golden_dataset_temporal_isolation_boundary():
    """Verify no positive relevant candidate (Grade 1, 2, 3) violates temporal precedence."""
    cases = get_golden_evaluation_cases()
    for case in cases:
        curr_time = case.context.current_payment.created_at
        hist_by_id = {hp.payment_id: hp for hp in case.context.historical_payments}

        for judgment in case.ground_truth:
            if judgment.payment_id in hist_by_id:
                cand = hist_by_id[judgment.payment_id]
                if cand.created_at is not None and cand.created_at > curr_time:
                    # Future candidate MUST be Grade 0 (irrelevant / invalid)
                    assert judgment.relevance_grade == 0, (
                        f"Temporal violation: future candidate {cand.payment_id} "
                        f"has positive grade {judgment.relevance_grade} in query {case.query_id}"
                    )


def test_golden_dataset_current_payment_exclusion():
    """Verify current payment is never marked as a positive relevant historical candidate."""
    cases = get_golden_evaluation_cases()
    for case in cases:
        curr_pid = case.context.current_payment.payment_id
        for judgment in case.ground_truth:
            if judgment.payment_id == curr_pid:
                assert judgment.relevance_grade == 0, (
                    f"Current payment {curr_pid} must have relevance_grade 0 in query {case.query_id}"
                )


def test_golden_dataset_customer_isolation():
    """Verify cross-customer candidates are strictly marked as Grade 0."""
    cases = get_golden_evaluation_cases()
    for case in cases:
        cust_id = case.context.customer.customer_id
        hist_ids = {hp.payment_id for hp in case.context.historical_payments}

        for judgment in case.ground_truth:
            # If payment is not in the customer's historical payments and not the current payment, it is cross-customer
            if judgment.payment_id not in hist_ids and judgment.payment_id != case.context.current_payment.payment_id:
                assert judgment.relevance_grade == 0, (
                    f"Cross-customer payment {judgment.payment_id} must have relevance_grade 0 in query {case.query_id}"
                )


def test_golden_dataset_no_retriever_imports():
    """Verify fixture does not import production retrieval algorithms."""
    import tests.fixtures.retrieval_golden_dataset as fixture_mod

    source = inspect.getsource(fixture_mod)
    assert "HistoricalRetriever" not in source
    assert "SemanticHistoricalRetriever" not in source
    assert "HybridHistoricalRetriever" not in source
    assert "retrieve_relevant_cases" not in source
