"""
Comprehensive Test Suite for HybridHistoricalRetriever and Reciprocal Rank Fusion (RRF).

Tests:
1. RRF formula calculation helper
2. Both retrievers returning overlapping cases (deduplication & score accumulation)
3. Case ranked highly in both retrievers achieves highest fused rank
4. Case appearing only in deterministic retrieval
5. Case appearing only in semantic retrieval
6. Empty deterministic results (semantic-only fallback)
7. Empty semantic results (deterministic-only fallback)
8. Both retrievers returning empty lists (returns [])
9. Deterministic tie-breaking policy
10. top_k parameter enforcement and invalid value rejection
11. Invalid rrf_k value rejection
12. Preservation of HistoricalCase data & immutability
13. Dependency injection with mock retrievers
14. End-to-end integration test with real HistoricalRetriever & SemanticHistoricalRetriever
"""

import math
import uuid
from datetime import datetime, timedelta, timezone

import pytest
from app.context import (
    CustomerContext,
    CustomerRecoveryContext,
    HistoricalPaymentContext,
    PaymentContext,
    RecoveryOpportunityContext,
)
from app.embedding_service import get_embedding_service
from app.historical_retrieval import HistoricalCase
from app.historical_retriever import HistoricalRetriever
from app.hybrid_historical_retriever import (
    DEFAULT_RRF_K,
    HybridHistoricalRetriever,
    compute_rrf_score,
    retrieve_hybrid_historical_cases,
)
from app.retrieval_document import historical_case_to_document
from app.semantic_historical_retriever import SemanticHistoricalRetriever
from app.vector_index import VectorIndex

# ============================================================================
# 1. RRF Formula & Score Calculation
# ============================================================================


def test_compute_rrf_score_values():
    """Verify compute_rrf_score calculates 1 / (k + rank) correctly."""
    assert math.isclose(compute_rrf_score(1, rrf_k=60), 1.0 / 61.0, rel_tol=1e-9)
    assert math.isclose(compute_rrf_score(2, rrf_k=60), 1.0 / 62.0, rel_tol=1e-9)
    assert math.isclose(compute_rrf_score(10, rrf_k=60), 1.0 / 70.0, rel_tol=1e-9)
    assert math.isclose(compute_rrf_score(1, rrf_k=20), 1.0 / 21.0, rel_tol=1e-9)

    with pytest.raises(ValueError):
        compute_rrf_score(0)


# ============================================================================
# 2. Mock Retrievers for Unit Testing
# ============================================================================


class MockRetriever:
    """Mock retriever returning a predefined list of HistoricalCase instances."""

    def __init__(self, cases: list[HistoricalCase]):
        self.cases = cases
        self.invoked = False

    def retrieve_relevant_cases(self, context, top_k=5):
        self.invoked = True
        return self.cases[:top_k]

    def retrieve(self, context, top_k=5):
        self.invoked = True
        return self.cases[:top_k]


def _create_case(
    pid: uuid.UUID,
    cid: uuid.UUID,
    amount: float,
    method: str,
    reason: str,
    rel_score: float = 0.5,
):
    return HistoricalCase(
        payment_id=pid,
        customer_id=cid,
        amount=amount,
        currency="INR",
        payment_method=method,
        failure_reason=reason,
        recovery_status="recovered",
        amount_recovered=amount,
        was_recovered=True,
        relevance_score=rel_score,
        created_at=datetime(2026, 1, 1, 10, 0, 0, tzinfo=timezone.utc),
    )


# ============================================================================
# 3. Fusion Scenarios (Overlapping, Unilateral, Accumulation)
# ============================================================================


def test_case_ranked_highly_in_both_achieves_top_fused_rank():
    """Verify a case appearing in both retrievers accumulates RRF score and ranks #1."""
    cid = uuid.uuid4()
    p_both = uuid.uuid4()  # In both: det rank 1, sem rank 1
    p_det_only = uuid.uuid4()  # In det only: rank 2
    p_sem_only = uuid.uuid4()  # In sem only: rank 2

    c_both = _create_case(p_both, cid, 1000.0, "card", "timeout", 0.9)
    c_det = _create_case(p_det_only, cid, 2000.0, "card", "timeout", 0.8)
    c_sem = _create_case(p_sem_only, cid, 3000.0, "upi", "timeout", 0.8)

    mock_det = MockRetriever([c_both, c_det])
    mock_sem = MockRetriever([c_both, c_sem])

    hybrid = HybridHistoricalRetriever(
        deterministic_retriever=mock_det,  # type: ignore
        semantic_retriever=mock_sem,  # type: ignore
        rrf_k=60,
    )

    context = CustomerRecoveryContext(
        customer=CustomerContext(customer_id=cid),
        current_payment=PaymentContext(
            payment_id=uuid.uuid4(),
            amount=1000.0,
            payment_method="card",
            status="failed",
        ),
    )

    results = hybrid.retrieve_relevant_cases(context, top_k=5)

    assert len(results) == 3
    # c_both must rank #1
    assert results[0].payment_id == p_both
    assert results[0].relevance_score == 1.0  # Normalized (1/61 + 1/61) / (2/61) = 1.0
    assert results[0].metadata["deterministic_rank"] == 1
    assert results[0].metadata["semantic_rank"] == 1
    assert results[0].metadata["raw_rrf_score"] == round(1 / 61 + 1 / 61, 6)

    # c_det and c_sem both have RRF score = 1/62
    remaining_ids = {results[1].payment_id, results[2].payment_id}
    assert remaining_ids == {p_det_only, p_sem_only}


def test_unilateral_cases_handled_correctly():
    """Verify cases appearing in only one retriever are included without error."""
    cid = uuid.uuid4()
    p1 = uuid.uuid4()
    p2 = uuid.uuid4()

    c1 = _create_case(p1, cid, 1000.0, "card", "timeout", 0.8)
    c2 = _create_case(p2, cid, 2000.0, "upi", "timeout", 0.8)

    mock_det = MockRetriever([c1])
    mock_sem = MockRetriever([c2])

    hybrid = HybridHistoricalRetriever(
        deterministic_retriever=mock_det,  # type: ignore
        semantic_retriever=mock_sem,  # type: ignore
        rrf_k=60,
    )

    context = CustomerRecoveryContext(
        customer=CustomerContext(customer_id=cid),
        current_payment=PaymentContext(
            payment_id=uuid.uuid4(),
            amount=1000.0,
            payment_method="card",
            status="failed",
        ),
    )

    results = hybrid.retrieve_relevant_cases(context, top_k=5)

    assert len(results) == 2
    assert {r.payment_id for r in results} == {p1, p2}
    # Both have rank 1 in their respective retriever -> same RRF score 1/61
    assert results[0].metadata["raw_rrf_score"] == round(1 / 61, 6)
    assert results[1].metadata["raw_rrf_score"] == round(1 / 61, 6)


def test_empty_deterministic_results():
    """Verify hybrid retriever works cleanly when deterministic retriever returns []."""
    cid = uuid.uuid4()
    p1 = uuid.uuid4()
    c1 = _create_case(p1, cid, 1000.0, "card", "timeout", 0.8)

    mock_det = MockRetriever([])
    mock_sem = MockRetriever([c1])

    hybrid = HybridHistoricalRetriever(
        deterministic_retriever=mock_det,  # type: ignore
        semantic_retriever=mock_sem,  # type: ignore
    )
    context = CustomerRecoveryContext(
        customer=CustomerContext(customer_id=cid),
        current_payment=PaymentContext(
            payment_id=uuid.uuid4(),
            amount=1000.0,
            payment_method="card",
            status="failed",
        ),
    )

    results = hybrid.retrieve_relevant_cases(context, top_k=5)
    assert len(results) == 1
    assert results[0].payment_id == p1
    assert results[0].metadata["deterministic_rank"] is None
    assert results[0].metadata["semantic_rank"] == 1


def test_empty_semantic_results():
    """Verify hybrid retriever works cleanly when semantic retriever returns []."""
    cid = uuid.uuid4()
    p1 = uuid.uuid4()
    c1 = _create_case(p1, cid, 1000.0, "card", "timeout", 0.8)

    mock_det = MockRetriever([c1])
    mock_sem = MockRetriever([])

    hybrid = HybridHistoricalRetriever(
        deterministic_retriever=mock_det,  # type: ignore
        semantic_retriever=mock_sem,  # type: ignore
    )
    context = CustomerRecoveryContext(
        customer=CustomerContext(customer_id=cid),
        current_payment=PaymentContext(
            payment_id=uuid.uuid4(),
            amount=1000.0,
            payment_method="card",
            status="failed",
        ),
    )

    results = hybrid.retrieve_relevant_cases(context, top_k=5)
    assert len(results) == 1
    assert results[0].payment_id == p1
    assert results[0].metadata["deterministic_rank"] == 1
    assert results[0].metadata["semantic_rank"] is None


def test_both_retrievers_empty_returns_empty_list():
    """Verify [] returned when both retrievers have no results."""
    mock_det = MockRetriever([])
    mock_sem = MockRetriever([])

    hybrid = HybridHistoricalRetriever(
        deterministic_retriever=mock_det,  # type: ignore
        semantic_retriever=mock_sem,  # type: ignore
    )
    context = CustomerRecoveryContext(
        customer=CustomerContext(customer_id=uuid.uuid4()),
        current_payment=PaymentContext(
            payment_id=uuid.uuid4(),
            amount=1000.0,
            payment_method="card",
            status="failed",
        ),
    )

    assert hybrid.retrieve_relevant_cases(context, top_k=5) == []


# ============================================================================
# 4. Deterministic Tie-Breaking
# ============================================================================


def test_deterministic_tie_breaking_policy():
    """Verify tie-breaking on identical RRF score uses best_rank -> score_sum -> recency -> payment_id."""
    cid = uuid.uuid4()
    now = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)

    id_a = uuid.UUID("00000000-0000-0000-0000-000000000001")
    id_b = uuid.UUID("00000000-0000-0000-0000-000000000002")

    # Case A: rank 1 in Det, none in Sem (RRF = 1/61)
    case_a = HistoricalCase(
        payment_id=id_a,
        customer_id=cid,
        amount=1000.0,
        currency="INR",
        payment_method="card",
        recovery_status="recovered",
        relevance_score=0.8,
        created_at=now,
    )

    # Case B: rank 1 in Sem, none in Det (RRF = 1/61)
    case_b = HistoricalCase(
        payment_id=id_b,
        customer_id=cid,
        amount=1000.0,
        currency="INR",
        payment_method="card",
        recovery_status="recovered",
        relevance_score=0.8,
        created_at=now,
    )

    # Add in reverse order
    mock_det = MockRetriever([case_a])
    mock_sem = MockRetriever([case_b])

    hybrid = HybridHistoricalRetriever(
        deterministic_retriever=mock_det,  # type: ignore
        semantic_retriever=mock_sem,  # type: ignore
    )
    context = CustomerRecoveryContext(
        customer=CustomerContext(customer_id=cid),
        current_payment=PaymentContext(
            payment_id=uuid.uuid4(),
            amount=1000.0,
            payment_method="card",
            status="failed",
        ),
    )

    results = hybrid.retrieve_relevant_cases(context, top_k=2)

    assert len(results) == 2
    # RRF scores are identical (1/61)
    assert results[0].metadata["raw_rrf_score"] == results[1].metadata["raw_rrf_score"]
    # Best ranks are identical (1 vs 1)
    # Raw scores are identical (0.8 vs 0.8)
    # Timestamps are identical
    # Tie broken by UUID string: id_a < id_b
    assert results[0].payment_id == id_a
    assert results[1].payment_id == id_b


# ============================================================================
# 5. Validation & Top-K Boundaries
# ============================================================================


def test_top_k_limits_and_validation():
    """Verify top_k limits results and invalid values raise ValueError."""
    cid = uuid.uuid4()
    cases = [
        _create_case(uuid.uuid4(), cid, float(100 * i), "card", "timeout", 0.5)
        for i in range(1, 10)
    ]

    mock_det = MockRetriever(cases)
    mock_sem = MockRetriever([])

    hybrid = HybridHistoricalRetriever(
        deterministic_retriever=mock_det,  # type: ignore
        semantic_retriever=mock_sem,  # type: ignore
    )
    context = CustomerRecoveryContext(
        customer=CustomerContext(customer_id=cid),
        current_payment=PaymentContext(
            payment_id=uuid.uuid4(),
            amount=1000.0,
            payment_method="card",
            status="failed",
        ),
    )

    results_3 = hybrid.retrieve_relevant_cases(context, top_k=3)
    assert len(results_3) == 3

    # Invalid top_k
    with pytest.raises(ValueError):
        hybrid.retrieve_relevant_cases(context, top_k=0)

    with pytest.raises(ValueError):
        hybrid.retrieve_relevant_cases(context, top_k=-5)

    with pytest.raises(ValueError):
        hybrid.retrieve_relevant_cases(context, top_k="3")  # type: ignore


def test_invalid_rrf_k_rejected():
    """Verify non-positive or non-int rrf_k raises ValueError."""
    with pytest.raises(ValueError):
        HybridHistoricalRetriever(rrf_k=0)

    with pytest.raises(ValueError):
        HybridHistoricalRetriever(rrf_k=-10)

    with pytest.raises(ValueError):
        HybridHistoricalRetriever(rrf_k="60")  # type: ignore


def test_invalid_context_type_raises_type_error():
    """Verify passing non-CustomerRecoveryContext raises TypeError."""
    hybrid = HybridHistoricalRetriever(
        deterministic_retriever=MockRetriever([]),  # type: ignore
        semantic_retriever=MockRetriever([]),  # type: ignore
    )
    with pytest.raises(TypeError):
        hybrid.retrieve_relevant_cases("invalid_context")  # type: ignore


# ============================================================================
# 6. End-to-End Integration Pipeline Test
# ============================================================================


def test_end_to_end_hybrid_retrieval_pipeline():
    """
    Verify complete hybrid retrieval flow with real component implementations:
    CustomerRecoveryContext
        -> deterministic retrieval (HistoricalRetriever)
        -> semantic retrieval (SemanticHistoricalRetriever via VectorIndex & EmbeddingService)
        -> RRF rank fusion
        -> final HistoricalCase[]
    """
    cid = uuid.uuid4()
    now = datetime(2026, 2, 1, 12, 0, 0, tzinfo=timezone.utc)

    # 1. Historical Case 1: Insufficient funds (Strong deterministic and semantic match)
    p1 = uuid.uuid4()
    c1 = HistoricalCase(
        payment_id=p1,
        customer_id=cid,
        amount=2500.0,
        currency="INR",
        payment_method="card",
        failure_reason="insufficient_funds",
        recovery_action="payment_link",
        recovery_status="recovered",
        amount_recovered=2500.0,
        was_recovered=True,
        created_at=now - timedelta(days=2),
    )

    # 2. Historical Case 2: Timeout on UPI (Semantic match on timeout, different method)
    p2 = uuid.uuid4()
    c2 = HistoricalCase(
        payment_id=p2,
        customer_id=cid,
        amount=500.0,
        currency="INR",
        payment_method="upi",
        failure_reason="bank_timeout",
        recovery_action="smart_retry",
        recovery_status="recovered",
        amount_recovered=500.0,
        was_recovered=True,
        created_at=now - timedelta(days=5),
    )

    # 3. Setup Semantic Retriever with Indexed Historical Documents
    emb_service = get_embedding_service()
    v_index = VectorIndex(dimension=emb_service.dimension)

    doc_1 = historical_case_to_document(c1)
    doc_2 = historical_case_to_document(c2)
    v_index.add(doc_1, emb_service.embed(doc_1.text))
    v_index.add(doc_2, emb_service.embed(doc_2.text))

    sem_retriever = SemanticHistoricalRetriever(
        vector_index=v_index,
        embedding_service=emb_service,
    )

    # 4. Setup Deterministic Retriever with Context Historical Payments
    h1_ctx = HistoricalPaymentContext(
        payment_id=p1,
        amount=2500.0,
        currency="INR",
        payment_method="card",
        status="succeeded",
        failure_reason="insufficient_funds",
        was_recovered=True,
        recovery_action="payment_link",
        created_at=now - timedelta(days=2),
    )
    h2_ctx = HistoricalPaymentContext(
        payment_id=p2,
        amount=500.0,
        currency="INR",
        payment_method="upi",
        status="succeeded",
        failure_reason="bank_timeout",
        was_recovered=True,
        recovery_action="smart_retry",
        created_at=now - timedelta(days=5),
    )

    det_retriever = HistoricalRetriever(db_session=None)

    # 5. Query Context: Failed Card Payment with Insufficient Funds
    query_pid = uuid.uuid4()
    context = CustomerRecoveryContext(
        customer=CustomerContext(customer_id=cid),
        current_payment=PaymentContext(
            payment_id=query_pid,
            amount=2500.0,
            currency="INR",
            payment_method="card",
            status="failed",
            failure_reason="insufficient_funds",
            created_at=now,
        ),
        current_opportunity=RecoveryOpportunityContext(
            opportunity_id=uuid.uuid4(),
            status="open",
            revenue_at_risk=2500.0,
            recommended_action="payment_link",
        ),
        historical_payments=[h2_ctx, h1_ctx],  # Reverse order
    )

    # 6. Execute Hybrid Retrieval with RRF
    results = retrieve_hybrid_historical_cases(
        context=context,
        deterministic_retriever=det_retriever,
        semantic_retriever=sem_retriever,
        top_k=2,
        rrf_k=60,
    )

    assert len(results) == 2
    # Case 1 (Card + Insufficient Funds) ranked #1 in both retrievers -> ranks #1 in fused output
    assert results[0].payment_id == p1
    assert results[0].relevance_score == 1.0  # Perfect agreement across both engines
    assert results[0].metadata["fusion_method"] == "rrf"
    assert results[0].metadata["deterministic_rank"] == 1
    assert results[0].metadata["semantic_rank"] == 1
    assert results[0].was_recovered is True
    assert results[0].recovery_action == "payment_link"

    # Case 2 ranks #2
    assert results[1].payment_id == p2
    assert results[1].relevance_score < results[0].relevance_score


def test_hybrid_retriever_without_semantic_retriever_deterministic_only():
    """Verify HybridHistoricalRetriever works in deterministic-only mode when semantic_retriever is None."""
    cid = uuid.uuid4()
    p1 = uuid.uuid4()
    c1 = _create_case(p1, cid, 1000.0, "card", "timeout", 0.8)

    mock_det = MockRetriever([c1])
    hybrid = HybridHistoricalRetriever(
        deterministic_retriever=mock_det,  # type: ignore
        semantic_retriever=None,
        rrf_k=60,
    )

    assert hybrid.semantic_retriever is None

    context = CustomerRecoveryContext(
        customer=CustomerContext(customer_id=cid),
        current_payment=PaymentContext(
            payment_id=uuid.uuid4(),
            amount=1000.0,
            payment_method="card",
            status="failed",
        ),
    )

    results = hybrid.retrieve_relevant_cases(context, top_k=5)
    assert len(results) == 1
    assert results[0].payment_id == p1
    assert results[0].metadata["deterministic_rank"] == 1
    assert results[0].metadata["semantic_rank"] is None
    assert mock_det.invoked is True


def test_hybrid_retriever_default_instantiation_without_arguments():
    """Verify HybridHistoricalRetriever() can be constructed with no arguments without raising TypeError."""
    hybrid = HybridHistoricalRetriever()
    assert hybrid.semantic_retriever is None
    assert hybrid.deterministic_retriever is not None
    assert hybrid.rrf_k == DEFAULT_RRF_K
