"""
Comprehensive Test Suite for SemanticHistoricalRetriever.

Tests:
1. Canonical query text construction matches canonical document format
2. Current context produces embedding and invokes vector search
3. Semantically similar historical case is retrieved as top result
4. top_k validation: rejects <=0, non-int, and limits results
5. Similarity score is preserved in HistoricalCase relevance_score
6. Deterministic ordering of results
7. Empty index returns []
8. Current payment is strictly excluded
9. Cross-customer / tenant records are strictly isolated
10. Temporal filtering excludes future cases
11. Strict dependency injection of VectorIndex and EmbeddingService
12. Malformed / invalid inputs raise clear exceptions
13. Embedding failure propagation (no silent swallowing)
14. Vector index failure propagation (no silent swallowing)
15. End-to-end pipeline integration test
"""

from datetime import datetime, timezone, timedelta
import uuid
import pytest

from app.context import (
    CustomerContext,
    CustomerRecoveryContext,
    PaymentContext,
    RecoveryOpportunityContext,
)
from app.embedding_service import (
    DeterministicLocalEmbeddingProvider,
    EmbeddingProvider,
    EmbeddingService,
    get_embedding_service,
)
from app.historical_retrieval import HistoricalCase
from app.retrieval_document import (
    RetrievalDocument,
    historical_case_to_document,
)
from app.semantic_historical_retriever import (
    SemanticHistoricalRetriever,
    construct_canonical_query_text,
)
from app.vector_index import VectorIndex


# ============================================================================
# 1. Canonical Query Representation
# ============================================================================


def test_canonical_query_text_construction():
    """Verify query text matches the exact 8-line format used by RetrievalDocument."""
    cust_id = uuid.uuid4()
    pay_id = uuid.uuid4()

    context = CustomerRecoveryContext(
        customer=CustomerContext(customer_id=cust_id),
        current_payment=PaymentContext(
            payment_id=pay_id,
            amount=2500.0,
            currency="INR",
            payment_method="card",
            status="failed",
            failure_reason="insufficient_funds",
        ),
        current_opportunity=RecoveryOpportunityContext(
            opportunity_id=uuid.uuid4(),
            status="open",
            revenue_at_risk=2500.0,
            recommended_action="payment_link",
        ),
    )

    query_text = construct_canonical_query_text(context)
    lines = query_text.split("\n")

    assert len(lines) == 8
    assert lines[0] == "failure_reason: insufficient_funds"
    assert lines[1] == "payment_method: card"
    assert lines[2] == "amount: 2500.00"
    assert lines[3] == "currency: INR"
    assert lines[4] == "recovery_action: payment_link"
    assert lines[5] == "recovery_status: open"
    assert lines[6] == "was_recovered: false"
    assert lines[7] == "amount_recovered: 0.00"


# ============================================================================
# 2. Semantic Similarity Retrieval & Ranking
# ============================================================================


def test_semantic_retrieval_ranks_most_similar_case_first():
    """Verify retriever ranks the most semantically similar historical case first."""
    customer_id = uuid.uuid4()
    embedding_service = get_embedding_service()
    vector_index = VectorIndex(dimension=embedding_service.dimension)

    t0 = datetime(2026, 1, 1, 10, 0, 0, tzinfo=timezone.utc)

    # Historical Case 1: High semantic match (same failure reason and payment method)
    case_match = HistoricalCase(
        payment_id=uuid.uuid4(),
        customer_id=customer_id,
        amount=1500.0,
        currency="INR",
        payment_method="upi",
        failure_reason="bank_timeout",
        recovery_action="smart_retry",
        recovery_status="recovered",
        amount_recovered=1500.0,
        was_recovered=True,
        created_at=t0,
    )

    # Historical Case 2: Low semantic match (different failure reason and payment method)
    case_other = HistoricalCase(
        payment_id=uuid.uuid4(),
        customer_id=customer_id,
        amount=9000.0,
        currency="INR",
        payment_method="card",
        failure_reason="card_expired",
        recovery_action="change_payment_method",
        recovery_status="failed",
        amount_recovered=0.0,
        was_recovered=False,
        created_at=t0,
    )

    # Index both historical cases
    doc_match = historical_case_to_document(case_match)
    doc_other = historical_case_to_document(case_other)
    vector_index.add(doc_match, embedding_service.embed(doc_match.text))
    vector_index.add(doc_other, embedding_service.embed(doc_other.text))

    # Query Context: Active failed UPI payment with bank_timeout
    curr_payment_id = uuid.uuid4()
    context = CustomerRecoveryContext(
        customer=CustomerContext(customer_id=customer_id),
        current_payment=PaymentContext(
            payment_id=curr_payment_id,
            amount=1500.0,
            currency="INR",
            payment_method="upi",
            status="failed",
            failure_reason="bank_timeout",
            created_at=t0 + timedelta(days=5),
        ),
        current_opportunity=RecoveryOpportunityContext(
            opportunity_id=uuid.uuid4(),
            status="open",
            revenue_at_risk=1500.0,
            recommended_action="smart_retry",
        ),
    )

    retriever = SemanticHistoricalRetriever(
        vector_index=vector_index,
        embedding_service=embedding_service,
    )

    results = retriever.retrieve(context, top_k=5)

    assert len(results) == 2
    assert results[0].payment_id == case_match.payment_id
    assert results[1].payment_id == case_other.payment_id
    assert results[0].relevance_score > results[1].relevance_score
    assert isinstance(results[0], HistoricalCase)
    assert results[0].was_recovered is True
    assert results[0].recovery_action == "smart_retry"


# ============================================================================
# 3. Isolation & Safety Boundaries
# ============================================================================


def test_current_payment_excluded():
    """Verify that current payment is never returned as historical evidence."""
    customer_id = uuid.uuid4()
    current_payment_id = uuid.uuid4()
    embedding_service = get_embedding_service()
    vector_index = VectorIndex(dimension=embedding_service.dimension)

    # Index a document with the current payment's ID
    current_case = HistoricalCase(
        payment_id=current_payment_id,
        customer_id=customer_id,
        amount=1000.0,
        currency="INR",
        payment_method="card",
        failure_reason="insufficient_funds",
        recovery_status="open",
    )
    doc_curr = historical_case_to_document(current_case)
    vector_index.add(doc_curr, embedding_service.embed(doc_curr.text))

    context = CustomerRecoveryContext(
        customer=CustomerContext(customer_id=customer_id),
        current_payment=PaymentContext(
            payment_id=current_payment_id,
            amount=1000.0,
            payment_method="card",
            status="failed",
            failure_reason="insufficient_funds",
        ),
    )

    retriever = SemanticHistoricalRetriever(
        vector_index=vector_index,
        embedding_service=embedding_service,
    )
    results = retriever.retrieve(context)

    # Current payment must be filtered out
    assert len(results) == 0


def test_cross_customer_tenant_isolation():
    """Verify that records belonging to another customer are never retrieved."""
    cust_a_id = uuid.uuid4()
    cust_b_id = uuid.uuid4()
    embedding_service = get_embedding_service()
    vector_index = VectorIndex(dimension=embedding_service.dimension)

    # Index Customer B's payment (perfect match)
    case_b = HistoricalCase(
        payment_id=uuid.uuid4(),
        customer_id=cust_b_id,
        amount=1000.0,
        currency="INR",
        payment_method="card",
        failure_reason="insufficient_funds",
        recovery_status="recovered",
        was_recovered=True,
    )
    doc_b = historical_case_to_document(case_b)
    vector_index.add(doc_b, embedding_service.embed(doc_b.text))

    # Query for Customer A
    context_a = CustomerRecoveryContext(
        customer=CustomerContext(customer_id=cust_a_id),
        current_payment=PaymentContext(
            payment_id=uuid.uuid4(),
            amount=1000.0,
            payment_method="card",
            status="failed",
            failure_reason="insufficient_funds",
        ),
    )

    retriever = SemanticHistoricalRetriever(
        vector_index=vector_index,
        embedding_service=embedding_service,
    )
    results = retriever.retrieve(context_a)

    # Must be 0 results (Cust B's data is isolated)
    assert len(results) == 0


def test_future_cases_excluded_by_temporal_boundary():
    """Verify historical cases with created_at timestamp after current payment are excluded."""
    customer_id = uuid.uuid4()
    embedding_service = get_embedding_service()
    vector_index = VectorIndex(dimension=embedding_service.dimension)

    t_curr = datetime(2026, 2, 1, 12, 0, 0, tzinfo=timezone.utc)

    # Historical Case created in the future relative to current payment
    future_case = HistoricalCase(
        payment_id=uuid.uuid4(),
        customer_id=customer_id,
        amount=1000.0,
        payment_method="card",
        failure_reason="timeout",
        recovery_status="recovered",
        created_at=t_curr + timedelta(days=10),
    )
    doc_future = historical_case_to_document(future_case)
    vector_index.add(doc_future, embedding_service.embed(doc_future.text))

    context = CustomerRecoveryContext(
        customer=CustomerContext(customer_id=customer_id),
        current_payment=PaymentContext(
            payment_id=uuid.uuid4(),
            amount=1000.0,
            payment_method="card",
            status="failed",
            failure_reason="timeout",
            created_at=t_curr,
        ),
    )

    retriever = SemanticHistoricalRetriever(
        vector_index=vector_index,
        embedding_service=embedding_service,
    )
    results = retriever.retrieve(context)

    assert len(results) == 0


# ============================================================================
# 4. Determinism, Top-K, and Validation
# ============================================================================


def test_empty_index_returns_empty_list():
    """Verify empty vector index returns empty list."""
    service = get_embedding_service()
    index = VectorIndex(dimension=service.dimension)
    context = CustomerRecoveryContext(
        customer=CustomerContext(customer_id=uuid.uuid4()),
        current_payment=PaymentContext(
            payment_id=uuid.uuid4(),
            amount=500.0,
            payment_method="upi",
            status="failed",
        ),
    )
    retriever = SemanticHistoricalRetriever(
        vector_index=index,
        embedding_service=service,
    )
    assert retriever.retrieve(context) == []


def test_top_k_validation_and_limits():
    """Verify top_k parameter validation and result limits."""
    customer_id = uuid.uuid4()
    embedding_service = get_embedding_service()
    vector_index = VectorIndex(dimension=embedding_service.dimension)

    for i in range(10):
        case = HistoricalCase(
            payment_id=uuid.uuid4(),
            customer_id=customer_id,
            amount=float(100 * (i + 1)),
            currency="INR",
            payment_method="card",
            failure_reason=f"error_{i}",
            recovery_status="recovered",
        )
        doc = historical_case_to_document(case)
        vector_index.add(doc, embedding_service.embed(doc.text))

    context = CustomerRecoveryContext(
        customer=CustomerContext(customer_id=customer_id),
        current_payment=PaymentContext(
            payment_id=uuid.uuid4(),
            amount=500.0,
            currency="INR",
            payment_method="card",
            status="failed",
        ),
    )

    retriever = SemanticHistoricalRetriever(
        vector_index=vector_index,
        embedding_service=embedding_service,
    )

    results_3 = retriever.retrieve(context, top_k=3)
    assert len(results_3) == 3

    # Invalid top_k rejected
    with pytest.raises(ValueError):
        retriever.retrieve(context, top_k=0)

    with pytest.raises(ValueError):
        retriever.retrieve(context, top_k=-1)

    with pytest.raises(ValueError):
        retriever.retrieve(context, top_k="five")  # type: ignore


def test_invalid_context_type_raises_type_error():
    """Verify passing non-CustomerRecoveryContext raises TypeError."""
    retriever = SemanticHistoricalRetriever(
        vector_index=VectorIndex(),
        embedding_service=get_embedding_service(),
    )
    with pytest.raises(TypeError):
        retriever.retrieve("invalid_context")  # type: ignore


def test_dependency_injection_type_validation():
    """Verify constructor rejects non-matching dependency types."""
    service = get_embedding_service()
    index = VectorIndex()

    with pytest.raises(TypeError):
        SemanticHistoricalRetriever(vector_index="not_an_index", embedding_service=service)  # type: ignore

    with pytest.raises(TypeError):
        SemanticHistoricalRetriever(vector_index=index, embedding_service="not_a_service")  # type: ignore


# ============================================================================
# 5. Infrastructure Failure Propagation
# ============================================================================


class FailingEmbeddingService(EmbeddingService):
    """Mock service that raises an infrastructure error."""

    def embed(self, text: str):
        raise RuntimeError("Embedding model connection timeout")


class FailingVectorIndex(VectorIndex):
    """Mock vector index that raises an infrastructure error."""

    def search(self, query_embedding, top_k=5):
        raise RuntimeError("Vector index corrupted")


def test_embedding_failure_propagates_exception():
    """Verify exceptions from embedding service are not silently swallowed."""
    failing_service = FailingEmbeddingService()
    index = VectorIndex()
    # Add dummy item to make size > 0
    doc = RetrievalDocument(case_id=uuid.uuid4(), text="dummy")
    index.add(doc, [0.1] * 64)

    context = CustomerRecoveryContext(
        customer=CustomerContext(customer_id=uuid.uuid4()),
        current_payment=PaymentContext(
            payment_id=uuid.uuid4(),
            amount=500.0,
            payment_method="card",
            status="failed",
        ),
    )

    retriever = SemanticHistoricalRetriever(
        vector_index=index,
        embedding_service=failing_service,
    )

    with pytest.raises(RuntimeError) as exc:
        retriever.retrieve(context)
    assert "Embedding model connection timeout" in str(exc.value)


def test_vector_index_failure_propagates_exception():
    """Verify exceptions from vector index are not silently swallowed."""
    service = get_embedding_service()
    failing_index = FailingVectorIndex(dimension=service.dimension)
    doc = RetrievalDocument(case_id=uuid.uuid4(), text="dummy")
    failing_index.add(doc, [0.1] * service.dimension)

    context = CustomerRecoveryContext(
        customer=CustomerContext(customer_id=uuid.uuid4()),
        current_payment=PaymentContext(
            payment_id=uuid.uuid4(),
            amount=500.0,
            payment_method="card",
            status="failed",
        ),
    )

    retriever = SemanticHistoricalRetriever(
        vector_index=failing_index,
        embedding_service=service,
    )

    with pytest.raises(RuntimeError) as exc:
        retriever.retrieve(context)
    assert "Vector index corrupted" in str(exc.value)


# ============================================================================
# 6. End-to-End Pipeline Integration Test
# ============================================================================


def test_end_to_end_semantic_retrieval_flow():
    """
    Verify complete flow:
    HistoricalCase
        ↓
    historical_case_to_document()
        ↓
    EmbeddingService
        ↓
    VectorIndex.add()

    Current CustomerRecoveryContext
        ↓
    canonical query representation
        ↓
    EmbeddingService
        ↓
    VectorIndex.search()
        ↓
    SemanticHistoricalRetriever
        ↓
    HistoricalCase
    """
    customer_id = uuid.uuid4()
    service = EmbeddingService(DeterministicLocalEmbeddingProvider(dimension=64))
    index = VectorIndex(dimension=64)

    # 1. Historical recovered case
    hist_case = HistoricalCase(
        payment_id=uuid.uuid4(),
        customer_id=customer_id,
        external_payment_id="ext_pay_1",
        amount=3000.0,
        currency="INR",
        payment_method="card",
        failure_reason="otp_expired",
        recovery_action="payment_link",
        recovery_status="recovered",
        amount_recovered=3000.0,
        was_recovered=True,
    )

    doc = historical_case_to_document(hist_case)
    emb = service.embed(doc.text)
    index.add(doc, emb)

    # 2. Context with similar OTP failure
    context = CustomerRecoveryContext(
        customer=CustomerContext(customer_id=customer_id),
        current_payment=PaymentContext(
            payment_id=uuid.uuid4(),
            amount=3000.0,
            currency="INR",
            payment_method="card",
            status="failed",
            failure_reason="otp_expired",
        ),
        current_opportunity=RecoveryOpportunityContext(
            opportunity_id=uuid.uuid4(),
            status="open",
            revenue_at_risk=3000.0,
            recommended_action="payment_link",
        ),
    )

    retriever = SemanticHistoricalRetriever(
        vector_index=index,
        embedding_service=service,
    )

    retrieved = retriever.retrieve(context, top_k=1)

    assert len(retrieved) == 1
    assert retrieved[0].payment_id == hist_case.payment_id
    assert retrieved[0].relevance_score is not None
    assert 0.0 <= retrieved[0].relevance_score <= 1.0
    assert retrieved[0].recovery_action == "payment_link"
    assert retrieved[0].was_recovered is True
