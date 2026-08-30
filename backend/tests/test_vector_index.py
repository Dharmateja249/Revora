"""
Comprehensive Test Suite for VectorIndex and Cosine Similarity Search.

Tests:
1. Adding single document and verifying index size and retrieval
2. Adding multiple documents and batch insertion
3. Exact vector match ranks first with similarity = 1.0
4. Cosine similarity correctly ranks intermediate matches
5. top_k parameter restricts result set length
6. top_k larger than index returns all available items without error
7. Zero / invalid vectors (NaN, Inf, empty, non-float) are rejected
8. Mismatched vector dimensions are rejected
9. Duplicate document IDs are handled via upsert policy (replace existing)
10. Deterministic tie-breaking on identical similarity scores
11. Search on empty index returns []
12. Batch insertion preserves document/embedding alignment and checks length match
13. VectorSearchResult immutability
14. Caller-owned list mutation does not corrupt stored embeddings
15. End-to-end integration pipeline:
    HistoricalCase -> historical_case_to_document() -> EmbeddingService -> VectorIndex.add() -> VectorIndex.search()
"""

import math
import uuid
import pytest
from pydantic import ValidationError

from app.embedding_service import get_embedding_service
from app.historical_retrieval import HistoricalCase
from app.retrieval_document import (
    RetrievalDocument,
    historical_case_to_document,
)
from app.vector_index import (
    VectorIndex,
    VectorSearchResult,
    calculate_cosine_similarity,
)


# ============================================================================
# 1. Cosine Similarity Calculation
# ============================================================================


def test_cosine_similarity_orthogonal_and_parallel():
    """Verify standard geometric vector angles."""
    # Parallel unit vectors -> 1.0
    assert calculate_cosine_similarity([1.0, 0.0], [1.0, 0.0]) == 1.0

    # Opposite vectors -> -1.0
    assert calculate_cosine_similarity([1.0, 0.0], [-1.0, 0.0]) == -1.0

    # Orthogonal vectors -> 0.0
    assert calculate_cosine_similarity([1.0, 0.0], [0.0, 1.0]) == 0.0

    # Zero vector -> 0.0
    assert calculate_cosine_similarity([0.0, 0.0], [1.0, 1.0]) == 0.0


def test_cosine_similarity_dimension_mismatch():
    """Verify mismatched dimensions raise ValueError."""
    with pytest.raises(ValueError):
        calculate_cosine_similarity([1.0, 2.0], [1.0, 2.0, 3.0])


# ============================================================================
# 2. Document Addition & Storage
# ============================================================================


def test_add_single_document():
    """Verify adding a single document sets dimension and increments index size."""
    index = VectorIndex()
    doc = RetrievalDocument(
        case_id=uuid.uuid4(),
        text="failure_reason: timeout",
    )
    vec = [0.1, 0.2, 0.3, 0.4]

    index.add(doc, vec)

    assert index.size == 1
    assert len(index) == 1
    assert index.dimension == 4

    stored = index.get(doc.case_id)
    assert stored is not None
    stored_vec, stored_doc = stored
    assert stored_vec == tuple(vec)
    assert stored_doc == doc


def test_add_batch_documents():
    """Verify batch insertion correctly stores all documents."""
    index = VectorIndex(dimension=2)
    docs = [
        RetrievalDocument(case_id=uuid.uuid4(), text="doc 1"),
        RetrievalDocument(case_id=uuid.uuid4(), text="doc 2"),
        RetrievalDocument(case_id=uuid.uuid4(), text="doc 3"),
    ]
    vectors = [
        [1.0, 0.0],
        [0.0, 1.0],
        [0.707, 0.707],
    ]

    index.add_batch(docs, vectors)

    assert index.size == 3
    for doc in docs:
        assert index.get(doc.case_id) is not None


def test_add_batch_mismatched_lengths_rejected():
    """Verify batch insertion with mismatched doc/vector counts raises ValueError."""
    index = VectorIndex()
    docs = [RetrievalDocument(case_id=uuid.uuid4(), text="doc 1")]
    vectors = [[1.0, 0.0], [0.0, 1.0]]

    with pytest.raises(ValueError) as exc:
        index.add_batch(docs, vectors)
    assert "count" in str(exc.value).lower()


def test_duplicate_case_id_upsert_policy():
    """Verify adding a document with existing case_id replaces the old entry."""
    index = VectorIndex()
    case_id = uuid.uuid4()

    doc_old = RetrievalDocument(case_id=case_id, text="old text")
    doc_new = RetrievalDocument(case_id=case_id, text="new updated text")

    index.add(doc_old, [1.0, 0.0])
    assert index.size == 1
    assert index.get(case_id)[1].text == "old text"  # type: ignore

    # Upsert with same case_id
    index.add(doc_new, [0.0, 1.0])
    assert index.size == 1
    assert index.get(case_id)[1].text == "new updated text"  # type: ignore
    assert index.get(case_id)[0] == (0.0, 1.0)  # type: ignore


# ============================================================================
# 3. Similarity Search & Ranking
# ============================================================================


def test_search_exact_match_ranks_first():
    """Verify exact match vector returns similarity = 1.0 as top result."""
    index = VectorIndex(dimension=3)
    target_id = uuid.uuid4()
    doc_target = RetrievalDocument(case_id=target_id, text="target match")
    doc_other = RetrievalDocument(case_id=uuid.uuid4(), text="other")

    index.add(doc_other, [0.0, 1.0, 0.0])
    index.add(doc_target, [1.0, 0.0, 0.0])

    results = index.search([1.0, 0.0, 0.0], top_k=2)

    assert len(results) == 2
    assert results[0].document.case_id == target_id
    assert math.isclose(results[0].similarity_score, 1.0, rel_tol=1e-5)
    assert results[1].similarity_score == 0.0


def test_search_top_k_limits():
    """Verify top_k parameter limits the number of returned results."""
    index = VectorIndex(dimension=2)
    for i in range(10):
        doc = RetrievalDocument(case_id=uuid.uuid4(), text=f"doc {i}")
        index.add(doc, [math.cos(i), math.sin(i)])

    results_3 = index.search([1.0, 0.0], top_k=3)
    assert len(results_3) == 3

    results_all = index.search([1.0, 0.0], top_k=50)
    assert len(results_all) == 10

    results_zero = index.search([1.0, 0.0], top_k=0)
    assert results_zero == []


def test_search_empty_index_returns_empty_list():
    """Verify search on unpopulated index gracefully returns []."""
    index = VectorIndex(dimension=4)
    results = index.search([1.0, 0.0, 0.0, 0.0], top_k=5)
    assert results == []


def test_deterministic_tie_breaking():
    """Verify identical similarity scores sort deterministically by case_id string."""
    index = VectorIndex(dimension=2)

    id_a = uuid.UUID("00000000-0000-0000-0000-000000000001")
    id_b = uuid.UUID("00000000-0000-0000-0000-000000000002")

    doc_b = RetrievalDocument(case_id=id_b, text="doc b")
    doc_a = RetrievalDocument(case_id=id_a, text="doc a")

    # Add in reverse order
    index.add(doc_b, [1.0, 1.0])
    index.add(doc_a, [1.0, 1.0])

    results = index.search([1.0, 1.0], top_k=2)

    assert len(results) == 2
    assert results[0].similarity_score == results[1].similarity_score
    # Lexicographical case_id tie-breaking: id_a < id_b
    assert results[0].document.case_id == id_a
    assert results[1].document.case_id == id_b


# ============================================================================
# 4. Validation & Dimension Constraints
# ============================================================================


def test_invalid_vector_values_rejected():
    """Verify NaN, Inf, empty, and non-numeric vectors raise errors."""
    index = VectorIndex()
    doc = RetrievalDocument(case_id=uuid.uuid4(), text="doc")

    # Empty vector
    with pytest.raises(ValueError):
        index.add(doc, [])

    # Vector with NaN
    with pytest.raises(ValueError):
        index.add(doc, [1.0, float("nan")])

    # Vector with Infinity
    with pytest.raises(ValueError):
        index.add(doc, [1.0, float("inf")])

    # Vector with boolean
    with pytest.raises(TypeError):
        index.add(doc, [True, 1.0])  # type: ignore

    # Non-sequence
    with pytest.raises(TypeError):
        index.add(doc, "not a vector")  # type: ignore


def test_dimension_mismatch_rejected():
    """Verify adding or querying vector with mismatched dimension is rejected."""
    index = VectorIndex(dimension=3)
    doc = RetrievalDocument(case_id=uuid.uuid4(), text="doc")

    # Add 2D vector to 3D index
    with pytest.raises(ValueError) as exc:
        index.add(doc, [1.0, 2.0])
    assert "dimension" in str(exc.value).lower()

    # Add valid 3D vector
    index.add(doc, [1.0, 2.0, 3.0])

    # Search with 4D query vector
    with pytest.raises(ValueError):
        index.search([1.0, 2.0, 3.0, 4.0])


# ============================================================================
# 5. Immutability & Mutation Isolation
# ============================================================================


def test_caller_mutation_does_not_corrupt_index():
    """Verify mutating caller-owned vector list after insertion does not alter stored vector."""
    index = VectorIndex()
    doc = RetrievalDocument(case_id=uuid.uuid4(), text="doc")
    source_vector = [1.0, 2.0, 3.0]

    index.add(doc, source_vector)

    # Mutate source vector
    source_vector[0] = 999.0

    stored = index.get(doc.case_id)
    assert stored is not None
    assert stored[0][0] == 1.0
    assert isinstance(stored[0], tuple)


def test_vector_search_result_immutability():
    """Verify VectorSearchResult contracts are frozen and immutable."""
    doc = RetrievalDocument(case_id=uuid.uuid4(), text="doc")
    result = VectorSearchResult(document=doc, similarity_score=0.85)

    with pytest.raises(ValidationError):
        result.similarity_score = 0.99  # type: ignore


def test_delete_and_clear():
    """Verify deletion and clear operations."""
    index = VectorIndex()
    doc1 = RetrievalDocument(case_id=uuid.uuid4(), text="doc 1")
    doc2 = RetrievalDocument(case_id=uuid.uuid4(), text="doc 2")

    index.add_batch([doc1, doc2], [[1.0, 0.0], [0.0, 1.0]])
    assert index.size == 2

    # Delete existing
    assert index.delete(doc1.case_id) is True
    assert index.size == 1
    assert index.get(doc1.case_id) is None

    # Delete non-existing
    assert index.delete(doc1.case_id) is False

    # Clear
    index.clear()
    assert index.size == 0


# ============================================================================
# 6. End-to-End Pipeline Integration
# ============================================================================


def test_end_to_end_historical_case_vector_search_pipeline():
    """
    Verify complete pipeline integration:
    HistoricalCase
        ↓
    historical_case_to_document()
        ↓
    EmbeddingService.embed()
        ↓
    VectorIndex.add()
        ↓
    VectorIndex.search()
    """
    embedding_service = get_embedding_service()
    vector_index = VectorIndex(dimension=embedding_service.dimension)

    # Historical Case 1: Insufficient funds on Card
    case_card_funds = HistoricalCase(
        payment_id=uuid.uuid4(),
        customer_id=uuid.uuid4(),
        amount=2500.0,
        currency="INR",
        payment_method="card",
        failure_reason="insufficient_funds",
        recovery_action="payment_link",
        recovery_status="recovered",
        amount_recovered=2500.0,
        was_recovered=True,
    )

    # Historical Case 2: Bank timeout on UPI
    case_upi_timeout = HistoricalCase(
        payment_id=uuid.uuid4(),
        customer_id=uuid.uuid4(),
        amount=500.0,
        currency="INR",
        payment_method="upi",
        failure_reason="bank_timeout",
        recovery_action="smart_retry",
        recovery_status="recovered",
        amount_recovered=500.0,
        was_recovered=True,
    )

    # 1. Convert to Canonical Documents
    doc_1 = historical_case_to_document(case_card_funds)
    doc_2 = historical_case_to_document(case_upi_timeout)

    # 2. Generate Embeddings
    vec_1 = embedding_service.embed(doc_1.text)
    vec_2 = embedding_service.embed(doc_2.text)

    # 3. Index Documents
    vector_index.add_batch([doc_1, doc_2], [vec_1, vec_2])
    assert vector_index.size == 2

    # 4. Search with Query matching Case 1 (Card + Insufficient Funds)
    query_case = HistoricalCase(
        payment_id=uuid.uuid4(),
        customer_id=uuid.uuid4(),
        amount=2400.0,
        currency="INR",
        payment_method="card",
        failure_reason="insufficient_funds",
        recovery_status="open",
    )
    query_doc = historical_case_to_document(query_case)
    query_vec = embedding_service.embed(query_doc.text)

    results = vector_index.search(query_vec, top_k=2)

    assert len(results) == 2
    # Case 1 must rank #1 with highest similarity score
    assert results[0].document.case_id == case_card_funds.payment_id
    assert results[0].similarity_score > results[1].similarity_score
    assert isinstance(results[0], VectorSearchResult)
