"""
Comprehensive Test Suite for EmbeddingService and Local Embedding Provider.

Tests:
1. Valid text produces a dense embedding vector
2. Vector has the expected fixed dimension
3. Determinism: identical input always produces identical vector
4. Distinctness: different text inputs produce different vectors
5. Normalization: leading/trailing whitespace variations produce identical vectors
6. Error handling: empty or whitespace-only text raises ValueError
7. Error handling: non-string input raises TypeError
8. Batch embedding: preserves input ordering
9. Batch embedding: output dimensions are consistent
10. Batch embedding: empty batch returns []
11. Batch error handling: invalid items inside batch raise appropriate errors
12. Custom provider injection / dimension configuration
13. Independence from database / ORM objects
14. End-to-end integration pipeline:
    HistoricalCase -> historical_case_to_document() -> RetrievalDocument.text -> EmbeddingService.embed()
"""

import math
import uuid

import pytest
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

# ============================================================================
# 1. Single Text Embedding & Vector Properties
# ============================================================================


def test_embed_valid_text():
    """Verify single valid text string generates dense unit vector of default dimension."""
    service = EmbeddingService()
    text = "failure_reason: insufficient_funds\npayment_method: card\namount: 2500.00"

    vector = service.embed(text)

    assert isinstance(vector, list)
    assert len(vector) == service.dimension
    assert service.dimension == 64
    assert all(isinstance(x, float) for x in vector)

    # Verify unit L2 normalization (length ≈ 1.0)
    norm = math.sqrt(sum(x * x for x in vector))
    assert math.isclose(norm, 1.0, rel_tol=1e-3)


def test_embedding_determinism():
    """Verify identical text inputs produce exact identical vectors across multiple calls."""
    service = EmbeddingService()
    text = (
        "failure_reason: bank_timeout\n"
        "payment_method: upi\n"
        "amount: 1500.00\n"
        "recovery_status: recovered"
    )

    vec_1 = service.embed(text)
    vec_2 = service.embed(text)
    vec_3 = service.embed(text)

    assert vec_1 == vec_2
    assert vec_2 == vec_3


def test_different_inputs_produce_distinct_vectors():
    """Verify distinct payment recovery text representations produce different vectors."""
    service = EmbeddingService()
    text_card = (
        "failure_reason: insufficient_funds\n"
        "payment_method: card\n"
        "amount: 2500.00\n"
        "recovery_action: payment_link"
    )
    text_upi = (
        "failure_reason: bank_timeout\n"
        "payment_method: upi\n"
        "amount: 500.00\n"
        "recovery_action: smart_retry"
    )

    vec_card = service.embed(text_card)
    vec_upi = service.embed(text_upi)

    assert vec_card != vec_upi

    # Cosine similarity between two unit vectors is simply dot product
    similarity = sum(a * b for a, b in zip(vec_card, vec_upi))
    assert -1.0 <= similarity <= 1.0
    assert similarity < 0.99


def test_whitespace_normalization_consistency():
    """Verify surrounding whitespace does not alter generated vector."""
    service = EmbeddingService()
    raw_text = "failure_reason: card_expired\npayment_method: card"
    padded_text = "   \n  failure_reason: card_expired\npayment_method: card  \n\n  "

    vec_raw = service.embed(raw_text)
    vec_padded = service.embed(padded_text)

    assert vec_raw == vec_padded


# ============================================================================
# 2. Validation & Error Handling
# ============================================================================


def test_empty_and_blank_text_raises_value_error():
    """Verify empty or blank text strings are rejected."""
    service = EmbeddingService()

    with pytest.raises(ValueError) as exc:
        service.embed("")
    assert "empty" in str(exc.value).lower()

    with pytest.raises(ValueError) as exc:
        service.embed("    \n\t  ")
    assert "empty" in str(exc.value).lower()


def test_non_string_text_raises_type_error():
    """Verify non-string input types raise TypeError."""
    service = EmbeddingService()

    with pytest.raises(TypeError):
        service.embed(12345)  # type: ignore

    with pytest.raises(TypeError):
        service.embed(None)  # type: ignore

    with pytest.raises(TypeError):
        service.embed(["list", "of", "strings"])  # type: ignore


# ============================================================================
# 3. Batch Embedding
# ============================================================================


def test_embed_batch_preserves_order_and_dimensions():
    """Verify batch embedding preserves input order and returns consistent dimensions."""
    service = EmbeddingService()
    texts = [
        "failure_reason: insufficient_funds\npayment_method: card",
        "failure_reason: bank_timeout\npayment_method: upi",
        "failure_reason: card_expired\npayment_method: card",
    ]

    vectors = service.embed_batch(texts)

    assert len(vectors) == len(texts)
    for vec in vectors:
        assert len(vec) == service.dimension

    # Individual embeddings must match batch outputs at same indices
    assert vectors[0] == service.embed(texts[0])
    assert vectors[1] == service.embed(texts[1])
    assert vectors[2] == service.embed(texts[2])


def test_embed_batch_empty_list():
    """Verify empty batch sequence returns empty list []."""
    service = EmbeddingService()
    assert service.embed_batch([]) == []
    assert service.embed_batch(()) == []


def test_embed_batch_invalid_items_rejected():
    """Verify invalid items in batch raise appropriate errors without silent failure."""
    service = EmbeddingService()

    # Batch with non-string item
    with pytest.raises(TypeError):
        service.embed_batch(["valid text", 42, "another valid"])  # type: ignore

    # Batch with blank item
    with pytest.raises(ValueError):
        service.embed_batch(["valid text", "   ", "another valid"])

    # Non-sequence batch
    with pytest.raises(TypeError):
        service.embed_batch("not a list")  # type: ignore


# ============================================================================
# 4. Custom Provider Injection & Configuration
# ============================================================================


class MockConstantEmbeddingProvider(EmbeddingProvider):
    """Custom provider for testing injection and pluggability."""

    def __init__(self, dimension: int = 16):
        self._dim = dimension

    @property
    def dimension(self) -> int:
        return self._dim

    def embed(self, text: str) -> list[float]:
        return [0.5] * self._dim

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [[0.5] * self._dim for _ in texts]


def test_custom_provider_injection():
    """Verify custom embedding provider is properly wrapped by EmbeddingService."""
    custom_provider = MockConstantEmbeddingProvider(dimension=16)
    service = EmbeddingService(provider=custom_provider)

    assert service.dimension == 16
    assert service.provider_name == "MockConstantEmbeddingProvider"

    vec = service.embed("test text")
    assert len(vec) == 16
    assert vec == [0.5] * 16


def test_custom_dimension_local_provider():
    """Verify DeterministicLocalEmbeddingProvider accepts custom dimension."""
    provider = DeterministicLocalEmbeddingProvider(dimension=128)
    service = EmbeddingService(provider=provider)

    assert service.dimension == 128
    vec = service.embed("test text")
    assert len(vec) == 128


def test_singleton_getter():
    """Verify get_embedding_service() returns a shared functional instance."""
    s1 = get_embedding_service()
    s2 = get_embedding_service()
    assert s1 is s2
    assert isinstance(s1, EmbeddingService)


# ============================================================================
# 5. Integration Pipeline: HistoricalCase -> Document -> Embedding
# ============================================================================


def test_end_to_end_case_to_embedding_pipeline():
    """
    Verify complete seamless flow:
    HistoricalCase
        ↓
    historical_case_to_document()
        ↓
    RetrievalDocument.text
        ↓
    EmbeddingService.embed()
    """
    payment_id = uuid.uuid4()
    customer_id = uuid.uuid4()

    case = HistoricalCase(
        payment_id=payment_id,
        customer_id=customer_id,
        amount=3500.0,
        currency="INR",
        payment_method="upi",
        failure_reason="bank_timeout",
        recovery_action="payment_link",
        recovery_status="recovered",
        amount_recovered=3500.0,
        was_recovered=True,
    )

    # 1. Convert to Canonical RetrievalDocument
    doc = historical_case_to_document(case)
    assert isinstance(doc, RetrievalDocument)
    assert doc.case_id == payment_id
    assert "failure_reason: bank_timeout" in doc.text

    # 2. Embed Canonical Document Text
    service = get_embedding_service()
    vector = service.embed(doc.text)

    assert isinstance(vector, list)
    assert len(vector) == service.dimension
    assert all(isinstance(v, float) for v in vector)

    # 3. Verify Batch Pipeline Flow
    batch_cases = [
        case,
        HistoricalCase(
            payment_id=uuid.uuid4(),
            customer_id=customer_id,
            amount=1000.0,
            payment_method="card",
            recovery_status="failed",
        ),
    ]

    docs = [historical_case_to_document(c) for c in batch_cases]
    batch_vectors = service.embed_batch([d.text for d in docs])

    assert len(batch_vectors) == 2
    assert batch_vectors[0] == vector
    assert batch_vectors[1] != vector
