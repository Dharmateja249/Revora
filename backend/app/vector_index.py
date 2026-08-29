"""
Revora In-Memory Vector Index and Cosine Similarity Search.

Provides a decoupled, thread-safe, in-memory vector storage and exact nearest-neighbor
cosine similarity search engine for RetrievalDocument representations without external dependencies.
"""

from collections import OrderedDict
import math
from typing import Any, Dict, List, Optional, Sequence, Tuple
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from app.retrieval_document import RetrievalDocument


class VectorSearchResult(BaseModel):
    """
    Immutable search result containing a retrieved RetrievalDocument and its similarity score.
    """

    model_config = ConfigDict(
        frozen=True,
        from_attributes=True,
        validate_default=True,
        arbitrary_types_allowed=True,
    )

    document: RetrievalDocument
    similarity_score: float = Field(ge=-1.0, le=1.0)


def calculate_cosine_similarity(
    vec_a: Sequence[float],
    vec_b: Sequence[float],
) -> float:
    """
    Calculate exact cosine similarity between two vectors.

    cos(a, b) = (a . b) / (||a|| * ||b||)

    Robust against unnormalized vectors and zero vectors.
    """
    if len(vec_a) != len(vec_b):
        raise ValueError(
            f"Vector dimensions do not match: {len(vec_a)} vs {len(vec_b)}"
        )

    dot_product = 0.0
    norm_a_sq = 0.0
    norm_b_sq = 0.0

    for a, b in zip(vec_a, vec_b):
        dot_product += a * b
        norm_a_sq += a * a
        norm_b_sq += b * b

    norm_a = math.sqrt(norm_a_sq)
    norm_b = math.sqrt(norm_b_sq)

    if norm_a <= 1e-12 or norm_b <= 1e-12:
        return 0.0

    raw_similarity = dot_product / (norm_a * norm_b)
    # Clamp to [-1.0, 1.0] to account for minor floating point rounding
    clamped = max(-1.0, min(1.0, raw_similarity))
    return round(clamped, 6)


class VectorIndex:
    """
    In-Memory Vector Index for RetrievalDocuments.

    Stores dense embeddings alongside document contracts, providing deterministic
    top-K cosine similarity search with exact tie-breaking.
    """

    def __init__(self, dimension: Optional[int] = None):
        if dimension is not None and dimension <= 0:
            raise ValueError(f"VectorIndex dimension must be positive, got {dimension}")
        self._dimension = dimension
        # Store entries mapped by case_id: (Tuple[float, ...], RetrievalDocument)
        self._entries: Dict[UUID, Tuple[Tuple[float, ...], RetrievalDocument]] = OrderedDict()

    @property
    def dimension(self) -> Optional[int]:
        """Return the vector dimension enforced by this index, or None if uninitialized."""
        return self._dimension

    @property
    def size(self) -> int:
        """Return total number of indexed documents."""
        return len(self._entries)

    def __len__(self) -> int:
        return self.size

    def _validate_vector(self, vector: Sequence[float]) -> Tuple[float, ...]:
        """Validate vector numeric integrity and dimension consistency."""
        if not isinstance(vector, (list, tuple)):
            raise TypeError(f"Vector must be a sequence of floats, got {type(vector).__name__}")
        if len(vector) == 0:
            raise ValueError("Vector cannot be empty.")

        # Check all elements are valid finite floats
        validated: List[float] = []
        for idx, val in enumerate(vector):
            if not isinstance(val, (int, float)) or isinstance(val, bool):
                raise TypeError(
                    f"Vector element at index {idx} must be a float, got {type(val).__name__}"
                )
            f_val = float(val)
            if math.isnan(f_val) or math.isinf(f_val):
                raise ValueError(f"Vector element at index {idx} is NaN or Infinite.")
            validated.append(f_val)

        # Enforce dimension consistency
        if self._dimension is None:
            self._dimension = len(validated)
        elif len(validated) != self._dimension:
            raise ValueError(
                f"Vector dimension {len(validated)} does not match index dimension {self._dimension}"
            )

        return tuple(validated)

    def add(
        self,
        document: RetrievalDocument,
        embedding: Sequence[float],
    ) -> None:
        """
        Add or update a RetrievalDocument with its embedding vector.

        If a document with the same case_id already exists, it is replaced (upsert policy).
        """
        if not isinstance(document, RetrievalDocument):
            raise TypeError(
                f"Expected RetrievalDocument instance, got {type(document).__name__}"
            )

        frozen_vector = self._validate_vector(embedding)
        self._entries[document.case_id] = (frozen_vector, document)

    def add_batch(
        self,
        documents: Sequence[RetrievalDocument],
        embeddings: Sequence[Sequence[float]],
    ) -> None:
        """
        Batch add documents and corresponding embeddings.

        Raises:
            ValueError: If lengths of documents and embeddings do not match.
        """
        if not isinstance(documents, (list, tuple)):
            raise TypeError(f"Expected documents sequence, got {type(documents).__name__}")
        if not isinstance(embeddings, (list, tuple)):
            raise TypeError(f"Expected embeddings sequence, got {type(embeddings).__name__}")
        if len(documents) != len(embeddings):
            raise ValueError(
                f"Documents count ({len(documents)}) does not match embeddings count ({len(embeddings)})"
            )

        for doc, emb in zip(documents, embeddings):
            self.add(doc, emb)

    def search(
        self,
        query_embedding: Sequence[float],
        top_k: int = 5,
    ) -> List[VectorSearchResult]:
        """
        Perform deterministic cosine similarity search against indexed vectors.

        Ranking order:
        1. Similarity score descending
        2. Case ID string ascending (deterministic tie-breaker)
        """
        if top_k <= 0:
            return []
        if len(self._entries) == 0:
            return []

        frozen_query = self._validate_vector(query_embedding)

        scored_results: List[Tuple[float, str, RetrievalDocument]] = []
        for case_id, (doc_vector, doc) in self._entries.items():
            similarity = calculate_cosine_similarity(frozen_query, doc_vector)
            scored_results.append((similarity, str(case_id), doc))

        # Sort deterministically: similarity score descending, case_id string ascending
        scored_results.sort(key=lambda item: (-item[0], item[1]))

        results: List[VectorSearchResult] = []
        for sim, _, doc in scored_results[:top_k]:
            results.append(
                VectorSearchResult(
                    document=doc,
                    similarity_score=sim,
                )
            )

        return results

    def get(self, case_id: UUID) -> Optional[Tuple[Tuple[float, ...], RetrievalDocument]]:
        """Retrieve stored embedding and document by case_id, or None if not found."""
        return self._entries.get(case_id)

    def delete(self, case_id: UUID) -> bool:
        """Remove a document by case_id. Returns True if deleted, False if not found."""
        if case_id in self._entries:
            del self._entries[case_id]
            return True
        return False

    def clear(self) -> None:
        """Clear all entries from the index."""
        self._entries.clear()
