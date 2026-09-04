"""
Revora Embedding Service Abstraction and Local Embedding Provider.

Provides a decoupled, production-style embedding abstraction that converts
RetrievalDocument text into dense vector representations.
"""

import hashlib
import math
import re
from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any


class EmbeddingProvider(ABC):
    """
    Abstract interface for embedding generation providers.
    Allows seamless swapping between local, OpenAI, SentenceTransformers, or custom models.
    """

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Return the fixed output embedding dimension."""

    @abstractmethod
    def embed(self, text: str) -> list[float]:
        """Convert a single normalized text string into a dense vector."""

    @abstractmethod
    def embed_batch(self, texts: Sequence[str]) -> list[list[float]]:
        """Convert a batch of normalized text strings into a list of dense vectors."""


class DeterministicLocalEmbeddingProvider(EmbeddingProvider):
    """
    Zero-dependency, deterministic local embedding provider.

    Uses feature hashing with semantic token weighting and L2 normalization
    to produce stable, reproducible dense vectors across environments without external APIs.
    """

    def __init__(self, dimension: int = 64):
        if dimension <= 0:
            raise ValueError(f"Embedding dimension must be positive, got {dimension}")
        self._dimension = dimension

    @property
    def dimension(self) -> int:
        return self._dimension

    def _normalize_text(self, text: str) -> str:
        if not isinstance(text, str):
            raise TypeError(f"Expected text to be a string, got {type(text).__name__}")
        cleaned = text.strip()
        if not cleaned:
            raise ValueError("Input text cannot be empty or whitespace-only.")
        return cleaned

    def _hash_token(self, token: str, seed: int = 0) -> int:
        """Deterministically hash a token string to an integer index."""
        raw = f"{seed}:{token}".encode()
        return int(hashlib.sha256(raw).hexdigest(), 16)

    def _generate_vector(self, text: str) -> list[float]:
        clean_text = self._normalize_text(text)
        vector = [0.0] * self._dimension

        # Tokenize lines and sub-tokens
        lines = [line.strip() for line in clean_text.split("\n") if line.strip()]
        for line_idx, line in enumerate(lines):
            # Line-level feature weighting
            line_hash = self._hash_token(line.lower(), seed=1)
            line_dim = line_hash % self._dimension
            line_sign = 1.0 if (line_hash >> 8) % 2 == 0 else -1.0
            vector[line_dim] += line_sign * 1.5

            # Key-value structured feature parsing
            if ":" in line:
                key, val = line.split(":", 1)
                key_norm = key.strip().lower()
                val_norm = val.strip().lower()
                kv_feature = f"{key_norm}={val_norm}"
                kv_hash = self._hash_token(kv_feature, seed=2)
                kv_dim = kv_hash % self._dimension
                kv_sign = 1.0 if (kv_hash >> 8) % 2 == 0 else -1.0
                vector[kv_dim] += kv_sign * 2.0

            # Sub-token n-grams and word features
            tokens = re.findall(r"\w+", line.lower())
            for t_idx, token in enumerate(tokens):
                t_hash = self._hash_token(token, seed=3)
                t_dim = t_hash % self._dimension
                t_sign = 1.0 if (t_hash >> 8) % 2 == 0 else -1.0
                vector[t_dim] += t_sign * 1.0

                # Bigram features
                if t_idx > 0:
                    bigram = f"{tokens[t_idx - 1]}_{token}"
                    b_hash = self._hash_token(bigram, seed=4)
                    b_dim = b_hash % self._dimension
                    b_sign = 1.0 if (b_hash >> 8) % 2 == 0 else -1.0
                    vector[b_dim] += b_sign * 1.2

        # L2-normalize to unit vector
        squared_sum = sum(x * x for x in vector)
        norm = math.sqrt(squared_sum)
        if norm > 1e-12:
            vector = [round(x / norm, 6) for x in vector]
        else:
            vector = [0.0] * self._dimension

        return vector

    def embed(self, text: str) -> list[float]:
        return self._generate_vector(text)

    def embed_batch(self, texts: Sequence[str]) -> list[list[float]]:
        if not isinstance(texts, (list, tuple)):
            raise TypeError(
                f"Expected texts to be a sequence, got {type(texts).__name__}"
            )
        if len(texts) == 0:
            return []
        return [self._generate_vector(t) for t in texts]


class EmbeddingService:
    """
    High-level embedding service for Revora.

    Encapsulates text normalization, validation, and batch processing while
    keeping the underlying provider decoupled and replaceable.
    """

    def __init__(self, provider: EmbeddingProvider | None = None):
        self._provider = provider or DeterministicLocalEmbeddingProvider()

    @property
    def dimension(self) -> int:
        """Return the dimension of embeddings generated by this service."""
        return self._provider.dimension

    @property
    def provider_name(self) -> str:
        """Return the class name of the underlying embedding provider."""
        return self._provider.__class__.__name__

    def embed(self, text: str) -> list[float]:
        """
        Embed a single text string into a dense unit vector.

        Raises:
            TypeError: If input is not a string.
            ValueError: If input string is empty or whitespace-only.
        """
        if not isinstance(text, str):
            raise TypeError(f"Input text must be a string, got {type(text).__name__}")
        if not text.strip():
            raise ValueError("Input text cannot be empty or whitespace-only.")

        vector = self._provider.embed(text)
        if len(vector) != self.dimension:
            raise ValueError(
                f"Provider returned vector of dimension {len(vector)}, expected {self.dimension}"
            )
        return vector

    def embed_document(self, doc: Any) -> list[float]:
        """Embed a document object with a canonical_text attribute or string conversion."""
        text = getattr(doc, "canonical_text", str(doc))
        return self.embed(text)

    def embed_batch(self, texts: Sequence[str]) -> list[list[float]]:
        """
        Embed a batch of text strings into a list of dense unit vectors, preserving order.

        Raises:
            TypeError: If texts is not a sequence or contains non-string items.
            ValueError: If any item in texts is empty or whitespace-only.
        """
        if not isinstance(texts, (list, tuple)):
            raise TypeError(
                f"Input texts must be a list or sequence, got {type(texts).__name__}"
            )
        if len(texts) == 0:
            return []

        # Validate all items prior to embedding
        for idx, item in enumerate(texts):
            if not isinstance(item, str):
                raise TypeError(
                    f"Item at index {idx} must be a string, got {type(item).__name__}"
                )
            if not item.strip():
                raise ValueError(
                    f"Item at index {idx} cannot be empty or whitespace-only."
                )

        vectors = self._provider.embed_batch(texts)
        if len(vectors) != len(texts):
            raise RuntimeError(
                f"Provider returned {len(vectors)} vectors for batch of size {len(texts)}"
            )

        for idx, vec in enumerate(vectors):
            if len(vec) != self.dimension:
                raise ValueError(
                    f"Vector at index {idx} has dimension {len(vec)}, expected {self.dimension}"
                )

        return vectors


# Global default service singleton for convenience
_default_embedding_service: EmbeddingService | None = None


def get_embedding_service() -> EmbeddingService:
    """Retrieve or initialize the global default EmbeddingService instance."""
    global _default_embedding_service
    if _default_embedding_service is None:
        _default_embedding_service = EmbeddingService()
    return _default_embedding_service
