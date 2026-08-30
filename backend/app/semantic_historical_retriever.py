"""
Revora Semantic Historical Recovery Case Retriever.

Retrieves and ranks relevant historical recovery cases for a CustomerRecoveryContext
using dense semantic embeddings, canonical query text representation, and VectorIndex
similarity search.
"""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from uuid import UUID

from app.context import CustomerRecoveryContext
from app.embedding_service import EmbeddingService
from app.historical_retrieval import HistoricalCase
from app.retrieval_document import RetrievalDocument
from app.vector_index import VectorIndex, VectorSearchResult


def _normalize_datetime(dt: Any) -> Optional[datetime]:
    """Safely parse and normalize datetime objects or ISO strings to UTC-aware datetime."""
    if isinstance(dt, datetime):
        if dt.tzinfo is None:
            return dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    if isinstance(dt, str) and dt.strip():
        raw_str = dt.strip()
        if raw_str.endswith("Z") or raw_str.endswith("z"):
            raw_str = raw_str[:-1] + "+00:00"
        try:
            parsed = datetime.fromisoformat(raw_str)
            if parsed.tzinfo is None:
                return parsed.replace(tzinfo=timezone.utc)
            return parsed.astimezone(timezone.utc)
        except (ValueError, TypeError):
            return None
    return None


def construct_canonical_query_text(context: CustomerRecoveryContext) -> str:
    """
    Construct the canonical text representation for a query CustomerRecoveryContext.

    Adheres strictly to the identical 8-line key-value format used by RetrievalDocument,
    ensuring that queries and stored historical cases inhabit the exact same embedding space.
    """
    current_payment = context.current_payment
    current_opp = context.current_opportunity

    failure_reason_norm = (
        current_payment.failure_reason.strip().lower()
        if current_payment and current_payment.failure_reason
        else "none"
    )
    payment_method_norm = (
        current_payment.payment_method.strip().lower()
        if current_payment and current_payment.payment_method
        else "unknown"
    )
    amount_norm = f"{current_payment.amount:.2f}" if current_payment else "0.00"
    currency_norm = (
        current_payment.currency.strip().upper()
        if current_payment and current_payment.currency
        else "INR"
    )
    recovery_action_norm = (
        current_opp.recommended_action.strip().lower()
        if current_opp and current_opp.recommended_action
        else "none"
    )
    recovery_status_norm = (
        current_opp.status.strip().lower()
        if current_opp and current_opp.status
        else (current_payment.status.strip().lower() if current_payment else "open")
    )
    was_recovered_norm = "false"  # Query payment is active and currently unresolved
    amount_recovered_norm = "0.00"

    lines = [
        f"failure_reason: {failure_reason_norm}",
        f"payment_method: {payment_method_norm}",
        f"amount: {amount_norm}",
        f"currency: {currency_norm}",
        f"recovery_action: {recovery_action_norm}",
        f"recovery_status: {recovery_status_norm}",
        f"was_recovered: {was_recovered_norm}",
        f"amount_recovered: {amount_recovered_norm}",
    ]

    return "\n".join(lines)


def _search_result_to_historical_case(result: VectorSearchResult) -> HistoricalCase:
    """
    Convert a VectorSearchResult containing a RetrievalDocument into a canonical HistoricalCase.
    """
    doc = result.document
    meta = doc.metadata

    # Extract mandatory customer and payment IDs (Fail closed: never fallback to payment_id)
    payment_id = doc.case_id
    raw_cust_id = meta.get("customer_id")
    if isinstance(raw_cust_id, UUID):
        customer_id = raw_cust_id
    elif isinstance(raw_cust_id, str):
        try:
            customer_id = UUID(raw_cust_id)
        except (ValueError, TypeError, AttributeError) as exc:
            raise ValueError(f"Invalid customer_id in document metadata: {raw_cust_id!r}") from exc
    else:
        raise ValueError(f"Missing or invalid customer_id in document metadata: {raw_cust_id!r}")

    # Parse timestamps if present in metadata
    created_at: Optional[datetime] = _normalize_datetime(meta.get("created_at"))
    completed_at: Optional[datetime] = _normalize_datetime(meta.get("completed_at"))

    # Relevance score bounded strictly to [0.0, 1.0]
    bounded_relevance = round(max(0.0, min(1.0, result.similarity_score)), 4)

    return HistoricalCase(
        payment_id=payment_id,
        customer_id=customer_id,
        external_payment_id=meta.get("external_payment_id"),
        external_customer_id=meta.get("external_customer_id"),
        amount=float(meta.get("amount", 0.0)),
        currency=str(meta.get("currency", "INR")),
        payment_method=str(meta.get("payment_method", "unknown")),
        failure_reason=meta.get("failure_reason"),
        recovery_action=meta.get("recovery_action"),
        recovery_status=str(meta.get("recovery_status", "failed")),
        amount_recovered=float(meta.get("amount_recovered", 0.0)),
        was_recovered=bool(meta.get("was_recovered", False)),
        relevance_score=bounded_relevance,
        created_at=created_at,
        completed_at=completed_at,
        metadata=dict(meta),
    )


class SemanticHistoricalRetriever:
    """
    Semantic Historical Recovery Case Retriever.

    Generates dense embeddings for a CustomerRecoveryContext query, executes
    cosine similarity search on an injected VectorIndex, applies tenant and
    temporal safety filters, and converts matching results into canonical HistoricalCase evidence.
    """

    def __init__(
        self,
        vector_index: VectorIndex,
        embedding_service: EmbeddingService,
    ):
        if not isinstance(vector_index, VectorIndex):
            raise TypeError(
                f"vector_index must be an instance of VectorIndex, got {type(vector_index).__name__}"
            )
        if not isinstance(embedding_service, EmbeddingService):
            raise TypeError(
                f"embedding_service must be an instance of EmbeddingService, got {type(embedding_service).__name__}"
            )
        self.vector_index = vector_index
        self.embedding_service = embedding_service

    def retrieve(
        self,
        context: CustomerRecoveryContext,
        top_k: int = 5,
    ) -> List[HistoricalCase]:
        """
        Retrieve top-k semantically relevant historical recovery cases for the given context.

        Args:
            context: CustomerRecoveryContext containing customer and active payment signals.
            top_k: Maximum number of ranked historical cases to return (must be positive integer).

        Returns:
            List[HistoricalCase] sorted by semantic similarity descending.
        """
        if not isinstance(context, CustomerRecoveryContext):
            raise TypeError(
                f"Expected context to be CustomerRecoveryContext, got {type(context).__name__}"
            )
        if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k <= 0:
            raise ValueError(f"top_k must be a positive integer, got: {top_k!r}")
        if self.vector_index.size == 0:
            return []

        # 1. Canonical Query Representation
        query_text = construct_canonical_query_text(context)

        # 2. Embedding Generation (Propagates any infrastructure exceptions)
        query_vector = self.embedding_service.embed(query_text)

        # 3. Vector Similarity Search
        # Fetch candidate pool (up to index size or top_k * 3) to allow post-filtering
        fetch_limit = max(top_k * 3, 20)
        search_results = self.vector_index.search(query_vector, top_k=fetch_limit)
        if not search_results:
            return []

        # 4. Safety Filtering & Isolation Boundaries
        # - Exclude current payment
        # - Strict customer / tenant isolation
        # - Temporal boundary (exclude future payments after query payment)
        current_payment = context.current_payment
        current_payment_id = current_payment.payment_id if current_payment else None
        current_customer_id_str = str(context.customer.customer_id)
        current_created_at = current_payment.created_at if current_payment else None

        filtered_cases: List[HistoricalCase] = []
        for result in search_results:
            doc = result.document
            meta = doc.metadata

            # Rule A: Exclude current payment
            if current_payment_id is not None and doc.case_id == current_payment_id:
                continue

            # Rule B: Tenant / Customer Isolation (Fail closed: reject missing, malformed, or mismatching)
            raw_cust_id = meta.get("customer_id")
            if not raw_cust_id:
                continue
            try:
                if isinstance(raw_cust_id, UUID):
                    doc_cust_uuid = raw_cust_id
                elif isinstance(raw_cust_id, str):
                    doc_cust_uuid = UUID(raw_cust_id)
                else:
                    continue
            except (ValueError, TypeError, AttributeError):
                continue

            if doc_cust_uuid != context.customer.customer_id:
                continue

            # Rule C: Temporal Isolation (Historical cases must precede or equal current payment; fail closed)
            if current_created_at is not None:
                doc_created_at = _normalize_datetime(meta.get("created_at"))
                norm_current_created_at = _normalize_datetime(current_created_at)
                if doc_created_at is None or norm_current_created_at is None:
                    continue
                if doc_created_at > norm_current_created_at:
                    continue

            # Convert to canonical HistoricalCase
            try:
                case = _search_result_to_historical_case(result)
            except ValueError:
                continue
            filtered_cases.append(case)

            if len(filtered_cases) >= top_k:
                break

        return filtered_cases
