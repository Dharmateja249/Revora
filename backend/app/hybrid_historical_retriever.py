"""
Revora Hybrid Historical Recovery Case Retriever using Reciprocal Rank Fusion (RRF).

Combines deterministic domain-aware retrieval and dense semantic vector retrieval
into a single, deduplicated, deterministically ranked list of HistoricalCase evidence.
"""

from typing import Any, Dict, List, Optional, Set, Tuple
from uuid import UUID

from app.context import CustomerRecoveryContext
from app.historical_retrieval import HistoricalCase
from app.historical_retriever import HistoricalRetriever
from app.semantic_historical_retriever import SemanticHistoricalRetriever


# Default RRF smoothing parameter (industry standard benchmark)
DEFAULT_RRF_K: int = 60


def compute_rrf_score(rank: int, rrf_k: int = DEFAULT_RRF_K) -> float:
    """
    Compute Reciprocal Rank Fusion score for a single 1-based rank position.

    Formula: 1 / (rrf_k + rank)
    """
    if rank < 1:
        raise ValueError(f"Rank must be 1-based (>= 1), got {rank}")
    return 1.0 / (rrf_k + rank)


class HybridHistoricalRetriever:
    """
    Hybrid Historical Case Retriever.

    Executes parallel retrieval across deterministic and semantic retrievers, fuses
    candidate rankings using Reciprocal Rank Fusion (RRF), and produces deduplicated,
    consistently ranked HistoricalCase evidence.
    """

    def __init__(
        self,
        deterministic_retriever: Optional[HistoricalRetriever] = None,
        semantic_retriever: Optional[SemanticHistoricalRetriever] = None,
        rrf_k: int = DEFAULT_RRF_K,
        deterministic_fetch_k: Optional[int] = None,
        semantic_fetch_k: Optional[int] = None,
    ):
        if not isinstance(rrf_k, int) or isinstance(rrf_k, bool) or rrf_k <= 0:
            raise ValueError(f"rrf_k must be a positive integer, got: {rrf_k!r}")

        self.deterministic_retriever = (
            deterministic_retriever
            if deterministic_retriever is not None
            else HistoricalRetriever()
        )
        self.semantic_retriever: Optional[SemanticHistoricalRetriever] = (
            semantic_retriever
        )
        self.rrf_k = rrf_k
        self.deterministic_fetch_k = deterministic_fetch_k
        self.semantic_fetch_k = semantic_fetch_k

    def retrieve_relevant_cases(
        self,
        context: CustomerRecoveryContext,
        top_k: int = 5,
    ) -> List[HistoricalCase]:
        """
        Retrieve and fuse historical cases using Reciprocal Rank Fusion.

        Args:
            context: CustomerRecoveryContext containing customer and active payment signals.
            top_k: Maximum number of ranked historical cases to return (positive integer).

        Returns:
            List[HistoricalCase] deduplicated and ordered by fused RRF score descending.
        """
        if not isinstance(context, CustomerRecoveryContext):
            raise TypeError(
                f"Expected context to be CustomerRecoveryContext, got {type(context).__name__}"
            )
        if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k <= 0:
            raise ValueError(f"top_k must be a positive integer, got: {top_k!r}")

        # 1. Determine over-fetch candidate pool size for each retriever
        fetch_k_det = self.deterministic_fetch_k or max(top_k * 2, 10)
        fetch_k_sem = self.semantic_fetch_k or max(top_k * 2, 10)

        # 2. Execute retrieval from both components
        det_cases = self.deterministic_retriever.retrieve_relevant_cases(
            context=context, top_k=fetch_k_det
        )
        sem_cases = (
            self.semantic_retriever.retrieve(
                context=context,
                top_k=fetch_k_sem,
            )
            if self.semantic_retriever is not None
            else []
        )

        # If both retrievers return empty, return immediately
        if not det_cases and not sem_cases:
            return []

        # 3. Accumulate RRF scores and track per-retriever rank positions
        # Map: payment_id -> dict of accumulator signals
        candidates: Dict[UUID, Dict[str, Any]] = {}

        for rank_1_based, case in enumerate(det_cases, start=1):
            pid = case.payment_id
            score_contribution = compute_rrf_score(rank_1_based, self.rrf_k)
            if pid not in candidates:
                candidates[pid] = {
                    "case": case,
                    "rrf_score": 0.0,
                    "deterministic_rank": None,
                    "semantic_rank": None,
                    "deterministic_score": None,
                    "semantic_score": None,
                }
            candidates[pid]["rrf_score"] += score_contribution
            candidates[pid]["deterministic_rank"] = rank_1_based
            candidates[pid]["deterministic_score"] = case.relevance_score
            # If case came from deterministic retriever, keep it as base reference
            candidates[pid]["case"] = case

        for rank_1_based, case in enumerate(sem_cases, start=1):
            pid = case.payment_id
            score_contribution = compute_rrf_score(rank_1_based, self.rrf_k)
            if pid not in candidates:
                candidates[pid] = {
                    "case": case,
                    "rrf_score": 0.0,
                    "deterministic_rank": None,
                    "semantic_rank": None,
                    "deterministic_score": None,
                    "semantic_score": None,
                }
            candidates[pid]["rrf_score"] += score_contribution
            candidates[pid]["semantic_rank"] = rank_1_based
            candidates[pid]["semantic_score"] = case.relevance_score
            # If not previously populated, set as base reference
            if candidates[pid]["case"] is None:
                candidates[pid]["case"] = case

        # 4. Sort Candidates Deterministically
        # Tie-breaking policy:
        # 1. RRF score descending
        # 2. Best individual rank ascending (lowest rank number)
        # 3. Sum of raw relevance scores descending
        # 4. Created_at timestamp descending (most recent first)
        # 5. Payment ID string ascending (lexicographical UUID)
        def sort_key(item: Tuple[UUID, Dict[str, Any]]):
            pid, data = item
            case: HistoricalCase = data["case"]
            rrf_score = data["rrf_score"]

            det_r = data["deterministic_rank"] or 999999
            sem_r = data["semantic_rank"] or 999999
            best_rank = min(det_r, sem_r)

            det_s = data["deterministic_score"] or 0.0
            sem_s = data["semantic_score"] or 0.0
            raw_score_sum = det_s + sem_s

            timestamp = case.created_at.timestamp() if case.created_at else 0.0
            return (-rrf_score, best_rank, -raw_score_sum, -timestamp, str(pid))

        sorted_candidates = sorted(candidates.items(), key=sort_key)

        # 5. Construct Normalized Final HistoricalCase Contracts
        # Theoretical maximum RRF score for 2 retrievers when a document ranks #1 in both
        max_theoretical_rrf = 2.0 / (self.rrf_k + 1.0)

        fused_cases: List[HistoricalCase] = []
        for pid, data in sorted_candidates[:top_k]:
            base_case: HistoricalCase = data["case"]
            raw_rrf = data["rrf_score"]

            # Normalize RRF score to [0.0, 1.0] for the HistoricalCase relevance_score contract
            normalized_relevance = round(
                max(0.0, min(1.0, raw_rrf / max_theoretical_rrf)), 4
            )

            # Build comprehensive fusion metadata
            fusion_meta: Dict[str, Any] = {
                **base_case.metadata,
                "fusion_method": "rrf",
                "rrf_k": self.rrf_k,
                "raw_rrf_score": round(raw_rrf, 6),
                "deterministic_rank": data["deterministic_rank"],
                "semantic_rank": data["semantic_rank"],
                "deterministic_score": data["deterministic_score"],
                "semantic_score": data["semantic_score"],
            }

            fused_case = HistoricalCase(
                payment_id=base_case.payment_id,
                customer_id=base_case.customer_id,
                external_payment_id=base_case.external_payment_id,
                external_customer_id=base_case.external_customer_id,
                amount=base_case.amount,
                currency=base_case.currency,
                payment_method=base_case.payment_method,
                failure_reason=base_case.failure_reason,
                recovery_action=base_case.recovery_action,
                recovery_status=base_case.recovery_status,
                amount_recovered=base_case.amount_recovered,
                was_recovered=base_case.was_recovered,
                relevance_score=normalized_relevance,
                created_at=base_case.created_at,
                completed_at=base_case.completed_at,
                metadata=fusion_meta,
            )
            fused_cases.append(fused_case)

        return fused_cases


def retrieve_hybrid_historical_cases(
    context: CustomerRecoveryContext,
    deterministic_retriever: Optional[HistoricalRetriever] = None,
    semantic_retriever: Optional[SemanticHistoricalRetriever] = None,
    top_k: int = 5,
    rrf_k: int = DEFAULT_RRF_K,
) -> List[HistoricalCase]:
    """
    Public entrypoint for hybrid historical case retrieval with Reciprocal Rank Fusion.
    """
    retriever = HybridHistoricalRetriever(
        deterministic_retriever=deterministic_retriever,
        semantic_retriever=semantic_retriever,
        rrf_k=rrf_k,
    )
    return retriever.retrieve_relevant_cases(context=context, top_k=top_k)
