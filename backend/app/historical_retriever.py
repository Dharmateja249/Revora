"""
Revora Deterministic Historical Case Retriever.

Retrieves and ranks relevant historical recovery cases for a given CustomerRecoveryContext
using interpretable, deterministic heuristic signals (failure category similarity, payment
method matching, amount proximity, recency, and recovery outcome) without ML or LLMs.
"""

from datetime import datetime
from typing import Any
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.orm import Session, selectinload

from app.context import (
    CustomerRecoveryContext,
)
from app.decision_engine import (
    CUSTOMER_INTERACTION_REASONS,
    INSUFFICIENT_FUNDS_REASONS,
    PERMANENT_FAILURE_REASONS,
    TRANSIENT_TECHNICAL_REASONS,
)
from app.historical_retrieval import HistoricalCase
from app.models import Payment, RecoveryOpportunity

# Weights for the deterministic scoring formula (Sum = 1.0)
WEIGHT_FAILURE_REASON = 0.35
WEIGHT_PAYMENT_METHOD = 0.25
WEIGHT_AMOUNT_SIMILARITY = 0.15
WEIGHT_RECENCY = 0.10
WEIGHT_RECOVERY_OUTCOME = 0.10
WEIGHT_CURRENCY = 0.05


def _classify_failure_category(reason: str | None) -> str | None:
    """Classify failure reason into deterministic domain category."""
    if not reason:
        return None
    cleaned = reason.strip().lower()
    if cleaned in PERMANENT_FAILURE_REASONS or any(
        term in cleaned for term in PERMANENT_FAILURE_REASONS
    ):
        return "permanent"
    if cleaned in TRANSIENT_TECHNICAL_REASONS or any(
        term in cleaned for term in TRANSIENT_TECHNICAL_REASONS
    ):
        return "transient"
    if cleaned in INSUFFICIENT_FUNDS_REASONS or any(
        term in cleaned for term in INSUFFICIENT_FUNDS_REASONS
    ):
        return "insufficient_funds"
    if cleaned in CUSTOMER_INTERACTION_REASONS or any(
        term in cleaned for term in CUSTOMER_INTERACTION_REASONS
    ):
        return "customer_interaction"
    return "other"


def _calculate_failure_similarity(
    query_reason: str | None,
    candidate_reason: str | None,
) -> float:
    """Calculate deterministic failure reason similarity in [0.0, 1.0]."""
    if not query_reason and not candidate_reason:
        return 0.5  # Neutral when neither is specified
    if not query_reason or not candidate_reason:
        return 0.1

    q_clean = query_reason.strip().lower()
    c_clean = candidate_reason.strip().lower()

    if q_clean == c_clean:
        return 1.0

    q_cat = _classify_failure_category(q_clean)
    c_cat = _classify_failure_category(c_clean)

    if q_cat and c_cat and q_cat == c_cat and q_cat != "other":
        return 0.75

    # Token overlap similarity (Jaccard on word tokens)
    q_tokens = set(q_clean.replace("_", " ").replace("-", " ").split())
    c_tokens = set(c_clean.replace("_", " ").replace("-", " ").split())
    if q_tokens and c_tokens:
        intersection = q_tokens.intersection(c_tokens)
        union = q_tokens.union(c_tokens)
        if intersection:
            return round(len(intersection) / len(union) * 0.6, 4)

    return 0.0


def _calculate_payment_method_similarity(
    query_method: str | None,
    candidate_method: str | None,
) -> float:
    """Calculate payment method match score in [0.0, 1.0]."""
    if not query_method or not candidate_method:
        return 0.0

    q_m = query_method.strip().lower()
    c_m = candidate_method.strip().lower()

    if q_m == c_m:
        return 1.0

    # Family similarities
    card_family = {"card", "credit_card", "debit_card", "prepaid_card"}
    upi_family = {"upi", "upi_collect", "upi_intent", "qr", "upi_qr"}
    netbanking_family = {"netbanking", "net_banking", "bank_transfer", "ach"}

    if (
        (q_m in card_family and c_m in card_family)
        or (q_m in upi_family and c_m in upi_family)
        or (q_m in netbanking_family and c_m in netbanking_family)
    ):
        return 0.6

    return 0.0


def _calculate_amount_similarity(
    query_amount: float,
    candidate_amount: float,
) -> float:
    """Calculate relative amount proximity score in [0.0, 1.0]."""
    if query_amount <= 0.0 and candidate_amount <= 0.0:
        return 1.0
    denom = max(query_amount, candidate_amount, 1.0)
    diff = abs(query_amount - candidate_amount)
    rel_diff = min(diff / denom, 1.0)
    return round(1.0 - rel_diff, 4)


def _calculate_recency_score(
    query_time: datetime | None,
    candidate_time: datetime | None,
    half_life_days: float = 30.0,
) -> float:
    """Calculate recency score using deterministic exponential decay."""
    if not query_time or not candidate_time:
        return 0.5  # Neutral fallback

    # Calculate difference in days (candidate should precede or equal query)
    delta_seconds = (query_time - candidate_time).total_seconds()
    if delta_seconds < 0:
        delta_seconds = 0.0

    days_diff = delta_seconds / 86400.0
    decay = 1.0 / (1.0 + (days_diff / half_life_days))
    return round(decay, 4)


class CandidateCase:
    """Internal lightweight structure representing a retrieval candidate."""

    def __init__(
        self,
        payment_id: UUID,
        customer_id: UUID,
        amount: float,
        currency: str,
        payment_method: str,
        failure_reason: str | None,
        recovery_status: str,
        amount_recovered: float,
        was_recovered: bool,
        recovery_action: str | None = None,
        external_payment_id: str | None = None,
        external_customer_id: str | None = None,
        created_at: datetime | None = None,
        completed_at: datetime | None = None,
        raw_metadata: dict[str, Any] | None = None,
    ):
        self.payment_id = payment_id
        self.customer_id = customer_id
        self.amount = amount
        self.currency = currency
        self.payment_method = payment_method
        self.failure_reason = failure_reason
        self.recovery_status = recovery_status
        self.amount_recovered = amount_recovered
        self.was_recovered = was_recovered
        self.recovery_action = recovery_action
        self.external_payment_id = external_payment_id
        self.external_customer_id = external_customer_id
        self.created_at = created_at
        self.completed_at = completed_at
        self.raw_metadata = raw_metadata or {}


class HistoricalRetriever:
    """
    Deterministic Historical Recovery Case Retriever.

    Retrieves and ranks historical payment recovery records for the current payment context
    using transparent, rule-based relevance scoring.
    """

    def __init__(self, db_session: Session | None = None):
        self.db_session = db_session

    def retrieve_relevant_cases(
        self,
        context: CustomerRecoveryContext,
        top_k: int = 5,
    ) -> list[HistoricalCase]:
        """
        Retrieve and rank top-k relevant historical recovery cases for the current context.

        Args:
            context: CustomerRecoveryContext representing the customer and failed payment.
            top_k: Maximum number of historical cases to return (default: 5).

        Returns:
            List[HistoricalCase] ordered by relevance score descending.
        """
        if top_k <= 0:
            return []

        # 1. Retrieve Candidate Historical Records
        candidates = self._retrieve_candidates(context)
        if not candidates:
            return []

        # 2. Score and Rank Candidates
        scored_candidates: list[tuple[float, dict[str, float], CandidateCase]] = []
        for candidate in candidates:
            score, breakdown = self._calculate_relevance(context, candidate)
            scored_candidates.append((score, breakdown, candidate))

        # 3. Sort Deterministically (Score desc, Recency desc, Payment ID asc for tie-breaking)
        ranked = self._rank_candidates(scored_candidates)

        # 4. Convert Top-K to Canonical HistoricalCase Contracts
        top_cases: list[HistoricalCase] = []
        for score, breakdown, cand in ranked[:top_k]:
            top_cases.append(self._to_historical_case(cand, score, breakdown))

        return top_cases

    def _retrieve_candidates(
        self,
        context: CustomerRecoveryContext,
    ) -> list[CandidateCase]:
        """
        Retrieve candidate historical payments respecting customer isolation and temporal ordering.
        """
        customer_id = context.customer.customer_id
        current_payment = context.current_payment
        current_payment_id = current_payment.payment_id if current_payment else None
        current_created_at = current_payment.created_at if current_payment else None

        candidates: list[CandidateCase] = []

        # Strategy A: Database Query with Eager Loading (Avoids N+1 queries)
        if self.db_session is not None:
            stmt = (
                select(Payment)
                .options(
                    selectinload(Payment.recovery_opportunity).selectinload(
                        RecoveryOpportunity.attempts
                    )
                )
                .where(
                    Payment.customer_id == customer_id,
                )
            )

            if current_payment_id is not None:
                stmt = stmt.where(Payment.id != current_payment_id)

            if current_created_at is not None:
                stmt = stmt.where(Payment.created_at <= current_created_at)

            # Order deterministically by created_at desc, id desc
            stmt = stmt.order_by(Payment.created_at.desc(), Payment.id.desc())

            payments = self.db_session.execute(stmt).scalars().all()

            for p in payments:
                opp = p.recovery_opportunity
                was_recovered = False
                recovery_action: str | None = None
                amount_recovered = 0.0
                recovery_status = "recovered" if p.status == "succeeded" else "failed"
                completed_at: datetime | None = None

                if opp is not None:
                    recovery_status = opp.status
                    if opp.status == "recovered":
                        was_recovered = True
                        amount_recovered = opp.expected_recovery or p.amount
                    recovery_action = opp.recommended_action

                    if opp.attempts:
                        for att in opp.attempts:
                            if att.status == "succeeded":
                                was_recovered = True
                                amount_recovered = max(
                                    amount_recovered, att.amount_recovered
                                )
                                recovery_action = att.action
                                if att.completed_at:
                                    completed_at = att.completed_at
                elif p.status == "succeeded":
                    was_recovered = True
                    amount_recovered = p.amount

                candidates.append(
                    CandidateCase(
                        payment_id=p.id,
                        customer_id=p.customer_id,
                        amount=p.amount,
                        currency=p.currency,
                        payment_method=p.payment_method,
                        failure_reason=p.failure_reason,
                        recovery_status=recovery_status,
                        amount_recovered=amount_recovered,
                        was_recovered=was_recovered,
                        recovery_action=recovery_action,
                        external_payment_id=p.external_payment_id,
                        external_customer_id=context.customer.external_customer_id,
                        created_at=p.created_at,
                        completed_at=completed_at,
                    )
                )

        # Strategy B: Context Historical Payments (Zero DB access fallback)
        else:
            for hist_p in context.historical_payments:
                if (
                    current_payment_id is not None
                    and hist_p.payment_id == current_payment_id
                ):
                    continue

                if (
                    current_created_at is not None
                    and hist_p.created_at is not None
                    and hist_p.created_at > current_created_at
                ):
                    continue

                recovery_status = (
                    "recovered"
                    if hist_p.was_recovered
                    else ("failed" if hist_p.status == "failed" else hist_p.status)
                )
                amount_recovered = hist_p.amount if hist_p.was_recovered else 0.0

                candidates.append(
                    CandidateCase(
                        payment_id=hist_p.payment_id,
                        customer_id=customer_id,
                        amount=hist_p.amount,
                        currency=hist_p.currency,
                        payment_method=hist_p.payment_method,
                        failure_reason=hist_p.failure_reason,
                        recovery_status=recovery_status,
                        amount_recovered=amount_recovered,
                        was_recovered=hist_p.was_recovered,
                        recovery_action=hist_p.recovery_action,
                        external_payment_id=hist_p.external_payment_id,
                        external_customer_id=context.customer.external_customer_id,
                        created_at=hist_p.created_at,
                        completed_at=None,
                        raw_metadata={
                            "recovery_attempts_count": hist_p.recovery_attempts_count,
                        },
                    )
                )

        return candidates

    def _calculate_relevance(
        self,
        context: CustomerRecoveryContext,
        candidate: CandidateCase,
    ) -> tuple[float, dict[str, float]]:
        """
        Compute deterministic relevance score in [0.0, 1.0] for a candidate.
        """
        current_payment = context.current_payment
        query_failure_reason = (
            current_payment.failure_reason if current_payment else None
        )
        query_payment_method = (
            current_payment.payment_method if current_payment else None
        )
        query_amount = current_payment.amount if current_payment else 0.0
        query_currency = current_payment.currency if current_payment else "INR"
        query_created_at = current_payment.created_at if current_payment else None

        # 1. Failure Reason Similarity
        failure_sim = _calculate_failure_similarity(
            query_failure_reason, candidate.failure_reason
        )

        # 2. Payment Method Match
        method_sim = _calculate_payment_method_similarity(
            query_payment_method, candidate.payment_method
        )

        # 3. Amount Similarity
        amount_sim = _calculate_amount_similarity(query_amount, candidate.amount)

        # 4. Currency Match
        currency_sim = (
            1.0 if query_currency.upper() == candidate.currency.upper() else 0.0
        )

        # 5. Recency
        recency_sim = _calculate_recency_score(query_created_at, candidate.created_at)

        # 6. Recovery Outcome (Recovered cases carry higher empirical resolution evidence)
        outcome_sim = 1.0 if candidate.was_recovered else 0.5

        # Weighted Total Score
        total_score = (
            (WEIGHT_FAILURE_REASON * failure_sim)
            + (WEIGHT_PAYMENT_METHOD * method_sim)
            + (WEIGHT_AMOUNT_SIMILARITY * amount_sim)
            + (WEIGHT_CURRENCY * currency_sim)
            + (WEIGHT_RECENCY * recency_sim)
            + (WEIGHT_RECOVERY_OUTCOME * outcome_sim)
        )

        # Ensure strict bounds [0.0, 1.0]
        final_score = round(max(0.0, min(1.0, total_score)), 4)

        breakdown = {
            "failure_similarity": failure_sim,
            "payment_method_similarity": method_sim,
            "amount_similarity": amount_sim,
            "currency_similarity": currency_sim,
            "recency_similarity": recency_sim,
            "outcome_similarity": outcome_sim,
        }

        return final_score, breakdown

    def _rank_candidates(
        self,
        scored_candidates: list[tuple[float, dict[str, float], CandidateCase]],
    ) -> list[tuple[float, dict[str, float], CandidateCase]]:
        """
        Deterministically sort candidates:
        1. Relevance score descending
        2. Recency timestamp descending
        3. Payment UUID string ascending (strict tie-breaking)
        """

        def sort_key(item: tuple[float, dict[str, float], CandidateCase]):
            score, _, cand = item
            timestamp = cand.created_at.timestamp() if cand.created_at else 0.0
            return (-score, -timestamp, str(cand.payment_id))

        return sorted(scored_candidates, key=sort_key)

    def _to_historical_case(
        self,
        candidate: CandidateCase,
        relevance_score: float,
        breakdown: dict[str, float],
    ) -> HistoricalCase:
        """
        Convert an internal candidate and score into an immutable HistoricalCase contract.
        """
        metadata = {
            **candidate.raw_metadata,
            "score_breakdown": breakdown,
        }

        return HistoricalCase(
            payment_id=candidate.payment_id,
            customer_id=candidate.customer_id,
            external_payment_id=candidate.external_payment_id,
            external_customer_id=candidate.external_customer_id,
            amount=candidate.amount,
            currency=candidate.currency,
            payment_method=candidate.payment_method,
            failure_reason=candidate.failure_reason,
            recovery_action=candidate.recovery_action,
            recovery_status=candidate.recovery_status,
            amount_recovered=candidate.amount_recovered,
            was_recovered=candidate.was_recovered,
            relevance_score=relevance_score,
            created_at=candidate.created_at,
            completed_at=candidate.completed_at,
            metadata=metadata,
        )


def retrieve_historical_cases(
    context: CustomerRecoveryContext,
    db_session: Session | None = None,
    top_k: int = 5,
) -> list[HistoricalCase]:
    """
    Public entrypoint function for retrieving relevant historical recovery cases.
    """
    retriever = HistoricalRetriever(db_session=db_session)
    return retriever.retrieve_relevant_cases(context=context, top_k=top_k)
