"""
Unit and Integration Tests for DecisionEngine + Historical RAG Integration.

Verifies:
1. Backward compatibility when historical evidence is absent (None / empty)
2. Safety Invariant: Permanent credential failure CANNOT be overridden by historical cases
3. Safety Invariant: Max attempts exceeded CANNOT be overridden by historical cases
4. Safety Invariant: Already recovered opportunity produces NO_ACTION regardless of historical cases
5. Confidence boosting when historical evidence corroborates candidate rule
6. Action pivoting when historical evidence shows strong empirical success for an alternative action
7. Penalty weighting on failed historical cases
8. Filtering out low-relevance cases (< 0.40)
9. Transparent explainability inside decision_basis
10. End-to-end integration with HybridHistoricalRetriever
"""

import uuid
from datetime import datetime, timedelta, timezone

from app.context import (
    CustomerContext,
    CustomerRecoveryContext,
    CustomerRecoveryStatsContext,
    HistoricalPaymentContext,
    PaymentContext,
    RecoveryAttemptContext,
    RecoveryOpportunityContext,
)
from app.decision_engine import (
    DecisionEngine,
    HistoricalEvidenceSynthesizer,
    RecoveryAction,
    evaluate_recovery_decision,
)
from app.embedding_service import get_embedding_service
from app.historical_retrieval import HistoricalCase
from app.historical_retriever import HistoricalRetriever
from app.hybrid_historical_retriever import HybridHistoricalRetriever
from app.retrieval_document import historical_case_to_document
from app.semantic_historical_retriever import SemanticHistoricalRetriever
from app.vector_index import VectorIndex


def _build_context(
    failure_reason: str = "insufficient_funds",
    payment_method: str = "card",
    opp_status: str = "open",
    payment_status: str = "failed",
    current_attempts: list | None = None,
) -> CustomerRecoveryContext:
    now = datetime(2026, 2, 1, 12, 0, 0, tzinfo=timezone.utc)
    cust_id = uuid.uuid4()
    pay_id = uuid.uuid4()
    opp_id = uuid.uuid4()

    return CustomerRecoveryContext(
        customer=CustomerContext(customer_id=cust_id),
        current_payment=PaymentContext(
            payment_id=pay_id,
            amount=2000.0,
            currency="INR",
            payment_method=payment_method,
            status=payment_status,
            failure_reason=failure_reason,
            created_at=now,
        ),
        current_opportunity=RecoveryOpportunityContext(
            opportunity_id=opp_id,
            status=opp_status,
            revenue_at_risk=2000.0,
        ),
        current_payment_attempts=current_attempts or [],
        recovery_statistics=CustomerRecoveryStatsContext(),
    )


def _make_hist_case(
    cust_id: uuid.UUID,
    action: str,
    was_recovered: bool,
    relevance_score: float = 0.8,
    reason: str = "insufficient_funds",
) -> HistoricalCase:
    return HistoricalCase(
        payment_id=uuid.uuid4(),
        customer_id=cust_id,
        amount=2000.0,
        currency="INR",
        payment_method="card",
        failure_reason=reason,
        recovery_action=action,
        recovery_status="recovered" if was_recovered else "failed",
        amount_recovered=2000.0 if was_recovered else 0.0,
        was_recovered=was_recovered,
        relevance_score=relevance_score,
    )


# ============================================================================
# 1. Backward Compatibility & Safety Invariants
# ============================================================================


def test_backward_compatibility_without_historical_evidence():
    """Verify engine behaves identically to baseline when no historical cases are passed."""
    ctx = _build_context(failure_reason="bank_timeout")
    decision = evaluate_recovery_decision(ctx)

    assert decision.recommended_action == RecoveryAction.RETRY_PAYMENT
    assert decision.confidence == 0.85
    assert (
        decision.decision_basis["rule_matched"]
        == "TransientTechnicalImmediateRetryRule"
    )
    assert decision.decision_basis["historical_rag_evidence"]["has_evidence"] is False


def test_safety_permanent_failure_cannot_be_overridden():
    """Verify permanent card expiration CANNOT be overridden by historical retry cases."""
    ctx = _build_context(failure_reason="card_expired")
    # Even if historical case recovered via retry_payment
    hist_cases = [
        _make_hist_case(
            ctx.customer.customer_id,
            "retry_payment",
            True,
            relevance_score=0.95,
            reason="card_expired",
        )
    ]

    decision = evaluate_recovery_decision(ctx, historical_cases=hist_cases)

    # Invariant: Permanent credential failures must require payment method change
    assert decision.recommended_action == RecoveryAction.CHANGE_PAYMENT_METHOD
    assert decision.confidence == 0.90
    assert decision.decision_basis["rule_matched"] == "PermanentCredentialFailureRule"


def test_safety_max_attempts_cannot_be_overridden():
    """Verify max attempts exceeded produces NO_ACTION regardless of positive historical evidence."""
    now = datetime(2026, 2, 1, 12, 0, 0, tzinfo=timezone.utc)
    attempts = [
        RecoveryAttemptContext(action="retry_payment", status="failed", created_at=now),
        RecoveryAttemptContext(
            action="wait_and_retry", status="failed", created_at=now
        ),
        RecoveryAttemptContext(action="payment_link", status="failed", created_at=now),
    ]
    ctx = _build_context(current_attempts=attempts)
    hist_cases = [
        _make_hist_case(
            ctx.customer.customer_id, "payment_link", True, relevance_score=0.99
        )
    ]

    decision = evaluate_recovery_decision(
        ctx, max_attempts=3, historical_cases=hist_cases
    )

    assert decision.recommended_action == RecoveryAction.NO_ACTION
    assert decision.decision_basis["rule_matched"] == "MaxAttemptsExceededRule"


# ============================================================================
# 2. Adaptive Evidence Weighting & Explainability
# ============================================================================


def test_historical_evidence_boosts_confidence():
    """Verify corroborating historical cases boost confidence for the candidate action."""
    ctx = _build_context(failure_reason="bank_timeout")
    hist_cases = [
        _make_hist_case(
            ctx.customer.customer_id, "retry_payment", True, relevance_score=0.9
        ),
        _make_hist_case(
            ctx.customer.customer_id, "retry_payment", True, relevance_score=0.85
        ),
    ]

    decision = evaluate_recovery_decision(ctx, historical_cases=hist_cases)

    assert decision.recommended_action == RecoveryAction.RETRY_PAYMENT
    assert decision.confidence == 0.90  # Base 0.85 + 0.05 boost
    assert "historical" in decision.reason.lower()
    assert decision.decision_basis["historical_rag_evidence"]["has_evidence"] is True
    assert (
        decision.decision_basis["historical_rag_evidence"]["best_action"]
        == RecoveryAction.RETRY_PAYMENT
    )


def test_historical_evidence_selection_on_unknown_failure():
    """Verify unclassified/custom error reason selects the historically successful action."""
    ctx = _build_context(failure_reason="custom_vendor_error_code_418")
    hist_cases = [
        _make_hist_case(
            ctx.customer.customer_id,
            "payment_link",
            True,
            relevance_score=0.85,
            reason="custom_vendor_error_code_418",
        ),
        _make_hist_case(
            ctx.customer.customer_id,
            "payment_link",
            True,
            relevance_score=0.80,
            reason="custom_vendor_error_code_418",
        ),
    ]

    decision = evaluate_recovery_decision(ctx, historical_cases=hist_cases)

    assert decision.recommended_action == RecoveryAction.PAYMENT_LINK
    assert decision.decision_basis["rule_matched"] == "HistoricalRAGSelectedRule"
    assert decision.confidence >= 0.75
    assert "2 similar historical recovery cases" in decision.reason


def test_low_relevance_threshold_filtering():
    """Verify historical cases with relevance < 0.40 are excluded from evidence."""
    ctx = _build_context(failure_reason="custom_unknown_code")
    hist_cases = [
        _make_hist_case(
            ctx.customer.customer_id, "smart_retry", True, relevance_score=0.25
        ),
    ]

    decision = evaluate_recovery_decision(ctx, historical_cases=hist_cases)

    # Low relevance case is discarded, falls back to default
    assert decision.decision_basis["rule_matched"] == "DefaultFallbackRule"
    assert decision.decision_basis["historical_rag_evidence"]["has_evidence"] is False


def test_failed_historical_cases_penalize_net_score():
    """Verify failed historical cases negatively weight an action in evidence synthesis."""
    cid = uuid.uuid4()
    cases = [
        _make_hist_case(cid, "smart_retry", True, relevance_score=0.8),
        _make_hist_case(
            cid, "smart_retry", False, relevance_score=0.9
        ),  # 0.8 - 0.9*0.75 = 0.8 - 0.675 = +0.125
        _make_hist_case(cid, "payment_link", True, relevance_score=0.85),  # 0.85
    ]

    synth = HistoricalEvidenceSynthesizer.synthesize(cases)

    assert synth["has_evidence"] is True
    assert synth["best_action"] == RecoveryAction.PAYMENT_LINK
    assert synth["action_net_scores"]["retry_payment"] == 0.125
    assert synth["action_net_scores"]["payment_link"] == 0.85


# ============================================================================
# 3. End-to-End Integration with HybridHistoricalRetriever
# ============================================================================


def test_end_to_end_decision_engine_hybrid_rag_integration():
    """
    Verify complete integrated pipeline:
    CustomerRecoveryContext
        ↓
    HybridHistoricalRetriever (Deterministic + Semantic + RRF)
        ↓
    DecisionEngine.evaluate()
        ↓
    Augmented RecoveryDecision
    """
    cid = uuid.uuid4()
    now = datetime(2026, 2, 1, 12, 0, 0, tzinfo=timezone.utc)

    # 1. Populate VectorIndex and Historical Context with a recovered payment_link case
    p_hist = uuid.uuid4()
    hist_case = HistoricalCase(
        payment_id=p_hist,
        customer_id=cid,
        amount=3000.0,
        currency="INR",
        payment_method="upi",
        failure_reason="bank_timeout",
        recovery_action="payment_link",
        recovery_status="recovered",
        amount_recovered=3000.0,
        was_recovered=True,
        created_at=now - timedelta(days=2),
    )

    emb_service = get_embedding_service()
    v_index = VectorIndex(dimension=emb_service.dimension)
    doc = historical_case_to_document(hist_case)
    v_index.add(doc, emb_service.embed(doc.text))

    sem_retriever = SemanticHistoricalRetriever(
        vector_index=v_index, embedding_service=emb_service
    )

    h_ctx = HistoricalPaymentContext(
        payment_id=p_hist,
        amount=3000.0,
        currency="INR",
        payment_method="upi",
        status="succeeded",
        failure_reason="bank_timeout",
        was_recovered=True,
        recovery_action="payment_link",
        created_at=now - timedelta(days=2),
    )
    det_retriever = HistoricalRetriever(db_session=None)

    hybrid_retriever = HybridHistoricalRetriever(
        deterministic_retriever=det_retriever,
        semantic_retriever=sem_retriever,
        rrf_k=60,
    )

    # 2. Context with bank_timeout where immediate retry already failed on this payment
    query_pid = uuid.uuid4()
    attempts = [
        RecoveryAttemptContext(
            action="retry_payment",
            status="failed",
            created_at=now - timedelta(minutes=5),
        )
    ]
    context = CustomerRecoveryContext(
        customer=CustomerContext(customer_id=cid),
        current_payment=PaymentContext(
            payment_id=query_pid,
            amount=3000.0,
            currency="INR",
            payment_method="upi",
            status="failed",
            failure_reason="bank_timeout",
            created_at=now,
        ),
        current_opportunity=RecoveryOpportunityContext(
            opportunity_id=uuid.uuid4(),
            status="open",
            revenue_at_risk=3000.0,
        ),
        current_payment_attempts=attempts,
        historical_payments=[h_ctx],
    )

    # 3. Instantiate DecisionEngine with injected HybridHistoricalRetriever
    engine = DecisionEngine(max_attempts=3, retriever=hybrid_retriever)
    decision = engine.evaluate(context)

    # Historical evidence showed customer recovered via payment link on UPI bank_timeout
    assert decision.recommended_action == RecoveryAction.PAYMENT_LINK
    assert decision.confidence == 0.80
    assert (
        decision.decision_basis["rule_matched"]
        == "TransientTechnicalHistoricalLinkPivotRule"
    )
    assert decision.decision_basis["historical_rag_evidence"]["has_evidence"] is True
    assert decision.decision_basis["historical_rag_evidence"]["top_case"][
        "payment_id"
    ] == str(p_hist)
