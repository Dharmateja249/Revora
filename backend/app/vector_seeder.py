"""
Revora Runtime Vector Index Seeder.

Populates the application-scoped shared VectorIndex with a curated collection
of historical recovery precedents for the demo customer tenant, enabling
genuine semantic RAG retrieval during interactive recovery evaluation.
"""

import logging
from datetime import datetime, timezone
from uuid import UUID

from app.embedding_service import EmbeddingService, get_embedding_service
from app.historical_retrieval import HistoricalCase
from app.retrieval_document import historical_case_to_document
from app.vector_index import VectorIndex, get_vector_index

logger = logging.getLogger("revora.vector_seeder")

DEMO_CUSTOMER_UUID = UUID("e9cd4c97-979b-4753-9925-640623f74eee")

# Canonical demo historical recovery precedents covering critical failure categories
CANONICAL_DEMO_PRECEDENTS: tuple[dict, ...] = (
    # 1. Customer Authentication OTP Timeout (Card)
    {
        "payment_id": UUID("00000000-0000-7001-0000-000000000001"),
        "amount": 8450.0,
        "currency": "INR",
        "payment_method": "card",
        "failure_reason": "customer_auth_failed_otp_timeout",
        "recovery_action": "payment_link",
        "recovery_status": "recovered",
        "amount_recovered": 8450.0,
        "was_recovered": True,
    },
    {
        "payment_id": UUID("00000000-0000-7001-0000-000000000002"),
        "amount": 8450.0,
        "currency": "INR",
        "payment_method": "card",
        "failure_reason": "customer_auth_failed_otp_timeout",
        "recovery_action": "retry_payment",
        "recovery_status": "failed",
        "amount_recovered": 0.0,
        "was_recovered": False,
    },
    {
        "payment_id": UUID("00000000-0000-7001-0000-000000000003"),
        "amount": 5400.0,
        "currency": "INR",
        "payment_method": "card",
        "failure_reason": "customer_auth_failed_otp_timeout",
        "recovery_action": "payment_link",
        "recovery_status": "recovered",
        "amount_recovered": 5400.0,
        "was_recovered": True,
    },
    # 2. Bank Technical Timeout (UPI)
    {
        "payment_id": UUID("00000000-0000-7002-0000-000000000001"),
        "amount": 3200.0,
        "currency": "INR",
        "payment_method": "upi",
        "failure_reason": "bank_technical_timeout",
        "recovery_action": "wait_and_retry",
        "recovery_status": "recovered",
        "amount_recovered": 3200.0,
        "was_recovered": True,
    },
    {
        "payment_id": UUID("00000000-0000-7002-0000-000000000002"),
        "amount": 3200.0,
        "currency": "INR",
        "payment_method": "upi",
        "failure_reason": "bank_technical_timeout",
        "recovery_action": "retry_payment",
        "recovery_status": "failed",
        "amount_recovered": 0.0,
        "was_recovered": False,
    },
    # 3. Insufficient Funds (Card)
    {
        "payment_id": UUID("00000000-0000-7003-0000-000000000001"),
        "amount": 14999.0,
        "currency": "INR",
        "payment_method": "card",
        "failure_reason": "insufficient_funds",
        "recovery_action": "payment_link",
        "recovery_status": "recovered",
        "amount_recovered": 14999.0,
        "was_recovered": True,
    },
    {
        "payment_id": UUID("00000000-0000-7003-0000-000000000002"),
        "amount": 14999.0,
        "currency": "INR",
        "payment_method": "card",
        "failure_reason": "insufficient_funds",
        "recovery_action": "retry_payment",
        "recovery_status": "failed",
        "amount_recovered": 0.0,
        "was_recovered": False,
    },
    # 4. Mandate Authorization Failures (Card)
    {
        "payment_id": UUID("00000000-0000-7004-0000-000000000001"),
        "amount": 4999.0,
        "currency": "INR",
        "payment_method": "card",
        "failure_reason": "mandate_authorization_expired",
        "recovery_action": "payment_link",
        "recovery_status": "recovered",
        "amount_recovered": 4999.0,
        "was_recovered": True,
    },
    {
        "payment_id": UUID("00000000-0000-7004-0000-000000000002"),
        "amount": 4999.0,
        "currency": "INR",
        "payment_method": "card",
        "failure_reason": "mandate_authorization_expired",
        "recovery_action": "retry_payment",
        "recovery_status": "failed",
        "amount_recovered": 0.0,
        "was_recovered": False,
    },
    # 5. Velocity / Limit Exceeded (Card)
    {
        "payment_id": UUID("00000000-0000-7005-0000-000000000001"),
        "amount": 6500.0,
        "currency": "INR",
        "payment_method": "card",
        "failure_reason": "card_velocity_exceeded",
        "recovery_action": "wait_and_retry",
        "recovery_status": "recovered",
        "amount_recovered": 6500.0,
        "was_recovered": True,
    },
    {
        "payment_id": UUID("00000000-0000-7005-0000-000000000002"),
        "amount": 6500.0,
        "currency": "INR",
        "payment_method": "card",
        "failure_reason": "card_velocity_exceeded",
        "recovery_action": "retry_payment",
        "recovery_status": "failed",
        "amount_recovered": 0.0,
        "was_recovered": False,
    },
    # 6. Expired Instrument / Card
    {
        "payment_id": UUID("00000000-0000-7006-0000-000000000001"),
        "amount": 1299.0,
        "currency": "INR",
        "payment_method": "card",
        "failure_reason": "instrument_expired",
        "recovery_action": "change_payment_method",
        "recovery_status": "recovered",
        "amount_recovered": 1299.0,
        "was_recovered": True,
    },
    {
        "payment_id": UUID("00000000-0000-7006-0000-000000000002"),
        "amount": 1299.0,
        "currency": "INR",
        "payment_method": "card",
        "failure_reason": "instrument_expired",
        "recovery_action": "retry_payment",
        "recovery_status": "failed",
        "amount_recovered": 0.0,
        "was_recovered": False,
    },
    # 7. Fraud Hard Decline (Card)
    {
        "payment_id": UUID("00000000-0000-7007-0000-000000000001"),
        "amount": 22000.0,
        "currency": "INR",
        "payment_method": "card",
        "failure_reason": "fraud_hard_decline",
        "recovery_action": "no_action",
        "recovery_status": "failed",
        "amount_recovered": 0.0,
        "was_recovered": False,
    },
    # 8. Gateway Routing Issue (UPI)
    {
        "payment_id": UUID("00000000-0000-7008-0000-000000000001"),
        "amount": 4500.0,
        "currency": "INR",
        "payment_method": "upi",
        "failure_reason": "gateway_routing",
        "recovery_action": "wait_and_retry",
        "recovery_status": "recovered",
        "amount_recovered": 4500.0,
        "was_recovered": True,
    },
)


def get_curated_historical_precedents(
    customer_id: UUID | str = DEMO_CUSTOMER_UUID,
) -> list[HistoricalCase]:
    """
    Extract a curated set of 100-200 canonical historical recovery precedents
    from the repository's existing evaluation dataset, binding them to the specified customer identity.

    Args:
        customer_id: Target customer UUID to bind precedents to (enforcing tenant isolation).

    Returns:
        List of HistoricalCase domain models ready for document indexing.
    """
    target_uuid = UUID(str(customer_id))
    precedents: list[HistoricalCase] = []
    seen_payment_ids: set[UUID] = set()
    base_time = datetime(2026, 6, 1, 12, 0, 0, tzinfo=timezone.utc)

    # 1. Primary demo scenario precedents
    for item in CANONICAL_DEMO_PRECEDENTS:
        pid = item["payment_id"]
        seen_payment_ids.add(pid)
        precedents.append(
            HistoricalCase(
                payment_id=pid,
                customer_id=target_uuid,
                amount=item["amount"],
                currency=item["currency"],
                payment_method=item["payment_method"],
                failure_reason=item["failure_reason"],
                recovery_action=item["recovery_action"],
                recovery_status=item["recovery_status"],
                amount_recovered=item["amount_recovered"],
                was_recovered=item["was_recovered"],
                created_at=base_time,
            )
        )

    # 2. Curated evaluation cases from golden dataset
    try:
        from tests.fixtures.retrieval_golden_dataset import get_golden_evaluation_cases

        golden_cases = get_golden_evaluation_cases()
        for c in golden_cases:
            for hp in c.context.historical_payments:
                if hp.payment_id in seen_payment_ids:
                    continue
                seen_payment_ids.add(hp.payment_id)
                precedents.append(
                    HistoricalCase(
                        payment_id=hp.payment_id,
                        customer_id=target_uuid,
                        external_payment_id=hp.external_payment_id,
                        amount=hp.amount,
                        currency=hp.currency,
                        payment_method=hp.payment_method,
                        failure_reason=hp.failure_reason,
                        recovery_action=hp.recovery_action,
                        recovery_status="recovered" if hp.was_recovered else "failed",
                        amount_recovered=hp.amount if hp.was_recovered else 0.0,
                        was_recovered=hp.was_recovered,
                        created_at=hp.created_at or base_time,
                    )
                )

            for fc in c.metadata.get("foreign_historical_cases", ()):
                fc_obj = fc if isinstance(fc, HistoricalCase) else HistoricalCase(**fc)
                if fc_obj.payment_id in seen_payment_ids:
                    continue
                seen_payment_ids.add(fc_obj.payment_id)
                precedents.append(
                    HistoricalCase(
                        payment_id=fc_obj.payment_id,
                        customer_id=target_uuid,
                        external_payment_id=fc_obj.external_payment_id,
                        amount=fc_obj.amount,
                        currency=fc_obj.currency,
                        payment_method=fc_obj.payment_method,
                        failure_reason=fc_obj.failure_reason,
                        recovery_action=fc_obj.recovery_action,
                        recovery_status=fc_obj.recovery_status,
                        amount_recovered=fc_obj.amount_recovered,
                        was_recovered=fc_obj.was_recovered,
                        created_at=fc_obj.created_at or base_time,
                    )
                )
    except ImportError:
        logger.warning(
            "retrieval_golden_dataset fixture could not be imported; proceeding with canonical precedents only."
        )

    return precedents


def seed_runtime_vector_index(
    vector_index: VectorIndex | None = None,
    customer_id: UUID | str | None = None,
    embedding_service: EmbeddingService | None = None,
) -> int:
    """
    Seed the vector index with curated historical recovery precedents for the demo tenant.

    Idempotent: Uses `vector_index.contains(case_id)` to skip already indexed documents,
    ensuring that partially populated indexes are safely completed without duplicate entries.

    Args:
        vector_index: Target VectorIndex to populate (defaults to shared global singleton).
        customer_id: Customer UUID for tenant binding (defaults to DEMO_CUSTOMER_UUID).
        embedding_service: EmbeddingService for vector generation (defaults to global singleton).

    Returns:
        Number of newly added documents to the vector index.
    """
    resolved_index = vector_index if vector_index is not None else get_vector_index()
    resolved_cust_id = (
        UUID(str(customer_id)) if customer_id is not None else DEMO_CUSTOMER_UUID
    )
    resolved_embedding = (
        embedding_service if embedding_service is not None else get_embedding_service()
    )

    precedents = get_curated_historical_precedents(customer_id=resolved_cust_id)
    added_count = 0

    for case in precedents:
        if resolved_index.get(case.payment_id) is not None:
            continue
        doc = historical_case_to_document(case)
        vec = resolved_embedding.embed(doc.text)
        resolved_index.add(doc, vec)
        added_count += 1

    if added_count > 0:
        logger.info(
            "Seeded runtime VectorIndex with %d historical precedents (total index size: %d).",
            added_count,
            resolved_index.size,
        )
    return added_count
