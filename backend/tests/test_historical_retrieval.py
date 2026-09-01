"""
Unit tests for HistoricalCase Retrieval Schema.

Verifies:
1. Valid construction with full and minimal fields
2. Validation bounds on monetary values (amount >= 0, amount_recovered >= 0, amount_recovered <= amount)
3. Relevance score boundary enforcement ([0.0, 1.0], None, <0.0 rejects, >1.0 rejects)
4. Required identifiers enforcement (payment_id, customer_id)
5. Valid recovery status and payment method validation
6. Immutability / Frozen behavior
7. External collection mutation isolation
8. Serialization roundtrip to/from dict and JSON
9. Zero / edge monetary values
"""

import json
import uuid
from datetime import datetime, timedelta, timezone

import pytest
from app.historical_retrieval import (
    HistoricalCase,
)
from pydantic import ValidationError

# ============================================================================
# 1. Valid Construction
# ============================================================================


def test_valid_historical_case_full_construction():
    """Verify clean instantiation of a fully populated HistoricalCase."""
    payment_id = uuid.uuid4()
    customer_id = uuid.uuid4()
    t0 = datetime(2026, 2, 1, 10, 0, 0, tzinfo=timezone.utc)
    t1 = t0 + timedelta(hours=1)

    case = HistoricalCase(
        payment_id=payment_id,
        customer_id=customer_id,
        external_payment_id="pay_ext_501",
        external_customer_id="cust_ext_501",
        amount=2500.0,
        currency="INR",
        payment_method="card",
        failure_reason="insufficient_funds",
        recovery_action="wait_and_retry",
        recovery_status="recovered",
        amount_recovered=2500.0,
        was_recovered=True,
        relevance_score=0.92,
        created_at=t0,
        completed_at=t1,
        metadata={"similarity_metric": "cosine", "cluster_id": 4},
    )

    assert case.payment_id == payment_id
    assert case.customer_id == customer_id
    assert case.external_payment_id == "pay_ext_501"
    assert case.external_customer_id == "cust_ext_501"
    assert case.amount == 2500.0
    assert case.currency == "INR"
    assert case.payment_method == "card"
    assert case.failure_reason == "insufficient_funds"
    assert case.recovery_action == "wait_and_retry"
    assert case.recovery_status == "recovered"
    assert case.amount_recovered == 2500.0
    assert case.was_recovered is True
    assert case.relevance_score == 0.92
    assert case.created_at == t0
    assert case.completed_at == t1
    assert case.metadata["similarity_metric"] == "cosine"


def test_valid_historical_case_minimal_defaults():
    """Verify construction with only required fields and sensible defaults."""
    payment_id = uuid.uuid4()
    customer_id = uuid.uuid4()

    case = HistoricalCase(
        payment_id=payment_id,
        customer_id=customer_id,
        amount=100.0,
        payment_method="upi",
        recovery_status="failed",
    )

    assert case.payment_id == payment_id
    assert case.customer_id == customer_id
    assert case.external_payment_id is None
    assert case.external_customer_id is None
    assert case.amount == 100.0
    assert case.currency == "INR"
    assert case.payment_method == "upi"
    assert case.failure_reason is None
    assert case.recovery_action is None
    assert case.recovery_status == "failed"
    assert case.amount_recovered == 0.0
    assert case.was_recovered is False
    assert case.relevance_score is None
    assert case.created_at is None
    assert case.completed_at is None
    assert case.metadata == {}


# ============================================================================
# 2. Validation Bounds on Monetary Values
# ============================================================================


def test_negative_payment_amount_rejected():
    """Verify negative payment amount raises ValidationError."""
    with pytest.raises(ValidationError) as exc_info:
        HistoricalCase(
            payment_id=uuid.uuid4(),
            customer_id=uuid.uuid4(),
            amount=-50.0,
            payment_method="card",
            recovery_status="failed",
        )
    assert "amount" in str(exc_info.value)


def test_negative_recovered_amount_rejected():
    """Verify negative amount_recovered raises ValidationError."""
    with pytest.raises(ValidationError) as exc_info:
        HistoricalCase(
            payment_id=uuid.uuid4(),
            customer_id=uuid.uuid4(),
            amount=500.0,
            payment_method="card",
            recovery_status="failed",
            amount_recovered=-1.0,
        )
    assert "amount_recovered" in str(exc_info.value)


def test_amount_recovered_exceeding_payment_amount_rejected():
    """Verify amount_recovered exceeding amount raises ValidationError."""
    with pytest.raises(ValidationError) as exc_info:
        HistoricalCase(
            payment_id=uuid.uuid4(),
            customer_id=uuid.uuid4(),
            amount=500.0,
            payment_method="card",
            recovery_status="recovered",
            amount_recovered=550.0,
        )
    assert "cannot exceed payment amount" in str(exc_info.value)


def test_zero_monetary_values_accepted():
    """Verify zero amount and zero amount_recovered are valid."""
    case = HistoricalCase(
        payment_id=uuid.uuid4(),
        customer_id=uuid.uuid4(),
        amount=0.0,
        payment_method="card",
        recovery_status="open",
        amount_recovered=0.0,
    )
    assert case.amount == 0.0
    assert case.amount_recovered == 0.0


# ============================================================================
# 3. Relevance Score Boundary Enforcement
# ============================================================================


def test_relevance_score_boundaries():
    """Verify relevance scores at exactly 0.0, 1.0, and intermediate values are valid."""
    payment_id = uuid.uuid4()
    customer_id = uuid.uuid4()

    case_zero = HistoricalCase(
        payment_id=payment_id,
        customer_id=customer_id,
        amount=100.0,
        payment_method="card",
        recovery_status="open",
        relevance_score=0.0,
    )
    assert case_zero.relevance_score == 0.0

    case_one = HistoricalCase(
        payment_id=payment_id,
        customer_id=customer_id,
        amount=100.0,
        payment_method="card",
        recovery_status="open",
        relevance_score=1.0,
    )
    assert case_one.relevance_score == 1.0

    case_none = HistoricalCase(
        payment_id=payment_id,
        customer_id=customer_id,
        amount=100.0,
        payment_method="card",
        recovery_status="open",
        relevance_score=None,
    )
    assert case_none.relevance_score is None


def test_relevance_score_below_zero_rejected():
    """Verify relevance_score < 0.0 is rejected."""
    with pytest.raises(ValidationError) as exc_info:
        HistoricalCase(
            payment_id=uuid.uuid4(),
            customer_id=uuid.uuid4(),
            amount=100.0,
            payment_method="card",
            recovery_status="open",
            relevance_score=-0.01,
        )
    assert "relevance_score" in str(exc_info.value)


def test_relevance_score_above_one_rejected():
    """Verify relevance_score > 1.0 is rejected."""
    with pytest.raises(ValidationError) as exc_info:
        HistoricalCase(
            payment_id=uuid.uuid4(),
            customer_id=uuid.uuid4(),
            amount=100.0,
            payment_method="card",
            recovery_status="open",
            relevance_score=1.01,
        )
    assert "relevance_score" in str(exc_info.value)


# ============================================================================
# 4. Required Identifiers Enforcement
# ============================================================================


def test_missing_payment_id_rejected():
    """Verify missing payment_id raises ValidationError."""
    with pytest.raises(ValidationError):
        HistoricalCase(
            customer_id=uuid.uuid4(),  # type: ignore
            amount=100.0,
            payment_method="card",
            recovery_status="open",
        )


def test_missing_customer_id_rejected():
    """Verify missing customer_id raises ValidationError."""
    with pytest.raises(ValidationError):
        HistoricalCase(
            payment_id=uuid.uuid4(),  # type: ignore
            amount=100.0,
            payment_method="card",
            recovery_status="open",
        )


# ============================================================================
# 5. Recovery Status & Payment Method Validation
# ============================================================================


def test_invalid_recovery_status_rejected():
    """Verify unsupported recovery_status raises ValidationError."""
    with pytest.raises(ValidationError) as exc_info:
        HistoricalCase(
            payment_id=uuid.uuid4(),
            customer_id=uuid.uuid4(),
            amount=100.0,
            payment_method="card",
            recovery_status="invalid_status_abc",
        )
    assert "Invalid or unsupported recovery status" in str(exc_info.value)


def test_empty_string_recovery_status_rejected():
    """Verify whitespace/empty recovery status is rejected."""
    with pytest.raises(ValidationError):
        HistoricalCase(
            payment_id=uuid.uuid4(),
            customer_id=uuid.uuid4(),
            amount=100.0,
            payment_method="card",
            recovery_status="   ",
        )


def test_empty_string_payment_method_rejected():
    """Verify whitespace/empty payment method is rejected."""
    with pytest.raises(ValidationError):
        HistoricalCase(
            payment_id=uuid.uuid4(),
            customer_id=uuid.uuid4(),
            amount=100.0,
            payment_method="   ",
            recovery_status="open",
        )


def test_status_case_normalization():
    """Verify case normalization for recovery_status."""
    case = HistoricalCase(
        payment_id=uuid.uuid4(),
        customer_id=uuid.uuid4(),
        amount=100.0,
        payment_method="card",
        recovery_status=" RECOVERED ",
    )
    assert case.recovery_status == "recovered"


# ============================================================================
# 6. Immutability
# ============================================================================


def test_historical_case_immutability():
    """Verify attempting to mutate any attribute raises ValidationError."""
    case = HistoricalCase(
        payment_id=uuid.uuid4(),
        customer_id=uuid.uuid4(),
        amount=100.0,
        payment_method="card",
        recovery_status="failed",
    )

    with pytest.raises(ValidationError):
        case.amount = 200.0  # type: ignore

    with pytest.raises(ValidationError):
        case.recovery_status = "recovered"  # type: ignore

    with pytest.raises(ValidationError):
        case.relevance_score = 0.95  # type: ignore


# ============================================================================
# 7. External Collection Mutation Isolation
# ============================================================================


def test_external_metadata_mutation_isolation():
    """Verify mutating external dict does not alter frozen metadata mapping."""
    source_meta = {"algorithm": "bm25", "nested": {"k1": 1.2}}

    case = HistoricalCase(
        payment_id=uuid.uuid4(),
        customer_id=uuid.uuid4(),
        amount=500.0,
        payment_method="card",
        recovery_status="recovered",
        metadata=source_meta,
    )

    # Mutate source dict
    source_meta["algorithm"] = "tfidf"
    source_meta["nested"]["k1"] = 9.9
    source_meta["added_key"] = "test"

    assert case.metadata["algorithm"] == "bm25"
    assert case.metadata["nested"]["k1"] == 1.2
    assert "added_key" not in case.metadata

    with pytest.raises(TypeError):
        case.metadata["algorithm"] = "mutated"  # type: ignore


# ============================================================================
# 8. Serialization Roundtrip
# ============================================================================


def test_serialization_roundtrip_dict_and_json():
    """Verify round-trip serialization to/from Python dict and JSON."""
    payment_id = uuid.uuid4()
    customer_id = uuid.uuid4()
    now = datetime(2026, 2, 15, 14, 30, 0, tzinfo=timezone.utc)

    case = HistoricalCase(
        payment_id=payment_id,
        customer_id=customer_id,
        external_payment_id="ext_p_88",
        external_customer_id="ext_c_88",
        amount=1200.0,
        currency="INR",
        payment_method="upi",
        failure_reason="timeout",
        recovery_action="payment_link",
        recovery_status="recovered",
        amount_recovered=1200.0,
        was_recovered=True,
        relevance_score=0.875,
        created_at=now,
        completed_at=now + timedelta(minutes=5),
        metadata={"retrieval_pass": 1, "tags": ["high_confidence"]},
    )

    # Model dump
    dumped = case.model_dump()
    assert dumped["payment_id"] == payment_id
    assert dumped["customer_id"] == customer_id
    assert dumped["relevance_score"] == 0.875
    assert dumped["metadata"]["tags"] == ["high_confidence"]

    reconstructed_from_dict = HistoricalCase.model_validate(dumped)
    assert reconstructed_from_dict == case

    # JSON dump
    json_str = case.model_dump_json()
    assert isinstance(json_str, str)
    parsed_json = json.loads(json_str)
    assert parsed_json["payment_id"] == str(payment_id)
    assert parsed_json["relevance_score"] == 0.875

    reconstructed_from_json = HistoricalCase.model_validate_json(json_str)
    assert reconstructed_from_json == case
