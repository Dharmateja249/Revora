"""
Comprehensive Test Suite for HistoricalRecoveryCase and Mappers.

Verifies:
1. Complete historical recovered case
2. Historical failed case
3. Case with no recovery attempts
4. Case with multiple attempts in chronological order
5. Amount validation (negative amounts, bounds, recovered > amount)
6. Invalid/unsupported status handling
7. Immutability
8. External mutation of source collections does not mutate the case
9. Serialization / Deserialization round-tripping (JSON & dict)
10. Cold-start / empty-history behavior
11. Deterministic mappers from domain/context schemas
12. Customer PII exclusion
"""

import json
import uuid
from datetime import datetime, timedelta, timezone

import pytest
from app.context import (
    CustomerContext,
    CustomerRecoveryContext,
    HistoricalPaymentContext,
    PaymentContext,
    RecoveryAttemptContext,
    RecoveryOpportunityContext,
)
from app.historical_case import (
    HistoricalAttempt,
    HistoricalRecoveryCase,
    map_context_to_historical_case,
    map_customer_recovery_context_to_cases,
    map_historical_payment_to_case,
)
from pydantic import ValidationError

# ============================================================================
# 1. Complete Historical Recovered Case
# ============================================================================


def test_complete_historical_recovered_case():
    """Verify clean instantiation of a fully populated recovered case."""
    payment_id = uuid.uuid4()
    opportunity_id = uuid.uuid4()
    cust_id = uuid.uuid4()
    att_id_1 = uuid.uuid4()
    att_id_2 = uuid.uuid4()

    t0 = datetime(2026, 1, 15, 10, 0, 0, tzinfo=timezone.utc)
    t1 = t0 + timedelta(hours=1)
    t2 = t0 + timedelta(hours=2)

    attempt1 = HistoricalAttempt(
        attempt_id=att_id_1,
        action="smart_retry",
        status="failed",
        amount_recovered=0.0,
        error_code="bank_timeout",
        external_reference="ref_001",
        created_at=t1,
        completed_at=t1 + timedelta(seconds=5),
    )
    attempt2 = HistoricalAttempt(
        attempt_id=att_id_2,
        action="payment_link",
        status="succeeded",
        amount_recovered=3500.0,
        external_reference="ref_002",
        created_at=t2,
        completed_at=t2 + timedelta(minutes=10),
    )

    case = HistoricalRecoveryCase(
        payment_id=payment_id,
        external_payment_id="pay_ext_100",
        opportunity_id=opportunity_id,
        customer_id=cust_id,
        external_customer_id="cust_ext_100",
        amount=3500.0,
        currency="INR",
        payment_method="upi",
        failure_reason="bank_timeout",
        recovery_status="recovered",
        amount_recovered=3500.0,
        successful_action="payment_link",
        attempts=(attempt1, attempt2),
        created_at=t0,
        completed_at=t2 + timedelta(minutes=10),
        metadata={"channel": "sms", "hours_to_recover": 2.16},
    )

    assert case.payment_id == payment_id
    assert case.external_payment_id == "pay_ext_100"
    assert case.opportunity_id == opportunity_id
    assert case.customer_id == cust_id
    assert case.external_customer_id == "cust_ext_100"
    assert case.amount == 3500.0
    assert case.currency == "INR"
    assert case.payment_method == "upi"
    assert case.failure_reason == "bank_timeout"
    assert case.recovery_status == "recovered"
    assert case.amount_recovered == 3500.0
    assert case.successful_action == "payment_link"
    assert len(case.attempts) == 2
    assert case.attempts[0].attempt_id == att_id_1
    assert case.attempts[1].attempt_id == att_id_2
    assert case.created_at == t0
    assert case.completed_at == t2 + timedelta(minutes=10)
    assert case.metadata["channel"] == "sms"


# ============================================================================
# 2. Historical Failed Case
# ============================================================================


def test_historical_failed_case():
    """Verify representation of an unrecovered historical failure."""
    payment_id = uuid.uuid4()
    t0 = datetime(2026, 1, 10, 14, 0, 0, tzinfo=timezone.utc)

    attempt = HistoricalAttempt(
        action="retry_payment",
        status="failed",
        amount_recovered=0.0,
        error_code="insufficient_funds",
        created_at=t0,
    )

    case = HistoricalRecoveryCase(
        payment_id=payment_id,
        amount=1200.0,
        currency="INR",
        payment_method="card",
        failure_reason="insufficient_funds",
        recovery_status="failed",
        amount_recovered=0.0,
        successful_action=None,
        attempts=(attempt,),
        created_at=t0,
    )

    assert case.recovery_status == "failed"
    assert case.amount_recovered == 0.0
    assert case.successful_action is None
    assert len(case.attempts) == 1
    assert case.attempts[0].status == "failed"


# ============================================================================
# 3. Case with No Recovery Attempts
# ============================================================================


def test_case_with_no_recovery_attempts():
    """Verify handling when no recovery attempts were made."""
    payment_id = uuid.uuid4()

    case = HistoricalRecoveryCase(
        payment_id=payment_id,
        amount=500.0,
        currency="INR",
        payment_method="netbanking",
        failure_reason="service_unavailable",
        recovery_status="open",
        attempts=(),
    )

    assert case.attempts == ()
    assert case.amount_recovered == 0.0
    assert case.successful_action is None
    assert case.metadata == {}


# ============================================================================
# 4. Chronological Attempt Ordering
# ============================================================================


def test_attempts_sorted_chronologically():
    """Verify attempts passed in out-of-order timestamps are sorted deterministically."""
    t0 = datetime(2026, 1, 1, 10, 0, 0, tzinfo=timezone.utc)
    t1 = t0 + timedelta(hours=1)
    t2 = t0 + timedelta(hours=2)
    t3 = t0 + timedelta(hours=3)

    att_1 = HistoricalAttempt(action="att_1", status="failed", created_at=t1)
    att_2 = HistoricalAttempt(action="att_2", status="failed", created_at=t2)
    att_3 = HistoricalAttempt(action="att_3", status="succeeded", created_at=t3)

    # Pass in jumbled order: att_3, att_1, att_2
    case = HistoricalRecoveryCase(
        payment_id=uuid.uuid4(),
        amount=1000.0,
        currency="INR",
        payment_method="card",
        recovery_status="recovered",
        amount_recovered=1000.0,
        attempts=(att_3, att_1, att_2),
    )

    assert [a.action for a in case.attempts] == ["att_1", "att_2", "att_3"]


# ============================================================================
# 5. Amount Validation Bounds
# ============================================================================


def test_amount_validation_negative_payment_amount():
    """Verify negative payment amount is rejected."""
    with pytest.raises(ValidationError) as exc_info:
        HistoricalRecoveryCase(
            payment_id=uuid.uuid4(),
            amount=-50.0,
            payment_method="card",
            recovery_status="failed",
        )
    assert "amount" in str(exc_info.value)


def test_amount_validation_negative_recovered_amount():
    """Verify negative amount_recovered is rejected."""
    with pytest.raises(ValidationError) as exc_info:
        HistoricalRecoveryCase(
            payment_id=uuid.uuid4(),
            amount=100.0,
            payment_method="card",
            recovery_status="failed",
            amount_recovered=-10.0,
        )
    assert "amount_recovered" in str(exc_info.value)


def test_amount_recovered_exceeding_payment_amount():
    """Verify amount_recovered cannot exceed original payment amount."""
    with pytest.raises(ValidationError) as exc_info:
        HistoricalRecoveryCase(
            payment_id=uuid.uuid4(),
            amount=500.0,
            payment_method="card",
            recovery_status="recovered",
            amount_recovered=600.0,
        )
    assert "cannot exceed payment amount" in str(exc_info.value)


def test_attempt_negative_amount_recovered():
    """Verify HistoricalAttempt rejects negative amount_recovered."""
    with pytest.raises(ValidationError) as exc_info:
        HistoricalAttempt(
            action="retry",
            status="failed",
            amount_recovered=-5.0,
        )
    assert "amount_recovered" in str(exc_info.value)


# ============================================================================
# 6. Invalid / Unsupported Status Handling
# ============================================================================


def test_invalid_recovery_status_rejected():
    """Verify unsupported recovery_status raises validation error with allowed set."""
    with pytest.raises(ValidationError) as exc_info:
        HistoricalRecoveryCase(
            payment_id=uuid.uuid4(),
            amount=100.0,
            payment_method="card",
            recovery_status="random_unsupported_status",
        )
    assert "Invalid or unsupported recovery status" in str(exc_info.value)


def test_invalid_attempt_status_rejected():
    """Verify unsupported attempt status raises validation error."""
    with pytest.raises(ValidationError) as exc_info:
        HistoricalAttempt(
            action="retry",
            status="some_unknown_status",
        )
    assert "Invalid or unsupported attempt status" in str(exc_info.value)


def test_empty_string_status_rejected():
    """Verify empty or whitespace-only status values are rejected."""
    with pytest.raises(ValidationError):
        HistoricalRecoveryCase(
            payment_id=uuid.uuid4(),
            amount=100.0,
            payment_method="card",
            recovery_status="   ",
        )

    with pytest.raises(ValidationError):
        HistoricalAttempt(
            action="retry",
            status="",
        )


def test_status_case_normalization():
    """Verify statuses provided in mixed case or with surrounding whitespace are normalized."""
    case = HistoricalRecoveryCase(
        payment_id=uuid.uuid4(),
        amount=100.0,
        payment_method="card",
        recovery_status=" RECOVERED ",
    )
    assert case.recovery_status == "recovered"

    attempt = HistoricalAttempt(
        action=" Retry ",
        status=" Succeeded ",
    )
    assert attempt.action == "Retry"
    assert attempt.status == "succeeded"


# ============================================================================
# 7. Immutability
# ============================================================================


def test_schema_immutability():
    """Verify frozen models raise ValidationError on attribute assignment."""
    case = HistoricalRecoveryCase(
        payment_id=uuid.uuid4(),
        amount=100.0,
        payment_method="card",
        recovery_status="failed",
    )

    with pytest.raises(ValidationError):
        case.amount = 200.0  # type: ignore

    with pytest.raises(ValidationError):
        case.recovery_status = "recovered"  # type: ignore

    attempt = HistoricalAttempt(action="retry", status="failed")
    with pytest.raises(ValidationError):
        attempt.status = "succeeded"  # type: ignore


# ============================================================================
# 8. Protection Against External Mutation of Source Collections
# ============================================================================


def test_external_mutation_of_attempts_list_does_not_affect_case():
    """Verify mutating the list passed into attempts does not alter the case."""
    att1 = HistoricalAttempt(action="retry_1", status="failed")
    source_attempts = [att1]

    case = HistoricalRecoveryCase(
        payment_id=uuid.uuid4(),
        amount=1000.0,
        payment_method="card",
        recovery_status="failed",
        attempts=source_attempts,
    )

    # Mutate external list
    att2 = HistoricalAttempt(
        action="retry_2", status="succeeded", amount_recovered=1000.0
    )
    source_attempts.append(att2)

    assert len(case.attempts) == 1
    assert case.attempts[0].action == "retry_1"
    assert isinstance(case.attempts, tuple)


def test_external_mutation_of_metadata_dict_does_not_affect_case():
    """Verify mutating the dict passed into metadata does not alter the case."""
    source_metadata = {"initial_key": "initial_value", "nested": {"counter": 1}}

    case = HistoricalRecoveryCase(
        payment_id=uuid.uuid4(),
        amount=1000.0,
        payment_method="card",
        recovery_status="failed",
        metadata=source_metadata,
    )

    # Mutate external dict
    source_metadata["initial_key"] = "mutated_value"
    source_metadata["new_key"] = "brand_new"
    source_metadata["nested"]["counter"] = 999

    assert case.metadata["initial_key"] == "initial_value"
    assert "new_key" not in case.metadata
    assert case.metadata["nested"]["counter"] == 1

    # In-place modification on MappingProxyType should raise TypeError
    with pytest.raises(TypeError):
        case.metadata["initial_key"] = "direct_mutation"  # type: ignore


# ============================================================================
# 9. Serialization & Deserialization
# ============================================================================


def test_serialization_roundtrip_dict_and_json():
    """Verify round-trip serialization to/from dictionary and JSON."""
    payment_id = uuid.uuid4()
    cust_id = uuid.uuid4()
    now = datetime(2026, 2, 1, 12, 30, 0, tzinfo=timezone.utc)

    attempt = HistoricalAttempt(
        attempt_id=uuid.uuid4(),
        action="payment_link",
        status="succeeded",
        amount_recovered=750.0,
        created_at=now,
    )

    case = HistoricalRecoveryCase(
        payment_id=payment_id,
        external_payment_id="ext_pay_750",
        customer_id=cust_id,
        external_customer_id="ext_cust_750",
        amount=750.0,
        currency="INR",
        payment_method="upi",
        failure_reason="timeout",
        recovery_status="recovered",
        amount_recovered=750.0,
        successful_action="payment_link",
        attempts=(attempt,),
        created_at=now,
        metadata={"rule": "TestRule", "tags": ["fast_recovery"]},
    )

    # Dict dump and validation
    dumped_dict = case.model_dump()
    assert dumped_dict["payment_id"] == payment_id
    assert dumped_dict["recovery_status"] == "recovered"
    assert len(dumped_dict["attempts"]) == 1
    assert dumped_dict["metadata"]["tags"] == ["fast_recovery"]

    reconstructed_from_dict = HistoricalRecoveryCase.model_validate(dumped_dict)
    assert reconstructed_from_dict == case

    # JSON dump and validation
    json_str = case.model_dump_json()
    assert isinstance(json_str, str)
    parsed_json = json.loads(json_str)
    assert parsed_json["payment_id"] == str(payment_id)
    assert parsed_json["amount"] == 750.0

    reconstructed_from_json = HistoricalRecoveryCase.model_validate_json(json_str)
    assert reconstructed_from_json.payment_id == case.payment_id
    assert reconstructed_from_json.amount == case.amount
    assert reconstructed_from_json.recovery_status == case.recovery_status
    assert reconstructed_from_json.attempts[0].action == case.attempts[0].action


# ============================================================================
# 10. Cold-Start / Empty-History Behavior
# ============================================================================


def test_cold_start_empty_history_defaults():
    """Verify safe handling of minimal historical case with default empty fields."""
    payment_id = uuid.uuid4()

    case = HistoricalRecoveryCase(
        payment_id=payment_id,
        amount=0.0,
        payment_method="unknown",
        recovery_status="pending",
    )

    assert case.payment_id == payment_id
    assert case.external_payment_id is None
    assert case.opportunity_id is None
    assert case.customer_id is None
    assert case.external_customer_id is None
    assert case.amount == 0.0
    assert case.currency == "INR"
    assert case.failure_reason is None
    assert case.recovery_status == "pending"
    assert case.amount_recovered == 0.0
    assert case.successful_action is None
    assert case.attempts == ()
    assert case.created_at is None
    assert case.completed_at is None
    assert case.metadata == {}


# ============================================================================
# 11. Deterministic Domain & Context Mappers
# ============================================================================


def test_mapper_from_payment_and_opportunity_context():
    """Verify deterministic mapping from PaymentContext and RecoveryOpportunityContext."""
    payment_id = uuid.uuid4()
    opp_id = uuid.uuid4()
    cust_id = uuid.uuid4()
    t0 = datetime(2026, 2, 10, 8, 0, 0, tzinfo=timezone.utc)

    pay_ctx = PaymentContext(
        payment_id=payment_id,
        external_payment_id="pay_999",
        amount=4000.0,
        currency="INR",
        payment_method="card",
        status="failed",
        failure_reason="card_expired",
        created_at=t0,
    )

    opp_ctx = RecoveryOpportunityContext(
        opportunity_id=opp_id,
        status="recovered",
        revenue_at_risk=4000.0,
        expected_recovery=4000.0,
        recommended_action="change_payment_method",
        confidence=0.9,
        created_at=t0,
    )

    att_ctx = RecoveryAttemptContext(
        attempt_id=uuid.uuid4(),
        action="change_payment_method",
        status="succeeded",
        amount_recovered=4000.0,
        created_at=t0 + timedelta(minutes=15),
        completed_at=t0 + timedelta(minutes=16),
    )

    case = map_context_to_historical_case(
        payment=pay_ctx,
        opportunity=opp_ctx,
        attempts=[att_ctx],
        customer_id=cust_id,
        external_customer_id="cust_ext_999",
    )

    assert case.payment_id == payment_id
    assert case.opportunity_id == opp_id
    assert case.customer_id == cust_id
    assert case.external_customer_id == "cust_ext_999"
    assert case.recovery_status == "recovered"
    assert case.amount == 4000.0
    assert case.amount_recovered == 4000.0
    assert case.successful_action == "change_payment_method"
    assert len(case.attempts) == 1
    assert case.attempts[0].action == "change_payment_method"
    assert case.completed_at == t0 + timedelta(minutes=16)
    assert case.metadata["confidence"] == 0.9


def test_mapper_from_historical_payment_context():
    """Verify deterministic mapping from HistoricalPaymentContext."""
    hist_id = uuid.uuid4()
    cust_id = uuid.uuid4()
    t0 = datetime(2026, 1, 5, 9, 0, 0, tzinfo=timezone.utc)

    hist_payment = HistoricalPaymentContext(
        payment_id=hist_id,
        external_payment_id="hist_pay_1",
        amount=1800.0,
        currency="INR",
        payment_method="upi",
        status="succeeded",
        failure_reason=None,
        created_at=t0,
        was_recovered=True,
        recovery_action="customer_prompt_upi",
        recovery_attempts_count=1,
    )

    case = map_historical_payment_to_case(
        historical_payment=hist_payment,
        customer_id=cust_id,
        external_customer_id="cust_ext_1",
    )

    assert case.payment_id == hist_id
    assert case.customer_id == cust_id
    assert case.external_customer_id == "cust_ext_1"
    assert case.recovery_status == "recovered"
    assert case.amount == 1800.0
    assert case.amount_recovered == 1800.0
    assert case.successful_action == "customer_prompt_upi"
    assert case.metadata["recovery_attempts_count"] == 1
    assert case.metadata["was_recovered"] is True


def test_map_customer_recovery_context_to_cases():
    """Verify mapping an entire CustomerRecoveryContext extracts all historical cases."""
    customer_id = uuid.uuid4()
    curr_payment_id = uuid.uuid4()
    hist_payment_id = uuid.uuid4()

    customer_ctx = CustomerContext(
        customer_id=customer_id,
        external_customer_id="cust_full",
        name="Alex Morgan",
        email="alex@secret-pii.com",
        total_payments=2,
        successful_payments=1,
        failed_payments=1,
    )

    curr_payment = PaymentContext(
        payment_id=curr_payment_id,
        amount=2000.0,
        currency="INR",
        payment_method="card",
        status="failed",
        failure_reason="gateway_timeout",
    )

    curr_opp = RecoveryOpportunityContext(
        opportunity_id=uuid.uuid4(),
        status="open",
        revenue_at_risk=2000.0,
        expected_recovery=0.0,
    )

    hist_payment = HistoricalPaymentContext(
        payment_id=hist_payment_id,
        amount=1500.0,
        currency="INR",
        payment_method="card",
        status="succeeded",
        was_recovered=True,
        recovery_action="smart_retry",
        recovery_attempts_count=1,
    )

    context = CustomerRecoveryContext(
        customer=customer_ctx,
        current_payment=curr_payment,
        current_opportunity=curr_opp,
        current_payment_attempts=[],
        historical_payments=[hist_payment],
    )

    cases = map_customer_recovery_context_to_cases(context)
    assert len(cases) == 2

    # Case 0 is current payment
    assert cases[0].payment_id == curr_payment_id
    assert cases[0].customer_id == customer_id
    assert cases[0].external_customer_id == "cust_full"
    assert cases[0].recovery_status == "open"

    # Case 1 is historical payment
    assert cases[1].payment_id == hist_payment_id
    assert cases[1].customer_id == customer_id
    assert cases[1].external_customer_id == "cust_full"
    assert cases[1].recovery_status == "recovered"
    assert cases[1].successful_action == "smart_retry"


# ============================================================================
# 12. PII Exclusion Verification
# ============================================================================


def test_pii_exclusion():
    """Verify that CustomerRecoveryCase has NO customer name or email fields."""
    customer_ctx = CustomerContext(
        customer_id=uuid.uuid4(),
        external_customer_id="cust_pii_test",
        name="Sensitive Customer Name",
        email="sensitive@pii-example.com",
    )
    payment_ctx = PaymentContext(
        payment_id=uuid.uuid4(),
        amount=500.0,
        currency="INR",
        payment_method="card",
        status="failed",
    )

    case = map_context_to_historical_case(
        payment=payment_ctx,
        customer=customer_ctx,
    )

    # Attributes check
    assert not hasattr(case, "name")
    assert not hasattr(case, "email")
    assert "name" not in HistoricalRecoveryCase.model_fields
    assert "email" not in HistoricalRecoveryCase.model_fields

    dumped = case.model_dump()
    assert "name" not in dumped
    assert "email" not in dumped
    assert "Sensitive Customer Name" not in str(dumped)
    assert "sensitive@pii-example.com" not in str(dumped)


def test_attempts_from_recovery_attempt_context_and_dict():
    """Verify attempts passed as RecoveryAttemptContext and dict objects are converted properly."""
    att_ctx = RecoveryAttemptContext(
        attempt_id=uuid.uuid4(),
        action="change_payment_method",
        status="succeeded",
        amount_recovered=2000.0,
    )
    raw_dict = {
        "action": "smart_retry",
        "status": "failed",
        "amount_recovered": 0.0,
        "error_code": "network_timeout",
    }

    case = HistoricalRecoveryCase(
        payment_id=uuid.uuid4(),
        amount=2000.0,
        currency="INR",
        payment_method="card",
        recovery_status="recovered",
        amount_recovered=2000.0,
        attempts=[att_ctx, raw_dict],
    )

    assert len(case.attempts) == 2
    assert all(isinstance(a, HistoricalAttempt) for a in case.attempts)
    assert case.attempts[0].action in ("change_payment_method", "smart_retry")


def test_empty_payment_method_rejected():
    """Verify empty payment_method string is rejected."""
    with pytest.raises(ValidationError):
        HistoricalRecoveryCase(
            payment_id=uuid.uuid4(),
            amount=100.0,
            payment_method="   ",
            recovery_status="open",
        )


def test_mapper_fallback_succeeded_payment():
    """Verify mapper handles succeeded payment with no opportunity or attempts."""
    payment_id = uuid.uuid4()
    pay_ctx = PaymentContext(
        payment_id=payment_id,
        amount=1500.0,
        currency="INR",
        payment_method="upi",
        status="succeeded",
    )

    case = map_context_to_historical_case(payment=pay_ctx)
    assert case.recovery_status == "recovered"
    assert case.amount_recovered == 1500.0
    assert case.attempts == ()
