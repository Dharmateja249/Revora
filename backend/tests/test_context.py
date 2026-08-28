"""
Tests for Revora Context Schemas and Domain Exceptions.

Verifies:
- Valid context construction and serialization
- Correct nested structures
- Graceful cold-start / zero-history customer handling
- Schema immutability and strict validation
- Proper raising and identification of domain exceptions
"""

from datetime import datetime, timezone
import uuid
import pytest
from pydantic import ValidationError

from app.context import (
    CustomerContext,
    PaymentContext,
    RecoveryOpportunityContext,
    RecoveryAttemptContext,
    HistoricalPaymentContext,
    CustomerRecoveryStatsContext,
    CustomerRecoveryContext,
    ContextRetrievalError,
    CustomerNotFoundError,
    PaymentNotFoundError,
    PaymentCustomerMismatchError,
    RecoveryOpportunityNotFoundError,
)


# ============================================================================
# Domain Exceptions Tests
# ============================================================================


def test_domain_exceptions_hierarchy():
    """Verify all custom exceptions inherit from ContextRetrievalError."""
    assert issubclass(CustomerNotFoundError, ContextRetrievalError)
    assert issubclass(PaymentNotFoundError, ContextRetrievalError)
    assert issubclass(PaymentCustomerMismatchError, ContextRetrievalError)
    assert issubclass(RecoveryOpportunityNotFoundError, ContextRetrievalError)


def test_customer_not_found_error():
    """Verify CustomerNotFoundError formatting and attribute retention."""
    cust_id = uuid.uuid4()
    err = CustomerNotFoundError(cust_id)
    assert err.identifier == str(cust_id)
    assert str(cust_id) in str(err)
    assert "was not found" in str(err)

    # String external ID support
    err_ext = CustomerNotFoundError("cust_ext_999")
    assert err_ext.identifier == "cust_ext_999"
    assert "cust_ext_999" in str(err_ext)


def test_payment_not_found_error():
    """Verify PaymentNotFoundError formatting and attribute retention."""
    pay_id = uuid.uuid4()
    err = PaymentNotFoundError(pay_id)
    assert err.identifier == str(pay_id)
    assert str(pay_id) in str(err)
    assert "was not found" in str(err)


def test_payment_customer_mismatch_error():
    """Verify PaymentCustomerMismatchError detects and formats tenant cross-contamination."""
    pay_id = uuid.uuid4()
    cust_a = uuid.uuid4()
    cust_b = uuid.uuid4()

    err = PaymentCustomerMismatchError(
        payment_id=pay_id,
        customer_id=cust_a,
        actual_customer_id=cust_b,
    )
    assert err.payment_id == str(pay_id)
    assert err.customer_id == str(cust_a)
    assert err.actual_customer_id == str(cust_b)
    assert str(cust_a) in str(err)
    assert str(cust_b) in str(err)


def test_recovery_opportunity_not_found_error():
    """Verify RecoveryOpportunityNotFoundError formatting."""
    opp_id = uuid.uuid4()
    err = RecoveryOpportunityNotFoundError(opp_id)
    assert err.identifier == str(opp_id)
    assert str(opp_id) in str(err)


# ============================================================================
# Context Schema Construction & Validation Tests
# ============================================================================


def test_valid_customer_recovery_context_construction():
    """Verify complete, valid context construction with all nested fields populated."""
    customer_id = uuid.uuid4()
    payment_id = uuid.uuid4()
    opportunity_id = uuid.uuid4()
    attempt_id = uuid.uuid4()
    hist_payment_id = uuid.uuid4()
    now = datetime.now(timezone.utc)

    customer_ctx = CustomerContext(
        customer_id=customer_id,
        external_customer_id="cust_ext_001",
        name="Alex Smith",
        email="alex@example.com",
        total_payments=10,
        successful_payments=8,
        failed_payments=2,
        historical_success_rate=0.8,
        created_at=now,
    )

    payment_ctx = PaymentContext(
        payment_id=payment_id,
        external_payment_id="pay_ext_001",
        amount=2500.0,
        currency="INR",
        payment_method="card",
        status="failed",
        failure_reason="insufficient_funds",
        created_at=now,
    )

    opportunity_ctx = RecoveryOpportunityContext(
        opportunity_id=opportunity_id,
        status="open",
        revenue_at_risk=2500.0,
        expected_recovery=2000.0,
        recommended_action="smart_retry_card",
        confidence=0.85,
        created_at=now,
    )

    attempt_ctx = RecoveryAttemptContext(
        attempt_id=attempt_id,
        action="smart_retry_card",
        status="failed",
        amount_recovered=0.0,
        error_code="insufficient_funds",
        external_reference="rec_1001",
        created_at=now,
    )

    hist_payment_ctx = HistoricalPaymentContext(
        payment_id=hist_payment_id,
        external_payment_id="pay_hist_001",
        amount=1500.0,
        currency="INR",
        payment_method="upi",
        status="succeeded",
        failure_reason=None,
        created_at=now,
        was_recovered=True,
        recovery_action="customer_prompt_upi",
        recovery_attempts_count=1,
    )

    stats_ctx = CustomerRecoveryStatsContext(
        total_recovery_opportunities=2,
        recovered_opportunities=1,
        failed_opportunities=1,
        recovery_rate=0.5,
        previously_successful_actions=["customer_prompt_upi"],
        previously_failed_actions=["smart_retry_card"],
        total_amount_recovered=1500.0,
    )

    context = CustomerRecoveryContext(
        customer=customer_ctx,
        current_payment=payment_ctx,
        current_opportunity=opportunity_ctx,
        current_payment_attempts=[attempt_ctx],
        historical_payments=[hist_payment_ctx],
        recovery_statistics=stats_ctx,
        retrieved_at=now,
    )

    assert context.customer.customer_id == customer_id
    assert context.current_payment.payment_id == payment_id
    assert context.current_opportunity.opportunity_id == opportunity_id
    assert len(context.current_payment_attempts) == 1
    assert context.current_payment_attempts[0].attempt_id == attempt_id
    assert len(context.historical_payments) == 1
    assert context.historical_payments[0].was_recovered is True
    assert context.recovery_statistics.recovery_rate == 0.5
    assert "customer_prompt_upi" in context.recovery_statistics.previously_successful_actions


def test_cold_start_new_customer_context():
    """Verify clean instantiation for a first-time customer with zero history."""
    customer_id = uuid.uuid4()
    payment_id = uuid.uuid4()
    opportunity_id = uuid.uuid4()

    customer_ctx = CustomerContext(
        customer_id=customer_id,
        external_customer_id="cust_new_101",
        name="Brand New Customer",
        email="new@example.com",
        total_payments=0,
        successful_payments=0,
        failed_payments=0,
        historical_success_rate=0.0,
    )

    payment_ctx = PaymentContext(
        payment_id=payment_id,
        amount=999.0,
        currency="INR",
        payment_method="upi",
        status="failed",
        failure_reason="timeout",
    )

    opportunity_ctx = RecoveryOpportunityContext(
        opportunity_id=opportunity_id,
        status="open",
        revenue_at_risk=999.0,
        expected_recovery=0.0,
    )

    context = CustomerRecoveryContext(
        customer=customer_ctx,
        current_payment=payment_ctx,
        current_opportunity=opportunity_ctx,
        # Omitted lists and stats default cleanly
    )

    assert context.customer.total_payments == 0
    assert context.customer.historical_success_rate == 0.0
    assert context.current_payment_attempts == []
    assert context.historical_payments == []
    assert context.recovery_statistics.total_recovery_opportunities == 0
    assert context.recovery_statistics.recovery_rate == 0.0
    assert context.recovery_statistics.previously_successful_actions == []
    assert context.recovery_statistics.total_amount_recovered == 0.0
    assert context.retrieved_at is not None


def test_customer_only_inspection_context():
    """Verify valid context creation when inspecting a customer profile without an active failed payment."""
    customer_id = uuid.uuid4()
    customer_ctx = CustomerContext(
        customer_id=customer_id,
        name="Standalone Customer",
        email="standalone@example.com",
        total_payments=5,
        successful_payments=5,
        failed_payments=0,
        historical_success_rate=1.0,
    )

    context = CustomerRecoveryContext(customer=customer_ctx)

    assert context.customer.customer_id == customer_id
    assert context.current_payment is None
    assert context.current_opportunity is None
    assert context.current_payment_attempts == []
    assert context.historical_payments == []


def test_schema_immutability():
    """Verify all context models are frozen / immutable to prevent accidental mutation."""
    customer_ctx = CustomerContext(
        customer_id=uuid.uuid4(),
        name="Frozen Test",
        email="frozen@example.com",
    )

    with pytest.raises(ValidationError):
        # Attempting to mutate a field on a frozen Pydantic model must raise ValidationError
        customer_ctx.name = "Mutated Name"  # type: ignore

    context = CustomerRecoveryContext(customer=customer_ctx)
    with pytest.raises(ValidationError):
        context.current_payment = None  # type: ignore


def test_invalid_data_rejection():
    """Verify validation constraints reject invalid counts, amounts, and out-of-range rates."""
    # Negative payment count
    with pytest.raises(ValidationError):
        CustomerContext(
            customer_id=uuid.uuid4(),
            total_payments=-1,
        )

    # Success rate > 1.0
    with pytest.raises(ValidationError):
        CustomerContext(
            customer_id=uuid.uuid4(),
            historical_success_rate=1.5,
        )

    # Negative payment amount
    with pytest.raises(ValidationError):
        PaymentContext(
            payment_id=uuid.uuid4(),
            amount=-100.0,
            payment_method="card",
            status="failed",
        )

    # Recovery rate < 0.0
    with pytest.raises(ValidationError):
        CustomerRecoveryStatsContext(
            recovery_rate=-0.1,
        )
