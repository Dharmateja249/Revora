"""
Deterministic Database Context Retrieval Module for Revora.

Retrieves and aggregates structured historical context from the relational database
for failed payment recovery evaluation, returning an immutable CustomerRecoveryContext.
"""

import uuid

from sqlalchemy import or_, select
from sqlalchemy.orm import Session, selectinload

from app.context import (
    CustomerContext,
    CustomerNotFoundError,
    CustomerRecoveryContext,
    CustomerRecoveryStatsContext,
    HistoricalPaymentContext,
    PaymentContext,
    PaymentCustomerMismatchError,
    PaymentNotFoundError,
    RecoveryAttemptContext,
    RecoveryOpportunityContext,
    RecoveryOpportunityNotFoundError,
)
from app.models import Customer, Payment, RecoveryOpportunity


def _resolve_customer(
    db_session: Session,
    customer_id: uuid.UUID | str,
) -> Customer:
    """
    Resolve a Customer model by internal UUID or external_customer_id.

    Raises:
        CustomerNotFoundError: If no customer matches the identifier.
    """
    if isinstance(customer_id, uuid.UUID):
        stmt = select(Customer).where(Customer.id == customer_id)
    else:
        # Check if the string is a valid UUID representation
        try:
            parsed_uuid = uuid.UUID(str(customer_id))
            stmt = select(Customer).where(
                or_(
                    Customer.id == parsed_uuid,
                    Customer.external_customer_id == str(customer_id),
                )
            )
        except ValueError:
            stmt = select(Customer).where(
                Customer.external_customer_id == str(customer_id)
            )

    customer = db_session.execute(stmt).scalars().first()
    if customer is None:
        raise CustomerNotFoundError(customer_id)
    return customer


def _resolve_payment(
    db_session: Session,
    payment_id: uuid.UUID | str,
) -> Payment:
    """
    Resolve a Payment model by internal UUID or external_payment_id,
    eagerly loading its recovery opportunity and attempts.

    Raises:
        PaymentNotFoundError: If no payment matches the identifier.
    """
    options = [
        selectinload(Payment.recovery_opportunity).selectinload(
            RecoveryOpportunity.attempts
        )
    ]

    if isinstance(payment_id, uuid.UUID):
        stmt = select(Payment).options(*options).where(Payment.id == payment_id)
    else:
        try:
            parsed_uuid = uuid.UUID(str(payment_id))
            stmt = (
                select(Payment)
                .options(*options)
                .where(
                    or_(
                        Payment.id == parsed_uuid,
                        Payment.external_payment_id == str(payment_id),
                    )
                )
            )
        except ValueError:
            stmt = (
                select(Payment)
                .options(*options)
                .where(Payment.external_payment_id == str(payment_id))
            )

    payment = db_session.execute(stmt).scalars().first()
    if payment is None:
        raise PaymentNotFoundError(payment_id)
    return payment


def get_customer_context(
    db_session: Session,
    customer_id: uuid.UUID | str,
    payment_id: uuid.UUID | str,
    history_limit: int | None = None,
) -> CustomerRecoveryContext:
    """
    Retrieve and assemble the complete deterministic context for a customer's failed payment.

    Args:
        db_session: Active SQLAlchemy database session.
        customer_id: Internal UUID or external string identifier for the customer.
        payment_id: Internal UUID or external string identifier for the failed payment.
        history_limit: Optional maximum number of historical payments to retrieve (default: all).

    Returns:
        CustomerRecoveryContext: An immutable snapshot containing customer facts,
        payment details, active opportunity, current attempt history, and prior aggregates.

    Raises:
        CustomerNotFoundError: If the customer does not exist.
        PaymentNotFoundError: If the payment does not exist.
        PaymentCustomerMismatchError: If the payment does not belong to the customer.
        RecoveryOpportunityNotFoundError: If the payment has no associated recovery opportunity.
    """
    # 1. Resolve Customer
    customer = _resolve_customer(db_session, customer_id)

    # 2. Resolve Payment
    payment = _resolve_payment(db_session, payment_id)

    # 3. Verify Customer-Payment Ownership Isolation
    if payment.customer_id != customer.id:
        raise PaymentCustomerMismatchError(
            payment_id=payment_id,
            customer_id=customer_id,
            actual_customer_id=payment.customer_id,
        )

    # 4. Resolve Recovery Opportunity for Current Payment
    opportunity = payment.recovery_opportunity
    if opportunity is None:
        raise RecoveryOpportunityNotFoundError(payment_id)

    # 5. Retrieve and Sort Current Recovery Attempts Deterministically
    raw_current_attempts = list(opportunity.attempts or [])
    raw_current_attempts.sort(
        key=lambda a: (
            a.created_at.timestamp() if a.created_at else 0.0,
            str(a.id),
        )
    )

    current_payment_attempts = [
        RecoveryAttemptContext(
            attempt_id=att.id,
            action=att.action,
            status=att.status,
            amount_recovered=att.amount_recovered,
            error_code=att.error_code,
            external_reference=att.external_reference,
            created_at=att.created_at,
            completed_at=att.completed_at,
        )
        for att in raw_current_attempts
    ]

    # 6. Retrieve Historical Payments for this Customer (Excluding Current Payment)
    hist_stmt = (
        select(Payment)
        .options(
            selectinload(Payment.recovery_opportunity).selectinload(
                RecoveryOpportunity.attempts
            )
        )
        .where(
            Payment.customer_id == customer.id,
            Payment.id != payment.id,
        )
        .order_by(Payment.created_at.desc(), Payment.id.desc())
    )

    if history_limit is not None and history_limit > 0:
        hist_stmt = hist_stmt.limit(history_limit)

    historical_payments_raw = db_session.execute(hist_stmt).scalars().all()

    # 7. Aggregate Historical Recovery Statistics
    total_opportunities = 0
    recovered_opportunities = 0
    failed_opportunities = 0
    total_amount_recovered = 0.0
    successful_actions: set[str] = set()
    failed_actions: set[str] = set()

    historical_payment_contexts = []
    for hist_p in historical_payments_raw:
        hist_opp = hist_p.recovery_opportunity
        was_recovered = False
        recovery_action: str | None = None
        attempts_count = 0

        if hist_opp is not None:
            total_opportunities += 1
            if hist_opp.status == "recovered":
                recovered_opportunities += 1
                was_recovered = True
            elif (
                hist_opp.status in ("failed", "abandoned") or hist_p.status == "failed"
            ):
                failed_opportunities += 1

            if hist_opp.attempts:
                attempts_count = len(hist_opp.attempts)
                for att in hist_opp.attempts:
                    if att.status == "succeeded":
                        successful_actions.add(att.action)
                        total_amount_recovered += att.amount_recovered
                        if recovery_action is None:
                            recovery_action = att.action
                    elif att.status == "failed":
                        failed_actions.add(att.action)

            if recovery_action is None and hist_opp.recommended_action:
                recovery_action = hist_opp.recommended_action
        elif hist_p.status == "succeeded":
            was_recovered = False

        historical_payment_contexts.append(
            HistoricalPaymentContext(
                payment_id=hist_p.id,
                external_payment_id=hist_p.external_payment_id,
                amount=hist_p.amount,
                currency=hist_p.currency,
                payment_method=hist_p.payment_method,
                status=hist_p.status,
                failure_reason=hist_p.failure_reason,
                created_at=hist_p.created_at,
                was_recovered=was_recovered,
                recovery_action=recovery_action,
                recovery_attempts_count=attempts_count,
            )
        )

    recovery_rate = (
        round(recovered_opportunities / total_opportunities, 4)
        if total_opportunities > 0
        else 0.0
    )

    recovery_stats_ctx = CustomerRecoveryStatsContext(
        total_recovery_opportunities=total_opportunities,
        recovered_opportunities=recovered_opportunities,
        failed_opportunities=failed_opportunities,
        recovery_rate=recovery_rate,
        previously_successful_actions=sorted(successful_actions),
        previously_failed_actions=sorted(failed_actions),
        total_amount_recovered=total_amount_recovered,
    )

    # 8. Assemble Customer and Payment Contexts
    total_cust_payments = customer.total_payments
    succ_cust_payments = customer.successful_payments
    hist_success_rate = (
        round(succ_cust_payments / total_cust_payments, 4)
        if total_cust_payments > 0
        else 0.0
    )

    customer_ctx = CustomerContext(
        customer_id=customer.id,
        external_customer_id=customer.external_customer_id,
        name=customer.name,
        email=customer.email,
        total_payments=total_cust_payments,
        successful_payments=succ_cust_payments,
        failed_payments=customer.failed_payments,
        historical_success_rate=hist_success_rate,
        created_at=customer.created_at,
    )

    payment_ctx = PaymentContext(
        payment_id=payment.id,
        external_payment_id=payment.external_payment_id,
        amount=payment.amount,
        currency=payment.currency,
        payment_method=payment.payment_method,
        status=payment.status,
        failure_reason=payment.failure_reason,
        created_at=payment.created_at,
    )

    opportunity_ctx = RecoveryOpportunityContext(
        opportunity_id=opportunity.id,
        status=opportunity.status,
        revenue_at_risk=opportunity.revenue_at_risk,
        expected_recovery=opportunity.expected_recovery,
        recommended_action=opportunity.recommended_action,
        confidence=opportunity.confidence,
        created_at=opportunity.created_at,
    )

    # 9. Return Complete Immutable Context
    return CustomerRecoveryContext(
        customer=customer_ctx,
        current_payment=payment_ctx,
        current_opportunity=opportunity_ctx,
        current_payment_attempts=current_payment_attempts,
        historical_payments=historical_payment_contexts,
        recovery_statistics=recovery_stats_ctx,
    )
