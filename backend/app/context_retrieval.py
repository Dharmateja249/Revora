"""
Deterministic Database Context Retrieval Module for Revora.

Retrieves and aggregates structured historical context from the relational database
for failed payment recovery evaluation, returning an immutable CustomerRecoveryContext.
"""

import uuid
from datetime import timedelta
from typing import Any

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
from app.models import (
    Customer,
    Payment,
    RecoveryAttempt,
    RecoveryOpportunity,
    utc_now,
)

DEMO_CUSTOMER_UUID = uuid.UUID("e9cd4c97-979b-4753-9925-640623f74eee")


def ensure_demo_customer_seeded(db_session: Session) -> Customer:
    """
    Ensure the canonical demo customer and baseline transaction history are seeded in the database.
    Creates 42 payments (40 successful, 2 failed with recovery records) reflecting the canonical
    customer profile so revora.db is the single source of truth.
    """
    demo_cust = db_session.get(Customer, DEMO_CUSTOMER_UUID)
    if demo_cust is not None:
        return demo_cust

    demo_cust = Customer(
        id=DEMO_CUSTOMER_UUID,
        external_customer_id="cust_demo_zepto",
        name="Zepto Enterprise",
        email="finance@zepto.in",
        total_payments=42,
        successful_payments=40,
        failed_payments=2,
    )
    db_session.add(demo_cust)
    db_session.flush()

    base_time = utc_now()

    # 40 successful payments across card and upi
    for i in range(40):
        amount = round(3500.0 + ((i * 370.0) % 8500.0), 2)
        p = Payment(
            id=uuid.uuid4(),
            external_payment_id=f"pay_demo_succ_{i:03d}",
            customer_id=DEMO_CUSTOMER_UUID,
            amount=amount,
            currency="INR",
            payment_method="card" if i % 2 == 0 else "upi",
            status="succeeded",
            failure_reason=None,
            created_at=base_time - timedelta(days=90 - i * 2, hours=i % 12),
        )
        db_session.add(p)

    # Failed payment 1: recovered via wait_and_retry
    p_fail1 = Payment(
        id=uuid.uuid4(),
        external_payment_id="pay_demo_fail_001",
        customer_id=DEMO_CUSTOMER_UUID,
        amount=6500.0,
        currency="INR",
        payment_method="card",
        status="succeeded",  # Recovered
        failure_reason="insufficient_funds",
        created_at=base_time - timedelta(days=12),
    )
    db_session.add(p_fail1)
    db_session.flush()

    opp1 = RecoveryOpportunity(
        id=uuid.uuid4(),
        payment_id=p_fail1.id,
        status="recovered",
        revenue_at_risk=6500.0,
        expected_recovery=6500.0,
        recommended_action="wait_and_retry",
        confidence=0.85,
        created_at=base_time - timedelta(days=12),
    )
    db_session.add(opp1)
    db_session.flush()

    att1 = RecoveryAttempt(
        id=uuid.uuid4(),
        opportunity_id=opp1.id,
        action="wait_and_retry",
        status="succeeded",
        amount_recovered=6500.0,
        external_reference="rec_demo_att_001",
        created_at=base_time - timedelta(days=11, hours=20),
        completed_at=base_time - timedelta(days=11, hours=19),
    )
    db_session.add(att1)

    # Failed payment 2: unrecovered gateway timeout
    p_fail2 = Payment(
        id=uuid.uuid4(),
        external_payment_id="pay_demo_fail_002",
        customer_id=DEMO_CUSTOMER_UUID,
        amount=4200.0,
        currency="INR",
        payment_method="upi",
        status="failed",
        failure_reason="bank_technical_timeout",
        created_at=base_time - timedelta(days=3),
    )
    db_session.add(p_fail2)
    db_session.flush()

    opp2 = RecoveryOpportunity(
        id=uuid.uuid4(),
        payment_id=p_fail2.id,
        status="failed",
        revenue_at_risk=4200.0,
        expected_recovery=0.0,
        recommended_action="retry_payment",
        confidence=0.70,
        created_at=base_time - timedelta(days=3),
    )
    db_session.add(opp2)
    db_session.flush()

    att2 = RecoveryAttempt(
        id=uuid.uuid4(),
        opportunity_id=opp2.id,
        action="retry_payment",
        status="failed",
        amount_recovered=0.0,
        external_reference="rec_demo_att_002",
        error_code="gateway_timeout",
        created_at=base_time - timedelta(days=2, hours=23),
        completed_at=base_time - timedelta(days=2, hours=22),
    )
    db_session.add(att2)

    db_session.commit()
    return demo_cust


def _resolve_customer(
    db_session: Session,
    customer_id: uuid.UUID | str,
) -> Customer:
    """
    Resolve a Customer model by internal UUID or external_customer_id.

    Raises:
        CustomerNotFoundError: If no customer matches the identifier.
    """
    parsed_uuid: uuid.UUID | None = None
    if isinstance(customer_id, uuid.UUID):
        parsed_uuid = customer_id
        stmt = select(Customer).where(Customer.id == customer_id)
    else:
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
        if parsed_uuid == DEMO_CUSTOMER_UUID:
            return ensure_demo_customer_seeded(db_session)
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
    payment_id: uuid.UUID | str | None = None,
    history_limit: int | None = None,
    current_payment_amount: float | None = None,
    current_payment_currency: str = "INR",
    current_payment_method: str | None = None,
    current_payment_failure_reason: str | None = None,
    current_payment_status: str = "failed",
    current_opportunity_status: str = "open",
    current_revenue_at_risk: float | None = None,
    current_attempts: list[RecoveryAttemptContext] | None = None,
) -> CustomerRecoveryContext:
    """
    Retrieve and assemble the complete deterministic context for a customer's failed payment
    directly from the relational database.

    Args:
        db_session: Active SQLAlchemy database session.
        customer_id: Internal UUID or external string identifier for the customer.
        payment_id: Optional internal UUID or external string identifier for the payment.
        history_limit: Optional maximum number of historical payments to retrieve.
        current_payment_amount: Optional amount if payment is newly arriving.
        current_payment_currency: Optional currency string.
        current_payment_method: Optional payment method string.
        current_payment_failure_reason: Optional failure reason string.
        current_payment_status: Payment status ('failed', 'succeeded', etc.).
        current_opportunity_status: Recovery opportunity status ('open', etc.).
        current_revenue_at_risk: Optional revenue at risk.
        current_attempts: Optional list of previous attempt contexts on current payment.

    Returns:
        CustomerRecoveryContext: An immutable snapshot containing customer facts,
        volume aggregates, recent payment behavior, active opportunity, current attempt history,
        and prior recovery statistics.

    Raises:
        CustomerNotFoundError: If the customer does not exist.
        PaymentNotFoundError: If payment_id was explicitly provided but does not exist in DB
                              and no current_payment_amount fallback was supplied.
        PaymentCustomerMismatchError: If the payment does not belong to the customer.
        RecoveryOpportunityNotFoundError: If the payment exists in DB but has no opportunity.
    """
    # 1. Resolve Customer
    customer = _resolve_customer(db_session, customer_id)

    # 2. Resolve Payment & Opportunity
    payment: Payment | None = None
    opportunity: RecoveryOpportunity | None = None
    current_payment_attempts: list[RecoveryAttemptContext] = []

    if payment_id is not None:
        try:
            payment = _resolve_payment(db_session, payment_id)
            if payment.customer_id != customer.id:
                raise PaymentCustomerMismatchError(
                    payment_id=payment_id,
                    customer_id=customer_id,
                    actual_customer_id=payment.customer_id,
                )
            opportunity = payment.recovery_opportunity
            if opportunity is None:
                raise RecoveryOpportunityNotFoundError(payment_id)

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
        except PaymentNotFoundError:
            if current_payment_amount is None:
                raise
            payment = None

    if payment is not None and opportunity is not None:
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
    else:
        # Construct current payment and opportunity context from request parameters
        assigned_payment_id = (
            payment_id if isinstance(payment_id, uuid.UUID) else uuid.uuid4()
        )
        assigned_amount = float(current_payment_amount or 0.0)
        payment_ctx = PaymentContext(
            payment_id=assigned_payment_id,
            amount=assigned_amount,
            currency=current_payment_currency,
            payment_method=current_payment_method or "card",
            status=current_payment_status,
            failure_reason=current_payment_failure_reason,
            created_at=utc_now(),
        )
        assigned_risk = (
            current_revenue_at_risk
            if current_revenue_at_risk is not None
            else assigned_amount
        )
        opportunity_ctx = RecoveryOpportunityContext(
            opportunity_id=uuid.uuid4(),
            status=current_opportunity_status,
            revenue_at_risk=assigned_risk,
            expected_recovery=0.0,
            created_at=utc_now(),
        )
        current_payment_attempts = list(current_attempts or [])

    # 3. Retrieve Historical Payments for this Customer (Excluding Current Payment if persisted)
    filter_conditions = [Payment.customer_id == customer.id]
    if payment is not None:
        filter_conditions.append(Payment.id != payment.id)

    hist_stmt = (
        select(Payment)
        .options(
            selectinload(Payment.recovery_opportunity).selectinload(
                RecoveryOpportunity.attempts
            )
        )
        .where(*filter_conditions)
        .order_by(Payment.created_at.desc(), Payment.id.desc())
    )

    if history_limit is not None and history_limit > 0:
        hist_stmt = hist_stmt.limit(history_limit)

    historical_payments_raw = db_session.execute(hist_stmt).scalars().all()

    # 4. Aggregate Historical Recovery Statistics & Payments
    total_opportunities = 0
    recovered_opportunities = 0
    failed_opportunities = 0
    total_amount_recovered = 0.0
    successful_actions: set[str] = set()
    failed_actions: set[str] = set()

    historical_payment_contexts: list[HistoricalPaymentContext] = []
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

    # 5. Compute Financial Volumes & Recent Payment Behavior
    total_tx_amount = round(sum(float(p.amount) for p in historical_payments_raw), 2)
    succ_tx_amount = round(
        sum(
            float(p.amount) for p in historical_payments_raw if p.status == "succeeded"
        ),
        2,
    )
    if payment is not None:
        total_tx_amount = round(total_tx_amount + float(payment.amount), 2)
        if payment.status == "succeeded":
            succ_tx_amount = round(succ_tx_amount + float(payment.amount), 2)

    total_payment_count = len(historical_payments_raw) + (
        1 if payment is not None else 0
    )
    avg_tx_amount = (
        round(total_tx_amount / total_payment_count, 2)
        if total_payment_count > 0
        else 0.0
    )

    recent_payment_behavior: list[dict[str, Any]] = [
        {
            "amount": float(p.amount),
            "currency": str(p.currency),
            "payment_method": str(p.payment_method),
            "status": str(p.status),
            "failure_reason": str(p.failure_reason) if p.failure_reason else None,
            "was_recovered": bool(p.was_recovered),
        }
        for p in historical_payment_contexts[:5]
    ]

    recovery_stats_ctx = CustomerRecoveryStatsContext(
        total_recovery_opportunities=total_opportunities,
        recovered_opportunities=recovered_opportunities,
        failed_opportunities=failed_opportunities,
        recovery_rate=recovery_rate,
        previously_successful_actions=sorted(successful_actions),
        previously_failed_actions=sorted(failed_actions),
        total_amount_recovered=round(total_amount_recovered, 2),
        total_transaction_amount=total_tx_amount,
        successful_transaction_amount=succ_tx_amount,
        average_transaction_amount=avg_tx_amount,
    )

    # 6. Assemble Customer Context
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
        total_transaction_amount=total_tx_amount,
        successful_transaction_amount=succ_tx_amount,
        average_transaction_amount=avg_tx_amount,
        recent_payment_behavior=recent_payment_behavior,
        created_at=customer.created_at,
    )

    # 7. Return Complete Immutable Context
    return CustomerRecoveryContext(
        customer=customer_ctx,
        current_payment=payment_ctx,
        current_opportunity=opportunity_ctx,
        current_payment_attempts=current_payment_attempts,
        historical_payments=historical_payment_contexts,
        recovery_statistics=recovery_stats_ctx,
    )
