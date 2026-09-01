"""
Revora Synthetic Recovery Dataset Fixture (100 Scenarios).

Provides 100 rich, independent synthetic failed payment scenarios for batch outcome simulation,
covering soft failures, hard declines, expired instruments, network timeouts, routing defects,
stopping rule edge cases, and high-value customer interventions.
"""

from datetime import datetime, timezone
from uuid import UUID, uuid5

from app.context import (
    CustomerContext,
    CustomerRecoveryContext,
    CustomerRecoveryStatsContext,
    PaymentContext,
    RecoveryAttemptContext,
    RecoveryOpportunityContext,
)
from app.decision_engine import RecoveryAction
from app.evaluation.recovery_schemas import RecoveryScenario

_DATASET_NAMESPACE = UUID("c8e6a1b2-3f4d-5e6a-7b8c-9d0e1f2a3b4c")


def _gen_uuid(name: str) -> UUID:
    return uuid5(_DATASET_NAMESPACE, name)


def _build_context(
    idx: int,
    amount: float,
    currency: str,
    failure_reason: str,
    attempt_count: int,
    last_action: str | None = None,
    risk_score: float = 0.05,
    is_vip: bool = False,
) -> CustomerRecoveryContext:
    cust_id = _gen_uuid(f"cust_{idx}")
    pay_id = _gen_uuid(f"pay_{idx}")
    opp_id = _gen_uuid(f"opp_{idx}")
    dt = datetime(2026, 3, 1, 10, 0, 0, tzinfo=timezone.utc)

    customer = CustomerContext(
        customer_id=cust_id,
        external_customer_id=f"ext_cust_{idx}",
        name=f"Customer {idx}",
        email=f"user_{idx}@example.com",
        total_payments=5 if not is_vip else 20,
        successful_payments=3 if not is_vip else 18,
        failed_payments=2,
        historical_success_rate=0.6 if not is_vip else 0.9,
        created_at=dt,
    )

    payment = PaymentContext(
        payment_id=pay_id,
        external_payment_id=f"ext_pay_{idx}",
        amount=amount,
        currency=currency,
        payment_method="card",
        status="failed",
        failure_reason=failure_reason,
        created_at=dt,
    )

    opp = RecoveryOpportunityContext(
        opportunity_id=opp_id,
        status="active",
        revenue_at_risk=amount,
        expected_recovery=amount,
        created_at=dt,
    )

    attempts = [
        RecoveryAttemptContext(
            attempt_id=_gen_uuid(f"attempt_{idx}_{a}"),
            action=last_action or RecoveryAction.RETRY_PAYMENT.value,
            status="failed",
            amount_recovered=0.0,
            created_at=dt,
        )
        for a in range(1, attempt_count)
    ]

    stats = CustomerRecoveryStatsContext(
        total_recovery_opportunities=3 if not is_vip else 10,
        recovered_opportunities=2 if not is_vip else 9,
        failed_opportunities=1,
        recovery_rate=0.67 if not is_vip else 0.90,
        previously_successful_actions=[RecoveryAction.RETRY_PAYMENT.value],
        previously_failed_actions=[last_action] if last_action else [],
        total_amount_recovered=amount * 2.0,
    )

    return CustomerRecoveryContext(
        customer=customer,
        current_payment=payment,
        current_opportunity=opp,
        current_payment_attempts=attempts,
        historical_payments=[],
        recovery_statistics=stats,
        retrieved_at=dt,
    )


def get_synthetic_recovery_dataset() -> list[RecoveryScenario]:
    """Generate the authoritative 100-scenario synthetic recovery evaluation dataset."""
    scenarios: list[RecoveryScenario] = []

    # 1. Technical / Network / Gateway Timeouts (Scenarios 1-25)
    # Recoverable via RETRY_PAYMENT or WAIT_AND_RETRY
    for i in range(1, 26):
        amt = float(1500.0 + (i * 250.0))
        ctx = _build_context(
            idx=i,
            amount=amt,
            currency="INR",
            failure_reason="network_timeout",
            attempt_count=1,
            risk_score=0.02,
        )
        scenarios.append(
            RecoveryScenario(
                scenario_id=f"REC-NET-{i:03d}",
                context=ctx,
                payment_amount=amt,
                failure_category="network_timeout",
                is_recoverable=True,
                expected_recoverable_amount=amt,
                max_allowed_attempts=4,
                current_attempt_count=1,
                allowed_actions=(
                    RecoveryAction.RETRY_PAYMENT,
                    RecoveryAction.WAIT_AND_RETRY,
                ),
                prohibited_actions=(RecoveryAction.CHANGE_PAYMENT_METHOD,),
                effective_policy_ids=("POL-TIMEOUT-RETRY",),
                success_action_rates={
                    RecoveryAction.RETRY_PAYMENT.value: 1.0,
                    RecoveryAction.WAIT_AND_RETRY.value: 0.95,
                },
                description=f"Temporary bank network timeout on ₹{amt:,.2f} payment.",
            )
        )

    # 2. Insufficient Funds (Scenarios 26-45)
    # Recoverable via RETRY_PAYMENT, WAIT_AND_RETRY, or PAYMENT_LINK
    for i in range(26, 46):
        amt = float(2500.0 + (i * 300.0))
        ctx = _build_context(
            idx=i,
            amount=amt,
            currency="INR",
            failure_reason="insufficient_funds",
            attempt_count=1,
            risk_score=0.05,
        )
        scenarios.append(
            RecoveryScenario(
                scenario_id=f"REC-NSF-{i:03d}",
                context=ctx,
                payment_amount=amt,
                failure_category="insufficient_funds",
                is_recoverable=True,
                expected_recoverable_amount=amt,
                max_allowed_attempts=4,
                current_attempt_count=1,
                allowed_actions=(
                    RecoveryAction.RETRY_PAYMENT,
                    RecoveryAction.WAIT_AND_RETRY,
                    RecoveryAction.PAYMENT_LINK,
                ),
                prohibited_actions=(),
                effective_policy_ids=("POL-NSF-RECOVERY",),
                success_action_rates={
                    RecoveryAction.RETRY_PAYMENT.value: 0.85,
                    RecoveryAction.WAIT_AND_RETRY.value: 0.85,
                    RecoveryAction.PAYMENT_LINK.value: 0.90,
                },
                description=f"Soft decline due to insufficient funds on ₹{amt:,.2f}.",
            )
        )

    # 3. Card Expired / Invalid Instrument Details (Scenarios 46-60)
    # Recoverable via PAYMENT_LINK or CHANGE_PAYMENT_METHOD, direct RETRY is prohibited
    for i in range(46, 61):
        amt = float(3000.0 + (i * 200.0))
        ctx = _build_context(
            idx=i,
            amount=amt,
            currency="INR",
            failure_reason="expired_card",
            attempt_count=1,
            risk_score=0.03,
        )
        scenarios.append(
            RecoveryScenario(
                scenario_id=f"REC-EXP-{i:03d}",
                context=ctx,
                payment_amount=amt,
                failure_category="expired_card",
                is_recoverable=True,
                expected_recoverable_amount=amt,
                max_allowed_attempts=4,
                current_attempt_count=1,
                allowed_actions=(
                    RecoveryAction.PAYMENT_LINK,
                    RecoveryAction.CHANGE_PAYMENT_METHOD,
                ),
                prohibited_actions=(
                    RecoveryAction.RETRY_PAYMENT,
                    RecoveryAction.WAIT_AND_RETRY,
                ),
                effective_policy_ids=("POL-EXPIRED-NO-RETRY",),
                success_action_rates={
                    RecoveryAction.PAYMENT_LINK.value: 0.95,
                    RecoveryAction.CHANGE_PAYMENT_METHOD.value: 0.90,
                },
                description=f"Expired card requires update payment link for ₹{amt:,.2f}.",
            )
        )

    # 4. Fraud / Stolen Card / Hard Decline (Scenarios 61-75)
    # Permanently NON-RECOVERABLE, retries prohibited, NO_ACTION expected
    for i in range(61, 76):
        amt = float(5000.0 + (i * 500.0))
        ctx = _build_context(
            idx=i,
            amount=amt,
            currency="INR",
            failure_reason="stolen_card",
            attempt_count=1,
            risk_score=0.95,
        )
        scenarios.append(
            RecoveryScenario(
                scenario_id=f"REC-FRD-{i:03d}",
                context=ctx,
                payment_amount=amt,
                failure_category="fraud_hard_decline",
                is_recoverable=False,
                expected_recoverable_amount=0.0,
                max_allowed_attempts=1,
                current_attempt_count=1,
                allowed_actions=(RecoveryAction.NO_ACTION,),
                prohibited_actions=(
                    RecoveryAction.RETRY_PAYMENT,
                    RecoveryAction.WAIT_AND_RETRY,
                    RecoveryAction.PAYMENT_LINK,
                    RecoveryAction.CHANGE_PAYMENT_METHOD,
                ),
                effective_policy_ids=("POL-FRAUD-PROHIBITED",),
                success_action_rates={},
                description=f"Fraudulent/stolen card hard decline for ₹{amt:,.2f}.",
            )
        )

    # 5. Processor / Gateway Routing Defects (Scenarios 76-85)
    # Recoverable via WAIT_AND_RETRY
    for i in range(76, 86):
        amt = float(4000.0 + (i * 400.0))
        ctx = _build_context(
            idx=i,
            amount=amt,
            currency="INR",
            failure_reason="gateway_rejected",
            attempt_count=1,
            risk_score=0.04,
        )
        scenarios.append(
            RecoveryScenario(
                scenario_id=f"REC-RTE-{i:03d}",
                context=ctx,
                payment_amount=amt,
                failure_category="gateway_routing",
                is_recoverable=True,
                expected_recoverable_amount=amt,
                max_allowed_attempts=4,
                current_attempt_count=1,
                allowed_actions=(
                    RecoveryAction.WAIT_AND_RETRY,
                    RecoveryAction.RETRY_PAYMENT,
                ),
                prohibited_actions=(RecoveryAction.CHANGE_PAYMENT_METHOD,),
                effective_policy_ids=("POL-GATEWAY-ROUTING",),
                success_action_rates={
                    RecoveryAction.WAIT_AND_RETRY.value: 1.0,
                    RecoveryAction.RETRY_PAYMENT.value: 0.50,
                },
                description=f"Gateway routing rejection on ₹{amt:,.2f}.",
            )
        )

    # 6. Max Attempts Reached / Terminal Stopping Rule (Scenarios 86-95)
    # Non-recoverable because attempts exhausted. NO_ACTION or PAYMENT_LINK only.
    for i in range(86, 96):
        amt = float(2000.0 + (i * 150.0))
        ctx = _build_context(
            idx=i,
            amount=amt,
            currency="INR",
            failure_reason="insufficient_funds",
            attempt_count=4,  # Current is 4th, max is 4
            last_action="retry_payment",
            risk_score=0.10,
        )
        scenarios.append(
            RecoveryScenario(
                scenario_id=f"REC-STP-{i:03d}",
                context=ctx,
                payment_amount=amt,
                failure_category="max_attempts_exhausted",
                is_recoverable=False,
                expected_recoverable_amount=0.0,
                max_allowed_attempts=4,
                current_attempt_count=4,
                allowed_actions=(
                    RecoveryAction.NO_ACTION,
                    RecoveryAction.PAYMENT_LINK,
                ),
                prohibited_actions=(
                    RecoveryAction.RETRY_PAYMENT,
                    RecoveryAction.WAIT_AND_RETRY,
                ),
                effective_policy_ids=("POL-STOPPING-RULE-EXHAUSTED",),
                success_action_rates={RecoveryAction.PAYMENT_LINK.value: 0.40},
                description=f"Maximum retry attempts (4/4) reached for ₹{amt:,.2f}.",
            )
        )

    # 7. High-Value VIP Customer Disputes / Outreach (Scenarios 96-100)
    # High payment amounts (₹1,00,000 - ₹2,00,000) recoverable via PAYMENT_LINK / CHANGE_PAYMENT_METHOD
    for i in range(96, 101):
        amt = float(75000.0 + ((i - 95) * 25000.0))
        ctx = _build_context(
            idx=i,
            amount=amt,
            currency="INR",
            failure_reason="customer_dispute_risk",
            attempt_count=1,
            risk_score=0.01,
            is_vip=True,
        )
        scenarios.append(
            RecoveryScenario(
                scenario_id=f"REC-VIP-{i:03d}",
                context=ctx,
                payment_amount=amt,
                failure_category="high_value_vip",
                is_recoverable=True,
                expected_recoverable_amount=amt,
                max_allowed_attempts=3,
                current_attempt_count=1,
                allowed_actions=(
                    RecoveryAction.PAYMENT_LINK,
                    RecoveryAction.CHANGE_PAYMENT_METHOD,
                ),
                prohibited_actions=(
                    RecoveryAction.RETRY_PAYMENT,
                    RecoveryAction.WAIT_AND_RETRY,
                ),
                effective_policy_ids=("POL-VIP-OUTREACH",),
                success_action_rates={
                    RecoveryAction.PAYMENT_LINK.value: 0.90,
                    RecoveryAction.CHANGE_PAYMENT_METHOD.value: 0.95,
                },
                description=f"High-value VIP payment ₹{amt:,.2f} requiring careful intervention.",
            )
        )

    return scenarios
