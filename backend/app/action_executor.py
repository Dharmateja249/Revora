"""
Revora Action Executor.

Orchestrates the external execution of policy-approved RecoveryActions.
Ensures that PolicyValidator constraints can never be bypassed and dispatches
supported actions to the RazorpayAdapter while failing safely for unsupported actions.
"""

import logging
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from app.context import CustomerRecoveryContext
from app.decision_engine import RecoveryAction
from app.policies.schemas import RecoveryPolicyContext
from app.razorpay_adapter import RazorpayAdapter, RazorpayError

logger = logging.getLogger("revora.action_executor")


class ActionResult(BaseModel):
    """
    Structured outcome of an external recovery action execution attempt.
    """

    model_config = ConfigDict(frozen=True)

    action: RecoveryAction = Field(
        ...,
        description="Recovery action that was executed or evaluated for execution.",
    )
    attempted: bool = Field(
        ...,
        description="Whether external or simulated execution was attempted.",
    )
    status: str = Field(
        ...,
        description="Execution status: 'success', 'simulated', 'failed', 'prohibited', 'skipped', 'unsupported'.",
    )
    success: bool = Field(
        ...,
        description="True if execution succeeded or was successfully simulated.",
    )
    reference_id: str | None = Field(
        default=None,
        description="External gateway reference ID (e.g., Razorpay 'plink_...' ID).",
    )
    resource_url: str | None = Field(
        default=None,
        description="Public URL to access the external recovery resource (e.g. short_url).",
    )
    message: str = Field(
        ...,
        description="Human-readable summary of the execution outcome.",
    )
    error: str | None = Field(
        default=None,
        description="Sanitized diagnostic error if execution failed.",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional execution telemetry and response data.",
    )


class ActionExecutor:
    """
    Executes policy-approved recovery actions against external payment providers.
    Enforces that PolicyValidator decisions are strictly respected before any execution.
    """

    def __init__(self, razorpay_adapter: RazorpayAdapter | None = None):
        self._razorpay_adapter = razorpay_adapter or RazorpayAdapter()

    @property
    def adapter(self) -> RazorpayAdapter:
        """Return the underlying Razorpay adapter."""
        return self._razorpay_adapter

    async def execute(
        self,
        approved_action: RecoveryAction,
        policy_context: RecoveryPolicyContext,
        context: CustomerRecoveryContext,
        reference_id: str | None = None,
    ) -> ActionResult:
        """
        Execute an approved recovery action through the appropriate provider adapter.

        Args:
            approved_action: RecoveryAction selected by orchestrator/validator.
            policy_context: Policy context defining allowed/prohibited constraints.
            context: Customer and payment recovery context.
            reference_id: Deterministic idempotency reference key for gateway deduplication.

        Returns:
            ActionResult indicating execution success, failure, or safe skip.
        """
        # 1. Authoritative Policy Verification (Never bypass PolicyValidator)
        if (
            approved_action not in policy_context.allowed_actions
            or approved_action in policy_context.prohibited_actions
        ):
            logger.error(
                "Execution rejected: Action '%s' is prohibited or not allowed by active policy.",
                approved_action,
            )
            return ActionResult(
                action=approved_action,
                attempted=False,
                status="prohibited",
                success=False,
                error="Action violates PolicyValidator constraints and cannot be executed.",
                message="Execution rejected: policy compliance violation.",
            )

        # 2. Extract payment & customer details safely
        amount = 0.0
        currency = "INR"
        payment_method = "unknown"
        customer_name: str | None = None
        customer_email: str | None = None

        if context.current_payment is not None:
            amount = context.current_payment.amount
            currency = context.current_payment.currency or "INR"
            payment_method = context.current_payment.payment_method or "unknown"

        if context.customer is not None:
            customer_name = context.customer.name
            customer_email = context.customer.email

        # 3. Dispatch to Supported Action Handlers
        if approved_action in (
            RecoveryAction.PAYMENT_LINK,
            RecoveryAction.CHANGE_PAYMENT_METHOD,
        ):
            if amount <= 0.0:
                logger.warning(
                    "Execution rejected: Cannot create payment link for non-positive amount %s",
                    amount,
                )
                return ActionResult(
                    action=approved_action,
                    attempted=False,
                    status="failed",
                    success=False,
                    error=f"Invalid recovery payment amount: {amount}. Amount must be positive.",
                    message="Payment link creation rejected due to invalid or non-positive amount.",
                )

            if approved_action == RecoveryAction.PAYMENT_LINK:
                return await self._execute_payment_link(
                    amount=amount,
                    currency=currency,
                    customer_name=customer_name,
                    customer_email=customer_email,
                    description=f"Payment recovery for failed {payment_method.upper()} payment",
                    reference_id=reference_id,
                )

            return await self._execute_payment_link(
                amount=amount,
                currency=currency,
                customer_name=customer_name,
                customer_email=customer_email,
                description="Complete payment by selecting a different payment method",
                is_method_change=True,
                reference_id=reference_id,
            )

        if approved_action == RecoveryAction.RETRY_PAYMENT:
            return self._execute_retry_payment(
                amount=amount,
                currency=currency,
                payment_method=payment_method,
            )

        if approved_action == RecoveryAction.WAIT_AND_RETRY:
            return ActionResult(
                action=approved_action,
                attempted=False,
                status="skipped",
                success=True,
                message="Action scheduled for delayed retry after cooling period; no immediate external call.",
            )

        if approved_action == RecoveryAction.NO_ACTION:
            return ActionResult(
                action=approved_action,
                attempted=False,
                status="skipped",
                success=True,
                message="No external recovery action required; opportunity marked terminal.",
            )

        # Unsupported Action Safety Net
        return ActionResult(
            action=approved_action,
            attempted=False,
            status="unsupported",
            success=False,
            error=f"RecoveryAction '{approved_action}' is not supported for automated execution.",
            message="Unsupported recovery action.",
        )

    async def _execute_payment_link(
        self,
        amount: float,
        currency: str,
        customer_name: str | None,
        customer_email: str | None,
        description: str,
        is_method_change: bool = False,
        reference_id: str | None = None,
    ) -> ActionResult:
        """Create an interactive Razorpay Payment Link."""
        target_action = (
            RecoveryAction.CHANGE_PAYMENT_METHOD
            if is_method_change
            else RecoveryAction.PAYMENT_LINK
        )
        try:
            res = await self._razorpay_adapter.create_payment_link(
                amount=amount,
                currency=currency,
                description=description,
                customer_name=customer_name,
                customer_email=customer_email,
                reference_id=reference_id,
            )
            is_sim = res.get("simulated", False)
            status_str = "simulated" if is_sim else "success"
            short_url = res.get("short_url")

            return ActionResult(
                action=target_action,
                attempted=True,
                status=status_str,
                success=True,
                reference_id=res.get("id"),
                resource_url=short_url,
                message=f"Razorpay Payment Link generated successfully: {short_url}",
                metadata=res,
            )
        except RazorpayError as exc:
            logger.warning("Razorpay payment link execution failed: %s", exc)
            return ActionResult(
                action=target_action,
                attempted=True,
                status="failed",
                success=False,
                error=str(exc),
                message="Razorpay payment link creation failed.",
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Unexpected error during payment link creation: %s", exc)
            return ActionResult(
                action=target_action,
                attempted=True,
                status="failed",
                success=False,
                error=str(exc),
                message="Unexpected error during payment link creation.",
            )

    def _execute_retry_payment(
        self,
        amount: float,
        currency: str,
        payment_method: str,
    ) -> ActionResult:
        """
        Execute or simulate an automated payment retry.

        In Razorpay, automated server-to-server retries without customer presence
        require an authorized recurring mandate or subscription token.
        In dry-run mode, simulated retry reference is generated.
        """
        if self._razorpay_adapter.dry_run:
            sim_ref = f"retry_sim_{uuid4().hex[:10]}"
            logger.info("Simulated payment retry: ref=%s amount=%s", sim_ref, amount)
            return ActionResult(
                action=RecoveryAction.RETRY_PAYMENT,
                attempted=True,
                status="simulated",
                success=True,
                reference_id=sim_ref,
                message="Automated payment retry simulated successfully in dry-run mode.",
                metadata={"simulated": True, "amount": amount, "currency": currency},
            )

        # In live mode without recurring mandate:
        return ActionResult(
            action=RecoveryAction.RETRY_PAYMENT,
            attempted=False,
            status="requires_customer_presence",
            success=False,
            error="Automated one-time payment retry requires customer-authorized mandate or checkout re-trigger in Razorpay.",
            message="Server-to-server payment retry cannot be executed without customer authentication token.",
        )
