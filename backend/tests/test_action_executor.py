"""
Unit Tests for Revora RazorpayAdapter and ActionExecutor.

Verifies:
1. RazorpayAdapter dry-run creates simulated payment links without outbound network calls.
2. RazorpayAdapter live execution requires credentials and makes authenticated requests.
3. RazorpayAdapter maps HTTP and network errors to sanitized RazorpayAPIError without leaking secrets.
4. ActionExecutor executes approved PAYMENT_LINK and CHANGE_PAYMENT_METHOD actions.
5. Invariant: ActionExecutor NEVER bypasses PolicyValidator (prohibited actions are rejected).
6. ActionExecutor fails safely on unsupported or unexecutable actions.
7. ActionExecutor gracefully catches and wraps adapter failures.
8. WAIT_AND_RETRY and NO_ACTION are safely marked skipped without external gateway calls.
"""

import re
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import httpx
import pytest
from app.action_executor import ActionExecutor, ActionResult
from app.context import (
    CustomerContext,
    CustomerRecoveryContext,
    PaymentContext,
    RecoveryOpportunityContext,
)
from app.decision_engine import RecoveryAction
from app.policies.schemas import RecoveryPolicyContext
from app.razorpay_adapter import (
    RazorpayAdapter,
    RazorpayAPIError,
    RazorpayConfigurationError,
)

# ============================================================================
# Helpers & Fixtures
# ============================================================================


@pytest.fixture
def sample_customer_context() -> CustomerRecoveryContext:
    """Fixture providing a standard CustomerRecoveryContext."""
    cust_id = uuid4()
    pay_id = uuid4()
    opp_id = uuid4()

    customer = CustomerContext(
        customer_id=cust_id,
        name="Aarav Sharma",
        email="aarav.sharma@example.com",
        total_payments=5,
        successful_payments=4,
        failed_payments=1,
        historical_success_rate=0.8,
    )
    payment = PaymentContext(
        payment_id=pay_id,
        amount=2500.0,
        currency="INR",
        payment_method="upi",
        status="failed",
        failure_reason="bank_technical_timeout",
    )
    opportunity = RecoveryOpportunityContext(
        opportunity_id=opp_id,
        status="open",
        revenue_at_risk=2500.0,
    )
    return CustomerRecoveryContext(
        customer=customer,
        current_payment=payment,
        current_opportunity=opportunity,
    )


@pytest.fixture
def permissive_policy_context() -> RecoveryPolicyContext:
    """Policy context permitting all standard actions."""
    return RecoveryPolicyContext(
        provider="razorpay",
        policy_version="2026.1",
        applicable_rules=(),
        allowed_actions=(
            RecoveryAction.PAYMENT_LINK,
            RecoveryAction.RETRY_PAYMENT,
            RecoveryAction.CHANGE_PAYMENT_METHOD,
            RecoveryAction.WAIT_AND_RETRY,
            RecoveryAction.NO_ACTION,
        ),
        prohibited_actions=(),
        mandatory_fallback_action=None,
    )


@pytest.fixture
def restrictive_policy_context() -> RecoveryPolicyContext:
    """Policy context prohibiting RETRY_PAYMENT and only permitting PAYMENT_LINK."""
    return RecoveryPolicyContext(
        provider="razorpay",
        policy_version="2026.1",
        applicable_rules=(),
        allowed_actions=(RecoveryAction.PAYMENT_LINK, RecoveryAction.NO_ACTION),
        prohibited_actions=(RecoveryAction.RETRY_PAYMENT,),
        mandatory_fallback_action=RecoveryAction.PAYMENT_LINK,
    )


# ============================================================================
# 1. RazorpayAdapter Tests
# ============================================================================


@pytest.mark.anyio
async def test_razorpay_adapter_dry_run_simulation():
    """Verify dry_run mode generates simulated payment link with valid schema without HTTP calls."""
    adapter = RazorpayAdapter(dry_run=True)
    assert adapter.dry_run is True

    result = await adapter.create_payment_link(
        amount=1500.0,
        currency="INR",
        description="Demo recovery link",
        customer_name="Rohan Gupta",
        customer_email="rohan@example.com",
    )

    assert result["simulated"] is True
    assert result["status"] == "created"
    assert result["id"].startswith("plink_sim_")
    assert result["short_url"].startswith("https://rzp.io/i/sim_")
    assert result["amount"] == 150000  # 1500 INR in paise
    assert result["currency"] == "INR"
    assert result["customer"]["name"] == "Rohan Gupta"


@pytest.mark.anyio
async def test_razorpay_adapter_live_missing_credentials_raises_error():
    """Verify live mode without credentials raises RazorpayConfigurationError."""
    adapter = RazorpayAdapter(key_id=None, key_secret=None, dry_run=False)

    with pytest.raises(
        RazorpayConfigurationError,
        match=re.escape(
            "Razorpay API credentials ('RAZORPAY_KEY_ID' and 'RAZORPAY_KEY_SECRET') must be configured for live payment link creation."
        ),
    ):
        await adapter.create_payment_link(amount=500.0)


@pytest.mark.anyio
async def test_razorpay_adapter_live_success_with_injected_client():
    """Verify live mode makes authenticated HTTP Basic Auth call and parses response."""
    mock_client = AsyncMock(spec=httpx.AsyncClient)
    mock_response = MagicMock(spec=httpx.Response)
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "id": "plink_test_live_98765",
        "short_url": "https://rzp.io/i/test_live_98765",
        "status": "created",
        "amount": 250000,
        "currency": "INR",
        "description": "Live recovery link",
    }
    mock_client.post.return_value = mock_response

    adapter = RazorpayAdapter(
        key_id="rzp_test_key123",
        key_secret="rzp_test_sec456",
        dry_run=False,
        client=mock_client,
    )

    res = await adapter.create_payment_link(
        amount=2500.0,
        currency="INR",
        description="Live recovery link",
    )

    assert res["id"] == "plink_test_live_98765"
    assert res["short_url"] == "https://rzp.io/i/test_live_98765"
    assert res["simulated"] is False

    # Verify basic auth and endpoint
    mock_client.post.assert_called_once()
    call_args, call_kwargs = mock_client.post.call_args
    assert "payment_links/" in call_args[0]
    assert call_kwargs["auth"] == ("rzp_test_key123", "rzp_test_sec456")
    assert call_kwargs["json"]["amount"] == 250000


@pytest.mark.anyio
async def test_razorpay_adapter_live_http_error_handled_cleanly():
    """Verify HTTP 400 from Razorpay is mapped to RazorpayAPIError without leaking secrets."""
    mock_client = AsyncMock(spec=httpx.AsyncClient)
    mock_response = MagicMock(spec=httpx.Response)
    mock_response.status_code = 400
    mock_response.json.return_value = {
        "error": {
            "code": "BAD_REQUEST_ERROR",
            "description": "Amount must be at least ₹1.00",
        }
    }
    mock_client.post.return_value = mock_response

    adapter = RazorpayAdapter(
        key_id="rzp_test_key123",
        key_secret="rzp_test_sec456",
        dry_run=False,
        client=mock_client,
    )

    with pytest.raises(
        RazorpayAPIError, match=re.escape("Amount must be at least ₹1.00")
    ):
        await adapter.create_payment_link(amount=0.50)


@pytest.mark.anyio
async def test_razorpay_adapter_live_invalid_json_raises_razorpay_api_error():
    """Verify that a 2xx response with invalid/corrupt JSON is safely mapped to RazorpayAPIError."""
    mock_client = AsyncMock(spec=httpx.AsyncClient)
    mock_response = MagicMock(spec=httpx.Response)
    mock_response.status_code = 200
    mock_response.json.side_effect = ValueError("Invalid JSON response payload")
    mock_client.post.return_value = mock_response

    adapter = RazorpayAdapter(
        key_id="rzp_test_key123",
        key_secret="rzp_test_sec456",
        dry_run=False,
        client=mock_client,
    )

    with pytest.raises(
        RazorpayAPIError,
        match=re.escape("Invalid JSON payload returned from Razorpay API (status 200)"),
    ) as exc_info:
        await adapter.create_payment_link(amount=100.0)

    assert exc_info.value.status_code == 200


def test_razorpay_adapter_accepts_default_and_custom_https_url():
    """Verify default and custom HTTPS URLs are accepted for live mode."""
    adapter_default = RazorpayAdapter(key_id="k", key_secret="s", dry_run=False)
    assert adapter_default.base_url == "https://api.razorpay.com/v1"

    adapter_custom = RazorpayAdapter(
        key_id="k",
        key_secret="s",
        base_url="https://api.custom-gateway.internal/v1",
        dry_run=False,
    )
    assert adapter_custom.base_url == "https://api.custom-gateway.internal/v1"


def test_razorpay_adapter_rejects_http_url_for_live_requests():
    """Verify non-HTTPS URLs are rejected before live requests can be dispatched."""
    with pytest.raises(
        RazorpayConfigurationError,
        match=re.escape("Live Razorpay API requests require a secure HTTPS base URL."),
    ):
        RazorpayAdapter(
            key_id="k",
            key_secret="s",
            base_url="http://api.insecure-gateway.com/v1",
            dry_run=False,
        )


@pytest.mark.anyio
async def test_razorpay_adapter_dry_run_unaffected_by_http_url():
    """Verify dry_run mode functions normally even if base_url is HTTP for local dev."""
    adapter = RazorpayAdapter(
        base_url="http://localhost:8000/mock-razorpay",
        dry_run=True,
    )
    assert adapter.dry_run is True
    res = await adapter.create_payment_link(amount=100.0)
    assert res["simulated"] is True
    assert res["id"].startswith("plink_sim_")


def test_razorpay_adapter_repr_masks_secret():
    """Verify __repr__ masks API key secret."""
    adapter = RazorpayAdapter(key_id="rzp_key", key_secret="super_secret_password")
    repr_str = repr(adapter)
    assert "super_secret_password" not in repr_str
    assert "***" in repr_str


# ============================================================================
# 2. ActionExecutor Tests
# ============================================================================


@pytest.mark.anyio
async def test_action_executor_payment_link_success(
    sample_customer_context, permissive_policy_context
):
    """Verify ActionExecutor dispatches approved PAYMENT_LINK to adapter."""
    executor = ActionExecutor(razorpay_adapter=RazorpayAdapter(dry_run=True))

    result: ActionResult = await executor.execute(
        approved_action=RecoveryAction.PAYMENT_LINK,
        policy_context=permissive_policy_context,
        context=sample_customer_context,
    )

    assert result.action == RecoveryAction.PAYMENT_LINK
    assert result.attempted is True
    assert result.success is True
    assert result.status == "simulated"
    assert result.reference_id is not None
    assert result.reference_id.startswith("plink_sim_")
    assert result.resource_url is not None
    assert "https://rzp.io/i/" in result.resource_url


@pytest.mark.anyio
async def test_action_executor_rejects_none_current_payment(
    sample_customer_context, permissive_policy_context
):
    """Verify ActionExecutor rejects payment link when current_payment is None without calling adapter."""
    mock_adapter = MagicMock(spec=RazorpayAdapter)
    mock_adapter.create_payment_link = AsyncMock()
    executor = ActionExecutor(razorpay_adapter=mock_adapter)

    # Set current_payment to None
    context_no_payment = CustomerRecoveryContext(
        customer=sample_customer_context.customer,
        current_payment=None,
        current_opportunity=sample_customer_context.current_opportunity,
    )

    result = await executor.execute(
        approved_action=RecoveryAction.PAYMENT_LINK,
        policy_context=permissive_policy_context,
        context=context_no_payment,
    )

    assert result.action == RecoveryAction.PAYMENT_LINK
    assert result.attempted is False
    assert result.status == "failed"
    assert result.success is False
    assert "Amount must be positive" in result.error
    mock_adapter.create_payment_link.assert_not_called()


@pytest.mark.anyio
@pytest.mark.parametrize("invalid_amount", [0.0, -1.0, -500.25])
async def test_action_executor_rejects_non_positive_payment_amount(
    sample_customer_context, permissive_policy_context, invalid_amount: float
):
    """Verify ActionExecutor rejects zero and negative amounts for both PAYMENT_LINK and CHANGE_PAYMENT_METHOD."""
    mock_adapter = MagicMock(spec=RazorpayAdapter)
    mock_adapter.create_payment_link = AsyncMock()
    executor = ActionExecutor(razorpay_adapter=mock_adapter)

    # Use model_construct to test zero and any negative amounts that might reach this layer
    payment = PaymentContext.model_construct(
        payment_id=uuid4(),
        amount=invalid_amount,
        currency="INR",
        payment_method="card",
        status="failed",
        failure_reason="timeout",
    )
    context_invalid_amount = CustomerRecoveryContext(
        customer=sample_customer_context.customer,
        current_payment=payment,
        current_opportunity=sample_customer_context.current_opportunity,
    )

    for action in (RecoveryAction.PAYMENT_LINK, RecoveryAction.CHANGE_PAYMENT_METHOD):
        result = await executor.execute(
            approved_action=action,
            policy_context=permissive_policy_context,
            context=context_invalid_amount,
        )

        assert result.action == action
        assert result.attempted is False
        assert result.status == "failed"
        assert result.success is False
        assert f"Invalid recovery payment amount: {invalid_amount}" in result.error
        mock_adapter.create_payment_link.assert_not_called()


@pytest.mark.anyio
async def test_action_executor_never_bypasses_policy_validator(
    sample_customer_context, restrictive_policy_context
):
    """
    CRITICAL INVARIANT: ActionExecutor refuses to execute prohibited actions.
    Even if a caller explicitly requests RETRY_PAYMENT, PolicyValidator constraints
    must be authoritative and prevent execution.
    """
    executor = ActionExecutor(razorpay_adapter=RazorpayAdapter(dry_run=True))

    result: ActionResult = await executor.execute(
        approved_action=RecoveryAction.RETRY_PAYMENT,
        policy_context=restrictive_policy_context,
        context=sample_customer_context,
    )

    assert result.action == RecoveryAction.RETRY_PAYMENT
    assert result.attempted is False
    assert result.success is False
    assert result.status == "prohibited"
    assert "PolicyValidator constraints" in result.error
    assert result.reference_id is None


@pytest.mark.anyio
async def test_action_executor_change_payment_method(
    sample_customer_context, permissive_policy_context
):
    """Verify CHANGE_PAYMENT_METHOD generates interactive payment link."""
    executor = ActionExecutor(razorpay_adapter=RazorpayAdapter(dry_run=True))

    result: ActionResult = await executor.execute(
        approved_action=RecoveryAction.CHANGE_PAYMENT_METHOD,
        policy_context=permissive_policy_context,
        context=sample_customer_context,
    )

    assert result.action == RecoveryAction.CHANGE_PAYMENT_METHOD
    assert result.attempted is True
    assert result.success is True
    assert result.reference_id is not None
    assert result.resource_url is not None


@pytest.mark.anyio
async def test_action_executor_wait_and_retry_skipped(
    sample_customer_context, permissive_policy_context
):
    """Verify WAIT_AND_RETRY is treated as an internal cooling schedule without external API calls."""
    executor = ActionExecutor(razorpay_adapter=RazorpayAdapter(dry_run=True))

    result: ActionResult = await executor.execute(
        approved_action=RecoveryAction.WAIT_AND_RETRY,
        policy_context=permissive_policy_context,
        context=sample_customer_context,
    )

    assert result.action == RecoveryAction.WAIT_AND_RETRY
    assert result.attempted is False
    assert result.status == "skipped"
    assert result.success is True
    assert "cooling period" in result.message


@pytest.mark.anyio
async def test_action_executor_no_action_skipped(
    sample_customer_context, permissive_policy_context
):
    """Verify NO_ACTION is treated as a terminal skip without external calls."""
    executor = ActionExecutor(razorpay_adapter=RazorpayAdapter(dry_run=True))

    result: ActionResult = await executor.execute(
        approved_action=RecoveryAction.NO_ACTION,
        policy_context=permissive_policy_context,
        context=sample_customer_context,
    )

    assert result.action == RecoveryAction.NO_ACTION
    assert result.attempted is False
    assert result.status == "skipped"
    assert result.success is True


@pytest.mark.anyio
async def test_action_executor_handles_adapter_failure(
    sample_customer_context, permissive_policy_context
):
    """Verify that when Razorpay adapter fails, ActionExecutor returns clean failed ActionResult without crashing."""
    failing_adapter = MagicMock(spec=RazorpayAdapter)
    failing_adapter.create_payment_link = AsyncMock(
        side_effect=RazorpayAPIError("Simulated Razorpay server 503 error")
    )

    executor = ActionExecutor(razorpay_adapter=failing_adapter)

    result: ActionResult = await executor.execute(
        approved_action=RecoveryAction.PAYMENT_LINK,
        policy_context=permissive_policy_context,
        context=sample_customer_context,
    )

    assert result.action == RecoveryAction.PAYMENT_LINK
    assert result.attempted is True
    assert result.status == "failed"
    assert result.success is False
    assert "Simulated Razorpay server 503 error" in result.error


@pytest.mark.anyio
async def test_action_executor_live_retry_payment_requires_customer_presence(
    sample_customer_context, permissive_policy_context
):
    """
    Verify that in live mode (dry_run=False) when customer mandate is unavailable:
    - attempted is False because no external payment attempt occurs
    - status is 'requires_customer_presence'
    - Razorpay is not called
    """
    mock_adapter = MagicMock(spec=RazorpayAdapter)
    mock_adapter.dry_run = False
    mock_adapter.create_payment_link = AsyncMock()

    executor = ActionExecutor(razorpay_adapter=mock_adapter)

    result: ActionResult = await executor.execute(
        approved_action=RecoveryAction.RETRY_PAYMENT,
        policy_context=permissive_policy_context,
        context=sample_customer_context,
    )

    assert result.action == RecoveryAction.RETRY_PAYMENT
    assert result.attempted is False
    assert result.status == "requires_customer_presence"
    assert result.success is False
    assert "customer-authorized mandate" in result.error
    mock_adapter.create_payment_link.assert_not_called()


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("input_amount", "expected_paise"),
    [
        (19.99, 1999),
        (10.005, 1001),  # ROUND_HALF_UP test
        (10.004, 1000),  # ROUND_HALF_UP test
        (0.01, 1),
        (2500.0, 250000),
        (99.995, 10000),
    ],
)
async def test_razorpay_adapter_decimal_currency_conversion_round_half_up(
    input_amount: float, expected_paise: int
):
    """Verify Decimal-based currency conversion with ROUND_HALF_UP converts to exact int paise."""
    adapter = RazorpayAdapter(dry_run=True)
    res = await adapter.create_payment_link(amount=input_amount)

    assert isinstance(res["amount"], int)
    assert res["amount"] == expected_paise
