"""
Unit tests for Revora Agent Context Builder.
"""

from datetime import datetime, timezone
import uuid
import pytest

from app.agent.context_builder import AgentContextBuilder
from app.agent.schemas import AgentDecisionPromptContext
from app.context import (
    CustomerContext,
    CustomerRecoveryContext,
    CustomerRecoveryStatsContext,
    PaymentContext,
    RecoveryAttemptContext,
    RecoveryOpportunityContext,
)
from app.decision_engine import RecoveryAction
from app.historical_retrieval import HistoricalCase
from app.policies.registry import (
    RZP_CUSTOMER_AUTH_2FA_REQUIRED_RULE,
    SAFETY_MAX_ATTEMPTS_RULE,
)
from app.policies.schemas import RecoveryPolicyContext


@pytest.fixture
def rich_customer_recovery_context():
    """Build a comprehensive CustomerRecoveryContext with sensitive PII fields populated."""
    cust_id = uuid.uuid4()
    pay_id = uuid.uuid4()
    opp_id = uuid.uuid4()
    now = datetime(2026, 8, 31, 14, 0, 0, tzinfo=timezone.utc)

    customer = CustomerContext(
        customer_id=cust_id,
        external_customer_id="CUST_SECRET_EXT_999",
        name="Sensitive Customer Name",
        email="confidential.customer@example.com",
        total_payments=10,
        successful_payments=8,
        failed_payments=2,
        historical_success_rate=0.80,
    )

    payment = PaymentContext(
        payment_id=pay_id,
        external_payment_id="PAY_SECRET_EXT_777",
        amount=1500.0,
        currency="INR",
        payment_method="card",
        status="failed",
        failure_reason="authentication_failed",
        created_at=now,
    )

    opportunity = RecoveryOpportunityContext(
        opportunity_id=opp_id,
        status="open",
        revenue_at_risk=1500.0,
        expected_recovery=1200.0,
        created_at=now,
    )

    attempts = [
        RecoveryAttemptContext(
            attempt_id=uuid.uuid4(),
            action="retry_payment",
            status="failed",
            amount_recovered=0.0,
            error_code="3ds_timeout",
            external_reference="TXN_ATT_SECRET_001",
            created_at=now,
        ),
        RecoveryAttemptContext(
            attempt_id=uuid.uuid4(),
            action="payment_link",
            status="pending",
            amount_recovered=0.0,
            error_code=None,
            external_reference="TXN_ATT_SECRET_002",
            created_at=now,
        ),
    ]

    stats = CustomerRecoveryStatsContext(
        total_recovery_opportunities=3,
        recovered_opportunities=2,
        failed_opportunities=1,
        recovery_rate=0.6667,
        previously_successful_actions=["payment_link", "retry_payment"],
        previously_failed_actions=["wait_and_retry"],
        total_amount_recovered=4500.0,
    )

    return CustomerRecoveryContext(
        customer=customer,
        current_payment=payment,
        current_opportunity=opportunity,
        current_payment_attempts=attempts,
        historical_payments=[],
        recovery_statistics=stats,
        retrieved_at=now,
    )


@pytest.fixture
def sample_historical_cases():
    """Build a list of historical recovery cases with varying relevance scores and metadata."""
    now = datetime(2026, 8, 15, 10, 0, 0, tzinfo=timezone.utc)
    cases = []
    scores = [0.75, 0.95, 0.85, 0.95, 0.60, 0.40]

    for idx, score in enumerate(scores):
        cases.append(
            HistoricalCase(
                payment_id=uuid.UUID(f"00000000-0000-0000-0000-00000000000{idx+1}"),
                customer_id=uuid.uuid4(),
                external_payment_id=f"HIST_EXT_{idx}",
                external_customer_id=f"HIST_CUST_{idx}",
                amount=1000.0 + idx * 100,
                currency="INR",
                payment_method="card",
                failure_reason="authentication_failed",
                recovery_action="payment_link",
                recovery_status="recovered",
                amount_recovered=1000.0 + idx * 100,
                was_recovered=True,
                relevance_score=score,
                metadata={"secret_token": f"sensitive_meta_{idx}"},
                created_at=now,
            )
        )
    return cases


@pytest.fixture
def sample_policy_context():
    """Build a valid RecoveryPolicyContext."""
    return RecoveryPolicyContext(
        provider="razorpay",
        policy_version="2026.1",
        applicable_rules=(RZP_CUSTOMER_AUTH_2FA_REQUIRED_RULE,),
        allowed_actions=(RecoveryAction.PAYMENT_LINK, RecoveryAction.CHANGE_PAYMENT_METHOD),
        prohibited_actions=(RecoveryAction.RETRY_PAYMENT, RecoveryAction.WAIT_AND_RETRY),
        mandatory_fallback_action=RecoveryAction.PAYMENT_LINK,
        metadata={"internal_policy_data": "secret"},
    )


# ============================================================================
# 1. Transformation and Mapping Tests
# ============================================================================


def test_build_prompt_context_complete_flow(
    rich_customer_recovery_context,
    sample_historical_cases,
    sample_policy_context,
):
    """Verify complete transformation pipeline returns a valid AgentDecisionPromptContext."""
    builder = AgentContextBuilder(max_historical_cases=5)
    prompt_ctx = builder.build_prompt_context(
        context=rich_customer_recovery_context,
        historical_cases=sample_historical_cases,
        policy_context=sample_policy_context,
    )

    assert isinstance(prompt_ctx, AgentDecisionPromptContext)
    assert prompt_ctx.current_payment["amount"] == 1500.0
    assert prompt_ctx.current_payment["payment_method"] == "card"
    assert prompt_ctx.current_payment["failure_reason"] == "authentication_failed"

    assert prompt_ctx.customer_profile["total_payments"] == 10
    assert prompt_ctx.customer_profile["recovery_rate"] == 0.6667
    assert prompt_ctx.customer_profile["lifetime_amount_recovered"] == 4500.0

    assert len(prompt_ctx.recovery_attempt_history) == 2
    assert len(prompt_ctx.historical_cases) == 5  # Capped at 5

    assert prompt_ctx.allowed_actions == ("change_payment_method", "payment_link")
    assert prompt_ctx.prohibited_actions == ("retry_payment", "wait_and_retry")
    assert prompt_ctx.mandatory_fallback == "payment_link"
    assert len(prompt_ctx.policy_constraints) == 1


def test_pii_fields_excluded_from_prompt_context(
    rich_customer_recovery_context,
    sample_historical_cases,
    sample_policy_context,
):
    """Verify strict PII allowlist: sensitive identifiers are nowhere in the prompt context."""
    builder = AgentContextBuilder()
    prompt_ctx = builder.build_prompt_context(
        context=rich_customer_recovery_context,
        historical_cases=sample_historical_cases,
        policy_context=sample_policy_context,
    )

    dumped = prompt_ctx.model_dump(mode="json")
    json_str = str(dumped)

    # Check direct customer PII
    assert "Sensitive Customer Name" not in json_str
    assert "confidential.customer@example.com" not in json_str
    assert "CUST_SECRET_EXT_999" not in json_str
    assert str(rich_customer_recovery_context.customer.customer_id) not in json_str

    # Check payment identifiers
    assert "PAY_SECRET_EXT_777" not in json_str
    assert str(rich_customer_recovery_context.current_payment.payment_id) not in json_str

    # Check attempt references
    assert "TXN_ATT_SECRET_001" not in json_str
    assert "TXN_ATT_SECRET_002" not in json_str

    # Check historical metadata
    assert "sensitive_meta" not in json_str
    assert "internal_policy_data" not in json_str


def test_payment_field_mapping(rich_customer_recovery_context):
    """Verify payment field mapping and defaults."""
    builder = AgentContextBuilder()
    prompt_ctx = builder.build_prompt_context(rich_customer_recovery_context)

    payment_data = prompt_ctx.current_payment
    assert payment_data["amount"] == 1500.0
    assert payment_data["currency"] == "INR"
    assert payment_data["payment_method"] == "card"
    assert payment_data["failure_reason"] == "authentication_failed"


def test_customer_profile_mapping(rich_customer_recovery_context):
    """Verify customer profile aggregation mapping."""
    builder = AgentContextBuilder()
    prompt_ctx = builder.build_prompt_context(rich_customer_recovery_context)

    profile = prompt_ctx.customer_profile
    assert profile["total_payments"] == 10
    assert profile["successful_payments"] == 8
    assert profile["failed_payments"] == 2
    assert profile["historical_success_rate"] == 0.8
    assert profile["total_recovery_opportunities"] == 3
    assert profile["recovered_opportunities"] == 2
    assert profile["failed_opportunities"] == 1
    assert profile["recovery_rate"] == 0.6667
    assert profile["previously_successful_actions"] == ("payment_link", "retry_payment")
    assert profile["previously_failed_actions"] == ("wait_and_retry",)
    assert profile["lifetime_amount_recovered"] == 4500.0

    # Also verify serialized dictionary converts tuples back to JSON lists
    json_profile = prompt_ctx.model_dump(mode="json")["customer_profile"]
    assert json_profile["previously_successful_actions"] == ["payment_link", "retry_payment"]
    assert json_profile["previously_failed_actions"] == ["wait_and_retry"]


def test_attempt_history_mapping(rich_customer_recovery_context):
    """Verify attempt sequence numbering and error codes."""
    builder = AgentContextBuilder()
    prompt_ctx = builder.build_prompt_context(rich_customer_recovery_context)

    attempts = prompt_ctx.recovery_attempt_history
    assert len(attempts) == 2
    assert attempts[0] == {
        "attempt_number": 1,
        "action": "retry_payment",
        "status": "failed",
        "error_code": "3ds_timeout",
    }
    assert attempts[1] == {
        "attempt_number": 2,
        "action": "payment_link",
        "status": "pending",
        "error_code": "none",
    }


def test_historical_cases_sorting_and_capping(
    rich_customer_recovery_context,
    sample_historical_cases,
):
    """Verify cases are sorted by relevance_score DESC and capped at max_historical_cases."""
    builder = AgentContextBuilder(max_historical_cases=3)
    prompt_ctx = builder.build_prompt_context(
        context=rich_customer_recovery_context,
        historical_cases=sample_historical_cases,
    )

    cases = prompt_ctx.historical_cases
    assert len(cases) == 3
    # First two should have 0.95 relevance score
    assert cases[0]["relevance_score"] == 0.95
    assert cases[1]["relevance_score"] == 0.95
    assert cases[2]["relevance_score"] == 0.85


def test_historical_cases_deterministic_tie_breaking(
    rich_customer_recovery_context,
):
    """Verify tie-breaking by case_id ASC when relevance scores are identical."""
    case_b = HistoricalCase(
        payment_id=uuid.UUID("00000000-0000-0000-0000-00000000000b"),
        customer_id=uuid.uuid4(),
        amount=500.0,
        payment_method="upi",
        recovery_status="recovered",
        relevance_score=0.90,
    )
    case_a = HistoricalCase(
        payment_id=uuid.UUID("00000000-0000-0000-0000-00000000000a"),
        customer_id=uuid.uuid4(),
        amount=500.0,
        payment_method="upi",
        recovery_status="recovered",
        relevance_score=0.90,
    )

    builder = AgentContextBuilder()
    # Pass in reverse order [case_b, case_a]
    prompt_ctx = builder.build_prompt_context(
        context=rich_customer_recovery_context,
        historical_cases=[case_b, case_a],
    )

    cases = prompt_ctx.historical_cases
    assert len(cases) == 2
    # case_a should come first due to tie-breaker
    assert cases[0]["case_id"] == "00000000-0000-0000-0000-00000000000a"
    assert cases[1]["case_id"] == "00000000-0000-0000-0000-00000000000b"


def test_policy_envelope_action_string_mapping(
    rich_customer_recovery_context,
    sample_policy_context,
):
    """Verify policy actions are converted to sorted string lists."""
    builder = AgentContextBuilder()
    prompt_ctx = builder.build_prompt_context(
        context=rich_customer_recovery_context,
        policy_context=sample_policy_context,
    )

    assert prompt_ctx.allowed_actions == ("change_payment_method", "payment_link")
    assert prompt_ctx.prohibited_actions == ("retry_payment", "wait_and_retry")
    assert prompt_ctx.mandatory_fallback == "payment_link"
    assert len(prompt_ctx.policy_constraints) == 1
    assert "2FA" in prompt_ctx.policy_constraints[0]


def test_missing_optional_payment_or_opportunity():
    """Verify fallback defaults when payment or stats are None."""
    cust_id = uuid.uuid4()
    context = CustomerRecoveryContext(
        customer=CustomerContext(customer_id=cust_id),
        current_payment=None,
        current_opportunity=None,
    )

    builder = AgentContextBuilder()
    prompt_ctx = builder.build_prompt_context(context)

    assert prompt_ctx.current_payment["amount"] == 0.0
    assert prompt_ctx.current_payment["payment_method"] == "unspecified"
    assert prompt_ctx.current_payment["failure_reason"] == "unspecified"
    assert prompt_ctx.allowed_actions == ()
    assert prompt_ctx.prohibited_actions == ()
    assert prompt_ctx.mandatory_fallback is None


def test_empty_historical_cases(rich_customer_recovery_context):
    """Verify empty tuple when historical cases are None or empty."""
    builder = AgentContextBuilder()
    prompt_ctx1 = builder.build_prompt_context(rich_customer_recovery_context, historical_cases=None)
    prompt_ctx2 = builder.build_prompt_context(rich_customer_recovery_context, historical_cases=[])

    assert prompt_ctx1.historical_cases == ()
    assert prompt_ctx2.historical_cases == ()


def test_deterministic_context_builder_output(
    rich_customer_recovery_context,
    sample_historical_cases,
    sample_policy_context,
):
    """Verify identical inputs produce identical outputs across multiple invocations."""
    builder = AgentContextBuilder()
    ctx1 = builder.build_prompt_context(
        rich_customer_recovery_context,
        sample_historical_cases,
        sample_policy_context,
    )
    ctx2 = builder.build_prompt_context(
        rich_customer_recovery_context,
        sample_historical_cases,
        sample_policy_context,
    )

    assert ctx1.model_dump(mode="json") == ctx2.model_dump(mode="json")


def test_nested_context_is_immutable(
    rich_customer_recovery_context,
    sample_historical_cases,
    sample_policy_context,
):
    """Verify the resulting AgentDecisionPromptContext cannot be mutated."""
    builder = AgentContextBuilder()
    prompt_ctx = builder.build_prompt_context(
        rich_customer_recovery_context,
        sample_historical_cases,
        sample_policy_context,
    )

    with pytest.raises(TypeError):
        prompt_ctx.current_payment["amount"] = 9999.0

    with pytest.raises(TypeError):
        prompt_ctx.customer_profile["total_payments"] = 9999


def test_source_objects_are_not_mutated(
    rich_customer_recovery_context,
    sample_historical_cases,
    sample_policy_context,
):
    """Verify input domain models remain completely unmodified."""
    original_amount = rich_customer_recovery_context.current_payment.amount
    original_cases_len = len(sample_historical_cases)

    builder = AgentContextBuilder(max_historical_cases=2)
    builder.build_prompt_context(
        rich_customer_recovery_context,
        sample_historical_cases,
        sample_policy_context,
    )

    assert rich_customer_recovery_context.current_payment.amount == original_amount
    assert len(sample_historical_cases) == original_cases_len


def test_invalid_max_historical_cases_rejected():
    """Verify validation on max_historical_cases parameter."""
    with pytest.raises(ValueError, match="positive integer"):
        AgentContextBuilder(max_historical_cases=0)

    with pytest.raises(ValueError, match="positive integer"):
        AgentContextBuilder(max_historical_cases=-1)

    with pytest.raises(TypeError, match="must be an integer"):
        AgentContextBuilder(max_historical_cases="five")  # type: ignore

    with pytest.raises(TypeError, match="must be an integer"):
        AgentContextBuilder(max_historical_cases=True)  # type: ignore
