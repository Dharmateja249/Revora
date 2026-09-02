"""
Unit tests for Revora Agent Prompt Generation.
"""

import json

import pytest

from app.agent.prompts import (
    REVORA_AGENT_SYSTEM_PROMPT,
    build_agent_messages,
)
from app.agent.schemas import AgentDecisionPromptContext


@pytest.fixture
def sample_prompt_context():
    """Create a sample AgentDecisionPromptContext fixture."""
    return AgentDecisionPromptContext(
        current_payment={
            "amount": 750.0,
            "currency": "INR",
            "payment_method": "card",
            "failure_reason": "authentication_failed",
        },
        customer_profile={
            "total_payments": 5,
            "successful_payments": 4,
            "failed_payments": 1,
            "historical_success_rate": 0.80,
            "recovery_rate": 1.0,
            "lifetime_amount_recovered": 750.0,
            "previously_successful_actions": ["payment_link"],
            "previously_failed_actions": [],
        },
        recovery_attempt_history=[
            {
                "attempt_number": 1,
                "action": "retry_payment",
                "status": "failed",
                "error_code": "3ds_timeout",
            }
        ],
        historical_cases=[
            {
                "case_id": "case_1",
                "amount": 750.0,
                "currency": "INR",
                "payment_method": "card",
                "failure_reason": "authentication_failed",
                "recovery_action": "payment_link",
                "recovery_status": "recovered",
                "amount_recovered": 750.0,
                "was_recovered": True,
                "relevance_score": 0.95,
            }
        ],
        allowed_actions=["change_payment_method", "payment_link"],
        prohibited_actions=["retry_payment", "wait_and_retry"],
        mandatory_fallback="payment_link",
        policy_constraints=[
            "RBI & 3DS cardholder presence mandate: silent server-side retry cannot complete cardholder verification."
        ],
    )


def test_system_prompt_contains_policy_invariants():
    """Verify system prompt contains core policy rules, action boundaries, and output contract."""
    prompt = REVORA_AGENT_SYSTEM_PROMPT

    assert "ALLOWED ACTIONS" in prompt
    assert "PROHIBITED ACTIONS" in prompt
    assert "policy constraint WINS" in prompt
    assert "empirical supporting evidence" in prompt
    assert "recommended_action" in prompt
    assert "confidence" in prompt
    assert "reasoning" in prompt
    assert "key_factors" in prompt
    assert "referenced_case_ids" in prompt


def test_build_messages_structure(sample_prompt_context):
    """Verify build_agent_messages returns standard system and user chat messages."""
    messages = build_agent_messages(sample_prompt_context)

    assert isinstance(messages, list)
    assert len(messages) == 2

    sys_msg = messages[0]
    assert sys_msg["role"] == "system"
    assert sys_msg["content"] == REVORA_AGENT_SYSTEM_PROMPT

    user_msg = messages[1]
    assert user_msg["role"] == "user"
    assert (
        "Analyze the following failed payment recovery scenario" in user_msg["content"]
    )


def test_user_message_contains_structured_sections(sample_prompt_context):
    """Verify user message contains all key context sections."""
    messages = build_agent_messages(sample_prompt_context)
    user_content = messages[1]["content"]

    assert "current_payment" in user_content
    assert "customer_recovery_profile" in user_content
    assert "attempt_budget" in user_content
    assert "prior_attempts_on_this_payment" in user_content
    assert "retrieved_historical_evidence" in user_content
    assert "policy_envelope" in user_content


def test_prompt_deterministic_generation(sample_prompt_context):
    """Verify prompt generation produces identical string across invocations."""
    msgs1 = build_agent_messages(sample_prompt_context)
    msgs2 = build_agent_messages(sample_prompt_context)

    assert msgs1 == msgs2
    assert msgs1[1]["content"] == msgs2[1]["content"]


def test_prompt_contains_allowed_and_prohibited_actions(sample_prompt_context):
    """Verify allowed and prohibited actions are clearly listed in the user payload."""
    messages = build_agent_messages(sample_prompt_context)
    user_content = messages[1]["content"]

    assert "change_payment_method" in user_content
    assert "payment_link" in user_content
    assert "retry_payment" in user_content
    assert "wait_and_retry" in user_content


def test_user_payload_is_valid_json(sample_prompt_context):
    """Extract and parse the JSON block inside the user message."""
    messages = build_agent_messages(sample_prompt_context)
    user_content = messages[1]["content"]

    # Extract JSON code block
    start_idx = user_content.find("```json\n") + len("```json\n")
    end_idx = user_content.find("\n```", start_idx)
    json_str = user_content[start_idx:end_idx]

    parsed = json.loads(json_str)
    assert parsed["current_payment"]["amount"] == 750.0
    assert parsed["current_payment"]["payment_method"] == "card"
    assert parsed["policy_envelope"]["mandatory_fallback"] == "payment_link"
    assert len(parsed["retrieved_historical_evidence"]) == 1


def test_invalid_prompt_context_type_raises_type_error():
    """Verify TypeError when invalid object passed to build_agent_messages."""
    with pytest.raises(TypeError, match="Expected AgentDecisionPromptContext"):
        build_agent_messages({"not": "a context"})  # type: ignore


def test_prompt_does_not_contain_pii_or_payment_ids(sample_prompt_context):
    """Verify generated messages contain anonymous case tokens and no raw UUIDs or PII."""
    messages = build_agent_messages(sample_prompt_context)
    user_content = messages[1]["content"]

    assert "case_1" in user_content
    # Ensure no raw database UUIDs or PII substrings are present
    assert "00000000-0000-0000-0000-000000000001" not in user_content
    assert "@" not in user_content


def test_build_agent_messages_rejects_empty_allowed_actions():
    """Regression test for Finding 2: build_agent_messages rejects prompt context with empty allowed_actions."""
    empty_allowed_ctx = AgentDecisionPromptContext(
        current_payment={"amount": 500.0, "currency": "INR"},
        customer_profile={"historical_success_rate": 0.8},
        allowed_actions=[],
    )
    with pytest.raises(ValueError, match="at least one allowed action"):
        build_agent_messages(empty_allowed_ctx)


def test_prompt_serialization_regression_for_historical_payment_ids():
    """Regression test for Finding 1: Historical payment UUIDs and customer UUIDs are never serialized in prompt."""
    import uuid
    from datetime import datetime, timezone

    from app.agent.context_builder import AgentContextBuilder
    from app.context import CustomerContext, CustomerRecoveryContext, PaymentContext
    from app.decision_engine import RecoveryAction
    from app.historical_retrieval import HistoricalCase
    from app.policies.schemas import RecoveryPolicyContext

    raw_payment_uuid = uuid.UUID("11111111-2222-3333-4444-555555555555")
    raw_customer_uuid = uuid.UUID("99999999-8888-7777-6666-555555555555")

    context = CustomerRecoveryContext(
        customer=CustomerContext(
            customer_id=uuid.uuid4(),
            name="Confidential Name",
            email="confidential@example.com",
        ),
        current_payment=PaymentContext(
            payment_id=uuid.uuid4(),
            amount=1200.0,
            currency="INR",
            payment_method="card",
            status="failed",
            failure_reason="authentication_failed",
            created_at=datetime.now(timezone.utc),
        ),
        current_opportunity=None,
    )

    hist_case = HistoricalCase(
        payment_id=raw_payment_uuid,
        customer_id=raw_customer_uuid,
        external_payment_id="LEAK_EXT_PAYMENT_123",
        external_customer_id="LEAK_EXT_CUST_123",
        amount=1200.0,
        currency="INR",
        payment_method="card",
        failure_reason="authentication_failed",
        recovery_action="payment_link",
        recovery_status="recovered",
        amount_recovered=1200.0,
        was_recovered=True,
        relevance_score=0.98,
        metadata={"private_key": "leak_secret"},
    )

    policy_ctx = RecoveryPolicyContext(
        provider="razorpay",
        policy_version="2026.1",
        applicable_rules=(),
        allowed_actions=(RecoveryAction.PAYMENT_LINK,),
    )

    builder = AgentContextBuilder()
    prompt_ctx = builder.build_prompt_context(
        context=context,
        historical_cases=[hist_case],
        policy_context=policy_ctx,
    )

    messages = build_agent_messages(prompt_ctx)
    user_prompt = messages[1]["content"]

    assert str(raw_payment_uuid) not in user_prompt
    assert str(raw_customer_uuid) not in user_prompt
    assert "LEAK_EXT_PAYMENT_123" not in user_prompt
    assert "LEAK_EXT_CUST_123" not in user_prompt
    assert "leak_secret" not in user_prompt
    assert "Confidential Name" not in user_prompt
    assert "confidential@example.com" not in user_prompt
    assert "case_1" in user_prompt


def test_prompt_attempt_budget_serialization():
    """Verify serialized prompt contains attempt_budget with expected fields."""
    ctx = AgentDecisionPromptContext(
        current_payment={"amount": 1000.0, "currency": "INR"},
        customer_profile={"total_payments": 3},
        attempt_budget={
            "current_attempt": 2,
            "max_attempts": 3,
            "remaining_attempts": 1,
        },
        allowed_actions=["payment_link"],
    )
    messages = build_agent_messages(ctx)
    user_content = messages[1]["content"]

    start_idx = user_content.find("```json\n") + len("```json\n")
    end_idx = user_content.find("\n```", start_idx)
    payload = json.loads(user_content[start_idx:end_idx])

    assert "attempt_budget" in payload
    assert payload["attempt_budget"] == {
        "current_attempt": 2,
        "max_attempts": 3,
        "remaining_attempts": 1,
    }


def test_prompt_does_not_leak_opportunity_or_attempt_ids():
    """Verify opportunity_id and attempt_id are never serialized in prompt."""
    from datetime import datetime, timezone
    from uuid import uuid4

    from app.agent.context_builder import AgentContextBuilder
    from app.context import (
        CustomerContext,
        CustomerRecoveryContext,
        PaymentContext,
        RecoveryAttemptContext,
        RecoveryOpportunityContext,
    )
    from app.decision_engine import RecoveryAction
    from app.policies.schemas import RecoveryPolicyContext

    opp_id = uuid4()
    att_id_1 = uuid4()
    att_id_2 = uuid4()

    ctx = CustomerRecoveryContext(
        customer=CustomerContext(customer_id=uuid4(), name="Customer Alpha"),
        current_payment=PaymentContext(
            payment_id=uuid4(),
            amount=2500.0,
            payment_method="card",
            status="failed",
            created_at=datetime.now(timezone.utc),
        ),
        current_opportunity=RecoveryOpportunityContext(
            opportunity_id=opp_id,
            status="in_progress",
            revenue_at_risk=2500.0,
        ),
        current_payment_attempts=[
            RecoveryAttemptContext(
                attempt_id=att_id_1,
                action="retry_payment",
                status="failed",
                error_code="do_not_honor",
            ),
            RecoveryAttemptContext(
                attempt_id=att_id_2,
                action="payment_link",
                status="failed",
            ),
        ],
    )

    policy_ctx = RecoveryPolicyContext(
        provider="razorpay",
        policy_version="2026.1",
        applicable_rules=(),
        allowed_actions=(RecoveryAction.CHANGE_PAYMENT_METHOD,),
    )

    builder = AgentContextBuilder()
    prompt_ctx = builder.build_prompt_context(ctx, policy_context=policy_ctx)
    messages = build_agent_messages(prompt_ctx)
    user_prompt = messages[1]["content"]

    assert str(opp_id) not in user_prompt
    assert str(att_id_1) not in user_prompt
    assert str(att_id_2) not in user_prompt
    assert "Customer Alpha" not in user_prompt

    start_idx = user_prompt.find("```json\n") + len("```json\n")
    end_idx = user_prompt.find("\n```", start_idx)
    payload = json.loads(user_prompt[start_idx:end_idx])

    assert payload["attempt_budget"] == {
        "current_attempt": 3,
        "max_attempts": 3,
        "remaining_attempts": 0,
    }
