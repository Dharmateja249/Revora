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
                "case_id": "00000000-0000-0000-0000-000000000001",
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
    assert "Analyze the following failed payment recovery scenario" in user_msg["content"]


def test_user_message_contains_structured_sections(sample_prompt_context):
    """Verify user message contains all key context sections."""
    messages = build_agent_messages(sample_prompt_context)
    user_content = messages[1]["content"]

    assert "current_payment" in user_content
    assert "customer_recovery_profile" in user_content
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
