"""
Unit tests for Revora LLM Provider Abstraction and Mock LLM Provider.
"""

import pytest
from app.agent.prompts import build_agent_messages
from app.agent.provider import (
    LLMProvider,
    LLMProviderError,
    LLMResponseValidationError,
    MockLLMProvider,
    validate_chat_messages,
)
from app.agent.schemas import (
    AgentDecisionPromptContext,
    AgentDecisionResult,
    LLMRecoveryRecommendation,
)
from app.decision_engine import RecoveryAction


@pytest.fixture
def valid_recommendation() -> LLMRecoveryRecommendation:
    """Create a sample valid LLMRecoveryRecommendation fixture."""
    return LLMRecoveryRecommendation(
        recommended_action=RecoveryAction.PAYMENT_LINK,
        confidence=0.85,
        reasoning="High probability of customer engagement via interactive payment link.",
        key_factors=["2fa_mandate_requires_interaction", "prior_link_success"],
        referenced_case_ids=["case_1", "case_2"],
    )


@pytest.fixture
def valid_messages() -> list[dict[str, str]]:
    """Create a standard valid 2-message chat sequence."""
    return [
        {"role": "system", "content": "You are Revora's recovery engine."},
        {"role": "user", "content": "Analyze recovery scenario for payment 123."},
    ]


# ============================================================================
# 1. Protocol & Type Safety Tests
# ============================================================================


def test_provider_protocol_shape(valid_recommendation):
    """Verify MockLLMProvider satisfies runtime Protocol check for LLMProvider."""
    provider = MockLLMProvider(recommendation=valid_recommendation)
    assert isinstance(provider, LLMProvider)
    assert hasattr(provider, "generate")
    assert callable(provider.generate)


# ============================================================================
# 2. Mock Provider Core Generation & Determinism
# ============================================================================


@pytest.mark.anyio
async def test_mock_provider_returns_recommendation(
    valid_recommendation, valid_messages
):
    """Verify configured valid LLMRecoveryRecommendation is returned upon generation."""
    provider = MockLLMProvider(recommendation=valid_recommendation)
    result = await provider.generate(valid_messages)

    assert result == valid_recommendation
    assert result.recommended_action == RecoveryAction.PAYMENT_LINK
    assert result.confidence == 0.85
    assert (
        result.reasoning
        == "High probability of customer engagement via interactive payment link."
    )


@pytest.mark.anyio
async def test_mock_provider_is_deterministic(valid_recommendation, valid_messages):
    """Verify identical input messages produce identical recommendations across calls."""
    provider = MockLLMProvider(recommendation=valid_recommendation)

    res1 = await provider.generate(valid_messages)
    res2 = await provider.generate(valid_messages)
    res3 = await provider.generate(valid_messages)

    assert res1 == res2 == res3 == valid_recommendation


@pytest.mark.anyio
async def test_mock_provider_records_messages(valid_recommendation, valid_messages):
    """Verify message recording in recorded_messages and last_messages."""
    provider = MockLLMProvider(
        recommendation=valid_recommendation, record_messages=True
    )
    assert provider.recorded_messages == []
    assert provider.last_messages is None

    await provider.generate(valid_messages)

    assert len(provider.recorded_messages) == 1
    assert provider.recorded_messages[0] == valid_messages
    assert provider.last_messages == valid_messages

    second_batch = [{"role": "user", "content": "Second scenario"}]
    await provider.generate(second_batch)

    assert len(provider.recorded_messages) == 2
    assert provider.recorded_messages[1] == second_batch
    assert provider.last_messages == second_batch


@pytest.mark.anyio
async def test_mock_provider_disable_recording(valid_recommendation, valid_messages):
    """Verify record_messages=False leaves recorded_messages empty."""
    provider = MockLLMProvider(
        recommendation=valid_recommendation, record_messages=False
    )
    await provider.generate(valid_messages)

    assert provider.recorded_messages == []
    assert provider.last_messages is None


# ============================================================================
# 3. Input Validation Boundary Tests
# ============================================================================


def test_validate_chat_messages_non_sequence():
    """Verify non-sequence inputs are rejected with TypeError."""
    with pytest.raises(TypeError, match="Expected Sequence"):
        validate_chat_messages(None)  # type: ignore

    with pytest.raises(TypeError, match="Expected Sequence"):
        validate_chat_messages("not a sequence of mappings")  # type: ignore


@pytest.mark.anyio
async def test_provider_rejects_empty_messages(valid_recommendation):
    """Verify empty message sequence raises ValueError."""
    provider = MockLLMProvider(recommendation=valid_recommendation)

    with pytest.raises(ValueError, match="Messages sequence cannot be empty"):
        await provider.generate([])


@pytest.mark.anyio
async def test_provider_rejects_malformed_messages(valid_recommendation):
    """Verify non-mapping elements or missing keys raise appropriate errors."""
    provider = MockLLMProvider(recommendation=valid_recommendation)

    with pytest.raises(TypeError, match="must be a mapping"):
        await provider.generate(["not a dict"])  # type: ignore

    with pytest.raises(ValueError, match="must contain both 'role' and 'content'"):
        await provider.generate([{"role": "user"}])

    with pytest.raises(ValueError, match="must contain both 'role' and 'content'"):
        await provider.generate([{"content": "hello"}])


@pytest.mark.anyio
async def test_provider_rejects_invalid_message_roles(valid_recommendation):
    """Verify unsupported message roles raise ValueError."""
    provider = MockLLMProvider(recommendation=valid_recommendation)

    with pytest.raises(ValueError, match="unsupported role 'admin'"):
        await provider.generate([{"role": "admin", "content": "hello"}])

    with pytest.raises(ValueError, match="invalid or empty role"):
        await provider.generate([{"role": "", "content": "hello"}])

    with pytest.raises(ValueError, match="invalid or empty role"):
        await provider.generate([{"role": "   ", "content": "hello"}])


@pytest.mark.anyio
async def test_provider_rejects_invalid_content(valid_recommendation):
    """Verify empty or non-string message content is rejected."""
    provider = MockLLMProvider(recommendation=valid_recommendation)

    with pytest.raises(TypeError, match="must be a string"):
        await provider.generate([{"role": "user", "content": 12345}])  # type: ignore

    with pytest.raises(ValueError, match="cannot be empty or whitespace-only"):
        await provider.generate([{"role": "user", "content": ""}])

    with pytest.raises(ValueError, match="cannot be empty or whitespace-only"):
        await provider.generate([{"role": "user", "content": "   \n\t  "}])


# ============================================================================
# 4. Response Contract & Separation Tests
# ============================================================================


@pytest.mark.anyio
async def test_mock_provider_returns_validated_contract(
    valid_recommendation, valid_messages
):
    """Verify return object strictly conforms to LLMRecoveryRecommendation schema."""
    provider = MockLLMProvider(recommendation=valid_recommendation)
    result = await provider.generate(valid_messages)

    assert isinstance(result, LLMRecoveryRecommendation)
    assert isinstance(result.recommended_action, RecoveryAction)
    assert 0.0 <= result.confidence <= 1.0
    assert len(result.reasoning) > 0


@pytest.mark.anyio
async def test_provider_does_not_return_agent_decision_result(
    valid_recommendation, valid_messages
):
    """Verify provider returns candidate recommendation and does NOT return AgentDecisionResult."""
    provider = MockLLMProvider(recommendation=valid_recommendation)
    result = await provider.generate(valid_messages)

    assert isinstance(result, LLMRecoveryRecommendation)
    assert not isinstance(result, AgentDecisionResult)


def test_mock_provider_rejects_invalid_recommendation_type():
    """Verify constructing MockLLMProvider with non-LLMRecoveryRecommendation raises TypeError."""
    with pytest.raises(
        TypeError,
        match="Expected recommendation to be an instance of LLMRecoveryRecommendation",
    ):
        MockLLMProvider(recommendation={"recommended_action": "payment_link"})  # type: ignore


# ============================================================================
# 5. Error Handling & Failure Simulation Tests
# ============================================================================


@pytest.mark.anyio
async def test_provider_error_contract():
    """Verify exception inheritance and explicit hierarchy."""
    assert issubclass(LLMResponseValidationError, LLMProviderError)
    assert issubclass(LLMProviderError, Exception)


@pytest.mark.anyio
async def test_mock_provider_failure_is_explicit(valid_recommendation, valid_messages):
    """Verify should_fail=True raises LLMProviderError."""
    provider = MockLLMProvider(recommendation=valid_recommendation, should_fail=True)

    with pytest.raises(LLMProviderError, match="configured execution failure"):
        await provider.generate(valid_messages)


@pytest.mark.anyio
async def test_mock_provider_custom_exception_propagation(
    valid_recommendation, valid_messages
):
    """Verify custom failure exception is propagated directly."""
    custom_err = LLMResponseValidationError("Malformed response payload.")
    provider = MockLLMProvider(
        recommendation=valid_recommendation,
        should_fail=True,
        failure_exception=custom_err,
    )

    with pytest.raises(LLMResponseValidationError, match="Malformed response payload"):
        await provider.generate(valid_messages)


# ============================================================================
# 6. Integration & Isolation Tests
# ============================================================================


@pytest.mark.anyio
async def test_mock_provider_integration_with_build_agent_messages(
    valid_recommendation,
):
    """Verify end-to-end integration from AgentDecisionPromptContext -> build_agent_messages -> provider."""
    ctx = AgentDecisionPromptContext(
        current_payment={"amount": 800.0, "currency": "INR", "payment_method": "card"},
        customer_profile={"historical_success_rate": 0.9},
        allowed_actions=["payment_link", "change_payment_method"],
    )

    messages = build_agent_messages(ctx)
    provider = MockLLMProvider(recommendation=valid_recommendation)

    recommendation = await provider.generate(messages)
    assert recommendation == valid_recommendation
    assert provider.last_messages == messages


@pytest.mark.anyio
async def test_provider_does_not_mutate_input_messages(
    valid_recommendation, valid_messages
):
    """Verify input message structures are not mutated during generation or recording."""
    original_messages = [dict(m) for m in valid_messages]
    provider = MockLLMProvider(
        recommendation=valid_recommendation, record_messages=True
    )

    await provider.generate(valid_messages)

    assert valid_messages == original_messages


@pytest.mark.anyio
async def test_recorded_messages_mutation_isolation(
    valid_recommendation, valid_messages
):
    """Regression test for Finding 1: mutating returned recorded_messages does not alter internal state."""
    provider = MockLLMProvider(
        recommendation=valid_recommendation, record_messages=True
    )
    await provider.generate(valid_messages)

    retrieved_recorded = provider.recorded_messages
    assert len(retrieved_recorded) == 1
    assert len(retrieved_recorded[0]) == 2

    # Mutate the dictionary and list returned from property
    retrieved_recorded[0][0]["content"] = "MUTATED_CONTENT"
    retrieved_recorded[0].append({"role": "user", "content": "NEW_INJECTED"})

    # Verify provider's internal state remains untouched
    fresh_recorded = provider.recorded_messages
    assert fresh_recorded[0][0]["content"] == valid_messages[0]["content"]
    assert len(fresh_recorded[0]) == len(valid_messages)


@pytest.mark.anyio
async def test_last_messages_mutation_isolation(valid_recommendation, valid_messages):
    """Regression test for Finding 1: mutating returned last_messages does not alter internal state."""
    provider = MockLLMProvider(
        recommendation=valid_recommendation, record_messages=True
    )
    await provider.generate(valid_messages)

    retrieved_last = provider.last_messages
    assert retrieved_last is not None

    # Mutate dictionary in returned list
    retrieved_last[0]["role"] = "MUTATED_ROLE"
    retrieved_last.pop()

    # Verify provider's internal state remains untouched
    fresh_last = provider.last_messages
    assert fresh_last is not None
    assert fresh_last[0]["role"] == valid_messages[0]["role"]
    assert len(fresh_last) == len(valid_messages)


def test_mock_provider_rejects_non_llm_provider_error_failure_exception(
    valid_recommendation,
):
    """Regression test for Finding 2: failure_exception must be an instance of LLMProviderError."""
    with pytest.raises(
        TypeError,
        match="Expected failure_exception to be an instance of LLMProviderError",
    ):
        MockLLMProvider(
            recommendation=valid_recommendation,
            should_fail=True,
            failure_exception=RuntimeError("Standard runtime error"),  # type: ignore
        )

    with pytest.raises(
        TypeError,
        match="Expected failure_exception to be an instance of LLMProviderError",
    ):
        MockLLMProvider(
            recommendation=valid_recommendation,
            should_fail=True,
            failure_exception=ValueError("Invalid value"),  # type: ignore
        )


@pytest.mark.anyio
async def test_mock_provider_accepts_valid_llm_provider_error_subclasses(
    valid_recommendation, valid_messages
):
    """Regression test for Finding 2: failure_exception accepts LLMProviderError and its subtypes."""
    base_err = LLMProviderError("Base provider failure")
    provider_base = MockLLMProvider(
        recommendation=valid_recommendation,
        should_fail=True,
        failure_exception=base_err,
    )
    with pytest.raises(LLMProviderError, match="Base provider failure"):
        await provider_base.generate(valid_messages)

    validation_err = LLMResponseValidationError("Response schema validation failure")
    provider_val = MockLLMProvider(
        recommendation=valid_recommendation,
        should_fail=True,
        failure_exception=validation_err,
    )
    with pytest.raises(
        LLMResponseValidationError, match="Response schema validation failure"
    ):
        await provider_val.generate(valid_messages)
