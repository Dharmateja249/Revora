"""
Unit tests for Revora OpenAILLMProvider and RealLLMProvider foundation.

Verifies:
1. Protocol conformance and substitution in AgentOrchestrator.
2. Structured output parsing and recommendation generation.
3. Message contract validation (empty, malformed, unsupported roles).
4. Exception mapping for authentication, rate limits, timeouts, connection errors, and status failures.
5. Response validation for truncation, model refusal, missing/null parsed content, and schema violations.
6. Configuration via constructor args, Settings, and fallback environment variables.
7. Security safeguards ensuring credentials are never exposed in repr or error messages.
"""

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import httpx
import openai
import pytest
from app.agent.openai_provider import (
    OpenAILLMProvider,
    RealLLMProvider,
)
from app.agent.orchestrator import AgentOrchestrator
from app.agent.provider import (
    LLMAuthenticationError,
    LLMConnectionError,
    LLMProvider,
    LLMProviderError,
    LLMRateLimitError,
    LLMResponseValidationError,
    LLMTimeoutError,
)
from app.agent.schemas import LLMRecoveryRecommendation
from app.config import Settings
from app.decision_engine import RecoveryAction
from pydantic import ValidationError

# ============================================================================
# Test Fixtures & Mocks
# ============================================================================


@pytest.fixture
def sample_recommendation() -> LLMRecoveryRecommendation:
    """Fixture providing a valid LLMRecoveryRecommendation."""
    return LLMRecoveryRecommendation(
        recommended_action=RecoveryAction.PAYMENT_LINK,
        confidence=0.88,
        reasoning="Customer requires interactive authentication link for payment completion.",
        key_factors=["two_factor_auth_required", "prior_link_success"],
        referenced_case_ids=["case_rec_101", "case_rec_102"],
    )


@pytest.fixture
def sample_messages() -> list[dict[str, str]]:
    """Fixture providing valid chat messages."""
    return [
        {
            "role": "system",
            "content": "You are Revora's recovery agent reasoning engine.",
        },
        {
            "role": "user",
            "content": "Evaluate failed payment scenario for customer cust_123.",
        },
    ]


def _build_mock_chat_completion(
    parsed: Any = None,
    content: str | None = None,
    refusal: str | None = None,
    finish_reason: str = "stop",
) -> MagicMock:
    """Helper creating a mock response structure matching OpenAI beta parse output."""
    message = MagicMock()
    message.parsed = parsed
    message.content = content
    message.refusal = refusal

    choice = MagicMock()
    choice.message = message
    choice.finish_reason = finish_reason

    response = MagicMock()
    response.choices = [choice]
    return response


def _create_mock_client(
    return_response: Any = None, side_effect: Exception | None = None
) -> MagicMock:
    """Helper building a mock AsyncOpenAI client."""
    client = MagicMock(spec=openai.AsyncOpenAI)
    parse_mock = AsyncMock()
    if side_effect is not None:
        parse_mock.side_effect = side_effect
    else:
        parse_mock.return_value = return_response
    client.beta.chat.completions.parse = parse_mock
    return client


# ============================================================================
# 1. Protocol Conformance & Orchestrator Substitution
# ============================================================================


def test_protocol_conformance(sample_recommendation):
    """Verify OpenAILLMProvider and RealLLMProvider satisfy LLMProvider runtime Protocol check."""
    mock_client = _create_mock_client()
    provider = OpenAILLMProvider(api_key="sk-test", client=mock_client)

    assert isinstance(provider, LLMProvider)
    assert hasattr(provider, "generate")
    assert callable(provider.generate)

    real_provider = RealLLMProvider(api_key="sk-test", client=mock_client)
    assert isinstance(real_provider, LLMProvider)
    assert RealLLMProvider is OpenAILLMProvider


def test_substitution_in_orchestrator():
    """Verify OpenAILLMProvider can be injected into AgentOrchestrator without error."""
    mock_client = _create_mock_client()
    provider = OpenAILLMProvider(api_key="sk-test", client=mock_client)

    orchestrator = AgentOrchestrator(provider=provider)
    assert orchestrator.provider is provider
    assert orchestrator.provider.provider_name == "openai"


# ============================================================================
# 2. Success Path & Structured Output
# ============================================================================


@pytest.mark.anyio
async def test_generate_success(sample_recommendation, sample_messages):
    """Verify successful generation returns validated LLMRecoveryRecommendation."""
    mock_response = _build_mock_chat_completion(parsed=sample_recommendation)
    mock_client = _create_mock_client(return_response=mock_response)

    provider = OpenAILLMProvider(
        api_key="sk-test",
        model="gpt-4o-mini",
        timeout_seconds=25.0,
        client=mock_client,
    )

    result = await provider.generate(sample_messages)

    assert result == sample_recommendation
    assert result.recommended_action == RecoveryAction.PAYMENT_LINK
    assert result.confidence == 0.88
    assert "interactive authentication" in result.reasoning

    # Verify parse was called with structured output arguments
    mock_client.beta.chat.completions.parse.assert_awaited_once_with(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": "You are Revora's recovery agent reasoning engine.",
            },
            {
                "role": "user",
                "content": "Evaluate failed payment scenario for customer cust_123.",
            },
        ],
        response_format=LLMRecoveryRecommendation,
        timeout=25.0,
    )


@pytest.mark.anyio
async def test_generate_success_from_dict_parsed(sample_messages):
    """Verify parsed dict output is properly validated into LLMRecoveryRecommendation."""
    dict_payload = {
        "recommended_action": "retry_payment",
        "confidence": 0.95,
        "reasoning": "Temporary bank downtime resolved; retry recommended.",
        "key_factors": ["soft_decline", "bank_online"],
        "referenced_case_ids": ["case_1"],
    }
    mock_response = _build_mock_chat_completion(parsed=dict_payload)
    mock_client = _create_mock_client(return_response=mock_response)

    provider = OpenAILLMProvider(api_key="sk-test", client=mock_client)
    result = await provider.generate(sample_messages)

    assert isinstance(result, LLMRecoveryRecommendation)
    assert result.recommended_action == RecoveryAction.RETRY_PAYMENT
    assert result.confidence == 0.95
    assert result.key_factors == ("soft_decline", "bank_online")


# ============================================================================
# 3. Message Contract Input Validation
# ============================================================================


@pytest.mark.anyio
async def test_input_validation_empty_messages():
    """Verify empty message list raises ValueError."""
    mock_client = _create_mock_client()
    provider = OpenAILLMProvider(api_key="sk-test", client=mock_client)

    with pytest.raises(ValueError, match="Messages sequence cannot be empty"):
        await provider.generate([])


@pytest.mark.anyio
async def test_input_validation_invalid_types():
    """Verify non-sequence or non-mapping elements raise TypeError."""
    mock_client = _create_mock_client()
    provider = OpenAILLMProvider(api_key="sk-test", client=mock_client)

    with pytest.raises(TypeError, match="Expected Sequence of message mappings"):
        await provider.generate("not a sequence")  # type: ignore[arg-type]

    with pytest.raises(TypeError, match="must be a mapping"):
        await provider.generate(["not a mapping"])  # type: ignore[list-item]


@pytest.mark.anyio
async def test_input_validation_missing_keys():
    """Verify missing role or content keys raise ValueError."""
    mock_client = _create_mock_client()
    provider = OpenAILLMProvider(api_key="sk-test", client=mock_client)

    with pytest.raises(ValueError, match="must contain both 'role' and 'content'"):
        await provider.generate([{"role": "user"}])

    with pytest.raises(ValueError, match="must contain both 'role' and 'content'"):
        await provider.generate([{"content": "hello"}])


@pytest.mark.anyio
async def test_input_validation_unsupported_role():
    """Verify unsupported roles raise ValueError."""
    mock_client = _create_mock_client()
    provider = OpenAILLMProvider(api_key="sk-test", client=mock_client)

    with pytest.raises(ValueError, match="unsupported role 'admin'"):
        await provider.generate([{"role": "admin", "content": "perform recovery"}])


@pytest.mark.anyio
async def test_input_validation_empty_content():
    """Verify empty or whitespace-only content raises ValueError."""
    mock_client = _create_mock_client()
    provider = OpenAILLMProvider(api_key="sk-test", client=mock_client)

    with pytest.raises(ValueError, match="cannot be empty or whitespace-only"):
        await provider.generate([{"role": "user", "content": "   "}])


# ============================================================================
# 4. Provider & Vendor Failure Mappings
# ============================================================================


@pytest.mark.anyio
async def test_failure_authentication_error(sample_messages):
    """Verify openai.AuthenticationError maps to LLMAuthenticationError."""
    req = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
    resp = httpx.Response(401, request=req)
    side_effect = openai.AuthenticationError(
        "Incorrect API key provided.", response=resp, body=None
    )

    mock_client = _create_mock_client(side_effect=side_effect)
    provider = OpenAILLMProvider(api_key="sk-test", client=mock_client)

    with pytest.raises(LLMAuthenticationError) as exc_info:
        await provider.generate(sample_messages)

    assert isinstance(exc_info.value, LLMProviderError)
    assert "OpenAI authentication failed" in str(exc_info.value)


@pytest.mark.anyio
async def test_failure_permission_denied(sample_messages):
    """Verify openai.PermissionDeniedError maps to LLMAuthenticationError."""
    req = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
    resp = httpx.Response(403, request=req)
    side_effect = openai.PermissionDeniedError(
        "Country not supported.", response=resp, body=None
    )

    mock_client = _create_mock_client(side_effect=side_effect)
    provider = OpenAILLMProvider(api_key="sk-test", client=mock_client)

    with pytest.raises(LLMAuthenticationError) as exc_info:
        await provider.generate(sample_messages)

    assert "permission denied" in str(exc_info.value).lower()


@pytest.mark.anyio
async def test_failure_rate_limit(sample_messages):
    """Verify openai.RateLimitError maps to LLMRateLimitError."""
    req = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
    resp = httpx.Response(429, request=req)
    side_effect = openai.RateLimitError(
        "Quota exceeded for the current month.", response=resp, body=None
    )

    mock_client = _create_mock_client(side_effect=side_effect)
    provider = OpenAILLMProvider(api_key="sk-test", client=mock_client)

    with pytest.raises(LLMRateLimitError) as exc_info:
        await provider.generate(sample_messages)

    assert isinstance(exc_info.value, LLMProviderError)
    assert "rate limit or quota exceeded" in str(exc_info.value)


@pytest.mark.anyio
async def test_failure_api_timeout(sample_messages):
    """Verify openai.APITimeoutError and asyncio.TimeoutError map to LLMTimeoutError."""
    req = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
    side_effect = openai.APITimeoutError(request=req)

    mock_client = _create_mock_client(side_effect=side_effect)
    provider = OpenAILLMProvider(api_key="sk-test", client=mock_client)

    with pytest.raises(LLMTimeoutError) as exc_info:
        await provider.generate(sample_messages)

    assert isinstance(exc_info.value, LLMProviderError)
    assert "timed out" in str(exc_info.value)

    # Also test standard asyncio.TimeoutError
    mock_client_async_timeout = _create_mock_client(side_effect=asyncio.TimeoutError())
    provider2 = OpenAILLMProvider(api_key="sk-test", client=mock_client_async_timeout)
    with pytest.raises(LLMTimeoutError):
        await provider2.generate(sample_messages)


@pytest.mark.anyio
async def test_failure_connection_error(sample_messages):
    """Verify openai.APIConnectionError and httpx.RequestError map to LLMConnectionError."""
    req = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
    side_effect = openai.APIConnectionError(request=req)

    mock_client = _create_mock_client(side_effect=side_effect)
    provider = OpenAILLMProvider(api_key="sk-test", client=mock_client)

    with pytest.raises(LLMConnectionError) as exc_info:
        await provider.generate(sample_messages)

    assert isinstance(exc_info.value, LLMProviderError)
    assert "connection error" in str(exc_info.value)

    # Also test httpx.ConnectError
    mock_client_httpx = _create_mock_client(side_effect=httpx.ConnectError("Refused"))
    provider2 = OpenAILLMProvider(api_key="sk-test", client=mock_client_httpx)
    with pytest.raises(LLMConnectionError):
        await provider2.generate(sample_messages)


@pytest.mark.anyio
async def test_failure_status_error(sample_messages):
    """Verify 500 status errors map to LLMProviderError."""
    req = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
    resp = httpx.Response(500, request=req)
    side_effect = openai.InternalServerError(
        "Internal server error", response=resp, body=None
    )

    mock_client = _create_mock_client(side_effect=side_effect)
    provider = OpenAILLMProvider(api_key="sk-test", client=mock_client)

    with pytest.raises(LLMProviderError) as exc_info:
        await provider.generate(sample_messages)

    assert "OpenAI API status error (500)" in str(exc_info.value)


@pytest.mark.anyio
async def test_failure_unexpected_exception(sample_messages):
    """Verify unexpected runtime exceptions map to LLMProviderError and do not leak."""
    side_effect = RuntimeError("Kernel failure")
    mock_client = _create_mock_client(side_effect=side_effect)
    provider = OpenAILLMProvider(api_key="sk-test", client=mock_client)

    with pytest.raises(LLMProviderError) as exc_info:
        await provider.generate(sample_messages)

    assert "Unexpected provider failure" in str(exc_info.value)


# ============================================================================
# 5. Response Validation & Malformed Outputs
# ============================================================================


@pytest.mark.anyio
async def test_response_validation_empty_choices(sample_messages):
    """Verify empty choices list raises LLMResponseValidationError."""
    empty_resp = MagicMock()
    empty_resp.choices = []

    mock_client = _create_mock_client(return_response=empty_resp)
    provider = OpenAILLMProvider(api_key="sk-test", client=mock_client)

    with pytest.raises(
        LLMResponseValidationError, match="empty response with no choices"
    ):
        await provider.generate(sample_messages)


@pytest.mark.anyio
async def test_response_validation_truncated_length(sample_messages):
    """Verify truncated response due to token length limit raises LLMResponseValidationError."""
    mock_resp = _build_mock_chat_completion(parsed=None, finish_reason="length")
    mock_client = _create_mock_client(return_response=mock_resp)
    provider = OpenAILLMProvider(api_key="sk-test", client=mock_client)

    with pytest.raises(
        LLMResponseValidationError, match="truncated due to maximum token limit"
    ):
        await provider.generate(sample_messages)


@pytest.mark.anyio
async def test_response_validation_model_refusal(sample_messages):
    """Verify model refusal raises LLMResponseValidationError with refusal reason."""
    mock_resp = _build_mock_chat_completion(
        refusal="I cannot assist with debt recovery."
    )
    mock_client = _create_mock_client(return_response=mock_resp)
    provider = OpenAILLMProvider(api_key="sk-test", client=mock_client)

    with pytest.raises(
        LLMResponseValidationError, match="refused the prompt: I cannot assist"
    ):
        await provider.generate(sample_messages)


@pytest.mark.anyio
async def test_response_validation_null_parsed_with_no_content(sample_messages):
    """Verify null parsed object without content raises LLMResponseValidationError."""
    mock_resp = _build_mock_chat_completion(parsed=None, content=None)
    mock_client = _create_mock_client(return_response=mock_resp)
    provider = OpenAILLMProvider(api_key="sk-test", client=mock_client)

    with pytest.raises(
        LLMResponseValidationError, match="returned null structured recommendation"
    ):
        await provider.generate(sample_messages)


@pytest.mark.anyio
async def test_response_validation_unparsed_prose_content(sample_messages):
    """Verify model returning raw prose instead of structured model raises LLMResponseValidationError."""
    mock_resp = _build_mock_chat_completion(
        parsed=None, content="I recommend retry_payment because..."
    )
    mock_client = _create_mock_client(return_response=mock_resp)
    provider = OpenAILLMProvider(api_key="sk-test", client=mock_client)

    with pytest.raises(LLMResponseValidationError, match="returned unparsed content"):
        await provider.generate(sample_messages)


@pytest.mark.anyio
async def test_response_validation_invalid_schema_fields(sample_messages):
    """Verify malformed schema fields (e.g. invalid action or out-of-bounds confidence) raise LLMResponseValidationError."""
    # Invalid recovery action
    bad_payload = {
        "recommended_action": "invalid_unknown_action",
        "confidence": 0.9,
        "reasoning": "Test",
    }
    mock_resp = _build_mock_chat_completion(parsed=bad_payload)
    mock_client = _create_mock_client(return_response=mock_resp)
    provider = OpenAILLMProvider(api_key="sk-test", client=mock_client)

    with pytest.raises(LLMResponseValidationError, match="validation"):
        await provider.generate(sample_messages)

    # Confidence out of bounds
    bad_payload2 = {
        "recommended_action": "retry_payment",
        "confidence": 1.5,
        "reasoning": "Confidence too high",
    }
    mock_resp2 = _build_mock_chat_completion(parsed=bad_payload2)
    mock_client2 = _create_mock_client(return_response=mock_resp2)
    provider2 = OpenAILLMProvider(api_key="sk-test", client=mock_client2)

    with pytest.raises(LLMResponseValidationError, match="validation"):
        await provider2.generate(sample_messages)


@pytest.mark.anyio
async def test_response_validation_pydantic_validation_error_from_parser(
    sample_messages,
):
    """Verify pydantic.ValidationError raised by parse() is mapped to LLMResponseValidationError."""
    try:
        LLMRecoveryRecommendation.model_validate({"confidence": 2.5})
    except ValidationError as err:
        validation_error = err

    mock_client = _create_mock_client(side_effect=validation_error)
    provider = OpenAILLMProvider(api_key="sk-test", client=mock_client)

    with pytest.raises(LLMResponseValidationError) as exc_info:
        await provider.generate(sample_messages)

    assert isinstance(exc_info.value, LLMProviderError)
    assert "schema validation failed" in str(exc_info.value).lower()


# ============================================================================
# 6. Configuration & Credential Safety
# ============================================================================


def test_configuration_explicit_arguments():
    """Verify constructor accepts and preserves explicit configuration parameters."""
    mock_client = _create_mock_client()
    provider = OpenAILLMProvider(
        api_key="sk-explicit-12345",
        model="gpt-4o",
        timeout_seconds=45.0,
        client=mock_client,
    )

    assert provider.model_name == "gpt-4o"
    assert provider.timeout_seconds == 45.0
    assert provider.provider_name == "openai"


def test_configuration_settings_resolution(monkeypatch):
    """Verify provider resolves API key and model from Settings when not explicitly supplied."""
    test_settings = Settings(
        OPENAI_API_KEY="sk-settings-key-999",
        OPENAI_MODEL="gpt-4o-custom",
        OPENAI_TIMEOUT_SECONDS=15.0,
    )
    monkeypatch.setattr("app.agent.openai_provider.get_settings", lambda: test_settings)

    mock_client = _create_mock_client()
    provider = OpenAILLMProvider(client=mock_client)

    assert provider.model_name == "gpt-4o-custom"
    assert provider.timeout_seconds == 15.0


def test_configuration_llm_prefix_fallback(monkeypatch):
    """Verify generic LLM_* settings are used when OPENAI_* settings are absent."""
    test_settings = Settings(
        OPENAI_API_KEY=None,
        LLM_API_KEY="sk-generic-llm-key",
        LLM_MODEL="gpt-4o-generic",
        LLM_TIMEOUT_SECONDS=20.0,
    )
    monkeypatch.setattr("app.agent.openai_provider.get_settings", lambda: test_settings)

    mock_client = _create_mock_client()
    provider = OpenAILLMProvider(client=mock_client)

    assert provider.model_name == "gpt-4o-generic"
    assert provider.timeout_seconds == 20.0


def test_configuration_missing_api_key_raises(monkeypatch):
    """Verify missing API key raises clear ValueError when client is not injected."""
    test_settings = Settings(
        OPENAI_API_KEY=None,
        LLM_API_KEY=None,
    )
    monkeypatch.setattr("app.agent.openai_provider.get_settings", lambda: test_settings)

    with pytest.raises(ValueError, match="OpenAI API key must be provided"):
        OpenAILLMProvider()


def test_configuration_empty_or_whitespace_key_raises(monkeypatch):
    """Verify blank or whitespace-only API key raises ValueError."""
    test_settings = Settings(OPENAI_API_KEY=None, LLM_API_KEY=None)
    monkeypatch.setattr("app.agent.openai_provider.get_settings", lambda: test_settings)

    with pytest.raises(ValueError, match="OpenAI API key must be provided"):
        OpenAILLMProvider(api_key="   ")


def test_configuration_invalid_timeout_raises():
    """Verify non-positive timeout values raise ValueError."""
    mock_client = _create_mock_client()

    with pytest.raises(ValueError, match="timeout_seconds must be a positive number"):
        OpenAILLMProvider(api_key="sk-test", timeout_seconds=0.0, client=mock_client)

    with pytest.raises(ValueError, match="timeout_seconds must be a positive number"):
        OpenAILLMProvider(api_key="sk-test", timeout_seconds=-5.0, client=mock_client)


def test_security_credentials_not_in_repr():
    """Verify API key is never exposed in repr string."""
    secret_key = "sk-super-secret-production-token-12345"
    mock_client = _create_mock_client()
    provider = OpenAILLMProvider(api_key=secret_key, client=mock_client)

    repr_str = repr(provider)
    assert secret_key not in repr_str
    assert "sk-super-secret" not in repr_str
    assert "OpenAILLMProvider" in repr_str
    assert "model=" in repr_str


@pytest.mark.anyio
async def test_security_credentials_not_in_error_messages(sample_messages):
    """Verify API key is never echoed into raised error messages."""
    secret_key = "sk-super-secret-token"
    req = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
    resp = httpx.Response(401, request=req)
    side_effect = openai.AuthenticationError(
        f"Unauthorized key {secret_key}",
        response=resp,
        body=None,
    )

    mock_client = _create_mock_client(side_effect=side_effect)
    provider = OpenAILLMProvider(api_key=secret_key, client=mock_client)

    with pytest.raises(LLMAuthenticationError) as exc_info:
        await provider.generate(sample_messages)

    assert secret_key not in str(exc_info.value)
