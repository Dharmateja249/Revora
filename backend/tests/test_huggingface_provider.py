"""
Unit Tests for Revora Hugging Face LLM Provider (HuggingFaceLLMProvider).

Verifies:
1. Successful response parsing into validated LLMRecoveryRecommendation domain instances.
2. Structured JSON response handling (including markdown fences).
3. Exception mapping:
   - HfHubHTTPError (401/403) -> LLMAuthenticationError
   - HfHubHTTPError (429) / OverloadedError -> LLMRateLimitError
   - InferenceTimeoutError / TimeoutError -> LLMTimeoutError
   - HfHubHTTPError (502/503) / httpx.RequestError / InferenceEndpointError -> LLMConnectionError
   - Malformed / Schema-violating response -> LLMResponseValidationError
4. Missing HF token raises clear ValueError (no silent fallback).
5. Factory resolution: create_llm_provider(provider="huggingface") correctly builds HuggingFaceLLMProvider.
6. Unsupported provider names fail explicitly with LLMProviderConfigurationError.
7. Secret masking: tokens are sanitized from repr(), logs, and exception messages.
8. Existing mock and OpenAI provider behavior remains unchanged.
"""

import json
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import httpx
import pytest
from huggingface_hub import AsyncInferenceClient
from huggingface_hub.errors import (
    HfHubHTTPError,
    InferenceEndpointError,
    InferenceTimeoutError,
    OverloadedError,
)

from app.agent.context_builder import AgentContextBuilder
from app.agent.factory import create_llm_provider
from app.agent.huggingface_provider import HuggingFaceLLMProvider
from app.agent.orchestrator import AgentOrchestrator
from app.agent.provider import (
    LLMAuthenticationError,
    LLMConnectionError,
    LLMProviderConfigurationError,
    LLMRateLimitError,
    LLMResponseValidationError,
    LLMTimeoutError,
    MockLLMProvider,
)
from app.agent.schemas import LLMRecoveryRecommendation
from app.config import Settings, get_settings
from app.context import (
    CustomerContext,
    CustomerRecoveryContext,
    PaymentContext,
    RecoveryOpportunityContext,
)
from app.decision_engine import RecoveryAction
from app.historical_retrieval import HistoricalCase
from app.policies.registry import RZP_CUSTOMER_AUTH_2FA_REQUIRED_RULE
from app.policies.schemas import RecoveryPolicyContext

# ============================================================================
# Helpers
# ============================================================================


def make_mock_chat_completion_output(
    action: str = "payment_link",
    confidence: float = 0.93,
    reasoning: str = "Card authentication failure requires interactive payment link.",
    key_factors: list[str] | None = None,
    referenced_case_ids: list[str] | None = None,
    wrap_in_markdown: bool = False,
) -> MagicMock:
    """Helper to construct a mock ChatCompletionOutput object matching huggingface_hub."""
    payload = {
        "recommended_action": action,
        "confidence": confidence,
        "reasoning": reasoning,
        "key_factors": key_factors or ["customer_auth_failed_otp_timeout"],
        "referenced_case_ids": referenced_case_ids or ["case_1"],
    }
    raw_text = json.dumps(payload)
    if wrap_in_markdown:
        raw_text = f"```json\n{raw_text}\n```"

    mock_choice = MagicMock()
    mock_choice.message.content = raw_text
    mock_output = MagicMock()
    mock_output.choices = [mock_choice]
    return mock_output


# ============================================================================
# 1. Success & Structured Parsing Tests
# ============================================================================


@pytest.mark.anyio
async def test_huggingface_provider_success_structured_output():
    """Verify HuggingFaceLLMProvider generates and validates LLMRecoveryRecommendation."""
    mock_client = AsyncMock(spec=AsyncInferenceClient)
    mock_client.chat_completion.return_value = make_mock_chat_completion_output(
        action="payment_link",
        confidence=0.91,
        reasoning="Payment link enables 2FA authorization.",
        key_factors=["auth_timeout"],
        referenced_case_ids=["case_1"],
    )

    provider = HuggingFaceLLMProvider(
        token="hf_mock_token_12345",
        model="Qwen/Qwen3-32B",
        client=mock_client,
    )

    messages = [
        {"role": "system", "content": "You are Revora recovery engine."},
        {"role": "user", "content": "Analyze payment failure."},
    ]

    rec = await provider.generate(messages)

    assert isinstance(rec, LLMRecoveryRecommendation)
    assert rec.recommended_action == RecoveryAction.PAYMENT_LINK
    assert rec.confidence == 0.91
    assert rec.reasoning == "Payment link enables 2FA authorization."
    assert rec.key_factors == ("auth_timeout",)
    assert rec.referenced_case_ids == ("case_1",)

    # Verify call parameters
    mock_client.chat_completion.assert_called_once()
    _, kwargs = mock_client.chat_completion.call_args
    assert kwargs["model"] == "Qwen/Qwen3-32B"
    assert kwargs["response_format"] == {"type": "json_object"}
    assert kwargs["messages"][0]["role"] == "system"
    assert kwargs["messages"][1]["role"] == "user"


@pytest.mark.anyio
async def test_huggingface_provider_handles_markdown_fenced_json():
    """Verify markdown code blocks (```json ... ```) are cleanly parsed."""
    mock_client = AsyncMock(spec=AsyncInferenceClient)
    mock_client.chat_completion.return_value = make_mock_chat_completion_output(
        action="change_payment_method",
        confidence=0.85,
        reasoning="Recurring mandate expired; alternate payment method required.",
        wrap_in_markdown=True,
    )

    provider = HuggingFaceLLMProvider(token="hf_token", client=mock_client)
    rec = await provider.generate([{"role": "user", "content": "Evaluate"}])

    assert rec.recommended_action == RecoveryAction.CHANGE_PAYMENT_METHOD
    assert rec.confidence == 0.85


# ============================================================================
# 2. Error Translation Tests
# ============================================================================


@pytest.mark.anyio
@pytest.mark.parametrize("status_code", [401, 403])
async def test_huggingface_provider_maps_auth_errors(status_code: int):
    """Verify HTTP 401/403 HfHubHTTPError maps to LLMAuthenticationError."""
    mock_client = AsyncMock(spec=AsyncInferenceClient)
    mock_response = MagicMock()
    mock_response.status_code = status_code

    error = HfHubHTTPError(
        f"{status_code} Client Error: Unauthorized for url",
        response=mock_response,
    )
    error.server_message = "Invalid credentials or token expired."
    mock_client.chat_completion.side_effect = error

    provider = HuggingFaceLLMProvider(token="hf_invalid_token", client=mock_client)

    with pytest.raises(
        LLMAuthenticationError, match="Hugging Face authentication failed"
    ):
        await provider.generate([{"role": "user", "content": "Evaluate"}])


@pytest.mark.anyio
async def test_huggingface_provider_maps_rate_limit_error():
    """Verify HTTP 429 HfHubHTTPError maps to LLMRateLimitError."""
    mock_client = AsyncMock(spec=AsyncInferenceClient)
    mock_response = MagicMock()
    mock_response.status_code = 429

    error = HfHubHTTPError(
        "429 Client Error: Rate Limit Exceeded", response=mock_response
    )
    error.server_message = "Rate limit reached."
    mock_client.chat_completion.side_effect = error

    provider = HuggingFaceLLMProvider(token="hf_test_token", client=mock_client)

    with pytest.raises(
        LLMRateLimitError, match="Hugging Face rate limit or quota exceeded"
    ):
        await provider.generate([{"role": "user", "content": "Evaluate"}])


@pytest.mark.anyio
async def test_huggingface_provider_maps_overloaded_error():
    """Verify OverloadedError maps to LLMRateLimitError."""
    mock_client = AsyncMock(spec=AsyncInferenceClient)
    mock_client.chat_completion.side_effect = OverloadedError(
        "Model is currently loading or busy"
    )

    provider = HuggingFaceLLMProvider(token="hf_test_token", client=mock_client)

    with pytest.raises(
        LLMRateLimitError, match="Hugging Face model is currently overloaded"
    ):
        await provider.generate([{"role": "user", "content": "Evaluate"}])


@pytest.mark.anyio
async def test_huggingface_provider_maps_timeout_error():
    """Verify InferenceTimeoutError maps to LLMTimeoutError."""
    mock_client = AsyncMock(spec=AsyncInferenceClient)
    mock_client.chat_completion.side_effect = InferenceTimeoutError(
        "Model execution exceeded 30s"
    )

    provider = HuggingFaceLLMProvider(
        token="hf_test_token", timeout_seconds=30.0, client=mock_client
    )

    with pytest.raises(
        LLMTimeoutError, match="Hugging Face request timed out after 30.0s."
    ):
        await provider.generate([{"role": "user", "content": "Evaluate"}])


@pytest.mark.anyio
async def test_huggingface_provider_maps_connection_error():
    """Verify InferenceEndpointError / ConnectionError maps to LLMConnectionError."""
    mock_client = AsyncMock(spec=AsyncInferenceClient)
    mock_client.chat_completion.side_effect = InferenceEndpointError(
        "Endpoint unreachable"
    )

    provider = HuggingFaceLLMProvider(token="hf_test_token", client=mock_client)

    with pytest.raises(LLMConnectionError, match="Hugging Face connection error"):
        await provider.generate([{"role": "user", "content": "Evaluate"}])


@pytest.mark.anyio
async def test_huggingface_provider_maps_malformed_json_response():
    """Verify invalid JSON content maps to LLMResponseValidationError."""
    mock_client = AsyncMock(spec=AsyncInferenceClient)
    mock_output = MagicMock()
    mock_choice = MagicMock()
    mock_choice.message.content = "Plain text without valid JSON structure"
    mock_output.choices = [mock_choice]
    mock_client.chat_completion.return_value = mock_output

    provider = HuggingFaceLLMProvider(token="hf_test_token", client=mock_client)

    with pytest.raises(
        LLMResponseValidationError, match="Failed to decode structured JSON"
    ):
        await provider.generate([{"role": "user", "content": "Evaluate"}])


@pytest.mark.anyio
async def test_huggingface_provider_maps_schema_validation_error():
    """Verify JSON with missing or invalid fields maps to LLMResponseValidationError."""
    mock_client = AsyncMock(spec=AsyncInferenceClient)
    mock_output = MagicMock()
    mock_choice = MagicMock()
    # Missing required 'confidence' and 'reasoning'
    mock_choice.message.content = json.dumps(
        {"recommended_action": "invalid_action_name"}
    )
    mock_output.choices = [mock_choice]
    mock_client.chat_completion.return_value = mock_output

    provider = HuggingFaceLLMProvider(token="hf_test_token", client=mock_client)

    with pytest.raises(LLMResponseValidationError, match="schema validation"):
        await provider.generate([{"role": "user", "content": "Evaluate"}])


# ============================================================================
# 3. Security & Missing Token Invariants
# ============================================================================


def test_huggingface_provider_missing_token_fails_clearly(monkeypatch):
    """Verify that omitting HF_TOKEN raises ValueError clearly without silent fallback."""
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    get_settings.cache_clear()

    try:
        with pytest.raises(ValueError, match="Hugging Face API token must be provided"):
            HuggingFaceLLMProvider(token=None)
    finally:
        get_settings.cache_clear()


def test_huggingface_provider_masks_token_in_repr():
    """Verify __repr__ never exposes the HF token."""
    secret_token = "hf_SecretSuperPrivateToken987654321"
    provider = HuggingFaceLLMProvider(
        token=secret_token,
        client=AsyncMock(spec=AsyncInferenceClient),
    )
    repr_str = repr(provider)
    assert secret_token not in repr_str
    assert "provider='huggingface'" in repr_str
    assert "model=" in repr_str


@pytest.mark.anyio
async def test_huggingface_provider_sanitizes_token_in_errors():
    """Verify error messages sanitize any occurrence of the HF token."""
    mock_client = AsyncMock(spec=AsyncInferenceClient)
    mock_client.chat_completion.side_effect = httpx.ConnectError(
        "Failed to connect using token hf_SecretTokenValue12345"
    )

    provider = HuggingFaceLLMProvider(
        token="hf_SecretTokenValue12345",
        client=mock_client,
    )

    with pytest.raises(LLMConnectionError) as exc_info:
        await provider.generate([{"role": "user", "content": "Evaluate"}])

    error_str = str(exc_info.value)
    assert "hf_SecretTokenValue12345" not in error_str
    assert "[REDACTED_HF_TOKEN]" in error_str


# ============================================================================
# 4. Factory & Configuration Tests
# ============================================================================


def test_factory_creates_huggingface_provider_explicit():
    """Verify create_llm_provider(provider='huggingface') returns HuggingFaceLLMProvider."""
    provider = create_llm_provider(
        provider="huggingface",
        api_key="hf_test_token_from_arg",
        model="Qwen/Qwen3-32B",
        client=AsyncMock(spec=AsyncInferenceClient),
    )
    assert isinstance(provider, HuggingFaceLLMProvider)
    assert provider.provider_name == "huggingface"
    assert provider.model_name == "Qwen/Qwen3-32B"


def test_factory_resolves_huggingface_from_settings():
    """Verify create_llm_provider configures HuggingFace from Settings instance."""
    settings = Settings(
        LLM_PROVIDER="huggingface",
        HF_TOKEN="hf_token_from_settings",
        HF_MODEL="Qwen/Qwen3-32B",
    )
    provider = create_llm_provider(
        config=settings, client=AsyncMock(spec=AsyncInferenceClient)
    )
    assert isinstance(provider, HuggingFaceLLMProvider)
    assert provider.provider_name == "huggingface"


def test_factory_resolves_huggingface_from_env(monkeypatch):
    """Verify create_llm_provider reads LLM_PROVIDER=huggingface and HF_TOKEN from environment."""
    get_settings.cache_clear()
    monkeypatch.setenv("LLM_PROVIDER", "huggingface")
    monkeypatch.setenv("HF_TOKEN", "hf_token_from_env_123")
    monkeypatch.setenv("HF_MODEL", "Qwen/Qwen3-32B")

    try:
        provider = create_llm_provider(client=AsyncMock(spec=AsyncInferenceClient))
        assert isinstance(provider, HuggingFaceLLMProvider)
        assert provider.provider_name == "huggingface"
        assert provider.model_name == "Qwen/Qwen3-32B"
    finally:
        get_settings.cache_clear()


def test_factory_unsupported_provider_fails_explicitly():
    """Verify that unsupported provider names still fail fast with LLMProviderConfigurationError."""
    with pytest.raises(
        LLMProviderConfigurationError, match="Unsupported or unknown LLM provider"
    ):
        create_llm_provider(provider="unsupported_custom_provider")


def test_existing_mock_and_openai_providers_remain_intact():
    """Verify mock and openai providers continue to resolve correctly."""
    mock_prov = create_llm_provider(provider="mock")
    assert isinstance(mock_prov, MockLLMProvider)
    assert mock_prov.provider_name == "mock"

    openai_prov = create_llm_provider(
        provider="openai",
        api_key="sk-mock-key-for-test",
        client=AsyncMock(),
    )
    assert openai_prov.provider_name == "openai"


# ============================================================================
# 5. Full Context & RAG Pipeline Delivery Verification
# ============================================================================


@pytest.mark.anyio
async def test_huggingface_receives_complete_rag_and_payment_context():
    """Verify AgentOrchestrator delivers RAG precedent, attempt budget, and failure reason to HF provider."""
    mock_client = AsyncMock(spec=AsyncInferenceClient)
    mock_client.chat_completion.return_value = make_mock_chat_completion_output(
        action="payment_link",
        confidence=0.96,
        reasoning="Historical precedents demonstrate high recovery rate with interactive payment links.",
        referenced_case_ids=["case_1"],
    )

    provider = HuggingFaceLLMProvider(
        token="hf_test_token",
        client=mock_client,
    )
    orchestrator = AgentOrchestrator(
        provider=provider,
        context_builder=AgentContextBuilder(),
    )

    customer = CustomerContext(
        customer_id=uuid4(),
        total_payments=15,
        successful_payments=14,
        failed_payments=1,
        historical_success_rate=0.933,
    )
    payment = PaymentContext(
        payment_id=uuid4(),
        amount=3200.0,
        currency="INR",
        payment_method="card",
        failure_reason="customer_auth_failed_otp_timeout",
        status="failed",
    )
    opportunity = RecoveryOpportunityContext(
        opportunity_id=uuid4(),
        status="open",
        revenue_at_risk=3200.0,
    )
    context = CustomerRecoveryContext(
        customer=customer,
        current_payment=payment,
        current_opportunity=opportunity,
    )
    policy_context = RecoveryPolicyContext(
        provider="razorpay",
        policy_version="2026.1",
        applicable_rules=(RZP_CUSTOMER_AUTH_2FA_REQUIRED_RULE,),
        allowed_actions=(
            RecoveryAction.PAYMENT_LINK,
            RecoveryAction.CHANGE_PAYMENT_METHOD,
        ),
        prohibited_actions=(RecoveryAction.RETRY_PAYMENT,),
    )

    hist_case = HistoricalCase(
        payment_id=uuid4(),
        customer_id=customer.customer_id,
        amount=3200.0,
        currency="INR",
        payment_method="card",
        failure_reason="customer_auth_failed_otp_timeout",
        recovery_action="payment_link",
        recovery_status="recovered",
        amount_recovered=3200.0,
        was_recovered=True,
        relevance_score=0.98,
    )

    result = await orchestrator.decide(
        context=context,
        policy_context=policy_context,
        historical_cases=[hist_case],
    )

    assert result.agent_used is True
    assert result.recommendation.recommended_action == RecoveryAction.PAYMENT_LINK

    # Verify message payload delivered to HF chat_completion
    mock_client.chat_completion.assert_called_once()
    _, kwargs = mock_client.chat_completion.call_args
    user_msg_content = kwargs["messages"][1]["content"]

    # Assert complete context is present in the prompt
    assert "case_1" in user_msg_content
    assert "customer_auth_failed_otp_timeout" in user_msg_content
    assert "attempt_budget" in user_msg_content
    assert "policy_envelope" in user_msg_content
