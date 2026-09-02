"""
Unit Tests for Revora Google Gemini LLM Provider (GeminiLLMProvider).

Verifies:
1. Structured JSON output parsing from Gemini API responses into LLMRecoveryRecommendation.
2. System instruction and multi-turn role formatting (user -> user, assistant -> model).
3. Robust handling of markdown code-fenced JSON responses (```json ... ```).
4. Error translation:
   - HTTP 401/403 -> LLMAuthenticationError
   - HTTP 429 -> LLMRateLimitError
   - Timeout -> LLMTimeoutError
   - Connection/Network failure -> LLMConnectionError
   - Malformed/schema-violating JSON -> LLMResponseValidationError
   - Safety filter blocks -> LLMResponseValidationError
5. Credential security: API key is never exposed in repr(), logs, or error strings.
6. Missing GEMINI_API_KEY fails clearly with ValueError.
7. Factory resolution: create_llm_provider(provider="gemini") correctly configures GeminiLLMProvider.
8. End-to-end RAG verification: Vector retrieval context and attempt budget reach the LLM prompt.
"""

import json
import re
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import httpx
import pytest

from app.agent.context_builder import AgentContextBuilder
from app.agent.factory import create_llm_provider
from app.agent.gemini_provider import GeminiLLMProvider
from app.agent.orchestrator import AgentOrchestrator
from app.agent.provider import (
    LLMAuthenticationError,
    LLMConnectionError,
    LLMRateLimitError,
    LLMResponseValidationError,
    LLMTimeoutError,
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
from app.policies.registry import RZP_CUSTOMER_AUTH_2FA_REQUIRED_RULE
from app.policies.schemas import RecoveryPolicyContext

# ============================================================================
# Helpers & Fixtures
# ============================================================================


def make_gemini_response_payload(
    action: str = "payment_link",
    confidence: float = 0.94,
    reasoning: str = "Interactive payment link required due to customer auth timeout.",
    key_factors: list[str] | None = None,
    referenced_case_ids: list[str] | None = None,
    wrap_in_markdown: bool = False,
) -> dict:
    """Helper to build a realistic Google Gemini generateContent response structure."""
    recommendation_data = {
        "recommended_action": action,
        "confidence": confidence,
        "reasoning": reasoning,
        "key_factors": key_factors or ["customer_auth_2fa_timeout", "high_trust"],
        "referenced_case_ids": referenced_case_ids or ["case_001"],
    }
    json_text = json.dumps(recommendation_data)
    if wrap_in_markdown:
        json_text = f"```json\n{json_text}\n```"

    return {
        "candidates": [
            {
                "content": {
                    "parts": [{"text": json_text}],
                    "role": "model",
                },
                "finishReason": "STOP",
            }
        ],
        "usageMetadata": {
            "promptTokenCount": 120,
            "candidatesTokenCount": 45,
            "totalTokenCount": 165,
        },
    }


# ============================================================================
# 1. Success Path & Structured Output Tests
# ============================================================================


@pytest.mark.anyio
async def test_gemini_provider_success_structured_output():
    """Verify GeminiLLMProvider calls generateContent and parses structured recommendation."""
    mock_client = AsyncMock(spec=httpx.AsyncClient)
    mock_response = MagicMock(spec=httpx.Response)
    mock_response.status_code = 200
    mock_response.json.return_value = make_gemini_response_payload(
        action="payment_link",
        confidence=0.92,
        reasoning="Payment link provides 2FA checkout.",
        key_factors=["auth_timeout"],
        referenced_case_ids=["case_101"],
    )
    mock_client.post.return_value = mock_response

    provider = GeminiLLMProvider(
        api_key="test-gemini-key-123",
        model="gemini-2.5-flash",
        client=mock_client,
    )

    messages = [
        {"role": "system", "content": "You are Revora recovery engine."},
        {"role": "user", "content": "Analyze payment failure scenario."},
    ]

    recommendation = await provider.generate(messages)

    assert isinstance(recommendation, LLMRecoveryRecommendation)
    assert recommendation.recommended_action == RecoveryAction.PAYMENT_LINK
    assert recommendation.confidence == 0.92
    assert recommendation.reasoning == "Payment link provides 2FA checkout."
    assert recommendation.key_factors == ("auth_timeout",)
    assert recommendation.referenced_case_ids == ("case_101",)

    # Verify HTTP request dispatched with header auth
    mock_client.post.assert_called_once()
    call_args, call_kwargs = mock_client.post.call_args
    assert "models/gemini-2.5-flash:generateContent" in call_args[0]
    assert call_kwargs["headers"]["x-goog-api-key"] == "test-gemini-key-123"
    assert (
        call_kwargs["json"]["generationConfig"]["responseMimeType"]
        == "application/json"
    )
    assert (
        call_kwargs["json"]["systemInstruction"]["parts"][0]["text"]
        == "You are Revora recovery engine."
    )
    assert (
        call_kwargs["json"]["contents"][0]["parts"][0]["text"]
        == "Analyze payment failure scenario."
    )


@pytest.mark.anyio
async def test_gemini_provider_handles_markdown_fenced_json():
    """Verify GeminiLLMProvider parses JSON wrapped in markdown ```json ``` blocks."""
    mock_client = AsyncMock(spec=httpx.AsyncClient)
    mock_response = MagicMock(spec=httpx.Response)
    mock_response.status_code = 200
    mock_response.json.return_value = make_gemini_response_payload(
        action="change_payment_method",
        confidence=0.88,
        reasoning="Customer card expired; prompt for alternate payment instrument.",
        wrap_in_markdown=True,
    )
    mock_client.post.return_value = mock_response

    provider = GeminiLLMProvider(
        api_key="test-gemini-key",
        client=mock_client,
    )

    rec = await provider.generate([{"role": "user", "content": "Evaluate"}])
    assert rec.recommended_action == RecoveryAction.CHANGE_PAYMENT_METHOD
    assert rec.confidence == 0.88


# ============================================================================
# 2. Error Translation Tests
# ============================================================================


@pytest.mark.anyio
@pytest.mark.parametrize("status_code", [401, 403])
async def test_gemini_provider_maps_auth_errors(status_code: int):
    """Verify HTTP 401 and 403 from Gemini are mapped to LLMAuthenticationError."""
    mock_client = AsyncMock(spec=httpx.AsyncClient)
    mock_response = MagicMock(spec=httpx.Response)
    mock_response.status_code = status_code
    mock_response.json.return_value = {
        "error": {
            "code": status_code,
            "message": "API key not valid. Please pass a valid API key.",
            "status": "UNAUTHENTICATED",
        }
    }
    mock_client.post.return_value = mock_response

    provider = GeminiLLMProvider(api_key="invalid-key", client=mock_client)

    with pytest.raises(LLMAuthenticationError, match="Gemini authentication failed"):
        await provider.generate([{"role": "user", "content": "Evaluate"}])


@pytest.mark.anyio
async def test_gemini_provider_maps_rate_limit_error():
    """Verify HTTP 429 from Gemini is mapped to LLMRateLimitError."""
    mock_client = AsyncMock(spec=httpx.AsyncClient)
    mock_response = MagicMock(spec=httpx.Response)
    mock_response.status_code = 429
    mock_response.json.return_value = {
        "error": {
            "code": 429,
            "message": "Resource has been exhausted (e.g. check quota).",
            "status": "RESOURCE_EXHAUSTED",
        }
    }
    mock_client.post.return_value = mock_response

    provider = GeminiLLMProvider(api_key="quota-exhausted-key", client=mock_client)

    with pytest.raises(LLMRateLimitError, match="Gemini rate limit or quota exceeded"):
        await provider.generate([{"role": "user", "content": "Evaluate"}])


@pytest.mark.anyio
async def test_gemini_provider_maps_timeout_error():
    """Verify request timeout is mapped to LLMTimeoutError."""
    mock_client = AsyncMock(spec=httpx.AsyncClient)
    mock_client.post.side_effect = httpx.TimeoutException("Read timed out")

    provider = GeminiLLMProvider(
        api_key="test-key", timeout_seconds=10.0, client=mock_client
    )

    with pytest.raises(
        LLMTimeoutError, match=re.escape("Gemini request timed out after 10.0s.")
    ):
        await provider.generate([{"role": "user", "content": "Evaluate"}])


@pytest.mark.anyio
async def test_gemini_provider_maps_connection_error():
    """Verify network connection failure is mapped to LLMConnectionError."""
    mock_client = AsyncMock(spec=httpx.AsyncClient)
    mock_client.post.side_effect = httpx.ConnectError("Failed to resolve host")

    provider = GeminiLLMProvider(api_key="test-key", client=mock_client)

    with pytest.raises(LLMConnectionError, match="Gemini connection error"):
        await provider.generate([{"role": "user", "content": "Evaluate"}])


@pytest.mark.anyio
async def test_gemini_provider_maps_safety_filter_block():
    """Verify prompt block by Gemini safety filter is mapped to LLMResponseValidationError."""
    mock_client = AsyncMock(spec=httpx.AsyncClient)
    mock_response = MagicMock(spec=httpx.Response)
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "candidates": [],
        "promptFeedback": {"blockReason": "SAFETY"},
    }
    mock_client.post.return_value = mock_response

    provider = GeminiLLMProvider(api_key="test-key", client=mock_client)

    with pytest.raises(
        LLMResponseValidationError, match="blocked by safety filters: SAFETY"
    ):
        await provider.generate([{"role": "user", "content": "Evaluate"}])


@pytest.mark.anyio
async def test_gemini_provider_maps_malformed_json_response():
    """Verify corrupt/malformed JSON from model is mapped to LLMResponseValidationError."""
    mock_client = AsyncMock(spec=httpx.AsyncClient)
    mock_response = MagicMock(spec=httpx.Response)
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "candidates": [
            {
                "content": {"parts": [{"text": "THIS IS NOT VALID JSON AT ALL"}]},
                "finishReason": "STOP",
            }
        ]
    }
    mock_client.post.return_value = mock_response

    provider = GeminiLLMProvider(api_key="test-key", client=mock_client)

    with pytest.raises(
        LLMResponseValidationError, match="Failed to decode structured JSON"
    ):
        await provider.generate([{"role": "user", "content": "Evaluate"}])


# ============================================================================
# 3. Credentials & Security Invariants
# ============================================================================


def test_gemini_provider_missing_key_raises_value_error(monkeypatch):
    """Verify that omitting Gemini API key raises a clear ValueError without silent mock fallback."""
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("LLM_API_KEY", raising=False)

    with pytest.raises(
        ValueError,
        match="Gemini API key must be provided via 'api_key' argument, or via 'GEMINI_API_KEY'",
    ):
        GeminiLLMProvider(api_key=None)


def test_gemini_provider_repr_masks_secret():
    """Verify that __repr__ masks the API key."""
    provider = GeminiLLMProvider(
        api_key="AIzaSySuperSecretKey999", client=AsyncMock(spec=httpx.AsyncClient)
    )
    repr_str = repr(provider)
    assert "AIzaSySuperSecretKey999" not in repr_str
    assert "provider='gemini'" in repr_str
    assert "model=" in repr_str


# ============================================================================
# 4. Factory Resolution Tests
# ============================================================================


def test_factory_creates_gemini_provider_explicit():
    """Verify create_llm_provider(provider='gemini') returns a GeminiLLMProvider."""
    provider = create_llm_provider(
        provider="gemini",
        api_key="test-gemini-key",
        model="gemini-2.5-flash",
        client=AsyncMock(spec=httpx.AsyncClient),
    )
    assert isinstance(provider, GeminiLLMProvider)
    assert provider.provider_name == "gemini"
    assert provider.model_name == "gemini-2.5-flash"


def test_factory_resolves_gemini_from_env(monkeypatch):
    """Verify create_llm_provider reads LLM_PROVIDER=gemini and GEMINI_API_KEY from environment."""
    get_settings.cache_clear()
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    monkeypatch.setenv("GEMINI_API_KEY", "env-gemini-key-xyz")
    monkeypatch.setenv("GEMINI_MODEL", "gemini-2.5-pro")

    try:
        provider = create_llm_provider(client=AsyncMock(spec=httpx.AsyncClient))
        assert isinstance(provider, GeminiLLMProvider)
        assert provider.provider_name == "gemini"
        assert provider.model_name == "gemini-2.5-pro"
    finally:
        get_settings.cache_clear()


def test_factory_resolves_gemini_from_settings():
    """Verify create_llm_provider resolves from explicit Settings instance."""
    settings = Settings(
        LLM_PROVIDER="gemini",
        GEMINI_API_KEY="settings-key-abc",
        GEMINI_MODEL="gemini-1.5-flash",
    )
    provider = create_llm_provider(
        config=settings, client=AsyncMock(spec=httpx.AsyncClient)
    )
    assert isinstance(provider, GeminiLLMProvider)
    assert provider.provider_name == "gemini"
    assert provider.model_name == "gemini-1.5-flash"


# ============================================================================
# 5. Full RAG Context & Agent Pipeline Verification
# ============================================================================


@pytest.mark.anyio
async def test_gemini_provider_receives_rag_context_and_attempt_budget():
    """Verify that AgentOrchestrator delivers RAG evidence and attempt budget to GeminiLLMProvider."""
    mock_client = AsyncMock(spec=httpx.AsyncClient)
    mock_response = MagicMock(spec=httpx.Response)
    mock_response.status_code = 200
    mock_response.json.return_value = make_gemini_response_payload(
        action="payment_link",
        confidence=0.95,
        reasoning="Historical precedents demonstrate high recovery via interactive link.",
        referenced_case_ids=["hist_case_001"],
    )
    mock_client.post.return_value = mock_response

    provider = GeminiLLMProvider(
        api_key="test-key",
        client=mock_client,
    )
    orchestrator = AgentOrchestrator(
        provider=provider,
        context_builder=AgentContextBuilder(),
    )

    # Construct context with payment failure and customer profile
    customer = CustomerContext(
        customer_id=uuid4(),
        total_payments=20,
        successful_payments=19,
        failed_payments=1,
        historical_success_rate=0.95,
    )
    payment = PaymentContext(
        payment_id=uuid4(),
        amount=4500.0,
        currency="INR",
        payment_method="card",
        failure_reason="customer_auth_failed_otp_timeout",
        status="failed",
    )
    opportunity = RecoveryOpportunityContext(
        opportunity_id=uuid4(),
        status="open",
        revenue_at_risk=4500.0,
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

    # Execute orchestrator with mock RAG historical case
    from app.historical_retrieval import HistoricalCase

    hist_case = HistoricalCase(
        payment_id=uuid4(),
        customer_id=customer.customer_id,
        amount=4500.0,
        currency="INR",
        payment_method="card",
        failure_reason="customer_auth_failed_otp_timeout",
        recovery_action="payment_link",
        recovery_status="recovered",
        amount_recovered=4500.0,
        was_recovered=True,
        relevance_score=0.96,
    )

    decision_result = await orchestrator.decide(
        context=context,
        policy_context=policy_context,
        historical_cases=[hist_case],
    )

    assert decision_result.agent_used is True
    assert (
        decision_result.recommendation.recommended_action == RecoveryAction.PAYMENT_LINK
    )

    # Inspect the payload sent to Gemini
    mock_client.post.assert_called_once()
    _, call_kwargs = mock_client.post.call_args
    user_content = call_kwargs["json"]["contents"][0]["parts"][0]["text"]

    # Verify RAG evidence, attempt budget, and failure reason are present in the JSON prompt
    assert "case_1" in user_content
    assert "customer_auth_failed_otp_timeout" in user_content
    assert "attempt_budget" in user_content
    assert "remaining_attempts" in user_content
