"""
Unit Tests for Revora LLM Provider Factory (Stage 8.2).

Verifies that create_llm_provider:
1. Returns MockLLMProvider by default and under LLM_PROVIDER=mock.
2. Returns OpenAILLMProvider under LLM_PROVIDER=openai when credentials/clients are supplied.
3. Does not silently fall back from explicit openai to mock on configuration errors.
4. Fails deterministically with LLMProviderConfigurationError on unknown provider names.
5. Preserves dependency inversion (returns LLMProvider protocol instances).
6. Never exposes sensitive credentials in error messages or logs.
7. Avoids global mutable state by creating independent provider instances.
"""

from unittest.mock import MagicMock

import pytest

from app.agent.factory import create_llm_provider
from app.agent.openai_provider import OpenAILLMProvider
from app.agent.provider import (
    LLMAuthenticationError,
    LLMProvider,
    LLMProviderConfigurationError,
    LLMProviderError,
    MockLLMProvider,
)
from app.agent.schemas import LLMRecoveryRecommendation
from app.config import Settings
from app.decision_engine import RecoveryAction


@pytest.fixture
def custom_recommendation():
    return LLMRecoveryRecommendation(
        recommended_action=RecoveryAction.RETRY_PAYMENT,
        confidence=0.92,
        reasoning="Network glitch likely resolved on retry.",
        key_factors=("transient_network_error",),
        referenced_case_ids=("case_101",),
    )


@pytest.fixture
def mock_async_openai_client():
    client = MagicMock()
    client.beta = MagicMock()
    client.beta.chat = MagicMock()
    client.beta.chat.completions = MagicMock()
    client.beta.chat.completions.parse = MagicMock()
    return client


# ============================================================================
# 1. Mock Provider Factory Tests
# ============================================================================


def test_factory_mock_default(monkeypatch):
    """Verify create_llm_provider defaults to MockLLMProvider with default recommendation."""
    monkeypatch.setattr(
        "app.agent.factory.get_settings",
        lambda: Settings(LLM_PROVIDER="mock"),
    )
    provider = create_llm_provider()

    assert isinstance(provider, LLMProvider)
    assert isinstance(provider, MockLLMProvider)
    assert provider.provider_name == "mock"
    assert provider.model_name == "mock-recovery-engine"
    # Default recommendation uses NO_ACTION safely
    assert provider.generate.__qualname__ == "MockLLMProvider.generate"


def test_factory_mock_explicit_string():
    """Verify create_llm_provider('mock') creates MockLLMProvider."""
    provider = create_llm_provider("mock")

    assert isinstance(provider, MockLLMProvider)
    assert provider.provider_name == "mock"


def test_factory_mock_with_custom_recommendation(custom_recommendation):
    """Verify create_llm_provider accepts a custom recommendation for MockLLMProvider."""
    provider = create_llm_provider(
        provider="mock",
        recommendation=custom_recommendation,
    )

    assert isinstance(provider, MockLLMProvider)
    assert provider._recommendation.recommended_action == RecoveryAction.RETRY_PAYMENT
    assert provider._recommendation.confidence == 0.92


def test_factory_mock_from_settings(custom_recommendation):
    """Verify Settings object with LLM_PROVIDER=mock constructs MockLLMProvider."""
    settings = Settings(LLM_PROVIDER="mock")
    provider = create_llm_provider(settings, recommendation=custom_recommendation)

    assert isinstance(provider, MockLLMProvider)


def test_factory_mock_from_dict():
    """Verify dictionary config with LLM_PROVIDER=mock constructs MockLLMProvider."""
    config = {"LLM_PROVIDER": "mock"}
    provider = create_llm_provider(config)

    assert isinstance(provider, MockLLMProvider)


def test_factory_mock_failure_parameters():
    """Verify failure parameters are passed into MockLLMProvider."""
    custom_exc = LLMAuthenticationError("Simulated mock auth failure")
    provider = create_llm_provider(
        provider="mock",
        should_fail=True,
        failure_exception=custom_exc,
        record_messages=False,
    )

    assert isinstance(provider, MockLLMProvider)
    assert provider._should_fail is True
    assert provider._failure_exception is custom_exc
    assert provider._record_messages is False


# ============================================================================
# 2. OpenAI Provider Factory Tests
# ============================================================================


def test_factory_openai_with_injected_client(mock_async_openai_client):
    """Verify create_llm_provider('openai') creates OpenAILLMProvider with injected client."""
    provider = create_llm_provider(
        provider="openai",
        client=mock_async_openai_client,
        model="gpt-4o",
        timeout_seconds=45.0,
    )

    assert isinstance(provider, LLMProvider)
    assert isinstance(provider, OpenAILLMProvider)
    assert provider.provider_name == "openai"
    assert provider.model_name == "gpt-4o"
    assert provider.timeout_seconds == 45.0


def test_factory_openai_from_settings(mock_async_openai_client):
    """Verify Settings object with LLM_PROVIDER=openai constructs OpenAILLMProvider."""
    settings = Settings(
        LLM_PROVIDER="openai",
        OPENAI_API_KEY="sk-test-fake-key-12345",
        OPENAI_MODEL="gpt-4o-mini",
        OPENAI_TIMEOUT_SECONDS=25.0,
    )
    provider = create_llm_provider(settings, client=mock_async_openai_client)

    assert isinstance(provider, OpenAILLMProvider)
    assert provider.provider_name == "openai"
    assert provider.model_name == "gpt-4o-mini"
    assert provider.timeout_seconds == 25.0


def test_factory_openai_from_dict(mock_async_openai_client):
    """Verify dictionary config constructs OpenAILLMProvider."""
    config = {
        "LLM_PROVIDER": "openai",
        "OPENAI_API_KEY": "sk-dict-key",
        "OPENAI_MODEL": "gpt-4o",
        "OPENAI_TIMEOUT_SECONDS": "35.0",
    }
    provider = create_llm_provider(config, client=mock_async_openai_client)

    assert isinstance(provider, OpenAILLMProvider)
    assert provider.model_name == "gpt-4o"
    assert provider.timeout_seconds == 35.0


def test_factory_openai_missing_api_key_raises_without_falling_back_to_mock(
    monkeypatch,
):
    """Verify explicit openai selection without credentials raises ValueError and does NOT fall back to mock."""
    settings = Settings(
        LLM_PROVIDER="openai",
        OPENAI_API_KEY=None,
        LLM_API_KEY=None,
    )
    monkeypatch.setattr("app.agent.factory.get_settings", lambda: settings)

    with pytest.raises(ValueError, match="OpenAI API key must be provided"):
        create_llm_provider(settings)


# ============================================================================
# 3. Unknown Provider & Configuration Error Tests
# ============================================================================


def test_factory_unknown_provider_raises_configuration_error():
    """Verify unsupported provider name raises LLMProviderConfigurationError."""
    with pytest.raises(
        LLMProviderConfigurationError, match="Unsupported or unknown LLM provider"
    ) as exc_info:
        create_llm_provider("anthropic")

    # Satisfies both ValueError and LLMProviderError
    assert isinstance(exc_info.value, ValueError)
    assert isinstance(exc_info.value, LLMProviderError)


def test_factory_unknown_provider_from_settings():
    """Verify Settings with invalid LLM_PROVIDER raises LLMProviderConfigurationError."""
    settings = Settings(LLM_PROVIDER="unsupported_vendor")
    with pytest.raises(LLMProviderConfigurationError, match="unsupported_vendor"):
        create_llm_provider(settings)


def test_factory_invalid_config_type_raises():
    """Verify non-Settings, non-dict, non-str config raises LLMProviderConfigurationError."""
    with pytest.raises(LLMProviderConfigurationError, match="Invalid config type"):
        create_llm_provider(config=12345)  # type: ignore


def test_factory_invalid_timeout_in_dict_raises():
    """Verify malformed timeout in dictionary raises LLMProviderConfigurationError."""
    config = {
        "LLM_PROVIDER": "mock",
        "OPENAI_TIMEOUT_SECONDS": "not-a-number",
    }
    with pytest.raises(LLMProviderConfigurationError, match="Invalid timeout value"):
        create_llm_provider(config)


# ============================================================================
# 4. Security & Isolation Tests
# ============================================================================


def test_factory_security_no_api_keys_leaked_in_exceptions():
    """Verify API keys are never included in exceptions raised by factory."""
    sensitive_key = "sk-super-secret-production-key-xyz"
    try:
        create_llm_provider(
            provider="unknown_provider",
            api_key=sensitive_key,
        )
    except LLMProviderConfigurationError as exc:
        assert sensitive_key not in str(exc)


def test_factory_creates_independent_instances():
    """Verify factory returns distinct instances with no shared mutable state."""
    p1 = create_llm_provider("mock")
    p2 = create_llm_provider("mock")

    assert p1 is not p2
    assert isinstance(p1, MockLLMProvider)
    assert isinstance(p2, MockLLMProvider)
