"""
Revora LLM Provider Factory.

Constructs configured LLMProvider instances based on application configuration
and dependency injection, adhering to the LLMProvider protocol.
"""

import logging
from collections.abc import Mapping
from typing import Any

from app.agent.provider import (
    LLMProvider,
    LLMProviderConfigurationError,
    LLMProviderError,
    MockLLMProvider,
)
from app.agent.schemas import LLMRecoveryRecommendation
from app.config import Settings, get_settings
from app.decision_engine import RecoveryAction

logger = logging.getLogger(__name__)

DEFAULT_MOCK_RECOMMENDATION = LLMRecoveryRecommendation(
    recommended_action=RecoveryAction.NO_ACTION,
    confidence=1.0,
    reasoning="Default MockLLMProvider recommendation from factory.",
    key_factors=("default_mock_provider",),
    referenced_case_ids=(),
)

SUPPORTED_PROVIDERS = frozenset({"mock", "openai", "gemini", "huggingface"})


def create_llm_provider(
    config: Settings | Mapping[str, Any] | str | None = None,
    *,
    provider: str | None = None,
    recommendation: LLMRecoveryRecommendation | None = None,
    should_fail: bool = False,
    failure_exception: LLMProviderError | None = None,
    record_messages: bool = True,
    api_key: str | None = None,
    model: str | None = None,
    timeout_seconds: float | None = None,
    client: Any | None = None,
    organization: str | None = None,
    base_url: str | None = None,
    **kwargs: Any,
) -> LLMProvider:
    """
    Construct an LLMProvider instance based on configuration and parameters.

    Args:
        config: Optional application Settings, a mapping of configuration options,
            a provider name string (e.g. "mock", "openai", "gemini", or "huggingface"), or None to use global settings.
        provider: Optional explicit provider name override ("mock", "openai", "gemini", or "huggingface").
        recommendation: Predefined recommendation for MockLLMProvider.
        should_fail: If True, MockLLMProvider raises on generate.
        failure_exception: Custom LLMProviderError for MockLLMProvider.
        record_messages: Whether MockLLMProvider records received messages.
        api_key: Explicit API key override for real LLM providers.
        model: Explicit model name override for real LLM providers.
        timeout_seconds: Explicit request timeout in seconds.
        client: Pre-configured client instance (useful for testing/mocking).
        organization: Optional organization ID.
        base_url: Optional alternative API endpoint URL.
        **kwargs: Additional provider-specific parameters.

    Returns:
        Configured instance implementing LLMProvider protocol.

    Raises:
        LLMProviderConfigurationError: If an unknown or invalid provider is configured.
        ValueError: If provider configuration parameters (e.g. missing API key) are invalid.
    """
    provider_name: str | None = provider
    resolved_settings: Settings | None = None

    if isinstance(config, str):
        if provider_name is None:
            provider_name = config
        resolved_settings = get_settings()
    elif isinstance(config, Settings):
        resolved_settings = config
        if provider_name is None:
            provider_name = config.LLM_PROVIDER
    elif isinstance(config, Mapping):
        if provider_name is None:
            provider_name = config.get("LLM_PROVIDER") or config.get("llm_provider")
    elif config is None:
        resolved_settings = get_settings()
        if provider_name is None:
            provider_name = resolved_settings.LLM_PROVIDER
    else:
        raise LLMProviderConfigurationError(
            f"Invalid config type provided to create_llm_provider: {type(config).__name__}"
        )

    # If provider is still not determined from config, fall back to global settings or "mock"
    if provider_name is None:
        if resolved_settings is None:
            resolved_settings = get_settings()
        provider_name = getattr(resolved_settings, "LLM_PROVIDER", "mock")

    clean_provider = (provider_name or "mock").strip().lower()

    if clean_provider == "mock":
        effective_rec = recommendation or DEFAULT_MOCK_RECOMMENDATION
        return MockLLMProvider(
            recommendation=effective_rec,
            should_fail=should_fail,
            failure_exception=failure_exception,
            record_messages=record_messages,
        )

    if clean_provider == "gemini":
        key_name = "GEMINI_API_KEY"
        model_name = "GEMINI_MODEL"
        timeout_name = "GEMINI_TIMEOUT_SECONDS"
    elif clean_provider == "huggingface":
        key_name = "HF_TOKEN"
        model_name = "HF_MODEL"
        timeout_name = "HF_TIMEOUT_SECONDS"
    elif clean_provider == "openai":
        key_name = "OPENAI_API_KEY"
        model_name = "OPENAI_MODEL"
        timeout_name = "OPENAI_TIMEOUT_SECONDS"
    else:
        raise LLMProviderConfigurationError(
            f"Unsupported or unknown LLM provider: {clean_provider!r}. "
            f"Supported providers are: {sorted(SUPPORTED_PROVIDERS)!r}."
        )

    # Resolve provider-specific configuration strictly for the selected provider
    config_key = None
    config_model = None
    config_timeout_raw = None
    if isinstance(config, Mapping):
        config_key = config.get(key_name) or config.get("LLM_API_KEY")
        config_model = config.get(model_name) or config.get("LLM_MODEL")
        config_timeout_raw = config.get(timeout_name) or config.get(
            "LLM_TIMEOUT_SECONDS"
        )

    parsed_config_timeout = None
    if config_timeout_raw is not None:
        try:
            parsed_config_timeout = float(config_timeout_raw)
        except (ValueError, TypeError) as exc:
            raise LLMProviderConfigurationError(
                f"Invalid timeout value in config: {config_timeout_raw!r}"
            ) from exc

    effective_key = (
        api_key
        or config_key
        or (getattr(resolved_settings, key_name, None) if resolved_settings else None)
        or (
            getattr(resolved_settings, "LLM_API_KEY", None)
            if resolved_settings
            else None
        )
    )

    effective_model = (
        model
        or config_model
        or (getattr(resolved_settings, model_name, None) if resolved_settings else None)
        or (
            getattr(resolved_settings, "LLM_MODEL", None) if resolved_settings else None
        )
    )

    effective_timeout = (
        timeout_seconds
        if timeout_seconds is not None
        else (
            parsed_config_timeout
            if parsed_config_timeout is not None
            else (
                (
                    getattr(resolved_settings, timeout_name, None)
                    if resolved_settings
                    else None
                )
                or (
                    getattr(resolved_settings, "LLM_TIMEOUT_SECONDS", None)
                    if resolved_settings
                    else None
                )
            )
        )
    )

    if clean_provider == "gemini":
        from app.agent.gemini_provider import GeminiLLMProvider

        return GeminiLLMProvider(
            api_key=effective_key,
            model=effective_model,
            timeout_seconds=effective_timeout,
            client=client,
            base_url=base_url,
        )

    if clean_provider == "huggingface":
        from app.agent.huggingface_provider import HuggingFaceLLMProvider

        return HuggingFaceLLMProvider(
            token=effective_key,
            model=effective_model,
            timeout_seconds=effective_timeout,
            client=client,
            base_url=base_url,
        )

    if clean_provider == "openai":
        from app.agent.openai_provider import OpenAILLMProvider

        return OpenAILLMProvider(
            api_key=effective_key,
            model=effective_model,
            timeout_seconds=effective_timeout,
            client=client,
            organization=organization,
            base_url=base_url,
        )
