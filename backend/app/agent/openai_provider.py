"""
Revora OpenAI LLM Provider Implementation.

Provides a production-ready LLM provider using OpenAI's asynchronous Structured
Outputs API to generate strictly validated LLMRecoveryRecommendation outcomes.
"""

import asyncio
import logging
from collections.abc import Mapping, Sequence

import httpx
import openai
from openai import AsyncOpenAI
from pydantic import ValidationError

from app.agent.provider import (
    LLMAuthenticationError,
    LLMConnectionError,
    LLMProviderError,
    LLMRateLimitError,
    LLMResponseValidationError,
    LLMTimeoutError,
    validate_chat_messages,
)
from app.agent.schemas import LLMRecoveryRecommendation
from app.config import get_settings

logger = logging.getLogger(__name__)

DEFAULT_OPENAI_MODEL = "gpt-4o-mini"
DEFAULT_OPENAI_TIMEOUT_SECONDS = 30.0


class OpenAILLMProvider:
    """
    Production-ready OpenAI LLM provider implementing the LLMProvider protocol.

    Executes asynchronous chat completions with OpenAI Structured Outputs,
    validating results against LLMRecoveryRecommendation.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        timeout_seconds: float | None = None,
        client: AsyncOpenAI | None = None,
        organization: str | None = None,
        base_url: str | None = None,
    ):
        """
        Initialize the OpenAILLMProvider.

        Args:
            api_key: Optional OpenAI API key. If omitted, resolved from settings/environment.
            model: Optional model name (defaults to settings or 'gpt-4o-mini').
            timeout_seconds: Optional request timeout in seconds (defaults to settings or 30.0s).
            client: Optional pre-configured AsyncOpenAI client (primarily for dependency injection/testing).
            organization: Optional OpenAI organization ID.
            base_url: Optional custom base URL for OpenAI-compatible gateways.

        Raises:
            ValueError: If api_key is empty/missing (when client not provided), or timeout is <= 0.
        """
        settings = get_settings()

        # Resolve timeout
        resolved_timeout = (
            timeout_seconds
            if timeout_seconds is not None
            else getattr(settings, "OPENAI_TIMEOUT_SECONDS", None)
            or getattr(settings, "LLM_TIMEOUT_SECONDS", None)
            or DEFAULT_OPENAI_TIMEOUT_SECONDS
        )
        if resolved_timeout <= 0:
            raise ValueError(
                f"timeout_seconds must be a positive number, got {resolved_timeout}"
            )
        self._timeout_seconds = float(resolved_timeout)

        # Resolve model
        self._model_name = (
            model.strip()
            if model and model.strip()
            else getattr(settings, "OPENAI_MODEL", None)
            or getattr(settings, "LLM_MODEL", None)
            or DEFAULT_OPENAI_MODEL
        )

        # Resolve client or API key
        if client is not None:
            self._client = client
            self._api_key = api_key or "injected-client-key"
        else:
            resolved_key = (
                api_key.strip()
                if api_key and api_key.strip()
                else (
                    getattr(settings, "OPENAI_API_KEY", None)
                    or getattr(settings, "LLM_API_KEY", None)
                    or ""
                )
            )
            if not resolved_key or not resolved_key.strip():
                raise ValueError(
                    "OpenAI API key must be provided via 'api_key' argument, "
                    "or via 'OPENAI_API_KEY' / 'LLM_API_KEY' setting or environment variable."
                )
            self._api_key = resolved_key.strip()
            self._client = AsyncOpenAI(
                api_key=self._api_key,
                timeout=self._timeout_seconds,
                organization=organization,
                base_url=base_url,
            )

        self._provider_name = "openai"

    @property
    def provider_name(self) -> str:
        """Return the provider identifier."""
        return self._provider_name

    @property
    def model_name(self) -> str:
        """Return the configured model identifier."""
        return self._model_name

    @property
    def timeout_seconds(self) -> float:
        """Return the configured timeout in seconds."""
        return self._timeout_seconds

    @property
    def client(self) -> AsyncOpenAI:
        """Return the underlying AsyncOpenAI client instance."""
        return self._client

    def __repr__(self) -> str:
        """Safe string representation without exposing sensitive API keys."""
        return (
            f"<{self.__class__.__name__}("
            f"provider='{self._provider_name}', "
            f"model='{self._model_name}', "
            f"timeout={self._timeout_seconds}s)>"
        )

    async def generate(
        self,
        messages: Sequence[Mapping[str, str]],
    ) -> LLMRecoveryRecommendation:
        """
        Generate a structured recovery recommendation using OpenAI Structured Outputs.

        Args:
            messages: Sequence of validated chat messages.

        Returns:
            Validated immutable LLMRecoveryRecommendation instance.

        Raises:
            TypeError: If messages is not a sequence of mappings.
            ValueError: If messages is empty or format is invalid.
            LLMAuthenticationError: On authentication or authorization failure.
            LLMRateLimitError: On rate limit or quota exceeded.
            LLMTimeoutError: On request timeout.
            LLMConnectionError: On network or connection failure.
            LLMResponseValidationError: If output is malformed, refused, or schema validation fails.
            LLMProviderError: On any other provider or API failure.
        """
        # 1. Validate generic message contract
        validate_chat_messages(messages)

        # 2. Format messages for OpenAI Chat Completion format
        formatted_messages = [
            {"role": msg["role"].strip().lower(), "content": msg["content"]}
            for msg in messages
        ]

        # 3. Call OpenAI Structured Outputs API
        try:
            response = await self._client.beta.chat.completions.parse(
                model=self._model_name,
                messages=formatted_messages,
                response_format=LLMRecoveryRecommendation,
                timeout=self._timeout_seconds,
            )
        except openai.AuthenticationError as exc:
            raise LLMAuthenticationError(
                "OpenAI authentication failed: invalid or unauthorized API key."
            ) from exc
        except openai.PermissionDeniedError as exc:
            raise LLMAuthenticationError(
                f"OpenAI permission denied: {exc.message}"
            ) from exc
        except openai.RateLimitError as exc:
            raise LLMRateLimitError(
                f"OpenAI rate limit or quota exceeded: {exc.message}"
            ) from exc
        except (openai.APITimeoutError, TimeoutError, asyncio.TimeoutError) as exc:
            raise LLMTimeoutError(
                f"OpenAI request timed out after {self._timeout_seconds}s."
            ) from exc
        except (openai.APIConnectionError, ConnectionError, httpx.RequestError) as exc:
            raise LLMConnectionError(
                "OpenAI connection error: unable to reach API endpoint."
            ) from exc
        except openai.LengthFinishReasonError as exc:
            raise LLMResponseValidationError(
                "OpenAI response truncated due to maximum token limit."
            ) from exc
        except (
            openai.ContentFilterFinishReasonError,
            openai.APIResponseValidationError,
        ) as exc:
            raise LLMResponseValidationError(
                f"OpenAI structured response validation error: {exc}"
            ) from exc
        except ValidationError as exc:
            raise LLMResponseValidationError(
                f"OpenAI structured response schema validation failed: {exc}"
            ) from exc
        except openai.APIStatusError as exc:
            raise LLMProviderError(
                f"OpenAI API status error ({exc.status_code}): {exc.message}"
            ) from exc
        except openai.OpenAIError as exc:
            raise LLMProviderError(f"OpenAI provider error: {exc}") from exc
        except Exception as exc:
            if isinstance(exc, ValidationError):
                raise LLMResponseValidationError(
                    f"OpenAI structured response schema validation failed: {exc}"
                ) from exc
            if isinstance(exc, (TypeError, ValueError, LLMProviderError)):
                raise
            raise LLMProviderError(f"Unexpected provider failure: {exc}") from exc

        # 4. Extract and validate structured outcome
        if not response or not getattr(response, "choices", None):
            raise LLMResponseValidationError(
                "OpenAI returned an empty response with no choices."
            )

        choice = response.choices[0]
        finish_reason = getattr(choice, "finish_reason", None)
        if finish_reason == "length":
            raise LLMResponseValidationError(
                "OpenAI response truncated due to maximum token limit."
            )

        message = getattr(choice, "message", None)
        if message is None:
            raise LLMResponseValidationError("OpenAI choice missing message object.")

        if getattr(message, "refusal", None):
            raise LLMResponseValidationError(
                f"OpenAI model refused the prompt: {message.refusal}"
            )

        parsed = getattr(message, "parsed", None)
        if parsed is None:
            content = getattr(message, "content", None)
            if content:
                raise LLMResponseValidationError(
                    "OpenAI model returned unparsed content instead of structured recommendation."
                )
            raise LLMResponseValidationError(
                "OpenAI model returned null structured recommendation."
            )

        if isinstance(parsed, LLMRecoveryRecommendation):
            return parsed

        try:
            if isinstance(parsed, dict):
                return LLMRecoveryRecommendation.model_validate(parsed)
            if hasattr(parsed, "model_dump"):
                return LLMRecoveryRecommendation.model_validate(parsed.model_dump())
            return LLMRecoveryRecommendation.model_validate(parsed)
        except ValidationError as exc:
            raise LLMResponseValidationError(
                f"Response failed LLMRecoveryRecommendation validation: {exc}"
            ) from exc
        except Exception as exc:
            raise LLMResponseValidationError(
                f"Failed to instantiate LLMRecoveryRecommendation from output: {exc}"
            ) from exc


# Convenient alias conforming to the architecture specification
RealLLMProvider = OpenAILLMProvider

__all__ = [
    "DEFAULT_OPENAI_MODEL",
    "DEFAULT_OPENAI_TIMEOUT_SECONDS",
    "OpenAILLMProvider",
    "RealLLMProvider",
]
