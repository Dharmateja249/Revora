"""
Revora Hugging Face LLM Provider Implementation.

Provides a production-grade LLM provider utilizing Hugging Face's Inference API
(AsyncInferenceClient) with structured JSON outputs to generate strictly validated
LLMRecoveryRecommendation outcomes.
"""

import asyncio
import json
import logging
import re
from collections.abc import Mapping, Sequence
from typing import Any

import httpx
from huggingface_hub import AsyncInferenceClient
from huggingface_hub.errors import (
    HfHubHTTPError,
    InferenceEndpointError,
    InferenceEndpointTimeoutError,
    InferenceTimeoutError,
    OverloadedError,
)
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

DEFAULT_HF_MODEL = "Qwen/Qwen3-32B"
DEFAULT_HF_TIMEOUT_SECONDS = 30.0

HF_TOKEN_REGEX = re.compile(r"hf_[A-Za-z0-9]+")


def _sanitize_error_message(msg: str, token: str | None = None) -> str:
    """Strip Hugging Face API tokens from error strings to prevent credential exposure."""
    sanitized = HF_TOKEN_REGEX.sub("[REDACTED_HF_TOKEN]", msg)
    if token and token.strip() and token in sanitized:
        sanitized = sanitized.replace(token.strip(), "[REDACTED_HF_TOKEN]")
    return sanitized


class HuggingFaceLLMProvider:
    """
    Hugging Face LLM provider implementing the decoupled LLMProvider protocol.

    Executes asynchronous chat completions against Hugging Face's Inference API
    using AsyncInferenceClient, returning strictly validated LLMRecoveryRecommendation instances.
    """

    def __init__(
        self,
        token: str | None = None,
        model: str | None = None,
        timeout_seconds: float | None = None,
        client: AsyncInferenceClient | None = None,
        base_url: str | None = None,
    ):
        """
        Initialize the HuggingFaceLLMProvider.

        Args:
            token: Optional Hugging Face access token. If omitted, resolved from settings/environment.
            model: Optional model identifier (defaults to settings or 'Qwen/Qwen3-32B').
            timeout_seconds: Optional request timeout in seconds (defaults to settings or 30.0s).
            client: Optional pre-configured AsyncInferenceClient (primarily for testing/mocking).
            base_url: Optional custom base URL / endpoint for dedicated Inference Endpoints.

        Raises:
            ValueError: If token is missing (when client not provided) or timeout is non-positive.
        """
        settings = get_settings()

        # 1. Resolve Timeout
        resolved_timeout = (
            timeout_seconds
            if timeout_seconds is not None
            else getattr(settings, "HF_TIMEOUT_SECONDS", None)
            or getattr(settings, "LLM_TIMEOUT_SECONDS", None)
            or DEFAULT_HF_TIMEOUT_SECONDS
        )
        if resolved_timeout <= 0:
            raise ValueError(
                f"timeout_seconds must be a positive number, got {resolved_timeout}"
            )
        self._timeout_seconds = float(resolved_timeout)

        # 2. Resolve Model
        self._model_name = (
            model.strip()
            if model and model.strip()
            else getattr(settings, "HF_MODEL", None)
            or getattr(settings, "LLM_MODEL", None)
            or DEFAULT_HF_MODEL
        )

        # 3. Resolve Token or Injected Client
        if client is not None:
            self._client = client
            self._token = token or "injected-hf-client-token"
        else:
            resolved_token = (
                token.strip()
                if token and token.strip()
                else (
                    getattr(settings, "HF_TOKEN", None)
                    or getattr(settings, "LLM_API_KEY", None)
                    or ""
                )
            )
            if not resolved_token or not resolved_token.strip():
                raise ValueError(
                    "Hugging Face API token must be provided via 'token' argument, "
                    "or via 'HF_TOKEN' / 'LLM_API_KEY' setting or environment variable."
                )
            self._token = resolved_token.strip()
            self._client = AsyncInferenceClient(
                model=self._model_name if not base_url else None,
                token=self._token,
                timeout=self._timeout_seconds,
                base_url=base_url,
            )

        self._provider_name = "huggingface"

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
    def client(self) -> AsyncInferenceClient:
        """Return the underlying AsyncInferenceClient instance."""
        return self._client

    def __repr__(self) -> str:
        """Safe string representation without exposing sensitive API tokens."""
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
        Generate a structured recovery recommendation using Hugging Face Inference API.

        Args:
            messages: Sequence of validated chat messages.

        Returns:
            Validated immutable LLMRecoveryRecommendation instance.

        Raises:
            TypeError: If messages is not a sequence of mappings.
            ValueError: If messages is empty or format is invalid.
            LLMAuthenticationError: On invalid or unauthorized Hugging Face token.
            LLMRateLimitError: On rate limit or service overload.
            LLMTimeoutError: On request timeout.
            LLMConnectionError: On network or connection failure.
            LLMResponseValidationError: If output is malformed or schema validation fails.
            LLMProviderError: On any other provider or API failure.
        """
        # 1. Validate generic message contract
        validate_chat_messages(messages)

        # 2. Format messages for Hugging Face Chat Completion format
        formatted_messages = [
            {"role": msg["role"].strip().lower(), "content": msg["content"]}
            for msg in messages
        ]

        # 3. Call Hugging Face Chat Completion API
        try:
            response = await self._client.chat_completion(
                messages=formatted_messages,
                model=self._model_name,
                response_format={"type": "json_object"},
                max_tokens=1000,
                temperature=0.1,
            )
        except HfHubHTTPError as exc:
            self._handle_hf_http_error(exc)
        except OverloadedError as exc:
            sanitized = _sanitize_error_message(str(exc), self._token)
            raise LLMRateLimitError(
                f"Hugging Face model is currently overloaded: {sanitized}"
            ) from exc
        except (
            InferenceTimeoutError,
            InferenceEndpointTimeoutError,
            httpx.TimeoutException,
            TimeoutError,
            asyncio.TimeoutError,
        ) as exc:
            raise LLMTimeoutError(
                f"Hugging Face request timed out after {self._timeout_seconds}s."
            ) from exc
        except (
            httpx.RequestError,
            ConnectionError,
            InferenceEndpointError,
        ) as exc:
            sanitized = _sanitize_error_message(str(exc), self._token)
            raise LLMConnectionError(
                f"Hugging Face connection error: unable to reach API endpoint ({sanitized})."
            ) from exc
        except Exception as exc:
            if isinstance(exc, (TypeError, ValueError, LLMProviderError)):
                raise
            sanitized = _sanitize_error_message(str(exc), self._token)
            raise LLMProviderError(
                f"Unexpected Hugging Face provider failure: {sanitized}"
            ) from exc

        # 4. Extract and validate structured JSON recommendation
        return self._parse_chat_response(response)

    def _handle_hf_http_error(self, exc: HfHubHTTPError) -> None:
        """Map HfHubHTTPError to domain LLMProviderError with secret sanitization."""
        status_code = getattr(exc.response, "status_code", None)
        server_message = getattr(exc, "server_message", None) or str(exc)
        sanitized = _sanitize_error_message(server_message, self._token)

        if status_code in (401, 403):
            raise LLMAuthenticationError(
                f"Hugging Face authentication failed: {sanitized}"
            ) from exc

        if status_code == 429:
            raise LLMRateLimitError(
                f"Hugging Face rate limit or quota exceeded: {sanitized}"
            ) from exc

        if status_code in (408, 504):
            raise LLMTimeoutError(
                f"Hugging Face request timed out: {sanitized}"
            ) from exc

        if status_code in (502, 503):
            raise LLMConnectionError(
                f"Hugging Face service unavailable ({status_code}): {sanitized}"
            ) from exc

        raise LLMProviderError(
            f"Hugging Face API error ({status_code}): {sanitized}"
        ) from exc

    def _parse_chat_response(self, response: Any) -> LLMRecoveryRecommendation:
        """Extract content from ChatCompletionOutput and validate against domain schema."""
        # Support both object attributes and dict indexing
        choices = getattr(response, "choices", None)
        if choices is None and isinstance(response, Mapping):
            choices = response.get("choices")

        if not choices or not isinstance(choices, Sequence) or len(choices) == 0:
            raise LLMResponseValidationError(
                "Hugging Face response contained no choices."
            )

        first_choice = choices[0]
        message = getattr(first_choice, "message", None)
        if message is None and isinstance(first_choice, Mapping):
            message = first_choice.get("message")

        if message is None:
            raise LLMResponseValidationError(
                "Hugging Face choice contained no message object."
            )

        content = getattr(message, "content", None)
        if content is None and isinstance(message, Mapping):
            content = message.get("content")

        if not content or not isinstance(content, str) or not content.strip():
            raise LLMResponseValidationError(
                "Hugging Face message contained empty or non-string content."
            )

        clean_text = content.strip()
        # Strip markdown fences if present
        if clean_text.startswith("```"):
            lines = clean_text.splitlines()
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].startswith("```"):
                lines = lines[:-1]
            clean_text = "\n".join(lines).strip()

        try:
            parsed_dict = json.loads(clean_text)
        except json.JSONDecodeError as exc:
            raise LLMResponseValidationError(
                f"Failed to decode structured JSON from Hugging Face response: {exc}"
            ) from exc

        if not isinstance(parsed_dict, dict):
            raise LLMResponseValidationError(
                f"Expected JSON object from Hugging Face, got {type(parsed_dict).__name__}"
            )

        try:
            return LLMRecoveryRecommendation.model_validate(parsed_dict)
        except ValidationError as exc:
            raise LLMResponseValidationError(
                f"Hugging Face output failed LLMRecoveryRecommendation schema validation: {exc}"
            ) from exc
