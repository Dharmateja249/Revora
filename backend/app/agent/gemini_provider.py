"""
Revora Google Gemini LLM Provider Implementation.

Provides a production-grade LLM provider utilizing Google's Gemini API with
structured JSON outputs to generate strictly validated LLMRecoveryRecommendation outcomes.
"""

import asyncio
import json
import logging
from collections.abc import Mapping, Sequence
from typing import Any

import httpx
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

DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"
DEFAULT_GEMINI_TIMEOUT_SECONDS = 30.0
GEMINI_API_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"


class GeminiLLMProvider:
    """
    Google Gemini LLM provider implementing the decoupled LLMProvider protocol.

    Executes asynchronous inference against Google Gemini's generateContent endpoint
    with responseMimeType='application/json' and structured schema validation.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        timeout_seconds: float | None = None,
        client: httpx.AsyncClient | None = None,
        base_url: str | None = None,
    ):
        """
        Initialize the GeminiLLMProvider.

        Args:
            api_key: Optional Gemini API key. If omitted, resolved from settings/environment.
            model: Optional model identifier (defaults to settings or 'gemini-2.5-flash').
            timeout_seconds: Optional request timeout in seconds (defaults to settings or 30.0s).
            client: Optional pre-configured httpx.AsyncClient (for testing/dependency injection).
            base_url: Optional custom base URL for Gemini API endpoint.

        Raises:
            ValueError: If api_key is missing (when client not provided) or timeout is non-positive.
        """
        settings = get_settings()

        # 1. Resolve Timeout
        resolved_timeout = (
            timeout_seconds
            if timeout_seconds is not None
            else getattr(settings, "GEMINI_TIMEOUT_SECONDS", None)
            or getattr(settings, "LLM_TIMEOUT_SECONDS", None)
            or DEFAULT_GEMINI_TIMEOUT_SECONDS
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
            else getattr(settings, "GEMINI_MODEL", None)
            or getattr(settings, "LLM_MODEL", None)
            or DEFAULT_GEMINI_MODEL
        )

        # 3. Resolve Base URL
        self._base_url = (base_url or GEMINI_API_BASE_URL).rstrip("/")

        # 4. Resolve API Key or Injected Client
        if client is not None:
            self._client = client
            self._api_key = api_key or "injected-test-key"
            self._owns_client = False
        else:
            resolved_key = (
                api_key.strip()
                if api_key and api_key.strip()
                else (
                    getattr(settings, "GEMINI_API_KEY", None)
                    or getattr(settings, "LLM_API_KEY", None)
                    or ""
                )
            )
            if not resolved_key or not resolved_key.strip():
                raise ValueError(
                    "Gemini API key must be provided via 'api_key' argument, "
                    "or via 'GEMINI_API_KEY' / 'LLM_API_KEY' setting or environment variable."
                )
            self._api_key = resolved_key.strip()
            self._client = httpx.AsyncClient(timeout=self._timeout_seconds)
            self._owns_client = True

        self._provider_name = "gemini"

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
    def base_url(self) -> str:
        """Return the Gemini API base URL."""
        return self._base_url

    def __repr__(self) -> str:
        """Safe string representation without exposing sensitive API keys."""
        return (
            f"<{self.__class__.__name__}("
            f"provider='{self._provider_name}', "
            f"model='{self._model_name}', "
            f"timeout={self._timeout_seconds}s)>"
        )

    async def close(self) -> None:
        """Close the underlying HTTP client if created internally."""
        if getattr(self, "_owns_client", False) and self._client is not None:
            await self._client.aclose()

    async def generate(
        self,
        messages: Sequence[Mapping[str, str]],
    ) -> LLMRecoveryRecommendation:
        """
        Generate a structured recovery recommendation using Gemini API.

        Args:
            messages: Sequence of validated chat messages.

        Returns:
            Validated immutable LLMRecoveryRecommendation instance.

        Raises:
            TypeError: If messages is not a sequence of mappings.
            ValueError: If messages is empty or format is invalid.
            LLMAuthenticationError: On authentication or permission failures.
            LLMRateLimitError: On rate limits or quota exhaustion.
            LLMTimeoutError: On request timeout.
            LLMConnectionError: On network or connection failure.
            LLMResponseValidationError: If output is malformed or schema validation fails.
            LLMProviderError: On any other provider or API failure.
        """
        # 1. Validate generic message contract
        validate_chat_messages(messages)

        # 2. Extract system instructions and user/assistant turns
        system_instruction_parts: list[dict[str, str]] = []
        contents: list[dict[str, Any]] = []

        for msg in messages:
            role = msg["role"].strip().lower()
            content = msg["content"]

            if role == "system":
                system_instruction_parts.append({"text": content})
            elif role == "user":
                contents.append({"role": "user", "parts": [{"text": content}]})
            elif role == "assistant":
                contents.append({"role": "model", "parts": [{"text": content}]})

        # 3. Build request payload with structured JSON configuration
        payload: dict[str, Any] = {
            "contents": contents,
            "generationConfig": {
                "responseMimeType": "application/json",
                "temperature": 0.1,
            },
        }

        if system_instruction_parts:
            payload["systemInstruction"] = {"parts": system_instruction_parts}

        endpoint = f"{self._base_url}/models/{self._model_name}:generateContent"
        # Secure header prevents API key from leaking into query parameters / access logs
        headers = {
            "Content-Type": "application/json",
            "x-goog-api-key": self._api_key,
        }

        # 4. Dispatch Async Request
        try:
            response = await self._client.post(
                endpoint,
                json=payload,
                headers=headers,
                timeout=self._timeout_seconds,
            )
        except (httpx.TimeoutException, TimeoutError, asyncio.TimeoutError) as exc:
            raise LLMTimeoutError(
                f"Gemini request timed out after {self._timeout_seconds}s."
            ) from exc
        except (httpx.RequestError, ConnectionError) as exc:
            raise LLMConnectionError(
                f"Gemini connection error: unable to reach API endpoint ({exc})."
            ) from exc
        except Exception as exc:
            raise LLMProviderError(
                f"Unexpected Gemini transport failure: {exc}"
            ) from exc

        # 5. Handle HTTP Errors & Map to Domain Exceptions
        if response.status_code != 200:
            self._handle_http_error(response)

        # 6. Parse and Validate Structured Response
        return self._parse_gemini_response(response)

    def _handle_http_error(self, response: httpx.Response) -> None:
        """Parse Gemini error payload and raise appropriate LLMProviderError."""
        error_msg = f"HTTP {response.status_code}"
        try:
            err_json = response.json()
            if isinstance(err_json, dict) and "error" in err_json:
                api_error = err_json["error"]
                message = api_error.get("message")
                status_str = api_error.get("status")
                if message:
                    error_msg = (
                        f"{message} (status: {status_str or response.status_code})"
                    )
        except (ValueError, KeyError, TypeError):
            error_msg = response.text or error_msg

        # Map by status code
        if response.status_code in (401, 403):
            raise LLMAuthenticationError(f"Gemini authentication failed: {error_msg}")
        if response.status_code == 429:
            raise LLMRateLimitError(f"Gemini rate limit or quota exceeded: {error_msg}")
        if response.status_code in (408, 504):
            raise LLMTimeoutError(f"Gemini request timed out: {error_msg}")
        if response.status_code in (502, 503):
            raise LLMConnectionError(f"Gemini service unavailable: {error_msg}")

        raise LLMProviderError(
            f"Gemini API error ({response.status_code}): {error_msg}"
        )

    def _parse_gemini_response(
        self, response: httpx.Response
    ) -> LLMRecoveryRecommendation:
        """Extract and validate LLMRecoveryRecommendation from Gemini response JSON."""
        try:
            data = response.json()
        except ValueError as exc:
            raise LLMResponseValidationError(
                f"Gemini returned invalid non-JSON payload: {exc}"
            ) from exc

        if not isinstance(data, dict):
            raise LLMResponseValidationError(
                f"Expected dict response from Gemini, got {type(data).__name__}"
            )

        candidates = data.get("candidates")
        if not candidates or not isinstance(candidates, list):
            prompt_feedback = data.get("promptFeedback", {})
            block_reason = prompt_feedback.get("blockReason")
            if block_reason:
                raise LLMResponseValidationError(
                    f"Gemini generation blocked by safety filters: {block_reason}"
                )
            raise LLMResponseValidationError(
                "Gemini response contained no candidate choices."
            )

        candidate = candidates[0]
        finish_reason = candidate.get("finishReason")
        if finish_reason in ("SAFETY", "RECITATION"):
            raise LLMResponseValidationError(
                f"Gemini response terminated early due to {finish_reason} filter."
            )

        content = candidate.get("content", {})
        parts = content.get("parts", [])
        if not parts:
            raise LLMResponseValidationError(
                "Gemini candidate content contained no parts."
            )

        raw_text = parts[0].get("text", "")
        if not raw_text or not raw_text.strip():
            raise LLMResponseValidationError(
                "Gemini candidate part contained empty text."
            )

        # Parse JSON structured recommendation
        clean_text = raw_text.strip()
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
                f"Failed to decode structured JSON from Gemini response: {exc}"
            ) from exc

        if not isinstance(parsed_dict, dict):
            raise LLMResponseValidationError(
                f"Expected JSON object from Gemini, got {type(parsed_dict).__name__}"
            )

        try:
            return LLMRecoveryRecommendation.model_validate(parsed_dict)
        except ValidationError as exc:
            raise LLMResponseValidationError(
                f"Gemini output failed LLMRecoveryRecommendation schema validation: {exc}"
            ) from exc
