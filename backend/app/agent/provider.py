"""
Revora LLM Provider Abstraction and Deterministic Mock Provider.

Defines the decoupled, provider-agnostic protocol for executing chat messages
and returning structured LLMRecoveryRecommendation contracts.
"""

from typing import Any, Dict, List, Mapping, Optional, Protocol, Sequence, runtime_checkable

from app.agent.schemas import LLMRecoveryRecommendation

SUPPORTED_MESSAGE_ROLES = frozenset({"system", "user", "assistant"})


class LLMProviderError(Exception):
    """Base exception for all LLM provider execution or communication failures."""

    pass


class LLMResponseValidationError(LLMProviderError):
    """Raised when an LLM provider response is malformed or fails schema validation."""

    pass


def validate_chat_messages(messages: Sequence[Mapping[str, str]]) -> None:
    """
    Validate that messages conform to the generic chat message contract.

    Args:
        messages: Sequence of chat message mappings with 'role' and 'content' keys.

    Raises:
        TypeError: If messages is not a sequence or if an individual message is not a mapping.
        ValueError: If messages is empty, or if role/content is missing, blank, or unsupported.
    """
    if not isinstance(messages, Sequence) or isinstance(messages, (str, bytes, bytearray)):
        raise TypeError(
            f"Expected Sequence of message mappings, got {type(messages).__name__}"
        )

    if len(messages) == 0:
        raise ValueError("Messages sequence cannot be empty.")

    for idx, msg in enumerate(messages):
        if not isinstance(msg, Mapping):
            raise TypeError(
                f"Message at index {idx} must be a mapping, got {type(msg).__name__}"
            )

        if "role" not in msg or "content" not in msg:
            raise ValueError(
                f"Message at index {idx} must contain both 'role' and 'content' keys."
            )

        role = msg["role"]
        if not isinstance(role, str) or not role.strip():
            raise ValueError(
                f"Message at index {idx} has invalid or empty role: {role!r}"
            )

        role_clean = role.strip().lower()
        if role_clean not in SUPPORTED_MESSAGE_ROLES:
            raise ValueError(
                f"Message at index {idx} has unsupported role '{role_clean}'. "
                f"Supported roles: {sorted(SUPPORTED_MESSAGE_ROLES)}"
            )

        content = msg["content"]
        if not isinstance(content, str):
            raise TypeError(
                f"Message content at index {idx} must be a string, got {type(content).__name__}"
            )

        if not content.strip():
            raise ValueError(f"Message content at index {idx} cannot be empty or whitespace-only.")


@runtime_checkable
class LLMProvider(Protocol):
    """
    Provider-agnostic protocol for executing LLM reasoning over generic chat messages.
    """

    async def generate(
        self,
        messages: Sequence[Mapping[str, str]],
    ) -> LLMRecoveryRecommendation:
        """
        Generate a structured recovery recommendation from generic chat messages.

        Args:
            messages: Sequence of validated chat messages.

        Returns:
            Validated immutable LLMRecoveryRecommendation instance.

        Raises:
            LLMProviderError: On execution, connection, or response validation errors.
        """
        ...


class MockLLMProvider:
    """
    Deterministic mock LLM provider for unit testing, offline evaluation, and local development.
    """

    def __init__(
        self,
        recommendation: LLMRecoveryRecommendation,
        should_fail: bool = False,
        failure_exception: Optional[LLMProviderError] = None,
        record_messages: bool = True,
    ):
        """
        Initialize the MockLLMProvider.

        Args:
            recommendation: Validated LLMRecoveryRecommendation to return.
            should_fail: If True, raises LLMProviderError during generate().
            failure_exception: Optional custom LLMProviderError to raise if should_fail is True.
            record_messages: If True, stores received message sequences in recorded_messages.
        """
        if not isinstance(recommendation, LLMRecoveryRecommendation):
            raise TypeError(
                f"Expected recommendation to be an instance of LLMRecoveryRecommendation, "
                f"got {type(recommendation).__name__}"
            )

        if failure_exception is not None and not isinstance(failure_exception, LLMProviderError):
            raise TypeError(
                f"Expected failure_exception to be an instance of LLMProviderError, "
                f"got {type(failure_exception).__name__}"
            )

        self._recommendation = recommendation
        self._should_fail = bool(should_fail)
        self._failure_exception = failure_exception
        self._record_messages = bool(record_messages)
        self._recorded_messages: List[List[Mapping[str, str]]] = []

    @property
    def recommendation(self) -> LLMRecoveryRecommendation:
        """Return the configured recommendation."""
        return self._recommendation

    @property
    def recorded_messages(self) -> List[List[Dict[str, str]]]:
        """Return an independent copy of the recorded message batches received by generate()."""
        return [[dict(m) for m in batch] for batch in self._recorded_messages]

    @property
    def last_messages(self) -> Optional[List[Dict[str, str]]]:
        """Return an independent copy of the most recently received message batch, if any."""
        if not self._recorded_messages:
            return None
        return [dict(m) for m in self._recorded_messages[-1]]

    async def generate(
        self,
        messages: Sequence[Mapping[str, str]],
    ) -> LLMRecoveryRecommendation:
        """
        Execute deterministic mock generation.

        Args:
            messages: Sequence of chat messages to validate and optionally record.

        Returns:
            The configured LLMRecoveryRecommendation.

        Raises:
            TypeError / ValueError: If input messages violate the contract.
            LLMProviderError: If configured to fail.
        """
        validate_chat_messages(messages)

        if self._record_messages:
            self._recorded_messages.append([dict(m) for m in messages])

        if self._should_fail:
            if self._failure_exception is not None:
                raise self._failure_exception
            raise LLMProviderError("MockLLMProvider configured execution failure.")

        return self._recommendation
