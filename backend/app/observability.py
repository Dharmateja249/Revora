"""
Revora Request-Level Observability & Correlation Tracking.

Provides request context isolation via ContextVar, deterministic request ID extraction
and validation, and lightweight structured logging helpers.
"""

import logging
from contextvars import ContextVar, Token
from uuid import uuid4

logger = logging.getLogger("revora.observability")

_REQUEST_ID_CTX_VAR: ContextVar[str | None] = ContextVar("request_id", default=None)
MAX_REQUEST_ID_LENGTH = 128


def get_request_id() -> str | None:
    """Retrieve the active request/correlation ID for the current async task context."""
    return _REQUEST_ID_CTX_VAR.get()


def set_request_id(request_id: str | None) -> Token:
    """Set the active request ID in the current ContextVar context and return the token."""
    return _REQUEST_ID_CTX_VAR.set(request_id)


def reset_request_id(token: Token) -> None:
    """Reset the ContextVar using the provided token."""
    _REQUEST_ID_CTX_VAR.reset(token)


def generate_request_id() -> str:
    """Generate a clean, unique server-side request ID."""
    return f"req_{uuid4().hex}"


def sanitize_request_id(header_val: str | None) -> str:
    """
    Validate and sanitize client-provided X-Request-ID header value.
    Rejects/replaces empty, whitespace-only, or values > 128 chars with a server-generated ID.
    """
    if not header_val:
        return generate_request_id()
    cleaned = str(header_val).strip()
    if not cleaned or len(cleaned) > MAX_REQUEST_ID_LENGTH:
        return generate_request_id()
    return cleaned
