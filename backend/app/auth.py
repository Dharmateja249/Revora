"""
Revora Authentication and Principal Security Dependencies.

Provides a clean, decoupled authentication dependency boundary for FastAPI endpoints.
Issues and validates cryptographically verifiable tokens bound to customer identities.
"""

import base64
import hashlib
import hmac
import time
from uuid import UUID

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, ConfigDict

from app.config import get_settings

# HTTP Bearer scheme with auto_error=False to allow custom 401 response handling
security_bearer = HTTPBearer(auto_error=False)

TOKEN_PREFIX = "rvra_tok_"
DEFAULT_TOKEN_EXPIRY_SECONDS = 86400  # 24 hours


class AuthenticatedPrincipal(BaseModel):
    """
    Identity of an authenticated caller / principal.
    """

    model_config = ConfigDict(frozen=True)

    customer_id: UUID


def is_known_demo_customer(customer_id: UUID | str) -> bool:
    """
    Check if the customer ID matches an authorized demo customer profile.
    """
    try:
        target_uuid = UUID(str(customer_id))
    except (ValueError, AttributeError):
        return False
    settings = get_settings()
    known = {UUID(str(cid)) for cid in settings.DEMO_CUSTOMER_IDS}
    return target_uuid in known


def create_access_token(
    customer_id: UUID | str,
    expires_in_seconds: int = DEFAULT_TOKEN_EXPIRY_SECONDS,
    secret_key: str | None = None,
) -> str:
    """
    Issue a cryptographically verifiable token bound to a customer ID.

    Format: rvra_tok_{base64url(customer_id:expiry_timestamp)}.{hmac_signature}
    """
    cust_uuid = UUID(str(customer_id))
    expiry = int(time.time()) + expires_in_seconds
    payload_str = f"{cust_uuid}:{expiry}"
    payload_b64 = (
        base64.urlsafe_b64encode(payload_str.encode("utf-8"))
        .decode("ascii")
        .rstrip("=")
    )

    key = (secret_key or get_settings().AUTH_SECRET_KEY).encode("utf-8")
    sig = hmac.new(key, payload_b64.encode("ascii"), hashlib.sha256).hexdigest()[:32]
    return f"{TOKEN_PREFIX}{payload_b64}.{sig}"


def verify_access_token(
    token: str,
    secret_key: str | None = None,
) -> UUID:
    """
    Verify and decode a server-issued token.

    Raises:
        ValueError: If token format is invalid, signature verification fails, or token has expired.
    """
    clean_token = token.strip()
    if not clean_token.startswith(TOKEN_PREFIX):
        raise ValueError(
            "Invalid token format: credentials must be a verifiable server-issued token."
        )

    token_body = clean_token[len(TOKEN_PREFIX) :]
    parts = token_body.split(".")
    if len(parts) != 2:
        raise ValueError("Malformed authentication token structure.")

    payload_b64, provided_sig = parts[0], parts[1]
    key = (secret_key or get_settings().AUTH_SECRET_KEY).encode("utf-8")
    expected_sig = hmac.new(
        key, payload_b64.encode("ascii"), hashlib.sha256
    ).hexdigest()[:32]

    if not hmac.compare_digest(provided_sig, expected_sig):
        raise ValueError("Authentication token signature verification failed.")

    # Re-pad base64
    padded_b64 = payload_b64 + "=" * (-len(payload_b64) % 4)
    try:
        decoded = base64.urlsafe_b64decode(padded_b64.encode("ascii")).decode("utf-8")
        cust_str, expiry_str = decoded.split(":", 1)
        expiry = int(expiry_str)
        customer_id = UUID(cust_str)
    except Exception as exc:
        raise ValueError(f"Failed to decode token payload: {exc}") from exc

    if time.time() > expiry:
        raise ValueError("Authentication token has expired.")

    return customer_id


def get_current_principal(
    credentials: HTTPAuthorizationCredentials | None = Depends(security_bearer),  # noqa: B008
) -> AuthenticatedPrincipal:
    """
    FastAPI dependency that extracts and validates the authenticated principal from Bearer token.

    Raises:
        HTTPException: 401 Unauthorized if credentials are missing, invalid, expired, or malformed.
    """
    if credentials is None or not credentials.credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication credentials were not provided.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    token = credentials.credentials.strip()
    try:
        principal_id = verify_access_token(token)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid authentication token: {exc}",
            headers={"WWW-Authenticate": "Bearer"},
        ) from exc

    return AuthenticatedPrincipal(customer_id=principal_id)
