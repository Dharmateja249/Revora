"""
Unit Tests for Revora Authentication and Verifiable Token System.

Verifies:
1. Token issuance: create_access_token generates HMAC-signed tokens bound to customer UUIDs.
2. Token verification: verify_access_token decodes valid tokens correctly.
3. Tampered signature rejection.
4. Expired token rejection.
5. Malformed / non-server token rejection.
6. get_current_principal FastAPI dependency behavior with 401 status.
"""

from uuid import UUID, uuid4

import pytest
from app.auth import (
    AuthenticatedPrincipal,
    create_access_token,
    get_current_principal,
    is_known_demo_customer,
    verify_access_token,
)
from fastapi import HTTPException
from fastapi.security import HTTPAuthorizationCredentials


def test_create_and_verify_valid_token():
    """Verify standard round-trip issuance and verification."""
    customer_id = uuid4()
    token = create_access_token(customer_id)

    assert token.startswith("rvra_tok_")
    extracted_id = verify_access_token(token)
    assert extracted_id == customer_id


def test_tampered_signature_rejected():
    """Verify that tampering with token body or signature fails verification."""
    customer_id = uuid4()
    token = create_access_token(customer_id)
    # Alter the signature
    tampered_token = token[:-4] + "dead"

    with pytest.raises(ValueError, match="signature verification failed"):
        verify_access_token(tampered_token)


def test_expired_token_rejected():
    """Verify that an expired token raises ValueError."""
    customer_id = uuid4()
    # Expire 10 seconds ago
    expired_token = create_access_token(customer_id, expires_in_seconds=-10)

    with pytest.raises(ValueError, match="token has expired"):
        verify_access_token(expired_token)


def test_raw_uuid_or_malformed_token_rejected():
    """Verify that raw customer UUID or non-Revora token string is rejected."""
    raw_uuid = str(uuid4())
    with pytest.raises(
        ValueError, match="credentials must be a verifiable server-issued token"
    ):
        verify_access_token(raw_uuid)

    with pytest.raises(
        ValueError, match="credentials must be a verifiable server-issued token"
    ):
        verify_access_token("Bearer some_random_token")


def test_get_current_principal_valid_token():
    """Verify get_current_principal returns AuthenticatedPrincipal for valid bearer credentials."""
    customer_id = uuid4()
    token = create_access_token(customer_id)
    creds = HTTPAuthorizationCredentials(scheme="Bearer", credentials=token)

    principal = get_current_principal(creds)
    assert isinstance(principal, AuthenticatedPrincipal)
    assert principal.customer_id == customer_id


def test_get_current_principal_missing_creds_raises_401():
    """Verify get_current_principal raises 401 when credentials are None."""
    with pytest.raises(HTTPException) as exc:
        get_current_principal(None)
    assert exc.value.status_code == 401
    assert "credentials were not provided" in exc.value.detail


def test_get_current_principal_invalid_token_raises_401():
    """Verify get_current_principal raises 401 when token is invalid."""
    creds = HTTPAuthorizationCredentials(scheme="Bearer", credentials="invalid-token")
    with pytest.raises(HTTPException) as exc:
        get_current_principal(creds)
    assert exc.value.status_code == 401
    assert "Invalid authentication token" in exc.value.detail


def test_is_known_demo_customer():
    """Verify is_known_demo_customer accurately identifies allowed demo profiles."""
    known_id = UUID("e9cd4c97-979b-4753-9925-640623f74eee")
    assert is_known_demo_customer(known_id) is True
    assert is_known_demo_customer(str(known_id)) is True

    # Random unknown UUID
    assert is_known_demo_customer(uuid4()) is False
    assert is_known_demo_customer("not-a-valid-uuid") is False
