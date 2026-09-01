"""
Revora Authentication and Principal Security Dependencies.

Provides a clean, decoupled authentication dependency boundary for FastAPI endpoints.
Validates incoming credentials and provides the AuthenticatedPrincipal identity.
"""

from uuid import UUID

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, ConfigDict

# HTTP Bearer scheme with auto_error=False to allow custom 401 response handling
security_bearer = HTTPBearer(auto_error=False)


class AuthenticatedPrincipal(BaseModel):
    """
    Identity of an authenticated caller / principal.
    """

    model_config = ConfigDict(frozen=True)

    customer_id: UUID


def get_current_principal(
    credentials: HTTPAuthorizationCredentials | None = Depends(security_bearer),  # noqa: B008
) -> AuthenticatedPrincipal:
    """
    FastAPI dependency that extracts and validates the authenticated principal from Bearer token.

    Raises:
        HTTPException: 401 Unauthorized if credentials are missing, invalid, or malformed.
    """
    if credentials is None or not credentials.credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication credentials were not provided.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    token = credentials.credentials.strip()
    try:
        principal_id = UUID(token)
    except (ValueError, TypeError, AttributeError) as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication token: credentials must represent a valid customer UUID.",
            headers={"WWW-Authenticate": "Bearer"},
        ) from exc

    return AuthenticatedPrincipal(customer_id=principal_id)
