"""
Tests for Request-Level Observability & Correlation Tracking.

Verifies:
1. A normal request receives a generated request ID (X-Request-ID header & response body).
2. Response X-Request-ID header strictly matches response body request_id.
3. A valid incoming X-Request-ID is preserved across header, context, and response payload.
4. An excessively long X-Request-ID (>128 chars) is rejected and replaced with a clean generated ID.
5. Existing recovery decision behavior remains identical and functional.
6. AuditEvent metadata contains request_id when recovery actions are executed.
7. ContextVar is properly sanitized and cleaned up after request lifecycle.
"""

from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

from app.action_executor import ActionExecutor
from app.agent.orchestrator import AgentOrchestrator
from app.agent.provider import MockLLMProvider
from app.agent.schemas import LLMRecoveryRecommendation
from app.auth import create_access_token
from app.context_retrieval import DEMO_CUSTOMER_UUID, ensure_demo_customer_seeded
from app.database import SessionLocal
from app.decision_engine import RecoveryAction
from app.main import app
from app.models import AuditEvent
from app.observability import (
    get_request_id,
    sanitize_request_id,
)
from app.razorpay_adapter import RazorpayAdapter
from app.recovery_decision_service import (
    RecoveryDecisionService,
    get_recovery_decision_service,
)
from fastapi import status
from fastapi.testclient import TestClient
from sqlalchemy import select

client = TestClient(app)
TEST_CUSTOMER_ID = DEMO_CUSTOMER_UUID
AUTH_HEADERS = {"Authorization": f"Bearer {create_access_token(TEST_CUSTOMER_ID)}"}


def test_normal_request_receives_generated_request_id():
    """Verify that a request without X-Request-ID receives a server-generated request ID."""
    db = SessionLocal()
    ensure_demo_customer_seeded(db)
    db.close()

    payload = {
        "amount": 1000.0,
        "currency": "INR",
        "payment_method": "upi",
        "failure_reason": "bank_timeout",
    }
    response = client.post(
        "/api/recovery/decision",
        json=payload,
        headers=AUTH_HEADERS,
    )
    assert response.status_code == status.HTTP_200_OK
    assert "x-request-id" in response.headers
    header_id = response.headers["x-request-id"]
    assert header_id.startswith("req_") or len(header_id) > 10

    data = response.json()
    assert data.get("request_id") == header_id


def test_response_header_matches_body_request_id():
    """Verify that the X-Request-ID header exactly matches the body's request_id field."""
    payload = {
        "amount": 1500.0,
        "currency": "INR",
        "payment_method": "card",
        "failure_reason": "customer_insufficient_funds",
    }
    response = client.post(
        "/api/recovery/decision",
        json=payload,
        headers=AUTH_HEADERS,
    )
    assert response.status_code == status.HTTP_200_OK
    header_id = response.headers.get("x-request-id")
    body_id = response.json().get("request_id")
    assert header_id is not None
    assert body_id is not None
    assert header_id == body_id


def test_valid_incoming_x_request_id_preserved():
    """Verify that a valid client-supplied X-Request-ID is preserved in headers and body."""
    custom_trace_id = "trace_client_session_abc123_xyz789"
    headers = {**AUTH_HEADERS, "X-Request-ID": custom_trace_id}

    payload = {
        "amount": 2000.0,
        "currency": "INR",
        "payment_method": "netbanking",
        "failure_reason": "gateway_timeout",
    }
    response = client.post(
        "/api/recovery/decision",
        json=payload,
        headers=headers,
    )
    assert response.status_code == status.HTTP_200_OK
    assert response.headers.get("x-request-id") == custom_trace_id
    assert response.json().get("request_id") == custom_trace_id


def test_excessively_long_x_request_id_rejected_and_replaced():
    """Verify that an X-Request-ID > 128 characters is rejected and replaced with a generated ID."""
    oversized_id = "oversized_trace_id_" + "x" * 200
    headers = {**AUTH_HEADERS, "X-Request-ID": oversized_id}

    payload = {
        "amount": 2500.0,
        "currency": "INR",
        "payment_method": "card",
        "failure_reason": "timeout",
    }
    response = client.post(
        "/api/recovery/decision",
        json=payload,
        headers=headers,
    )
    assert response.status_code == status.HTTP_200_OK
    result_id = response.headers.get("x-request-id")
    assert result_id != oversized_id
    assert len(result_id) <= 64
    assert response.json().get("request_id") == result_id


def test_audit_metadata_contains_request_id_on_execution():
    """Verify that executed recovery actions record request_id in the AuditEvent metadata payload."""
    db = SessionLocal()
    ensure_demo_customer_seeded(db)

    mock_adapter = MagicMock(spec=RazorpayAdapter)
    mock_adapter.create_payment_link = AsyncMock(
        return_value={
            "id": "plink_obs_audit_123",
            "short_url": "https://rzp.io/i/obs_audit_123",
            "amount": 300000,
            "currency": "INR",
            "status": "created",
        }
    )
    mock_adapter.dry_run = False
    executor = ActionExecutor(razorpay_adapter=mock_adapter)

    provider = MockLLMProvider(
        recommendation=LLMRecoveryRecommendation(
            recommended_action=RecoveryAction.PAYMENT_LINK,
            confidence=0.92,
            reasoning="Issue payment link for failed transaction.",
        )
    )
    orchestrator = AgentOrchestrator(provider=provider)
    service = RecoveryDecisionService(
        agent_orchestrator=orchestrator,
        action_executor=executor,
    )

    custom_trace_id = f"trace_audit_run_{uuid4().hex[:12]}"
    headers = {**AUTH_HEADERS, "X-Request-ID": custom_trace_id}

    app.dependency_overrides[get_recovery_decision_service] = lambda: service
    try:
        payload = {
            "amount": 3000.0,
            "currency": "INR",
            "payment_method": "card",
            "failure_reason": "customer_auth_failed_otp_timeout",
            "execute_action": True,
            "idempotency_key": f"idem_audit_{uuid4().hex[:12]}",
        }
        response = client.post(
            "/api/recovery/decision",
            json=payload,
            headers=headers,
        )
        assert response.status_code == status.HTTP_200_OK
        assert response.json().get("request_id") == custom_trace_id

        # Verify DB AuditEvent
        latest_audit = (
            db.execute(
                select(AuditEvent)
                .where(AuditEvent.event_type == "recovery_action_executed")
                .order_by(AuditEvent.created_at.desc())
            )
            .scalars()
            .first()
        )
        assert latest_audit is not None
        assert latest_audit.metadata_payload.get("request_id") == custom_trace_id
    finally:
        app.dependency_overrides.pop(get_recovery_decision_service, None)
        db.close()


def test_sanitize_request_id_unit_semantics():
    """Unit test verify sanitation semantics of sanitize_request_id function."""
    # None or empty returns generated
    assert sanitize_request_id(None).startswith("req_")
    assert sanitize_request_id("").startswith("req_")
    assert sanitize_request_id("   ").startswith("req_")

    # Valid string is trimmed and preserved
    assert sanitize_request_id("  my-trace-id  ") == "my-trace-id"

    # Excessively long string returns generated
    assert sanitize_request_id("a" * 129).startswith("req_")
    assert sanitize_request_id("a" * 128) == "a" * 128


def test_context_var_isolated_outside_request():
    """Verify that outside an active request context, get_request_id returns None."""
    assert get_request_id() is None
