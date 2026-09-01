"""
Unit Tests for Recovery Evaluation API Request and Response DTO Schemas.

Tests:
1. Valid request creation and UUID parsing
2. Default value for use_rag is True
3. Invalid UUIDs in request raise ValidationError
4. Extra fields in request are forbidden
5. Invalid types in request raise ValidationError
6. Valid response construction and field preservation
7. Response JSON serialization (model_dump_json) and dictionary export (model_dump)
8. Response confidence bounds enforcement [0.0, 1.0]
9. Negative retrieved_evidence_count rejection
10. Response immutability (frozen=True)
"""

import json
import uuid
from datetime import datetime

import pytest
from app.decision_engine import RecoveryAction
from app.schemas.recovery import (
    RecoveryEvaluationRequest,
    RecoveryEvaluationResponse,
)
from pydantic import ValidationError

# ============================================================================
# 1. RecoveryEvaluationRequest Tests
# ============================================================================


def test_valid_request_with_uuid_objects():
    """Verify request parses UUID objects correctly."""
    cust_id = uuid.uuid4()
    pay_id = uuid.uuid4()

    req = RecoveryEvaluationRequest(
        customer_id=cust_id,
        payment_id=pay_id,
        use_rag=False,
    )

    assert req.customer_id == cust_id
    assert req.payment_id == pay_id
    assert req.use_rag is False


def test_valid_request_with_string_uuids():
    """Verify request coerces valid UUID strings to UUID objects."""
    cust_str = "12345678-1234-5678-1234-567812345678"
    pay_str = "87654321-4321-8765-4321-876543218765"

    req = RecoveryEvaluationRequest(
        customer_id=cust_str,  # type: ignore
        payment_id=pay_str,  # type: ignore
    )

    assert isinstance(req.customer_id, uuid.UUID)
    assert str(req.customer_id) == cust_str
    assert isinstance(req.payment_id, uuid.UUID)
    assert str(req.payment_id) == pay_str
    # Verify default use_rag is True
    assert req.use_rag is True


def test_request_invalid_uuid_rejected():
    """Verify malformed UUID strings raise ValidationError."""
    with pytest.raises(ValidationError) as exc:
        RecoveryEvaluationRequest(
            customer_id="not-a-valid-uuid",  # type: ignore
            payment_id=uuid.uuid4(),
        )
    assert "customer_id" in str(exc.value)

    with pytest.raises(ValidationError) as exc:
        RecoveryEvaluationRequest(
            customer_id=uuid.uuid4(),
            payment_id=12345,  # type: ignore
        )
    assert "payment_id" in str(exc.value)


def test_request_extra_fields_forbidden():
    """Verify extra attributes are forbidden to prevent malformed payloads."""
    with pytest.raises(ValidationError) as exc:
        RecoveryEvaluationRequest(
            customer_id=uuid.uuid4(),
            payment_id=uuid.uuid4(),
            unexpected_field="disallowed",  # type: ignore
        )
    assert (
        "extra_forbidden" in str(exc.value).lower()
        or "unexpected_field" in str(exc.value).lower()
    )


def test_request_immutability():
    """Verify request instance cannot be mutated."""
    req = RecoveryEvaluationRequest(
        customer_id=uuid.uuid4(),
        payment_id=uuid.uuid4(),
    )
    with pytest.raises(ValidationError):
        req.use_rag = False  # type: ignore


# ============================================================================
# 2. RecoveryEvaluationResponse Tests
# ============================================================================


def test_valid_response_construction():
    """Verify response DTO accepts all required and optional fields."""
    pay_id = uuid.uuid4()
    cust_id = uuid.uuid4()
    opp_id = uuid.uuid4()

    resp = RecoveryEvaluationResponse(
        payment_id=pay_id,
        customer_id=cust_id,
        opportunity_id=opp_id,
        recommended_action=RecoveryAction.PAYMENT_LINK,
        reason="Customer historically recovered via payment link.",
        confidence=0.85,
        decision_basis={
            "rule_matched": "InsufficientFundsHistoricalLinkRule",
            "historical_evidence_used": True,
        },
        historical_rag_used=True,
        retrieved_evidence_count=3,
    )

    assert resp.payment_id == pay_id
    assert resp.customer_id == cust_id
    assert resp.opportunity_id == opp_id
    assert resp.recommended_action == RecoveryAction.PAYMENT_LINK
    assert resp.confidence == 0.85
    assert resp.historical_rag_used is True
    assert resp.retrieved_evidence_count == 3
    assert isinstance(resp.evaluated_at, datetime)
    assert resp.decision_basis["rule_matched"] == "InsufficientFundsHistoricalLinkRule"


def test_response_serialization():
    """Verify response serializes cleanly to JSON and dictionary format."""
    pay_id = uuid.uuid4()
    cust_id = uuid.uuid4()

    resp = RecoveryEvaluationResponse(
        payment_id=pay_id,
        customer_id=cust_id,
        recommended_action=RecoveryAction.RETRY_PAYMENT,
        reason="Transient network timeout.",
        confidence=0.90,
    )

    # model_dump()
    data = resp.model_dump()
    assert data["payment_id"] == pay_id
    assert data["recommended_action"] == RecoveryAction.RETRY_PAYMENT
    assert data["opportunity_id"] is None
    assert data["historical_rag_used"] is False
    assert data["retrieved_evidence_count"] == 0

    # model_dump_json()
    json_str = resp.model_dump_json()
    parsed = json.loads(json_str)
    assert parsed["payment_id"] == str(pay_id)
    assert parsed["recommended_action"] == "retry_payment"
    assert parsed["confidence"] == 0.90


def test_response_confidence_bounds_enforced():
    """Verify confidence outside [0.0, 1.0] raises ValidationError."""
    pay_id = uuid.uuid4()
    cust_id = uuid.uuid4()

    with pytest.raises(ValidationError):
        RecoveryEvaluationResponse(
            payment_id=pay_id,
            customer_id=cust_id,
            recommended_action=RecoveryAction.NO_ACTION,
            reason="Test",
            confidence=1.1,  # Out of bounds
        )

    with pytest.raises(ValidationError):
        RecoveryEvaluationResponse(
            payment_id=pay_id,
            customer_id=cust_id,
            recommended_action=RecoveryAction.NO_ACTION,
            reason="Test",
            confidence=-0.01,  # Out of bounds
        )


def test_response_negative_evidence_count_rejected():
    """Verify negative retrieved_evidence_count is rejected."""
    with pytest.raises(ValidationError):
        RecoveryEvaluationResponse(
            payment_id=uuid.uuid4(),
            customer_id=uuid.uuid4(),
            recommended_action=RecoveryAction.NO_ACTION,
            reason="Test",
            confidence=0.5,
            retrieved_evidence_count=-1,
        )


def test_response_immutability():
    """Verify response DTO is frozen."""
    resp = RecoveryEvaluationResponse(
        payment_id=uuid.uuid4(),
        customer_id=uuid.uuid4(),
        recommended_action=RecoveryAction.RETRY_PAYMENT,
        reason="Test",
        confidence=0.8,
    )
    with pytest.raises(ValidationError):
        resp.confidence = 0.99  # type: ignore


def test_request_use_agent_defaults_to_none():
    """Verify use_agent in request defaults to None when omitted."""
    req = RecoveryEvaluationRequest(
        customer_id=uuid.uuid4(),
        payment_id=uuid.uuid4(),
    )
    assert req.use_agent is None


def test_request_use_agent_explicit_boolean_values():
    """Verify use_agent explicitly parses True, False, and None."""
    cust_id = uuid.uuid4()
    pay_id = uuid.uuid4()

    req_true = RecoveryEvaluationRequest(
        customer_id=cust_id,
        payment_id=pay_id,
        use_agent=True,
    )
    assert req_true.use_agent is True

    req_false = RecoveryEvaluationRequest(
        customer_id=cust_id,
        payment_id=pay_id,
        use_agent=False,
    )
    assert req_false.use_agent is False

    req_none = RecoveryEvaluationRequest(
        customer_id=cust_id,
        payment_id=pay_id,
        use_agent=None,
    )
    assert req_none.use_agent is None


def test_response_agent_telemetry_defaults():
    """Verify default values for agent telemetry fields in response DTO."""
    resp = RecoveryEvaluationResponse(
        payment_id=uuid.uuid4(),
        customer_id=uuid.uuid4(),
        recommended_action=RecoveryAction.RETRY_PAYMENT,
        reason="Default evaluation",
        confidence=0.75,
    )
    assert resp.agent_used is False
    assert resp.is_fallback is False
    assert resp.fallback_reason is None

    dumped = resp.model_dump()
    assert dumped["agent_used"] is False
    assert dumped["is_fallback"] is False
    assert dumped["fallback_reason"] is None


def test_response_agent_telemetry_explicit_values():
    """Verify explicit values for agent telemetry fields in response DTO."""
    resp = RecoveryEvaluationResponse(
        payment_id=uuid.uuid4(),
        customer_id=uuid.uuid4(),
        recommended_action=RecoveryAction.PAYMENT_LINK,
        reason="Deterministic fallback applied",
        confidence=0.0,
        agent_used=False,
        is_fallback=True,
        fallback_reason="LLM provider failure; deterministic fallback applied",
    )
    assert resp.agent_used is False
    assert resp.is_fallback is True
    assert (
        resp.fallback_reason == "LLM provider failure; deterministic fallback applied"
    )

    # Verify JSON serialization round-trip
    parsed = json.loads(resp.model_dump_json())
    assert parsed["agent_used"] is False
    assert parsed["is_fallback"] is True
    assert (
        parsed["fallback_reason"]
        == "LLM provider failure; deterministic fallback applied"
    )


def test_settings_enable_agent_decision_engine_default_and_env_override(monkeypatch):
    """Verify Settings ENABLE_AGENT_DECISION_ENGINE defaults to False and respects env override."""
    from app.config import Settings

    # Default check
    settings_default = Settings(_env_file=None)
    assert settings_default.ENABLE_AGENT_DECISION_ENGINE is False

    # Env override check
    monkeypatch.setenv("ENABLE_AGENT_DECISION_ENGINE", "true")
    settings_enabled = Settings(_env_file=None)
    assert settings_enabled.ENABLE_AGENT_DECISION_ENGINE is True
