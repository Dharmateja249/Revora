"""
API Boundary Tests for Revora Decision API (POST /api/recovery/decision).

Tests:
1. Health endpoint regression: GET /health returns 200 and healthy status.
2. Successful recovery decision: POST /api/recovery/decision returns 200 with full explainability.
3. Request validation failures:
   - Missing required fields (amount, payment_method, failure_reason) -> 422
   - Non-positive amount -> 422
   - Extra fields -> 422
4. Deterministic fallback when LLM provider fails -> 200 with is_fallback=True, agent_used=False.
5. Policy override: LLM recommending prohibited action is overridden by PolicyValidator -> policy_overridden=True.
6. Multi-attempt history and attempt budget propagation.
7. Unexpected service exception returns clean 500 without leaking secrets or stack traces.
"""

from unittest.mock import AsyncMock, MagicMock

from fastapi import status
from fastapi.testclient import TestClient

from app.agent.orchestrator import AgentOrchestrator
from app.agent.provider import LLMProviderError, MockLLMProvider
from app.agent.schemas import LLMRecoveryRecommendation
from app.decision_engine import RecoveryAction
from app.main import app
from app.recovery_decision_service import (
    RecoveryDecisionService,
    get_recovery_decision_service,
)

client = TestClient(app)


# ============================================================================
# 1. Health Endpoint Tests
# ============================================================================


def test_health_endpoint():
    """Verify that GET /health returns 200 and valid JSON health status."""
    response = client.get("/health")
    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert data["status"] == "healthy"
    assert data["app"] == "Revora"
    assert "version" in data


# ============================================================================
# 2. Successful Decision Evaluation
# ============================================================================


def test_recovery_decision_success():
    """Verify POST /api/recovery/decision produces 200 with full decision details."""
    payload = {
        "amount": 4999.0,
        "currency": "INR",
        "payment_method": "upi",
        "failure_reason": "bank_technical_timeout",
        "payment_status": "failed",
        "customer": {
            "total_payments": 10,
            "successful_payments": 9,
            "failed_payments": 1,
            "historical_success_rate": 0.9,
        },
        "previous_attempts": [],
        "opportunity_status": "open",
        "max_attempts": 3,
    }

    response = client.post("/api/recovery/decision", json=payload)
    assert response.status_code == status.HTTP_200_OK

    data = response.json()
    assert "recommended_action" in data
    assert data["recommended_action"] in [a.value for a in RecoveryAction]
    assert isinstance(data["confidence"], float)
    assert 0.0 <= data["confidence"] <= 1.0
    assert isinstance(data["reasoning"], str)
    assert len(data["reasoning"]) > 0
    assert isinstance(data["key_factors"], list)
    assert isinstance(data["referenced_case_ids"], list)
    assert data["agent_used"] is True
    assert data["policy_overridden"] is False
    assert data["is_fallback"] is False
    assert data["fallback_reason"] is None

    # Verify action execution telemetry
    assert "execution" in data
    assert data["execution"] is not None
    assert data["execution"]["action"] == data["recommended_action"]
    assert data["execution"]["status"] in ("skipped", "simulated", "success")


def test_recovery_decision_payment_link_execution():
    """Verify approved PAYMENT_LINK recommendation executes via RazorpayAdapter."""
    stub_recommendation = LLMRecoveryRecommendation(
        recommended_action=RecoveryAction.PAYMENT_LINK,
        confidence=0.92,
        reasoning="Customer requires interactive payment link to complete transaction.",
        key_factors=["interactive_link_suitable"],
    )
    provider = MockLLMProvider(recommendation=stub_recommendation)
    orchestrator = AgentOrchestrator(provider=provider)
    service = RecoveryDecisionService(agent_orchestrator=orchestrator)

    app.dependency_overrides[get_recovery_decision_service] = lambda: service
    try:
        payload = {
            "amount": 2500.0,
            "currency": "INR",
            "payment_method": "upi",
            "failure_reason": "bank_timeout",
            "execute_action": True,
        }
        response = client.post("/api/recovery/decision", json=payload)
        assert response.status_code == status.HTTP_200_OK

        data = response.json()
        assert data["recommended_action"] == RecoveryAction.PAYMENT_LINK.value
        assert data["execution"] is not None
        assert data["execution"]["action"] == RecoveryAction.PAYMENT_LINK.value
        assert data["execution"]["attempted"] is True
        assert data["execution"]["success"] is True
        assert data["execution"]["status"] in ("simulated", "success")
        assert data["execution"]["reference_id"] is not None
        assert "plink_" in data["execution"]["reference_id"]
        assert data["execution"]["resource_url"] is not None
        assert "https://rzp.io/i/" in data["execution"]["resource_url"]
    finally:
        app.dependency_overrides.pop(get_recovery_decision_service, None)


def test_recovery_decision_execute_action_flag_false():
    """Verify that when execute_action is False, decision is returned without executing action."""
    payload = {
        "amount": 1000.0,
        "currency": "INR",
        "payment_method": "upi",
        "failure_reason": "bank_timeout",
        "execute_action": False,
    }
    response = client.post("/api/recovery/decision", json=payload)
    assert response.status_code == status.HTTP_200_OK

    data = response.json()
    assert data["execution"] is None


# ============================================================================
# 3. Request Validation Failures (422)
# ============================================================================


def test_recovery_decision_missing_required_fields():
    """Verify missing amount, payment_method, or failure_reason returns 422."""
    # Missing amount
    res1 = client.post(
        "/api/recovery/decision",
        json={"payment_method": "upi", "failure_reason": "timeout"},
    )
    assert res1.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    # Missing payment_method
    res2 = client.post(
        "/api/recovery/decision",
        json={"amount": 100.0, "failure_reason": "timeout"},
    )
    assert res2.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    # Missing failure_reason
    res3 = client.post(
        "/api/recovery/decision",
        json={"amount": 100.0, "payment_method": "card"},
    )
    assert res3.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY


def test_recovery_decision_invalid_amount():
    """Verify non-positive amounts return 422."""
    res_zero = client.post(
        "/api/recovery/decision",
        json={"amount": 0.0, "payment_method": "card", "failure_reason": "timeout"},
    )
    assert res_zero.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    res_neg = client.post(
        "/api/recovery/decision",
        json={"amount": -50.0, "payment_method": "card", "failure_reason": "timeout"},
    )
    assert res_neg.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY


def test_recovery_decision_extra_fields_forbidden():
    """Verify unexpected fields trigger 422 Unprocessable Content."""
    payload = {
        "amount": 1500.0,
        "payment_method": "card",
        "failure_reason": "timeout",
        "unauthorized_field": "injected_data",
    }
    response = client.post("/api/recovery/decision", json=payload)
    assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY


# ============================================================================
# 4. Deterministic Fallback on LLM Provider Failure
# ============================================================================


def test_recovery_decision_fallback_on_provider_failure():
    """Verify that when LLM provider fails, system returns 200 with deterministic fallback."""
    failing_provider = MockLLMProvider(
        recommendation=LLMRecoveryRecommendation(
            recommended_action=RecoveryAction.RETRY_PAYMENT,
            confidence=0.5,
            reasoning="Default mock recommendation",
        ),
        should_fail=True,
        failure_exception=LLMProviderError("Simulated LLM gateway failure"),
    )
    failing_orchestrator = AgentOrchestrator(provider=failing_provider)
    failing_service = RecoveryDecisionService(agent_orchestrator=failing_orchestrator)

    app.dependency_overrides[get_recovery_decision_service] = lambda: failing_service
    try:
        payload = {
            "amount": 2000.0,
            "currency": "INR",
            "payment_method": "upi",
            "failure_reason": "gateway_timeout",
        }
        response = client.post("/api/recovery/decision", json=payload)
        assert response.status_code == status.HTTP_200_OK

        data = response.json()
        assert data["is_fallback"] is True
        assert data["agent_used"] is False
        assert "LLM provider failure" in data["fallback_reason"]
        assert data["recommended_action"] in [a.value for a in RecoveryAction]
    finally:
        app.dependency_overrides.pop(get_recovery_decision_service, None)


# ============================================================================
# 5. Policy Override Verification
# ============================================================================


def test_recovery_decision_policy_override():
    """
    Verify PolicyValidator overrides candidate action when LLM recommends prohibited action.
    For card payment requiring 2FA (Razorpay rule), RETRY_PAYMENT is prohibited;
    only PAYMENT_LINK is permitted.
    """
    # LLM recommends RETRY_PAYMENT
    stub_recommendation = LLMRecoveryRecommendation(
        recommended_action=RecoveryAction.RETRY_PAYMENT,
        confidence=0.95,
        reasoning="Attempting direct automated retry.",
        key_factors=["customer_retry_likely"],
    )
    provider = MockLLMProvider(recommendation=stub_recommendation)
    orchestrator = AgentOrchestrator(provider=provider)
    service = RecoveryDecisionService(agent_orchestrator=orchestrator)

    app.dependency_overrides[get_recovery_decision_service] = lambda: service
    try:
        # Failure reason that triggers RZP_CUSTOMER_AUTH_2FA_REQUIRED_RULE
        payload = {
            "amount": 3500.0,
            "currency": "INR",
            "payment_method": "card",
            "failure_reason": "customer_auth_failed_otp_timeout",
        }
        response = client.post("/api/recovery/decision", json=payload)
        assert response.status_code == status.HTTP_200_OK

        data = response.json()
        assert data["policy_overridden"] is True
        # Overridden to compliant action (PAYMENT_LINK)
        assert data["recommended_action"] == RecoveryAction.PAYMENT_LINK.value
        assert data["execution"] is not None
        assert data["execution"]["action"] == RecoveryAction.PAYMENT_LINK.value
        assert data["execution"]["success"] is True
        assert data["execution"]["reference_id"] is not None
    finally:
        app.dependency_overrides.pop(get_recovery_decision_service, None)


# ============================================================================
# 6. Previous Attempts and Lifecycle State
# ============================================================================


def test_recovery_decision_with_previous_attempts():
    """Verify previous attempts are accepted and attempt budget is honored."""
    payload = {
        "amount": 1000.0,
        "currency": "INR",
        "payment_method": "upi",
        "failure_reason": "insufficient_funds",
        "previous_attempts": [
            {
                "action": "retry_payment",
                "status": "failed",
                "error_code": "INSUFFICIENT_FUNDS",
            },
            {
                "action": "payment_link",
                "status": "failed",
                "error_code": "LINK_EXPIRED",
            },
        ],
        "max_attempts": 3,
    }

    response = client.post("/api/recovery/decision", json=payload)
    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert data["recommended_action"] in [a.value for a in RecoveryAction]


# ============================================================================
# 7. Unexpected Service Error Masking (500)
# ============================================================================


def test_recovery_decision_unexpected_error_masks_details():
    """Verify unexpected application exceptions return clean 500 without leaking stack traces or secrets."""
    mock_service = MagicMock(spec=RecoveryDecisionService)
    mock_service.evaluate_decision = AsyncMock(
        side_effect=RuntimeError("Internal database crash with api_key_secret_99999")
    )

    app.dependency_overrides[get_recovery_decision_service] = lambda: mock_service
    try:
        payload = {
            "amount": 500.0,
            "currency": "INR",
            "payment_method": "card",
            "failure_reason": "internal_error",
        }
        response = client.post("/api/recovery/decision", json=payload)
        assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR

        data = response.json()
        assert (
            data["detail"]
            == "An unexpected error occurred while evaluating the recovery decision."
        )
        assert "api_key_secret_99999" not in response.text
        assert "RuntimeError" not in response.text
    finally:
        app.dependency_overrides.pop(get_recovery_decision_service, None)


def test_recovery_decision_reports_execution_failure():
    """Verify that when action execution fails, API returns 200 with failed execution status."""
    from app.action_executor import ActionExecutor
    from app.razorpay_adapter import RazorpayAdapter, RazorpayAPIError

    mock_adapter = MagicMock(spec=RazorpayAdapter)
    mock_adapter.create_payment_link = AsyncMock(
        side_effect=RazorpayAPIError("Razorpay gateway temporary 502 Bad Gateway")
    )
    mock_adapter.dry_run = False
    executor = ActionExecutor(razorpay_adapter=mock_adapter)

    # Re-use real orchestrator with mock provider
    provider = MockLLMProvider(
        recommendation=LLMRecoveryRecommendation(
            recommended_action=RecoveryAction.PAYMENT_LINK,
            confidence=0.9,
            reasoning="Payment link needed.",
        )
    )
    orchestrator = AgentOrchestrator(provider=provider)
    service = RecoveryDecisionService(
        agent_orchestrator=orchestrator,
        action_executor=executor,
    )

    app.dependency_overrides[get_recovery_decision_service] = lambda: service
    try:
        payload = {
            "amount": 1200.0,
            "currency": "INR",
            "payment_method": "upi",
            "failure_reason": "bank_timeout",
            "execute_action": True,
        }
        response = client.post("/api/recovery/decision", json=payload)
        assert response.status_code == status.HTTP_200_OK

        data = response.json()
        assert data["recommended_action"] == RecoveryAction.PAYMENT_LINK.value
        assert data["execution"] is not None
        assert data["execution"]["attempted"] is True
        assert data["execution"]["success"] is False
        assert data["execution"]["status"] == "failed"
        assert "502 Bad Gateway" in data["execution"]["error"]
    finally:
        app.dependency_overrides.pop(get_recovery_decision_service, None)
