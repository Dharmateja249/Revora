"""
API Boundary Tests for FastAPI Recovery Evaluation Router (/v1/recovery/evaluate-decision).

Tests:
1. Happy path: POST /v1/recovery/evaluate-decision returns 200 with complete RecoveryEvaluationResponse
2. Request validation: missing customer_id, missing payment_id, malformed UUID, extra fields (422)
3. Domain exception mappings:
   - CustomerNotFoundError -> HTTP 404
   - PaymentNotFoundError -> HTTP 404
   - RecoveryOpportunityNotFoundError -> HTTP 404
   - PaymentCustomerMismatchError -> HTTP 403
4. RAG disabled (use_rag=False): returns 200 with historical_rag_used=False
5. Service isolation: dependency override of get_recovery_service for isolated HTTP boundary testing
6. Health endpoint regression: GET /health returns 200
7. OpenAPI schema regression: /openapi.json contains POST /v1/recovery/evaluate-decision
"""

from datetime import datetime, timezone, timedelta
import uuid
from fastapi import status
from fastapi.testclient import TestClient
import pytest
from sqlalchemy import create_engine
from sqlalchemy.pool import StaticPool
from sqlalchemy.orm import sessionmaker

from app.context import (
    CustomerNotFoundError,
    PaymentCustomerMismatchError,
    PaymentNotFoundError,
    RecoveryOpportunityNotFoundError,
)
from app.database import Base, get_db
from app.decision_engine import RecoveryAction
from app.main import app
from app.models import Customer, Payment, RecoveryOpportunity, utc_now
from app.recovery_service import RecoveryService
from app.routers.recovery import get_recovery_service
from app.schemas.recovery import (
    RecoveryEvaluationRequest,
    RecoveryEvaluationResponse,
)


# ============================================================================
# Database & TestClient Fixtures
# ============================================================================


@pytest.fixture
def test_db_session():
    """Create a temporary in-memory database session."""
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine, autocommit=False, autoflush=False)
    session = Session()
    try:
        yield session
    finally:
        session.close()
        Base.metadata.drop_all(engine)


@pytest.fixture
def client(test_db_session):
    """Create a TestClient with database dependency override."""
    app.dependency_overrides[get_db] = lambda: test_db_session
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


def _seed_customer_payment_opportunity(
    session,
    failure_reason="bank_timeout",
    payment_method="upi",
    amount=3000.0,
):
    """Seed test customer, failed payment, and recovery opportunity in the DB."""
    cust_id = uuid.uuid4()
    pay_id = uuid.uuid4()
    opp_id = uuid.uuid4()
    now = utc_now()

    ext_suffix = uuid.uuid4().hex[:8]
    customer = Customer(
        id=cust_id,
        external_customer_id=f"cust_ext_{ext_suffix}",
        name="Bob Verma",
        email=f"bob_{ext_suffix}@example.com",
        total_payments=3,
        successful_payments=2,
        failed_payments=1,
        created_at=now - timedelta(days=20),
    )
    payment = Payment(
        id=pay_id,
        external_payment_id=f"pay_ext_{ext_suffix}",
        customer_id=cust_id,
        amount=amount,
        currency="INR",
        payment_method=payment_method,
        status="failed",
        failure_reason=failure_reason,
        created_at=now,
    )
    opportunity = RecoveryOpportunity(
        id=opp_id,
        payment_id=pay_id,
        status="open",
        revenue_at_risk=amount,
        expected_recovery=amount * 0.75,
        created_at=now,
    )

    session.add_all([customer, payment, opportunity])
    session.commit()
    return customer, payment, opportunity


# ============================================================================
# 1. Happy Path & RAG Disabled Tests
# ============================================================================


def test_evaluate_decision_endpoint_happy_path(client, test_db_session):
    """Verify POST /v1/recovery/evaluate-decision returns 200 with complete response."""
    customer, payment, opportunity = _seed_customer_payment_opportunity(test_db_session)

    payload = {
        "customer_id": str(customer.id),
        "payment_id": str(payment.id),
        "use_rag": True,
    }

    response = client.post("/v1/recovery/evaluate-decision", json=payload)

    assert response.status_code == status.HTTP_200_OK
    data = response.json()

    assert data["payment_id"] == str(payment.id)
    assert data["customer_id"] == str(customer.id)
    assert data["opportunity_id"] == str(opportunity.id)
    assert data["recommended_action"] in [a.value for a in RecoveryAction]
    assert 0.0 <= data["confidence"] <= 1.0
    assert len(data["reason"]) > 0
    assert isinstance(data["decision_basis"], dict)
    assert data["historical_rag_used"] is True
    assert data["retrieved_evidence_count"] >= 0
    assert "evaluated_at" in data

    # Verify no PII fields
    assert "email" not in data
    assert "name" not in data


def test_evaluate_decision_endpoint_rag_disabled(client, test_db_session):
    """Verify POST /v1/recovery/evaluate-decision with use_rag=False."""
    customer, payment, opportunity = _seed_customer_payment_opportunity(test_db_session)

    payload = {
        "customer_id": str(customer.id),
        "payment_id": str(payment.id),
        "use_rag": False,
    }

    response = client.post("/v1/recovery/evaluate-decision", json=payload)

    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert data["historical_rag_used"] is False
    assert data["retrieved_evidence_count"] == 0


# ============================================================================
# 2. Request Validation Tests (HTTP 422)
# ============================================================================


def test_request_validation_missing_fields(client):
    """Verify missing required identifiers return 422 Unprocessable Entity."""
    # Missing payment_id
    resp = client.post("/v1/recovery/evaluate-decision", json={"customer_id": str(uuid.uuid4())})
    assert resp.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    # Missing customer_id
    resp = client.post("/v1/recovery/evaluate-decision", json={"payment_id": str(uuid.uuid4())})
    assert resp.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    # Empty payload
    resp = client.post("/v1/recovery/evaluate-decision", json={})
    assert resp.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY


def test_request_validation_malformed_uuid(client):
    """Verify malformed UUID string returns 422."""
    payload = {
        "customer_id": "not-a-uuid",
        "payment_id": str(uuid.uuid4()),
    }
    resp = client.post("/v1/recovery/evaluate-decision", json=payload)
    assert resp.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY


def test_request_validation_extra_fields(client):
    """Verify unexpected/extra fields in payload return 422."""
    payload = {
        "customer_id": str(uuid.uuid4()),
        "payment_id": str(uuid.uuid4()),
        "unexpected_extra_key": "forbidden",
    }
    resp = client.post("/v1/recovery/evaluate-decision", json=payload)
    assert resp.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY


# ============================================================================
# 3. Domain Exception HTTP Mappings
# ============================================================================


def test_customer_not_found_returns_404(client, test_db_session):
    """Verify CustomerNotFoundError translates to HTTP 404."""
    non_existent_cust_id = str(uuid.uuid4())
    payload = {
        "customer_id": non_existent_cust_id,
        "payment_id": str(uuid.uuid4()),
    }
    resp = client.post("/v1/recovery/evaluate-decision", json=payload)
    assert resp.status_code == status.HTTP_404_NOT_FOUND
    assert non_existent_cust_id in resp.json()["detail"]


def test_payment_not_found_returns_404(client, test_db_session):
    """Verify PaymentNotFoundError translates to HTTP 404."""
    customer, _, _ = _seed_customer_payment_opportunity(test_db_session)
    non_existent_pay_id = str(uuid.uuid4())

    payload = {
        "customer_id": str(customer.id),
        "payment_id": non_existent_pay_id,
    }
    resp = client.post("/v1/recovery/evaluate-decision", json=payload)
    assert resp.status_code == status.HTTP_404_NOT_FOUND
    assert non_existent_pay_id in resp.json()["detail"]


def test_payment_customer_mismatch_returns_403(client, test_db_session):
    """Verify PaymentCustomerMismatchError translates to HTTP 403 Forbidden."""
    customer1, payment1, _ = _seed_customer_payment_opportunity(test_db_session)
    customer2, _, _ = _seed_customer_payment_opportunity(test_db_session)

    payload = {
        "customer_id": str(customer2.id),
        "payment_id": str(payment1.id),
    }
    resp = client.post("/v1/recovery/evaluate-decision", json=payload)
    assert resp.status_code == status.HTTP_403_FORBIDDEN
    assert "belongs to customer" in resp.json()["detail"]


def test_opportunity_not_found_returns_404(client, test_db_session):
    """Verify RecoveryOpportunityNotFoundError translates to HTTP 404."""
    customer, payment, opportunity = _seed_customer_payment_opportunity(test_db_session)
    test_db_session.delete(opportunity)
    test_db_session.commit()

    payload = {
        "customer_id": str(customer.id),
        "payment_id": str(payment.id),
    }
    resp = client.post("/v1/recovery/evaluate-decision", json=payload)
    assert resp.status_code == status.HTTP_404_NOT_FOUND
    assert "Recovery opportunity" in resp.json()["detail"]


# ============================================================================
# 4. Service Isolation via Dependency Override
# ============================================================================


class MockRecoveryService:
    """Mock RecoveryService for isolated HTTP layer testing."""

    def __init__(self, stub_response: RecoveryEvaluationResponse):
        self.stub_response = stub_response
        self.invoked = False

    def evaluate_recovery(self, db_session, request):
        self.invoked = True
        return self.stub_response


def test_service_isolation_with_dependency_override(client):
    """Verify router cleanly delegates to RecoveryService dependency."""
    stub = RecoveryEvaluationResponse(
        payment_id=uuid.uuid4(),
        customer_id=uuid.uuid4(),
        opportunity_id=uuid.uuid4(),
        recommended_action=RecoveryAction.CHANGE_PAYMENT_METHOD,
        reason="Isolated test stub reason",
        confidence=0.95,
        decision_basis={"mock": True},
        historical_rag_used=True,
        retrieved_evidence_count=2,
    )
    mock_svc = MockRecoveryService(stub)

    app.dependency_overrides[get_recovery_service] = lambda: mock_svc

    payload = {
        "customer_id": str(stub.customer_id),
        "payment_id": str(stub.payment_id),
    }
    resp = client.post("/v1/recovery/evaluate-decision", json=payload)

    assert resp.status_code == status.HTTP_200_OK
    assert mock_svc.invoked is True
    data = resp.json()
    assert data["recommended_action"] == "change_payment_method"
    assert data["confidence"] == 0.95
    assert data["reason"] == "Isolated test stub reason"


# ============================================================================
# 5. Health & OpenAPI Regressions
# ============================================================================


def test_health_check_endpoint_regression(client):
    """Verify GET /health remains operational."""
    resp = client.get("/health")
    assert resp.status_code == status.HTTP_200_OK
    data = resp.json()
    assert data["status"] == "healthy"
    assert "app" in data


def test_openapi_schema_contains_recovery_endpoint(client):
    """Verify GET /openapi.json registers POST /v1/recovery/evaluate-decision."""
    resp = client.get("/openapi.json")
    assert resp.status_code == status.HTTP_200_OK
    schema = resp.json()

    paths = schema.get("paths", {})
    assert "/v1/recovery/evaluate-decision" in paths
    recovery_post = paths["v1/recovery/evaluate-decision"]["post"] if "v1/recovery/evaluate-decision" in paths else paths["/v1/recovery/evaluate-decision"]["post"]

    assert "Recovery" in recovery_post.get("tags", [])
    assert recovery_post.get("summary") == "Evaluate Failed Payment Recovery Decision"
