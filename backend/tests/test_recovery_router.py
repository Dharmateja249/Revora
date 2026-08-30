"""
API Boundary Tests for FastAPI Recovery Evaluation Router (/v1/recovery/evaluate-decision).

Tests:
1. Happy path: POST /v1/recovery/evaluate-decision returns 200 with complete RecoveryEvaluationResponse
2. Security & Tenant Authorization:
   - Unauthenticated requests return 401 Unauthorized
   - Invalid token format returns 401 Unauthorized
   - Cross-customer requests return 403 Forbidden
   - Authorization failures cause zero DB mutations
3. Request validation: missing customer_id, missing payment_id, malformed UUID, extra fields (422)
4. Domain exception mappings:
   - CustomerNotFoundError -> HTTP 404
   - PaymentNotFoundError -> HTTP 404
   - RecoveryOpportunityNotFoundError -> HTTP 404
   - PaymentCustomerMismatchError -> HTTP 403
5. RAG disabled (use_rag=False): returns 200 with historical_rag_used=False
6. Service isolation: dependency override of get_recovery_service for isolated HTTP boundary testing
7. Shared VectorIndex: persistence across requests and service resolutions
8. Health endpoint regression: GET /health returns 200
9. OpenAPI schema regression: /openapi.json contains POST /v1/recovery/evaluate-decision
"""

from datetime import datetime, timezone, timedelta
import uuid
from fastapi import status
from fastapi.testclient import TestClient
import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.pool import StaticPool
from sqlalchemy.orm import sessionmaker

from app.auth import AuthenticatedPrincipal, get_current_principal
from app.context import (
    CustomerNotFoundError,
    PaymentCustomerMismatchError,
    PaymentNotFoundError,
    RecoveryOpportunityNotFoundError,
)
from app.database import Base, get_db
from app.decision_engine import RecoveryAction
from app.embedding_service import get_embedding_service
from app.main import app
from app.models import AuditEvent, Customer, Payment, RecoveryOpportunity, utc_now
from app.recovery_service import RecoveryService
from app.retrieval_document import RetrievalDocument
from app.routers.recovery import get_recovery_service
from app.schemas.recovery import (
    RecoveryEvaluationRequest,
    RecoveryEvaluationResponse,
)
from app.vector_index import VectorIndex, get_vector_index


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
    """Verify POST /v1/recovery/evaluate-decision returns 200 with complete response for authenticated user."""
    customer, payment, opportunity = _seed_customer_payment_opportunity(test_db_session)

    payload = {
        "customer_id": str(customer.id),
        "payment_id": str(payment.id),
        "use_rag": True,
    }
    headers = {"Authorization": f"Bearer {customer.id}"}

    response = client.post("/v1/recovery/evaluate-decision", json=payload, headers=headers)

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
    headers = {"Authorization": f"Bearer {customer.id}"}

    response = client.post("/v1/recovery/evaluate-decision", json=payload, headers=headers)

    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert data["historical_rag_used"] is False
    assert data["retrieved_evidence_count"] == 0


# ============================================================================
# 2. Security & Tenant Authorization Tests
# ============================================================================


def test_unauthenticated_request_rejected_401(client, test_db_session):
    """Verify unauthenticated request without token returns 401 Unauthorized."""
    customer, payment, _ = _seed_customer_payment_opportunity(test_db_session)
    payload = {
        "customer_id": str(customer.id),
        "payment_id": str(payment.id),
    }

    # No Authorization header
    resp = client.post("/v1/recovery/evaluate-decision", json=payload)
    assert resp.status_code == status.HTTP_401_UNAUTHORIZED
    assert "credentials were not provided" in resp.json()["detail"]


def test_invalid_token_format_rejected_401(client, test_db_session):
    """Verify malformed bearer token returns 401 Unauthorized."""
    customer, payment, _ = _seed_customer_payment_opportunity(test_db_session)
    payload = {
        "customer_id": str(customer.id),
        "payment_id": str(payment.id),
    }
    headers = {"Authorization": "Bearer not-a-uuid-token"}

    resp = client.post("/v1/recovery/evaluate-decision", json=payload, headers=headers)
    assert resp.status_code == status.HTTP_401_UNAUTHORIZED
    assert "Invalid authentication token" in resp.json()["detail"]


def test_cross_customer_request_rejected_403(client, test_db_session):
    """Verify authenticated customer requesting another customer's ID returns 403 Forbidden."""
    cust_a, pay_a, _ = _seed_customer_payment_opportunity(test_db_session)
    cust_b, pay_b, _ = _seed_customer_payment_opportunity(test_db_session)

    # Caller authenticated as Customer A, but requests evaluation for Customer B
    payload = {
        "customer_id": str(cust_b.id),
        "payment_id": str(pay_b.id),
    }
    headers = {"Authorization": f"Bearer {cust_a.id}"}

    resp = client.post("/v1/recovery/evaluate-decision", json=payload, headers=headers)
    assert resp.status_code == status.HTTP_403_FORBIDDEN
    assert "Cross-tenant access forbidden" in resp.json()["detail"]


def test_authorization_failure_causes_zero_database_mutation(client, test_db_session):
    """Verify that unauthorized requests perform zero database mutations or audit event creation."""
    cust_a, pay_a, _ = _seed_customer_payment_opportunity(test_db_session)
    cust_b, pay_b, opp_b = _seed_customer_payment_opportunity(test_db_session)

    initial_action = opp_b.recommended_action
    initial_audits_count = test_db_session.query(AuditEvent).count()

    # Unauthorized cross-tenant call
    payload = {
        "customer_id": str(cust_b.id),
        "payment_id": str(pay_b.id),
    }
    headers = {"Authorization": f"Bearer {cust_a.id}"}

    resp = client.post("/v1/recovery/evaluate-decision", json=payload, headers=headers)
    assert resp.status_code == status.HTTP_403_FORBIDDEN

    # Verify opportunity is completely untouched
    test_db_session.refresh(opp_b)
    assert opp_b.recommended_action == initial_action

    # Verify no audit event was created
    final_audits_count = test_db_session.query(AuditEvent).count()
    assert final_audits_count == initial_audits_count


# ============================================================================
# 3. Request Validation Tests (HTTP 422)
# ============================================================================


def test_request_validation_missing_fields(client):
    """Verify missing required identifiers return 422 Unprocessable Entity."""
    cust_id = uuid.uuid4()
    headers = {"Authorization": f"Bearer {cust_id}"}

    # Missing payment_id
    resp = client.post(
        "/v1/recovery/evaluate-decision",
        json={"customer_id": str(cust_id)},
        headers=headers,
    )
    assert resp.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    # Missing customer_id
    resp = client.post(
        "/v1/recovery/evaluate-decision",
        json={"payment_id": str(uuid.uuid4())},
        headers=headers,
    )
    assert resp.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    # Empty payload
    resp = client.post("/v1/recovery/evaluate-decision", json={}, headers=headers)
    assert resp.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY


def test_request_validation_malformed_uuid(client):
    """Verify malformed UUID string returns 422."""
    cust_id = uuid.uuid4()
    headers = {"Authorization": f"Bearer {cust_id}"}
    payload = {
        "customer_id": "not-a-uuid",
        "payment_id": str(uuid.uuid4()),
    }
    resp = client.post("/v1/recovery/evaluate-decision", json=payload, headers=headers)
    assert resp.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY


def test_request_validation_extra_fields(client):
    """Verify unexpected/extra fields in payload return 422."""
    cust_id = uuid.uuid4()
    headers = {"Authorization": f"Bearer {cust_id}"}
    payload = {
        "customer_id": str(cust_id),
        "payment_id": str(uuid.uuid4()),
        "unexpected_extra_key": "forbidden",
    }
    resp = client.post("/v1/recovery/evaluate-decision", json=payload, headers=headers)
    assert resp.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY


# ============================================================================
# 4. Domain Exception HTTP Mappings
# ============================================================================


def test_customer_not_found_returns_404(client, test_db_session):
    """Verify CustomerNotFoundError translates to HTTP 404."""
    non_existent_cust_id = str(uuid.uuid4())
    payload = {
        "customer_id": non_existent_cust_id,
        "payment_id": str(uuid.uuid4()),
    }
    headers = {"Authorization": f"Bearer {non_existent_cust_id}"}
    resp = client.post("/v1/recovery/evaluate-decision", json=payload, headers=headers)
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
    headers = {"Authorization": f"Bearer {customer.id}"}
    resp = client.post("/v1/recovery/evaluate-decision", json=payload, headers=headers)
    assert resp.status_code == status.HTTP_404_NOT_FOUND
    assert non_existent_pay_id in resp.json()["detail"]


def test_payment_customer_mismatch_returns_403(client, test_db_session):
    """Verify PaymentCustomerMismatchError translates to HTTP 403 Forbidden."""
    customer1, payment1, _ = _seed_customer_payment_opportunity(test_db_session)
    customer2, _, _ = _seed_customer_payment_opportunity(test_db_session)

    # Customer 2 authorized to request customer2, but payment1 belongs to customer1
    payload = {
        "customer_id": str(customer2.id),
        "payment_id": str(payment1.id),
    }
    headers = {"Authorization": f"Bearer {customer2.id}"}
    resp = client.post("/v1/recovery/evaluate-decision", json=payload, headers=headers)
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
    headers = {"Authorization": f"Bearer {customer.id}"}
    resp = client.post("/v1/recovery/evaluate-decision", json=payload, headers=headers)
    assert resp.status_code == status.HTTP_404_NOT_FOUND
    assert "Recovery opportunity" in resp.json()["detail"]


# ============================================================================
# 5. Service Isolation via Dependency Override
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
    headers = {"Authorization": f"Bearer {stub.customer_id}"}
    resp = client.post("/v1/recovery/evaluate-decision", json=payload, headers=headers)

    assert resp.status_code == status.HTTP_200_OK
    assert mock_svc.invoked is True
    data = resp.json()
    assert data["recommended_action"] == "change_payment_method"
    assert data["confidence"] == 0.95
    assert data["reason"] == "Isolated test stub reason"


# ============================================================================
# 6. Shared Application-Scoped VectorIndex Tests
# ============================================================================


def test_shared_vector_index_persistence_across_requests(client, test_db_session):
    """Verify that multiple requests/service resolutions use the same shared VectorIndex."""
    shared_index = get_vector_index()
    embedding_service = get_embedding_service()

    customer, payment, _ = _seed_customer_payment_opportunity(
        test_db_session, failure_reason="insufficient_funds", payment_method="card"
    )

    # 1. Add historical document to shared index
    hist_doc = RetrievalDocument(
        case_id=uuid.uuid4(),
        text="failure_reason: insufficient_funds\npayment_method: card\namount: 3000.00\ncurrency: INR\nrecovery_action: payment_link\nrecovery_status: recovered\nwas_recovered: true\namount_recovered: 3000.00",
        metadata={
            "customer_id": str(customer.id),
            "amount": 3000.0,
            "was_recovered": True,
            "recovery_action": "payment_link",
            "created_at": (utc_now() - timedelta(days=2)).isoformat(),
        },
    )
    shared_index.add(hist_doc, embedding_service.embed(hist_doc.text))

    # 2. First HTTP request retrieves RAG evidence from shared index
    payload = {
        "customer_id": str(customer.id),
        "payment_id": str(payment.id),
        "use_rag": True,
    }
    headers = {"Authorization": f"Bearer {customer.id}"}

    resp1 = client.post("/v1/recovery/evaluate-decision", json=payload, headers=headers)
    assert resp1.status_code == status.HTTP_200_OK
    data1 = resp1.json()
    assert data1["historical_rag_used"] is True
    assert data1["retrieved_evidence_count"] >= 1

    # 3. Second HTTP request also retrieves from the same populated shared index
    resp2 = client.post("/v1/recovery/evaluate-decision", json=payload, headers=headers)
    assert resp2.status_code == status.HTTP_200_OK
    data2 = resp2.json()
    assert data2["historical_rag_used"] is True
    assert data2["retrieved_evidence_count"] >= 1

    # 4. Resolving RecoveryService again does not recreate/wipe the shared index
    new_service = get_recovery_service()
    assert new_service.vector_index is shared_index
    assert new_service.vector_index.size == shared_index.size


# ============================================================================
# 7. Health & OpenAPI Regressions
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
    recovery_post = (
        paths["v1/recovery/evaluate-decision"]["post"]
        if "v1/recovery/evaluate-decision" in paths
        else paths["/v1/recovery/evaluate-decision"]["post"]
    )

    assert "Recovery" in recovery_post.get("tags", [])
    assert recovery_post.get("summary") == "Evaluate Failed Payment Recovery Decision"
