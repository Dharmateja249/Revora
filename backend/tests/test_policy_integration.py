"""
Integration tests for Policy Subsystem across Decision Engine, Recovery Service, and API Router.
"""

from datetime import datetime, timezone
import uuid
import pytest
from fastapi import status
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.database import Base, get_db
from app.decision_engine import DecisionEngine, RecoveryAction
from app.historical_retrieval import HistoricalCase
from app.main import app
from app.models import AuditEvent, Customer, Payment, RecoveryAttempt, RecoveryOpportunity
from app.policies.registry import DEFAULT_POLICY_VERSION
from app.policies.resolver import resolve_policy_context
from app.recovery_service import RecoveryService
from app.schemas.recovery import RecoveryEvaluationRequest


@pytest.fixture
def db_session(tmp_path):
    """Fixture providing an isolated SQLite database session."""
    db_file = tmp_path / "test_policy_integration.db"
    engine = create_engine(f"sqlite:///{db_file}", connect_args={"check_same_thread": False})
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    Base.metadata.create_all(bind=engine)

    session = TestingSessionLocal()
    try:
        yield session
    finally:
        session.close()


@pytest.fixture
def seeded_customer_and_auth_payment(db_session):
    """Seed customer and payment with authentication_failed error."""
    now = datetime(2026, 6, 15, 12, 0, 0, tzinfo=timezone.utc)
    customer = Customer(
        name="Policy Test Customer",
        email="policy@example.com",
        total_payments=5,
        successful_payments=4,
        failed_payments=1,
    )
    db_session.add(customer)
    db_session.commit()

    # Prior payment where retry_payment succeeded historically
    hist_payment = Payment(
        customer_id=customer.id,
        amount=500.0,
        currency="INR",
        payment_method="card",
        status="succeeded",
        failure_reason="bank_timeout",
        created_at=now,
    )
    db_session.add(hist_payment)
    db_session.commit()

    hist_opp = RecoveryOpportunity(
        payment_id=hist_payment.id,
        status="recovered",
        revenue_at_risk=500.0,
        expected_recovery=500.0,
        recommended_action="retry_payment",
        created_at=now,
    )
    db_session.add(hist_opp)
    db_session.commit()

    hist_att = RecoveryAttempt(
        opportunity_id=hist_opp.id,
        action="retry_payment",
        status="succeeded",
        amount_recovered=500.0,
        created_at=now,
    )
    db_session.add(hist_att)
    db_session.commit()

    # Current failed payment with authentication failure
    curr_payment = Payment(
        customer_id=customer.id,
        amount=750.0,
        currency="INR",
        payment_method="card",
        status="failed",
        failure_reason="authentication_failed",
        created_at=now,
    )
    db_session.add(curr_payment)
    db_session.commit()

    curr_opp = RecoveryOpportunity(
        payment_id=curr_payment.id,
        status="open",
        revenue_at_risk=750.0,
        expected_recovery=0.0,
        created_at=now,
    )
    db_session.add(curr_opp)
    db_session.commit()

    return customer, curr_payment, curr_opp


def test_customer_and_rag_proposing_prohibited_action_overridden_by_policy(
    db_session, seeded_customer_and_auth_payment
):
    """
    Verify that even when customer history and historical RAG cases recommend 'retry_payment',
    the provider policy constraint (2FA required) deterministically overrides the recommendation to 'payment_link'.
    """
    customer, curr_payment, curr_opp = seeded_customer_and_auth_payment
    service = RecoveryService()

    # Create artificial RAG cases that strongly recommend retry_payment
    fake_rag_case = HistoricalCase(
        payment_id=uuid.uuid4(),
        customer_id=customer.id,
        amount=750.0,
        payment_method="card",
        failure_reason="authentication_failed",
        recovery_action="retry_payment",
        recovery_status="recovered",
        amount_recovered=750.0,
        was_recovered=True,
        relevance_score=0.95,
    )

    req = RecoveryEvaluationRequest(
        customer_id=customer.id,
        payment_id=curr_payment.id,
        use_rag=False,
    )

    response = service.evaluate_recovery(db_session=db_session, request=req)

    # Effective action must be payment_link (policy wins)
    assert response.recommended_action == RecoveryAction.PAYMENT_LINK
    assert response.provider == "razorpay"
    assert response.policy_version == DEFAULT_POLICY_VERSION
    assert "RZP_CUSTOMER_AUTH_2FA_REQUIRED" in response.applied_policy_ids

    # Audit event should record policy validation telemetry
    audit = (
        db_session.query(AuditEvent)
        .filter(AuditEvent.opportunity_id == curr_opp.id)
        .first()
    )
    assert audit is not None
    assert "applied_policy_ids" in audit.metadata_payload
    assert "policy_validation" in audit.metadata_payload


def test_api_endpoint_returns_policy_telemetry(db_session, seeded_customer_and_auth_payment):
    """Verify POST /v1/recovery/evaluate-decision HTTP endpoint returns policy metadata."""
    customer, curr_payment, _ = seeded_customer_and_auth_payment

    def override_get_db():
        yield db_session

    app.dependency_overrides[get_db] = override_get_db
    client = TestClient(app)

    try:
        response = client.post(
            "/v1/recovery/evaluate-decision",
            headers={"Authorization": f"Bearer {customer.id}"},
            json={
                "customer_id": str(customer.id),
                "payment_id": str(curr_payment.id),
                "use_rag": False,
            },
        )
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["recommended_action"] == "payment_link"
        assert data["provider"] == "razorpay"
        assert data["policy_version"] == DEFAULT_POLICY_VERSION
        assert isinstance(data["applied_policy_ids"], list)
        assert "RZP_CUSTOMER_AUTH_2FA_REQUIRED" in data["applied_policy_ids"]
        assert isinstance(data["policy_overridden"], bool)
    finally:
        app.dependency_overrides.clear()
