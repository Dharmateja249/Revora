"""
Tests for Revora Dashboard Metrics API (GET /api/dashboard/metrics).

Verifies:
1. Dashboard metrics computation with normal recovery data.
2. Recovery rate percentage calculation.
3. Total amount recovered calculation.
4. Recovered, failed, and pending case counts.
5. Policy override count aggregation from audit logs.
6. Execution success rate calculation.
7. Zero-case empty state handling (no DivisionByZero or NaN).
8. Strict multi-tenant isolation.
9. RAG runtime precedent count accuracy.
10. Average confidence and fallback decision metrics.
"""

from uuid import uuid4

from app.auth import create_access_token
from app.context_retrieval import DEMO_CUSTOMER_UUID, ensure_demo_customer_seeded
from app.database import SessionLocal
from app.main import app
from app.models import (
    AuditEvent,
    Customer,
    Payment,
    RecoveryAttempt,
    RecoveryOpportunity,
)
from app.retrieval_document import RetrievalDocument
from app.vector_index import get_vector_index
from fastapi import status
from fastapi.testclient import TestClient

client = TestClient(app)


def test_zero_case_handling_empty_customer():
    """Verify that a customer with zero payment records receives clean 0.0 metrics without errors."""
    db = SessionLocal()
    empty_cust_id = uuid4()
    empty_cust = Customer(
        id=empty_cust_id,
        name="Empty Tenant",
        email=f"empty_{empty_cust_id.hex[:8]}@example.com",
    )
    db.add(empty_cust)
    db.commit()
    db.close()

    auth_headers = {"Authorization": f"Bearer {create_access_token(empty_cust_id)}"}
    response = client.get("/api/dashboard/metrics", headers=auth_headers)
    assert response.status_code == status.HTTP_200_OK

    data = response.json()
    assert data["total_cases"] == 0
    assert data["recovered_cases"] == 0
    assert data["failed_cases"] == 0
    assert data["pending_cases"] == 0
    assert data["recovery_rate"] == 0.0
    assert data["amount_recovered"] == 0.0
    assert data["total_executions"] == 0
    assert data["successful_executions"] == 0
    assert data["failed_executions"] == 0
    assert data["execution_success_rate"] == 0.0
    assert data["policy_overrides"] == 0
    assert data["fallback_decisions"] == 0
    assert data["average_confidence"] == 0.0
    assert isinstance(data["rag_precedents"], int)


def test_dashboard_metrics_normal_recovery_data():
    """Verify metrics calculation for a tenant with a mix of recovered, failed, and in-progress cases."""
    db = SessionLocal()
    tenant_id = uuid4()
    tenant = Customer(
        id=tenant_id,
        name="Acme Corp",
        email=f"acme_{tenant_id.hex[:8]}@example.com",
    )
    db.add(tenant)

    # Case 1: Recovered payment (amount 5000)
    pay_1 = Payment(
        id=uuid4(),
        customer_id=tenant_id,
        amount=5000.0,
        currency="INR",
        payment_method="card",
        status="succeeded",
    )
    opp_1 = RecoveryOpportunity(
        id=uuid4(),
        payment_id=pay_1.id,
        status="recovered",
        revenue_at_risk=5000.0,
        expected_recovery=5000.0,
        recommended_action="payment_link",
        confidence=0.95,
    )
    att_1 = RecoveryAttempt(
        id=uuid4(),
        opportunity_id=opp_1.id,
        action="payment_link",
        status="succeeded",
        amount_recovered=5000.0,
        idempotency_key=f"idem_norm_1_{uuid4().hex[:8]}",
    )
    audit_1 = AuditEvent(
        id=uuid4(),
        opportunity_id=opp_1.id,
        event_type="recovery_action_executed",
        description="Payment link executed.",
        metadata_payload={"policy_overridden": False, "agent_used": True},
    )

    # Case 2: Failed payment (amount 3000)
    pay_2 = Payment(
        id=uuid4(),
        customer_id=tenant_id,
        amount=3000.0,
        currency="INR",
        payment_method="upi",
        status="failed",
    )
    opp_2 = RecoveryOpportunity(
        id=uuid4(),
        payment_id=pay_2.id,
        status="failed",
        revenue_at_risk=3000.0,
        expected_recovery=0.0,
        recommended_action="no_action",
        confidence=0.85,
    )
    att_2 = RecoveryAttempt(
        id=uuid4(),
        opportunity_id=opp_2.id,
        action="no_action",
        status="failed",
        amount_recovered=0.0,
        idempotency_key=f"idem_norm_2_{uuid4().hex[:8]}",
    )
    audit_2 = AuditEvent(
        id=uuid4(),
        opportunity_id=opp_2.id,
        event_type="recovery_action_executed",
        description="Policy override enforced.",
        metadata_payload={"policy_overridden": True, "agent_used": True},
    )

    # Case 3: In-Progress/Pending recovery (amount 2000)
    pay_3 = Payment(
        id=uuid4(),
        customer_id=tenant_id,
        amount=2000.0,
        currency="INR",
        payment_method="card",
        status="failed",
    )
    opp_3 = RecoveryOpportunity(
        id=uuid4(),
        payment_id=pay_3.id,
        status="in_progress",
        revenue_at_risk=2000.0,
        expected_recovery=0.0,
        recommended_action="wait_and_retry",
        confidence=0.90,
    )
    att_3 = RecoveryAttempt(
        id=uuid4(),
        opportunity_id=opp_3.id,
        action="wait_and_retry",
        status="in_progress",
        amount_recovered=0.0,
        idempotency_key=f"idem_norm_3_{uuid4().hex[:8]}",
    )
    audit_3 = AuditEvent(
        id=uuid4(),
        opportunity_id=opp_3.id,
        event_type="recovery_action_executed",
        description="Fallback action applied.",
        metadata_payload={"is_fallback": True, "agent_used": False},
    )

    db.add_all(
        [
            pay_1,
            opp_1,
            att_1,
            audit_1,
            pay_2,
            opp_2,
            att_2,
            audit_2,
            pay_3,
            opp_3,
            att_3,
            audit_3,
        ]
    )
    db.commit()
    db.close()

    auth_headers = {"Authorization": f"Bearer {create_access_token(tenant_id)}"}
    response = client.get("/api/dashboard/metrics", headers=auth_headers)
    assert response.status_code == status.HTTP_200_OK

    data = response.json()
    assert data["total_cases"] == 3
    assert data["recovered_cases"] == 1
    assert data["failed_cases"] == 1
    assert data["pending_cases"] == 1
    assert data["revenue_at_risk"] == 2000.0
    # Recovery rate = 1 / 3 = 33.3%
    assert data["recovery_rate"] == 33.3
    # Amount recovered = 5000.0
    assert data["amount_recovered"] == 5000.0
    # Total executions = 3 (1 succeeded, 1 failed, 1 in_progress)
    assert data["total_executions"] == 3
    assert data["successful_executions"] == 1
    assert data["failed_executions"] == 1
    assert data["pending_executions"] == 1
    # Execution success rate = 1 / 3 = 33.3%
    assert data["execution_success_rate"] == 33.3
    # Policy overrides = 1
    assert data["policy_overrides"] == 1
    # Fallback decisions = 1
    assert data["fallback_decisions"] == 1
    # Average confidence = (0.95 + 0.85 + 0.90) / 3 = 0.90
    assert data["average_confidence"] == 0.90


def test_strict_tenant_isolation_in_dashboard_metrics():
    """Verify that Tenant A cannot see Tenant B's metrics under any circumstance."""
    db = SessionLocal()
    cust_a_id = uuid4()
    cust_b_id = uuid4()

    cust_a = Customer(
        id=cust_a_id,
        name="Tenant Alpha",
        email=f"alpha_{cust_a_id.hex[:8]}@example.com",
    )
    cust_b = Customer(
        id=cust_b_id,
        name="Tenant Beta",
        email=f"beta_{cust_b_id.hex[:8]}@example.com",
    )
    db.add(cust_a)
    db.add(cust_b)

    # Add 10 recovered cases for Tenant B
    for _ in range(10):
        pay = Payment(
            id=uuid4(),
            customer_id=cust_b_id,
            amount=1000.0,
            currency="INR",
            payment_method="upi",
            status="succeeded",
        )
        opp = RecoveryOpportunity(
            id=uuid4(),
            payment_id=pay.id,
            status="recovered",
            revenue_at_risk=1000.0,
            expected_recovery=1000.0,
        )
        att = RecoveryAttempt(
            id=uuid4(),
            opportunity_id=opp.id,
            action="payment_link",
            status="succeeded",
            amount_recovered=1000.0,
            idempotency_key=f"idem_b_{uuid4().hex[:8]}",
        )
        db.add(pay)
        db.add(opp)
        db.add(att)

    db.commit()
    db.close()

    # Query metrics for Tenant A (should be 0)
    auth_a = {"Authorization": f"Bearer {create_access_token(cust_a_id)}"}
    resp_a = client.get("/api/dashboard/metrics", headers=auth_a)
    assert resp_a.status_code == status.HTTP_200_OK
    data_a = resp_a.json()
    assert data_a["total_cases"] == 0
    assert data_a["recovered_cases"] == 0
    assert data_a["amount_recovered"] == 0.0

    # Query metrics for Tenant B (should be 10 cases, 10,000 INR)
    auth_b = {"Authorization": f"Bearer {create_access_token(cust_b_id)}"}
    resp_b = client.get("/api/dashboard/metrics", headers=auth_b)
    assert resp_b.status_code == status.HTTP_200_OK
    data_b = resp_b.json()
    assert data_b["total_cases"] == 10
    assert data_b["recovered_cases"] == 10
    assert data_b["amount_recovered"] == 10000.0
    assert data_b["recovery_rate"] == 100.0


def test_rag_precedent_count_reflects_actual_vector_index():
    """Verify that rag_precedents dynamically matches the active VectorIndex size."""
    db = SessionLocal()
    ensure_demo_customer_seeded(db)
    db.close()

    vector_index = get_vector_index()
    initial_size = vector_index.size

    # Add a unique test document using standard embedding service
    doc_id = uuid4()
    test_doc = RetrievalDocument(
        case_id=doc_id,
        text="Test precedent case for dashboard metric test.",
        metadata={"test": "true"},
    )
    from app.embedding_service import get_embedding_service

    emb = get_embedding_service().embed(test_doc.canonical_text)
    vector_index.add(test_doc, emb)

    try:
        auth = {"Authorization": f"Bearer {create_access_token(DEMO_CUSTOMER_UUID)}"}
        resp = client.get("/api/dashboard/metrics", headers=auth)
        assert resp.status_code == status.HTTP_200_OK
        data = resp.json()
        assert data["rag_precedents"] == initial_size + 1
    finally:
        # Cleanup test document
        vector_index.delete(doc_id)
