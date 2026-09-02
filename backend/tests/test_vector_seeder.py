"""
Unit and regression tests for runtime VectorIndex seeder.

Verifies:
- Seeding an empty index populates it with canonical precedents
- Seeded documents have the expected embedding dimension
- Seeding is idempotent and does not produce duplicate documents
- Semantic retrieval for an OTP timeout scenario returns relevant historical precedents
- Semantic retrieval respects the demo customer tenant
- Documents belonging to another customer are strictly filtered out (tenant isolation)
- Citation tokens formatted by AgentContextBuilder remain deterministic and safe
"""

from uuid import UUID, uuid4

import pytest

from app.agent.context_builder import AgentContextBuilder
from app.context import (
    CustomerContext,
    CustomerRecoveryContext,
    PaymentContext,
    RecoveryOpportunityContext,
)
from app.embedding_service import get_embedding_service
from app.semantic_historical_retriever import SemanticHistoricalRetriever
from app.vector_index import VectorIndex
from app.vector_seeder import (
    DEMO_CUSTOMER_UUID,
    get_curated_historical_precedents,
    seed_runtime_vector_index,
)


def test_seed_empty_index_populates_precedents():
    """Verify that seeding an empty VectorIndex populates it with 100-200 precedents."""
    index = VectorIndex()
    assert index.size == 0

    added = seed_runtime_vector_index(vector_index=index)
    assert added >= 100
    assert added <= 200
    assert index.size == added


def test_seeded_documents_have_expected_embedding_dimension():
    """Verify that all seeded documents possess vectors matching the embedding service dimension."""
    svc = get_embedding_service()
    index = VectorIndex(dimension=svc.dimension)

    seed_runtime_vector_index(vector_index=index, embedding_service=svc)
    assert index.size > 0

    # Inspect entries
    for case_id in list(index._entries.keys())[:10]:
        entry = index.get(case_id)
        assert entry is not None
        vector, doc = entry
        assert len(vector) == svc.dimension
        assert doc.case_id == case_id
        assert doc.metadata.get("customer_id") == str(DEMO_CUSTOMER_UUID)


def test_seeding_is_idempotent_no_duplicates():
    """Verify that running seed multiple times does not add duplicate documents."""
    index = VectorIndex()
    first_added = seed_runtime_vector_index(vector_index=index)
    initial_size = index.size
    assert first_added > 0

    second_added = seed_runtime_vector_index(vector_index=index)
    assert second_added == 0
    assert index.size == initial_size


def test_retrieval_for_otp_timeout_scenario():
    """Verify semantic retrieval returns relevant precedents for an OTP timeout failure."""
    svc = get_embedding_service()
    index = VectorIndex(dimension=svc.dimension)
    seed_runtime_vector_index(vector_index=index, embedding_service=svc)

    retriever = SemanticHistoricalRetriever(vector_index=index, embedding_service=svc)

    ctx = CustomerRecoveryContext(
        customer=CustomerContext(
            customer_id=DEMO_CUSTOMER_UUID,
            total_payments=20,
            successful_payments=18,
            failed_payments=2,
            historical_success_rate=0.90,
        ),
        current_payment=PaymentContext(
            payment_id=uuid4(),
            amount=8450.0,
            currency="INR",
            payment_method="card",
            status="failed",
            failure_reason="customer_auth_failed_otp_timeout",
        ),
        current_opportunity=RecoveryOpportunityContext(
            opportunity_id=uuid4(),
            status="open",
            revenue_at_risk=8450.0,
            expected_recovery=0.0,
        ),
        current_payment_attempts=[],
    )

    results = retriever.retrieve(ctx, top_k=3)
    assert len(results) > 0

    top_case = results[0]
    assert top_case.customer_id == DEMO_CUSTOMER_UUID
    assert top_case.relevance_score is not None
    assert top_case.relevance_score > 0.0
    # Top match should identify authentication/OTP failure pattern
    assert any(
        kw in (top_case.failure_reason or "").lower()
        for kw in ("otp", "auth", "timeout", "expired", "3ds")
    )


def test_retrieval_respects_demo_customer_tenant():
    """Verify that retrieval only returns documents matching the query customer."""
    svc = get_embedding_service()
    index = VectorIndex(dimension=svc.dimension)
    seed_runtime_vector_index(vector_index=index, embedding_service=svc)

    retriever = SemanticHistoricalRetriever(vector_index=index, embedding_service=svc)

    ctx = CustomerRecoveryContext(
        customer=CustomerContext(
            customer_id=DEMO_CUSTOMER_UUID,
            total_payments=10,
            successful_payments=9,
            failed_payments=1,
            historical_success_rate=0.9,
        ),
        current_payment=PaymentContext(
            payment_id=uuid4(),
            amount=3200.0,
            currency="INR",
            payment_method="upi",
            status="failed",
            failure_reason="bank_technical_timeout",
        ),
        current_opportunity=RecoveryOpportunityContext(
            opportunity_id=uuid4(),
            status="open",
            revenue_at_risk=3200.0,
            expected_recovery=0.0,
        ),
        current_payment_attempts=[],
    )

    results = retriever.retrieve(ctx, top_k=5)
    assert len(results) > 0
    for case in results:
        assert case.customer_id == DEMO_CUSTOMER_UUID


def test_documents_belonging_to_another_customer_not_returned():
    """Verify that foreign customer queries strictly drop all seeded demo documents (tenant isolation)."""
    svc = get_embedding_service()
    index = VectorIndex(dimension=svc.dimension)
    seed_runtime_vector_index(vector_index=index, embedding_service=svc)

    retriever = SemanticHistoricalRetriever(vector_index=index, embedding_service=svc)

    foreign_customer_id = UUID("11111111-2222-3333-4444-555555555555")
    ctx = CustomerRecoveryContext(
        customer=CustomerContext(
            customer_id=foreign_customer_id,
            total_payments=5,
            successful_payments=4,
            failed_payments=1,
            historical_success_rate=0.8,
        ),
        current_payment=PaymentContext(
            payment_id=uuid4(),
            amount=8450.0,
            currency="INR",
            payment_method="card",
            status="failed",
            failure_reason="customer_auth_failed_otp_timeout",
        ),
        current_opportunity=RecoveryOpportunityContext(
            opportunity_id=uuid4(),
            status="open",
            revenue_at_risk=8450.0,
            expected_recovery=0.0,
        ),
        current_payment_attempts=[],
    )

    results = retriever.retrieve(ctx, top_k=5)
    assert len(results) == 0, "Foreign customer must not receive another tenant's precedents"


def test_citation_tokens_remain_deterministic_and_safe():
    """Verify that AgentContextBuilder converts retrieved precedents into safe citation tokens."""
    svc = get_embedding_service()
    index = VectorIndex(dimension=svc.dimension)
    seed_runtime_vector_index(vector_index=index, embedding_service=svc)

    retriever = SemanticHistoricalRetriever(vector_index=index, embedding_service=svc)

    ctx = CustomerRecoveryContext(
        customer=CustomerContext(
            customer_id=DEMO_CUSTOMER_UUID,
            total_payments=10,
            successful_payments=9,
            failed_payments=1,
            historical_success_rate=0.9,
        ),
        current_payment=PaymentContext(
            payment_id=uuid4(),
            amount=8450.0,
            currency="INR",
            payment_method="card",
            status="failed",
            failure_reason="customer_auth_failed_otp_timeout",
        ),
        current_opportunity=RecoveryOpportunityContext(
            opportunity_id=uuid4(),
            status="open",
            revenue_at_risk=8450.0,
            expected_recovery=0.0,
        ),
        current_payment_attempts=[],
    )

    results = retriever.retrieve(ctx, top_k=3)
    assert len(results) > 0

    from app.decision_engine import RecoveryAction
    from app.policies.schemas import RecoveryPolicyContext

    policy_ctx = RecoveryPolicyContext(
        provider="default",
        policy_version="1.0.0",
        allowed_actions=(RecoveryAction.PAYMENT_LINK, RecoveryAction.RETRY_PAYMENT),
    )
    builder = AgentContextBuilder(max_historical_cases=3)
    prompt_context = builder.build_prompt_context(
        ctx, historical_cases=results, policy_context=policy_ctx
    )

    # Verify citation scheme
    for idx, case_dict in enumerate(prompt_context.historical_cases):
        expected_id = f"case_{idx + 1}"
        assert case_dict["case_id"] == expected_id
        # Ensure raw UUIDs are not leaked into prompt dictionary
        assert "payment_id" not in case_dict
        assert "customer_id" not in case_dict
        assert "id" not in case_dict


def test_fastapi_lifespan_seeds_shared_vector_index():
    """Verify that starting FastAPI application via lifespan seeds the shared vector index."""
    from fastapi.testclient import TestClient
    from app.main import app
    from app.vector_index import get_vector_index

    with TestClient(app) as client:
        resp = client.get("/health")
        assert resp.status_code == 200
        idx = get_vector_index()
        assert idx.size >= 100
