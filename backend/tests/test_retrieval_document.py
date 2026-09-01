"""
Unit and Integration Tests for RetrievalDocument and Canonical Text Construction.

Tests:
1. Valid HistoricalCase converts successfully to RetrievalDocument
2. Output is a valid, immutable RetrievalDocument
3. Identical inputs produce byte-for-byte identical text (deterministic conversion)
4. Field ordering is strictly deterministic
5. Missing optional fields are handled consistently ('none')
6. Whitespace and case normalization works
7. Numeric formatting is deterministic (2 decimal places)
8. PII fields (customer name, email) are strictly absent from text and metadata
9. Metadata is deeply immutable (MappingProxyType)
10. Mutating the original input case/metadata does not mutate the document
11. Different historical cases produce appropriately distinct text representations
12. Serialization round-tripping to/from dictionary and JSON
"""

import json
import uuid
from datetime import datetime, timedelta, timezone

import pytest
from app.historical_retrieval import HistoricalCase
from app.retrieval_document import (
    RetrievalDocument,
    construct_canonical_case_text,
    historical_case_to_document,
)
from pydantic import ValidationError

# ============================================================================
# 1. Valid Construction & Conversion
# ============================================================================


def test_valid_historical_case_conversion():
    """Verify clean conversion from HistoricalCase to RetrievalDocument."""
    payment_id = uuid.uuid4()
    customer_id = uuid.uuid4()
    t0 = datetime(2026, 2, 1, 10, 0, 0, tzinfo=timezone.utc)

    case = HistoricalCase(
        payment_id=payment_id,
        customer_id=customer_id,
        external_payment_id="pay_ext_100",
        external_customer_id="cust_ext_100",
        amount=2500.0,
        currency="INR",
        payment_method="card",
        failure_reason="insufficient_funds",
        recovery_action="payment_link",
        recovery_status="recovered",
        amount_recovered=2500.0,
        was_recovered=True,
        relevance_score=0.95,
        created_at=t0,
        completed_at=t0 + timedelta(minutes=15),
        metadata={"channel": "sms", "nested": {"key": "val"}},
    )

    doc = historical_case_to_document(case)

    assert isinstance(doc, RetrievalDocument)
    assert doc.case_id == payment_id
    assert isinstance(doc.text, str)
    assert len(doc.text) > 0
    assert doc.metadata["payment_id"] == str(payment_id)
    assert doc.metadata["customer_id"] == str(customer_id)
    assert doc.metadata["channel"] == "sms"
    assert doc.metadata["nested"]["key"] == "val"


# ============================================================================
# 2. Deterministic Canonical Text & Field Ordering
# ============================================================================


def test_identical_inputs_produce_identical_text():
    """Verify that converting the same HistoricalCase multiple times yields exact identical text."""
    payment_id = uuid.uuid4()
    customer_id = uuid.uuid4()

    case = HistoricalCase(
        payment_id=payment_id,
        customer_id=customer_id,
        amount=1500.5,
        currency="INR",
        payment_method="upi",
        failure_reason="bank_timeout",
        recovery_action="smart_retry",
        recovery_status="recovered",
        amount_recovered=1500.5,
        was_recovered=True,
    )

    text_1 = construct_canonical_case_text(case)
    text_2 = construct_canonical_case_text(case)
    doc_1 = historical_case_to_document(case)
    doc_2 = historical_case_to_document(case)

    assert text_1 == text_2
    assert doc_1.text == doc_2.text
    assert doc_1.text == (
        "failure_reason: bank_timeout\n"
        "payment_method: upi\n"
        "amount: 1500.50\n"
        "currency: INR\n"
        "recovery_action: smart_retry\n"
        "recovery_status: recovered\n"
        "was_recovered: true\n"
        "amount_recovered: 1500.50"
    )


def test_canonical_text_field_ordering():
    """Verify the exact deterministic order of lines in canonical text."""
    case = HistoricalCase(
        payment_id=uuid.uuid4(),
        customer_id=uuid.uuid4(),
        amount=500.0,
        payment_method="card",
        recovery_status="failed",
    )

    text = construct_canonical_case_text(case)
    lines = text.split("\n")

    assert len(lines) == 8
    assert lines[0].startswith("failure_reason:")
    assert lines[1].startswith("payment_method:")
    assert lines[2].startswith("amount:")
    assert lines[3].startswith("currency:")
    assert lines[4].startswith("recovery_action:")
    assert lines[5].startswith("recovery_status:")
    assert lines[6].startswith("was_recovered:")
    assert lines[7].startswith("amount_recovered:")


# ============================================================================
# 3. Missing Fields & Normalization
# ============================================================================


def test_missing_optional_fields_handled_consistently():
    """Verify missing failure_reason or recovery_action normalized to 'none'."""
    case = HistoricalCase(
        payment_id=uuid.uuid4(),
        customer_id=uuid.uuid4(),
        amount=0.0,
        currency="INR",
        payment_method="card",
        failure_reason=None,
        recovery_action=None,
        recovery_status="pending",
        amount_recovered=0.0,
        was_recovered=False,
    )

    doc = historical_case_to_document(case)
    assert "failure_reason: none" in doc.text
    assert "recovery_action: none" in doc.text
    assert "was_recovered: false" in doc.text
    assert "amount: 0.00" in doc.text
    assert "amount_recovered: 0.00" in doc.text


def test_whitespace_and_casing_normalization():
    """Verify whitespace is stripped and casing normalized across fields."""
    case = HistoricalCase(
        payment_id=uuid.uuid4(),
        customer_id=uuid.uuid4(),
        amount=99.9,
        currency=" inr ",
        payment_method=" CARD ",
        failure_reason="  Bank_Timeout  ",
        recovery_action=" SMART_RETRY ",
        recovery_status=" RECOVERED ",
        amount_recovered=99.9,
        was_recovered=True,
    )

    doc = historical_case_to_document(case)
    assert "failure_reason: bank_timeout" in doc.text
    assert "payment_method: card" in doc.text
    assert "currency: INR" in doc.text
    assert "recovery_action: smart_retry" in doc.text
    assert "recovery_status: recovered" in doc.text


def test_numeric_formatting_precision():
    """Verify float values format strictly to 2 decimal places."""
    case = HistoricalCase(
        payment_id=uuid.uuid4(),
        customer_id=uuid.uuid4(),
        amount=100.12345,
        payment_method="upi",
        recovery_status="recovered",
        amount_recovered=100.12345,
    )

    text = construct_canonical_case_text(case)
    assert "amount: 100.12" in text
    assert "amount_recovered: 100.12" in text


# ============================================================================
# 4. Strict PII Exclusion Guarantees
# ============================================================================


def test_pii_fields_strictly_absent():
    """Verify customer names, emails, and sensitive identifiers never appear in document text or metadata."""
    case = HistoricalCase(
        payment_id=uuid.uuid4(),
        customer_id=uuid.uuid4(),
        amount=500.0,
        payment_method="card",
        recovery_status="failed",
        metadata={
            "name": "Secret Customer Name",
            "email": "sensitive@example.com",
            "phone": "+123456789",
            "address": "123 Main St",
            "valid_key": "safe_data",
        },
    )

    doc = historical_case_to_document(case)

    # Document text check
    assert "Secret Customer Name" not in doc.text
    assert "sensitive@example.com" not in doc.text
    assert "+123456789" not in doc.text
    assert "123 Main St" not in doc.text

    # Document metadata check
    assert "name" not in doc.metadata
    assert "email" not in doc.metadata
    assert "phone" not in doc.metadata
    assert "address" not in doc.metadata
    assert doc.metadata["valid_key"] == "safe_data"


def test_compound_and_mixed_case_pii_keys_rejected():
    """Verify compound and mixed-case PII keys are filtered out of document metadata."""
    case = HistoricalCase(
        payment_id=uuid.uuid4(),
        customer_id=uuid.uuid4(),
        amount=500.0,
        payment_method="card",
        recovery_status="failed",
        metadata={
            "customer_name": "Alice Smith",
            "Customer_Email": "alice@example.com",
            "billing_address": "456 Market St",
            "phone_number": "+987654321",
            "CUSTOMER_PHONE": "+1122334455",
            "shipping_address_line1": "Warehouse 7",
            "user_name": "asmith",
            "legitimate_tag": "high_priority",
            "attempt_count": 3,
        },
    )

    doc = historical_case_to_document(case)

    # None of the PII keys should be in metadata (case-insensitive check)
    for forbidden in [
        "customer_name",
        "customer_email",
        "billing_address",
        "phone_number",
        "customer_phone",
        "shipping_address_line1",
        "user_name",
    ]:
        assert forbidden not in doc.metadata

    # Legitimate non-PII metadata must be preserved
    assert doc.metadata["legitimate_tag"] == "high_priority"
    assert doc.metadata["attempt_count"] == 3


def test_caller_metadata_cannot_override_canonical_fields():
    """Verify caller-supplied metadata cannot overwrite canonical document fields."""
    pid = uuid.uuid4()
    cid = uuid.uuid4()
    spoofed_pid = str(uuid.uuid4())
    spoofed_cid = str(uuid.uuid4())

    case = HistoricalCase(
        payment_id=pid,
        customer_id=cid,
        amount=500.0,
        currency="INR",
        payment_method="card",
        recovery_status="failed",
        amount_recovered=0.0,
        was_recovered=False,
        metadata={
            "payment_id": spoofed_pid,
            "customer_id": spoofed_cid,
            "was_recovered": True,
            "amount_recovered": 9999.0,
            "amount": 1.0,
            "custom_campaign_id": "summer_sale",
        },
    )

    doc = historical_case_to_document(case)

    # Canonical fields must retain their authentic values
    assert doc.metadata["payment_id"] == str(pid)
    assert doc.metadata["customer_id"] == str(cid)
    assert doc.metadata["was_recovered"] is False
    assert doc.metadata["amount_recovered"] == 0.0
    assert doc.metadata["amount"] == 500.0
    # Custom non-canonical field is preserved
    assert doc.metadata["custom_campaign_id"] == "summer_sale"


# ============================================================================
# 5. Immutability & Mutation Isolation
# ============================================================================


def test_retrieval_document_immutability():
    """Verify attempting to reassign attributes on RetrievalDocument raises ValidationError."""
    doc = RetrievalDocument(
        case_id=uuid.uuid4(),
        text="canonical text",
        metadata={"k": "v"},
    )

    with pytest.raises(ValidationError):
        doc.text = "new text"  # type: ignore

    with pytest.raises(ValidationError):
        doc.case_id = uuid.uuid4()  # type: ignore


def test_nested_metadata_mutation_isolation():
    """Verify mutating source dictionary does not alter the document's metadata."""
    source_meta = {"key": "original", "nested": {"count": 1}}

    case = HistoricalCase(
        payment_id=uuid.uuid4(),
        customer_id=uuid.uuid4(),
        amount=100.0,
        payment_method="card",
        recovery_status="open",
        metadata=source_meta,
    )

    doc = historical_case_to_document(case)

    # Mutate source dict
    source_meta["key"] = "mutated"
    source_meta["nested"]["count"] = 999
    source_meta["new_key"] = "added"

    assert doc.metadata["key"] == "original"
    assert doc.metadata["nested"]["count"] == 1
    assert "new_key" not in doc.metadata

    # In-place modification on MappingProxyType must raise TypeError
    with pytest.raises(TypeError):
        doc.metadata["key"] = "direct_mutation"  # type: ignore


# ============================================================================
# 6. Differentiation & Serialization
# ============================================================================


def test_different_cases_produce_different_text():
    """Verify different historical cases produce distinct canonical representations."""
    case_a = HistoricalCase(
        payment_id=uuid.uuid4(),
        customer_id=uuid.uuid4(),
        amount=1000.0,
        payment_method="card",
        failure_reason="insufficient_funds",
        recovery_action="wait_and_retry",
        recovery_status="recovered",
        amount_recovered=1000.0,
        was_recovered=True,
    )

    case_b = HistoricalCase(
        payment_id=uuid.uuid4(),
        customer_id=uuid.uuid4(),
        amount=5000.0,
        payment_method="upi",
        failure_reason="bank_timeout",
        recovery_action="payment_link",
        recovery_status="failed",
        amount_recovered=0.0,
        was_recovered=False,
    )

    doc_a = historical_case_to_document(case_a)
    doc_b = historical_case_to_document(case_b)

    assert doc_a.text != doc_b.text
    assert "card" in doc_a.text and "upi" in doc_b.text
    assert "insufficient_funds" in doc_a.text and "bank_timeout" in doc_b.text
    assert "was_recovered: true" in doc_a.text and "was_recovered: false" in doc_b.text


def test_retrieval_document_serialization_roundtrip():
    """Verify round-trip serialization to/from Python dict and JSON."""
    case_id = uuid.uuid4()
    doc = RetrievalDocument(
        case_id=case_id,
        text="canonical text content",
        metadata={"source": "unit_test", "tags": ["search", "recovery"]},
    )

    # Model dump
    dumped = doc.model_dump()
    assert dumped["case_id"] == case_id
    assert dumped["text"] == "canonical text content"
    assert dumped["metadata"]["tags"] == ["search", "recovery"]

    reconstructed_from_dict = RetrievalDocument.model_validate(dumped)
    assert reconstructed_from_dict == doc

    # JSON dump
    json_str = doc.model_dump_json()
    assert isinstance(json_str, str)
    parsed = json.loads(json_str)
    assert parsed["case_id"] == str(case_id)
    assert parsed["text"] == "canonical text content"

    reconstructed_from_json = RetrievalDocument.model_validate_json(json_str)
    assert reconstructed_from_json == doc
