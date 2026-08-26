import pytest
from app.synthetic_data import (
    FAILURE_REASONS,
    PAYMENT_METHODS,
    RECOVERY_ACTIONS,
    calculate_dataset_statistics,
    generate_synthetic_recovery_dataset,
)


@pytest.fixture(scope="module")
def dataset():
    """Generate deterministic 5,000 record dataset for testing."""
    return generate_synthetic_recovery_dataset(num_records=5000, seed=42)


def test_dataset_size_and_uniqueness(dataset):
    """Verify record count is 5,000 and all record_ids are unique."""
    assert len(dataset) == 5000
    record_ids = [r["record_id"] for r in dataset]
    assert len(set(record_ids)) == 5000


def test_id_validity(dataset):
    """Verify customer_id and payment_id are non-empty and well-formed."""
    for r in dataset:
        assert isinstance(r["customer_id"], str) and len(r["customer_id"]) > 0
        assert isinstance(r["payment_id"], str) and len(r["payment_id"]) > 0
        assert r["customer_id"].startswith("cust_")
        assert r["payment_id"].startswith("pay_")


def test_controlled_categorical_sets(dataset):
    """Verify all categorical fields belong to allowed sets."""
    allowed_reasons = set(FAILURE_REASONS)
    allowed_methods = set(PAYMENT_METHODS)
    allowed_actions = set(RECOVERY_ACTIONS)

    for r in dataset:
        assert r["failure_reason"] in allowed_reasons
        assert r["payment_method"] in allowed_methods
        assert r["action_taken"] in allowed_actions
        assert r["currency"] == "INR"


def test_numerical_bounds_and_relationships(dataset):
    """Verify numerical field constraints and recovery outcome invariants."""
    for r in dataset:
        assert r["payment_amount"] > 0
        assert r["amount_recovered"] >= 0
        assert r["amount_recovered"] <= r["payment_amount"]
        assert r["attempt_number"] >= 1
        assert r["previous_attempt_count"] == r["attempt_number"] - 1
        assert r["hours_since_failure"] >= 0
        assert r["recovery_time_hours"] >= 0
        assert 0.0 <= r["customer_success_rate"] <= 1.0
        assert r["customer_previous_failures"] >= 0
        assert r["customer_payment_count"] >= 1

        # Outcome integrity
        if r["recovered"]:
            assert r["amount_recovered"] > 0
            assert r["amount_recovered"] == r["payment_amount"]
            assert r["recovery_time_hours"] > 0
        else:
            assert r["amount_recovered"] == 0.0
            assert r["recovery_time_hours"] == 0.0

        # Action integrity
        if r["action_taken"] == "STOP":
            assert r["recovered"] is False
            assert r["amount_recovered"] == 0.0


def test_reproducibility():
    """Verify that the same random seed produces bitwise identical datasets."""
    ds1 = generate_synthetic_recovery_dataset(num_records=100, seed=123)
    ds2 = generate_synthetic_recovery_dataset(num_records=100, seed=123)
    assert ds1 == ds2


def test_statistical_patterns(dataset):
    """Verify that expected domain probabilistic tendencies exist in the data."""
    stats = calculate_dataset_statistics(dataset)
    cross_tab = stats["cross_tab_reason_action"]
    by_attempt = stats["recovery_rate_by_attempt_number"]

    # 1. RETRY is significantly more effective on bank_timeout than on insufficient_funds
    bank_timeout_retry = cross_tab["bank_timeout"]["RETRY"]
    insufficient_funds_retry = cross_tab["insufficient_funds"]["RETRY"]
    assert bank_timeout_retry > 0.50
    assert insufficient_funds_retry < 0.30
    assert bank_timeout_retry > insufficient_funds_retry

    # 2. PAYMENT_LINK outperforms RETRY for authentication_failed
    auth_retry = cross_tab["authentication_failed"]["RETRY"]
    auth_link = cross_tab["authentication_failed"]["PAYMENT_LINK"]
    assert auth_link > auth_retry
    assert auth_link > 0.50

    # 3. PAYMENT_LINK outperforms RETRY for insufficient_funds
    funds_link = cross_tab["insufficient_funds"]["PAYMENT_LINK"]
    assert funds_link > insufficient_funds_retry

    # 4. Attempt decay: early attempts have higher recovery rate than late attempts
    attempt_1_rate = by_attempt[1]["recovery_rate"]
    attempt_2_rate = by_attempt[2]["recovery_rate"]
    assert attempt_1_rate > attempt_2_rate
