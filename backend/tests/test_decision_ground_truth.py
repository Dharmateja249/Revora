"""
Revora Decision Ground Truth & Dataset Contracts Tests.

Validates Stage 6.1 decision evaluation contracts:
- DecisionGroundTruth model construction, invariants, and serialization
- DecisionEvalResult model construction, immutability, and serialization
- EvaluationCase backward compatibility
- Golden dataset enrichment (50/50 cases) and policy safety invariants
"""

from datetime import datetime, timezone
from uuid import uuid4

import pytest
from app.context import (
    CustomerContext,
    CustomerRecoveryContext,
    CustomerRecoveryStatsContext,
    PaymentContext,
    RecoveryOpportunityContext,
)
from app.decision_engine import RecoveryAction
from app.evaluation.schemas import (
    DecisionEvalResult,
    DecisionGroundTruth,
    EvaluationCase,
    GroundTruthJudgment,
)
from pydantic import ValidationError

from tests.fixtures.retrieval_golden_dataset import (
    GOLDEN_EVALUATION_CASES,
    get_golden_evaluation_cases,
)


def _make_dummy_context() -> CustomerRecoveryContext:
    cust_id = uuid4()
    pay_id = uuid4()
    t_now = datetime.now(timezone.utc)
    return CustomerRecoveryContext(
        customer=CustomerContext(
            customer_id=cust_id,
            external_customer_id="cust_ext_001",
            risk_tier="low",
            created_at=t_now,
        ),
        current_payment=PaymentContext(
            payment_id=pay_id,
            external_payment_id="pay_ext_001",
            amount=500.0,
            currency="INR",
            payment_method="upi",
            status="failed",
            failure_reason="bank_timeout",
            created_at=t_now,
        ),
        current_opportunity=RecoveryOpportunityContext(
            opportunity_id=uuid4(),
            status="failed",
            revenue_at_risk=500.0,
        ),
        current_payment_attempts=[],
        historical_payments=[],
        recovery_statistics=CustomerRecoveryStatsContext(
            total_recovery_opportunities=0,
            recovered_opportunities=0,
            failed_opportunities=0,
            recovery_rate=0.0,
            previously_successful_actions=[],
            previously_failed_actions=[],
            total_amount_recovered=0.0,
        ),
        retrieved_at=t_now,
    )


# =============================================================================
# Test Group A: DecisionGroundTruth Model Construction & Invariants
# =============================================================================


def test_decision_ground_truth_valid_minimal_construction():
    truth = DecisionGroundTruth(expected_action=RecoveryAction.RETRY_PAYMENT)

    assert truth.expected_action == RecoveryAction.RETRY_PAYMENT
    assert truth.acceptable_actions == (RecoveryAction.RETRY_PAYMENT,)
    assert truth.expected_policy_ids == ()
    assert truth.prohibited_actions == ()
    assert truth.rationale is None
    assert truth.expected_reasoning_factors == ()


def test_decision_ground_truth_string_coercion():
    truth = DecisionGroundTruth(
        expected_action="retry_payment",
        acceptable_actions=["retry_payment", "wait_and_retry"],
        prohibited_actions=["payment_link"],
    )

    assert truth.expected_action == RecoveryAction.RETRY_PAYMENT
    assert truth.acceptable_actions == (
        RecoveryAction.RETRY_PAYMENT,
        RecoveryAction.WAIT_AND_RETRY,
    )
    assert truth.prohibited_actions == (RecoveryAction.PAYMENT_LINK,)


def test_decision_ground_truth_expected_action_required():
    with pytest.raises(ValidationError):
        DecisionGroundTruth()


def test_decision_ground_truth_invalid_action_rejected():
    with pytest.raises(ValidationError):
        DecisionGroundTruth(expected_action="invalid_action_name")


def test_decision_ground_truth_automatically_includes_expected_in_acceptable():
    truth = DecisionGroundTruth(
        expected_action=RecoveryAction.RETRY_PAYMENT,
        acceptable_actions=(RecoveryAction.WAIT_AND_RETRY,),
    )

    assert RecoveryAction.RETRY_PAYMENT in truth.acceptable_actions
    assert truth.acceptable_actions == (
        RecoveryAction.RETRY_PAYMENT,
        RecoveryAction.WAIT_AND_RETRY,
    )


def test_decision_ground_truth_removes_duplicate_actions_preserving_order():
    truth = DecisionGroundTruth(
        expected_action=RecoveryAction.PAYMENT_LINK,
        acceptable_actions=[
            RecoveryAction.PAYMENT_LINK,
            RecoveryAction.PAYMENT_LINK,
            RecoveryAction.CHANGE_PAYMENT_METHOD,
            RecoveryAction.PAYMENT_LINK,
        ],
        prohibited_actions=[
            RecoveryAction.RETRY_PAYMENT,
            RecoveryAction.RETRY_PAYMENT,
            RecoveryAction.WAIT_AND_RETRY,
        ],
    )

    assert truth.acceptable_actions == (
        RecoveryAction.PAYMENT_LINK,
        RecoveryAction.CHANGE_PAYMENT_METHOD,
    )
    assert truth.prohibited_actions == (
        RecoveryAction.RETRY_PAYMENT,
        RecoveryAction.WAIT_AND_RETRY,
    )


def test_decision_ground_truth_rejects_expected_in_prohibited():
    with pytest.raises(ValidationError, match="cannot be in prohibited_actions"):
        DecisionGroundTruth(
            expected_action=RecoveryAction.RETRY_PAYMENT,
            prohibited_actions=(RecoveryAction.RETRY_PAYMENT,),
        )


def test_decision_ground_truth_rejects_acceptable_and_prohibited_overlap():
    with pytest.raises(ValidationError, match="overlap on actions"):
        DecisionGroundTruth(
            expected_action=RecoveryAction.PAYMENT_LINK,
            acceptable_actions=(
                RecoveryAction.PAYMENT_LINK,
                RecoveryAction.CHANGE_PAYMENT_METHOD,
            ),
            prohibited_actions=(RecoveryAction.CHANGE_PAYMENT_METHOD,),
        )


def test_decision_ground_truth_is_frozen_and_immutable():
    truth = DecisionGroundTruth(expected_action=RecoveryAction.RETRY_PAYMENT)

    with pytest.raises(ValidationError):
        truth.expected_action = RecoveryAction.PAYMENT_LINK


def test_decision_ground_truth_collections_are_immutable_tuples():
    truth = DecisionGroundTruth(
        expected_action=RecoveryAction.PAYMENT_LINK,
        acceptable_actions=[
            RecoveryAction.PAYMENT_LINK,
            RecoveryAction.CHANGE_PAYMENT_METHOD,
        ],
        expected_policy_ids=["RZP_CUSTOMER_AUTH_2FA_REQUIRED"],
        prohibited_actions=[RecoveryAction.RETRY_PAYMENT],
        expected_reasoning_factors=["2fa_required", "interactive_link"],
    )

    assert isinstance(truth.acceptable_actions, tuple)
    assert isinstance(truth.prohibited_actions, tuple)
    assert isinstance(truth.expected_policy_ids, tuple)
    assert isinstance(truth.expected_reasoning_factors, tuple)


# =============================================================================
# Test Group B: Serialization & Deserialization Round-Trip
# =============================================================================


def test_decision_ground_truth_json_roundtrip():
    original = DecisionGroundTruth(
        expected_action=RecoveryAction.PAYMENT_LINK,
        acceptable_actions=(
            RecoveryAction.PAYMENT_LINK,
            RecoveryAction.CHANGE_PAYMENT_METHOD,
        ),
        expected_policy_ids=("RZP_CUSTOMER_AUTH_2FA_REQUIRED",),
        prohibited_actions=(RecoveryAction.RETRY_PAYMENT,),
        rationale="2FA failure requires interactive payment link.",
        expected_reasoning_factors=("otp_expired", "2fa_required"),
    )

    dumped = original.model_dump(mode="json")
    assert dumped["expected_action"] == "payment_link"
    assert dumped["acceptable_actions"] == ["payment_link", "change_payment_method"]
    assert dumped["prohibited_actions"] == ["retry_payment"]
    assert dumped["expected_policy_ids"] == ["RZP_CUSTOMER_AUTH_2FA_REQUIRED"]

    restored = DecisionGroundTruth.model_validate(dumped)
    assert restored == original


def test_evaluation_case_with_decision_ground_truth_roundtrip():
    context = _make_dummy_context()
    decision_gt = DecisionGroundTruth(
        expected_action=RecoveryAction.RETRY_PAYMENT,
        acceptable_actions=(
            RecoveryAction.RETRY_PAYMENT,
            RecoveryAction.WAIT_AND_RETRY,
        ),
        rationale="Prior retry recovery succeeded.",
    )
    case = EvaluationCase(
        query_id=uuid4(),
        context=context,
        ground_truth=(
            GroundTruthJudgment(
                payment_id=uuid4(), relevance_grade=3, rationale="Exact match"
            ),
        ),
        decision_ground_truth=decision_gt,
        description="UPI Timeout Scenario",
    )

    dumped = case.model_dump(mode="json")
    restored = EvaluationCase.model_validate(dumped)

    assert restored.decision_ground_truth is not None
    assert (
        restored.decision_ground_truth.expected_action == RecoveryAction.RETRY_PAYMENT
    )
    assert restored.decision_ground_truth.acceptable_actions == (
        RecoveryAction.RETRY_PAYMENT,
        RecoveryAction.WAIT_AND_RETRY,
    )
    assert restored.decision_ground_truth.rationale == "Prior retry recovery succeeded."


def test_decision_eval_result_construction_and_serialization():
    q_id = uuid4()
    result = DecisionEvalResult(
        query_id=q_id,
        pipeline_name="agent_rag_pipeline",
        predicted_action=RecoveryAction.PAYMENT_LINK,
        expected_action=RecoveryAction.PAYMENT_LINK,
        acceptable_actions=(
            RecoveryAction.PAYMENT_LINK,
            RecoveryAction.CHANGE_PAYMENT_METHOD,
        ),
        prohibited_actions=(RecoveryAction.RETRY_PAYMENT,),
        expected_policy_ids=("RZP_CUSTOMER_AUTH_2FA_REQUIRED",),
        is_exact_match=True,
        is_acceptable_match=True,
        confidence=0.92,
        policy_overridden=False,
        is_fallback=False,
        applied_policy_ids=("RZP_CUSTOMER_AUTH_2FA_REQUIRED",),
        referenced_case_ids=("case_001", "case_002"),
        key_factors=("otp_timeout", "interactive_link"),
        latency_ms=145.5,
        metadata={"model": "mock_llm"},
    )

    assert result.query_id == q_id
    assert result.is_exact_match is True
    assert result.is_acceptable_match is True
    assert result.confidence == 0.92
    assert result.latency_ms == 145.5

    dumped = result.model_dump(mode="json")
    restored = DecisionEvalResult.model_validate(dumped)
    assert restored == result


# =============================================================================
# Test Group C: EvaluationCase Backward Compatibility
# =============================================================================


def test_evaluation_case_without_decision_ground_truth_defaults_to_none():
    context = _make_dummy_context()
    case = EvaluationCase(
        query_id=uuid4(),
        context=context,
        ground_truth=(),
    )

    assert case.decision_ground_truth is None


def test_evaluation_case_preserves_retrieval_ground_truth():
    context = _make_dummy_context()
    p1 = uuid4()
    p2 = uuid4()
    case = EvaluationCase(
        query_id=uuid4(),
        context=context,
        ground_truth=(
            GroundTruthJudgment(payment_id=p1, relevance_grade=3),
            GroundTruthJudgment(payment_id=p2, relevance_grade=0),
        ),
        decision_ground_truth=DecisionGroundTruth(
            expected_action=RecoveryAction.RETRY_PAYMENT
        ),
    )

    assert len(case.ground_truth) == 2
    assert case.ground_truth[0].payment_id == p1
    assert case.ground_truth[0].relevance_grade == 3
    assert case.decision_ground_truth is not None
    assert case.decision_ground_truth.expected_action == RecoveryAction.RETRY_PAYMENT


# =============================================================================
# Test Group D: Golden Dataset Completeness & Consistency (50 Scenarios)
# =============================================================================


def test_golden_dataset_cardinality():
    cases = get_golden_evaluation_cases()
    assert len(cases) == 50
    assert len(GOLDEN_EVALUATION_CASES) == 50


def test_all_50_cases_have_decision_ground_truth():
    cases = get_golden_evaluation_cases()
    for idx, case in enumerate(cases):
        assert case.decision_ground_truth is not None, (
            f"Case #{idx + 1} (scenario {case.metadata.get('scenario_id')}) lacks decision_ground_truth"
        )


def test_all_50_cases_satisfy_expected_in_acceptable():
    cases = get_golden_evaluation_cases()
    for case in cases:
        dgt = case.decision_ground_truth
        assert dgt is not None
        assert isinstance(dgt.expected_action, RecoveryAction)
        assert dgt.expected_action in dgt.acceptable_actions


# =============================================================================
# Safety Invariant Tests across Golden Dataset
# =============================================================================

PERMANENT_FAILURE_REASONS = {
    "card_expired",
    "expired_card",
    "invalid_cvv",
    "invalid_card_number",
    "blocked_account",
    "do_not_honor",
    "account_closed",
    "invalid_account",
}

CUSTOMER_2FA_AUTH_REASONS = {
    "otp_expired",
    "otp_timeout",
    "3ds_failed",
    "authentication_failed",
    "pin_incorrect",
    "user_cancelled",
    "customer_cancelled",
    "declined_by_user",
}


def test_permanent_failures_prohibit_retry_across_golden_dataset():
    cases = get_golden_evaluation_cases()
    permanent_cases_found = 0

    for case in cases:
        failure_reason = (
            case.context.current_payment.failure_reason
            if case.context.current_payment
            else None
        )
        if failure_reason in PERMANENT_FAILURE_REASONS:
            permanent_cases_found += 1
            dgt = case.decision_ground_truth
            assert dgt is not None, f"Scenario for failure {failure_reason} lacks dgt"
            assert RecoveryAction.RETRY_PAYMENT not in dgt.acceptable_actions, (
                f"Permanent failure '{failure_reason}' must never allow RETRY_PAYMENT in acceptable_actions"
            )
            assert RecoveryAction.RETRY_PAYMENT in dgt.prohibited_actions, (
                f"Permanent failure '{failure_reason}' must explicitly prohibit RETRY_PAYMENT"
            )

    assert permanent_cases_found >= 8, (
        f"Expected at least 8 permanent failure cases, found {permanent_cases_found}"
    )


def test_2fa_failures_prohibit_retry_across_golden_dataset():
    cases = get_golden_evaluation_cases()
    auth_cases_found = 0

    for case in cases:
        failure_reason = (
            case.context.current_payment.failure_reason
            if case.context.current_payment
            else None
        )
        if failure_reason in CUSTOMER_2FA_AUTH_REASONS:
            auth_cases_found += 1
            dgt = case.decision_ground_truth
            assert dgt is not None, f"Scenario for failure {failure_reason} lacks dgt"
            assert RecoveryAction.RETRY_PAYMENT not in dgt.acceptable_actions, (
                f"2FA failure '{failure_reason}' must never allow RETRY_PAYMENT in acceptable_actions"
            )
            assert RecoveryAction.RETRY_PAYMENT in dgt.prohibited_actions, (
                f"2FA failure '{failure_reason}' must explicitly prohibit RETRY_PAYMENT"
            )

    assert auth_cases_found >= 8, (
        f"Expected at least 8 2FA/auth failure cases, found {auth_cases_found}"
    )
