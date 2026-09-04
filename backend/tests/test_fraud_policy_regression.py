"""
Regression tests for fraud_hard_decline stopping policy and safety invariants.

Validates end-to-end:
1. fraud_hard_decline + LLM recommendation CHANGE_PAYMENT_METHOD -> overridden to NO_ACTION
2. fraud_hard_decline + LLM recommendation PAYMENT_LINK -> overridden to NO_ACTION
3. fraud_hard_decline + LLM recommendation RETRY_PAYMENT -> overridden to NO_ACTION
4. fraud_hard_decline -> no executable action (attempted=False, skipped)
5. Legitimate recoverable failures (insufficient_funds, expired_card) retain their intended actions.
"""

from uuid import uuid4

from app.action_executor import ActionExecutor
from app.agent.context_builder import AgentContextBuilder
from app.agent.orchestrator import AgentOrchestrator
from app.agent.provider import MockLLMProvider
from app.agent.schemas import LLMRecoveryRecommendation
from app.context import (
    CustomerContext,
    CustomerRecoveryContext,
    PaymentContext,
    RecoveryOpportunityContext,
)
from app.decision_engine import RecoveryAction
from app.policies.resolver import resolve_policy_context
from app.policies.validator import PolicyValidator
from app.recovery_decision_service import RecoveryDecisionService
from app.schemas.decision import RecoveryDecisionRequest


def _build_recovery_context(failure_reason: str) -> CustomerRecoveryContext:
    return CustomerRecoveryContext(
        customer=CustomerContext(
            customer_id=uuid4(),
            total_payments=20,
            successful_payments=18,
            failed_payments=2,
            historical_success_rate=0.90,
        ),
        current_payment=PaymentContext(
            payment_id=uuid4(),
            amount=15000.0,
            currency="INR",
            payment_method="card",
            status="failed",
            failure_reason=failure_reason,
        ),
        current_opportunity=RecoveryOpportunityContext(
            opportunity_id=uuid4(),
            status="open",
            revenue_at_risk=15000.0,
            expected_recovery=0.0,
        ),
        current_payment_attempts=[],
    )


class TestFraudHardDeclinePolicyRegression:
    """Test suite ensuring zero-tolerance policy enforcement on fraud declines."""

    def test_fraud_hard_decline_policy_resolution(self):
        """Verify resolve_policy_context enforces strict safety invariants for fraud."""
        context = _build_recovery_context("fraud_hard_decline")
        policy = resolve_policy_context(context)

        applied_ids = [r.policy_id for r in policy.applicable_rules]
        assert "SAFETY_FRAUD_SECURITY_DECLINE" in applied_ids
        assert policy.allowed_actions == (RecoveryAction.NO_ACTION,)
        assert RecoveryAction.RETRY_PAYMENT in policy.prohibited_actions
        assert RecoveryAction.WAIT_AND_RETRY in policy.prohibited_actions
        assert RecoveryAction.PAYMENT_LINK in policy.prohibited_actions
        assert RecoveryAction.CHANGE_PAYMENT_METHOD in policy.prohibited_actions
        assert policy.mandatory_fallback_action == RecoveryAction.NO_ACTION

    def test_fraud_hard_decline_llm_change_payment_method_overridden_to_no_action(self):
        """fraud_hard_decline + LLM candidate CHANGE_PAYMENT_METHOD -> overridden to NO_ACTION."""
        context = _build_recovery_context("fraud_hard_decline")
        policy = resolve_policy_context(context)
        validator = PolicyValidator()

        val_result = validator.validate_decision(
            RecoveryAction.CHANGE_PAYMENT_METHOD, policy
        )
        assert val_result.is_valid is False
        assert val_result.was_overridden is True
        assert val_result.effective_action == RecoveryAction.NO_ACTION
        assert "SAFETY_FRAUD_SECURITY_DECLINE" in val_result.violated_policy_ids

    def test_fraud_hard_decline_llm_payment_link_overridden_to_no_action(self):
        """fraud_hard_decline + LLM candidate PAYMENT_LINK -> overridden to NO_ACTION."""
        context = _build_recovery_context("fraud_hard_decline")
        policy = resolve_policy_context(context)
        validator = PolicyValidator()

        val_result = validator.validate_decision(RecoveryAction.PAYMENT_LINK, policy)
        assert val_result.is_valid is False
        assert val_result.was_overridden is True
        assert val_result.effective_action == RecoveryAction.NO_ACTION
        assert "SAFETY_FRAUD_SECURITY_DECLINE" in val_result.violated_policy_ids

    def test_fraud_hard_decline_llm_retry_payment_overridden_to_no_action(self):
        """fraud_hard_decline + LLM candidate RETRY_PAYMENT -> overridden to NO_ACTION."""
        context = _build_recovery_context("fraud_hard_decline")
        policy = resolve_policy_context(context)
        validator = PolicyValidator()

        val_result = validator.validate_decision(RecoveryAction.RETRY_PAYMENT, policy)
        assert val_result.is_valid is False
        assert val_result.was_overridden is True
        assert val_result.effective_action == RecoveryAction.NO_ACTION
        assert "SAFETY_FRAUD_SECURITY_DECLINE" in val_result.violated_policy_ids

    def test_agent_orchestrator_overrides_llm_for_fraud_hard_decline(self):
        """AgentOrchestrator rejects unsafe LLM output and enforces NO_ACTION."""
        import asyncio

        context = _build_recovery_context("fraud_hard_decline")
        policy = resolve_policy_context(context)

        mock_provider = MockLLMProvider(
            recommendation=LLMRecoveryRecommendation(
                recommended_action=RecoveryAction.CHANGE_PAYMENT_METHOD,
                confidence=0.85,
                reasoning="Suggesting alternative payment method despite fraud flag.",
                key_factors=("customer_request",),
                referenced_case_ids=(),
            )
        )
        orchestrator = AgentOrchestrator(
            provider=mock_provider,
            policy_validator=PolicyValidator(),
            context_builder=AgentContextBuilder(),
        )

        decision = asyncio.run(orchestrator.decide(context, policy))
        assert decision.recommendation.recommended_action == RecoveryAction.NO_ACTION
        assert decision.metadata["policy_overridden"] is True
        assert (
            "SAFETY_FRAUD_SECURITY_DECLINE" in decision.metadata["violated_policy_ids"]
        )

    def test_fraud_hard_decline_no_executable_action(self):
        """fraud_hard_decline produces NO_ACTION and refuses automated execution."""
        import asyncio

        context = _build_recovery_context("fraud_hard_decline")
        policy = resolve_policy_context(context)

        # 1. ActionExecutor skips NO_ACTION
        executor = ActionExecutor()
        result = asyncio.run(
            executor.execute(RecoveryAction.NO_ACTION, policy, context)
        )
        assert result.action == RecoveryAction.NO_ACTION
        assert result.attempted is False
        assert result.status == "skipped"

        # 2. RecoveryDecisionService end-to-end with execute_action=True
        mock_provider = MockLLMProvider(
            recommendation=LLMRecoveryRecommendation(
                recommended_action=RecoveryAction.CHANGE_PAYMENT_METHOD,
                confidence=0.88,
                reasoning="Unsafe recommendation.",
                key_factors=(),
                referenced_case_ids=(),
            )
        )
        orchestrator = AgentOrchestrator(
            provider=mock_provider,
            policy_validator=PolicyValidator(),
            context_builder=AgentContextBuilder(),
        )
        service = RecoveryDecisionService(
            agent_orchestrator=orchestrator,
            action_executor=executor,
        )

        request = RecoveryDecisionRequest(
            amount=15000.0,
            currency="INR",
            payment_method="card",
            failure_reason="fraud_hard_decline",
            payment_status="failed",
            customer={
                "customer_id": str(context.customer.customer_id),
                "total_payments": 20,
                "successful_payments": 18,
                "failed_payments": 2,
                "historical_success_rate": 0.90,
            },
            previous_attempts=[],
            opportunity_status="open",
            revenue_at_risk=15000.0,
            max_attempts=3,
            execute_action=True,
        )
        resp = asyncio.run(service.evaluate_decision(request))
        assert resp.recommended_action == RecoveryAction.NO_ACTION
        assert resp.policy_overridden is True
        assert resp.execution is not None
        assert resp.execution.attempted is False
        assert resp.execution.status == "skipped"

    def test_legitimate_recoverable_insufficient_funds_retains_recovery_actions(self):
        """insufficient_funds allows WAIT_AND_RETRY and PAYMENT_LINK, not blocked by fraud safety."""
        context = _build_recovery_context("insufficient_funds")
        policy = resolve_policy_context(context)

        assert "SAFETY_FRAUD_SECURITY_DECLINE" not in [
            r.policy_id for r in policy.applicable_rules
        ]
        assert RecoveryAction.WAIT_AND_RETRY in policy.allowed_actions
        assert RecoveryAction.PAYMENT_LINK in policy.allowed_actions
        assert policy.mandatory_fallback_action == RecoveryAction.WAIT_AND_RETRY

        validator = PolicyValidator()
        val_result = validator.validate_decision(RecoveryAction.WAIT_AND_RETRY, policy)
        assert val_result.is_valid is True
        assert val_result.effective_action == RecoveryAction.WAIT_AND_RETRY
        assert val_result.was_overridden is False

    def test_legitimate_recoverable_expired_card_retains_recovery_actions(self):
        """expired_card allows CHANGE_PAYMENT_METHOD, not blocked by fraud safety."""
        context = _build_recovery_context("expired_card")
        policy = resolve_policy_context(context)

        assert "SAFETY_FRAUD_SECURITY_DECLINE" not in [
            r.policy_id for r in policy.applicable_rules
        ]
        assert RecoveryAction.CHANGE_PAYMENT_METHOD in policy.allowed_actions
        assert RecoveryAction.PAYMENT_LINK in policy.allowed_actions
        assert policy.mandatory_fallback_action == RecoveryAction.CHANGE_PAYMENT_METHOD

        validator = PolicyValidator()
        val_result = validator.validate_decision(
            RecoveryAction.CHANGE_PAYMENT_METHOD, policy
        )
        assert val_result.is_valid is True
        assert val_result.effective_action == RecoveryAction.CHANGE_PAYMENT_METHOD
        assert val_result.was_overridden is False

    def test_decision_engine_rules_for_fraud_and_recoverable_failures(self):
        """Verify baseline DecisionEngine rules for fraud vs recoverable failures."""
        from app.decision_engine import evaluate_recovery_decision

        # Fraud decline -> NO_ACTION
        fraud_ctx = _build_recovery_context("fraud_hard_decline")
        fraud_dec = evaluate_recovery_decision(fraud_ctx)
        assert fraud_dec.recommended_action == RecoveryAction.NO_ACTION
        assert fraud_dec.decision_basis["rule_matched"] == "FraudSecurityDeclineRule"

        # Insufficient funds -> WAIT_AND_RETRY
        funds_ctx = _build_recovery_context("insufficient_funds")
        funds_dec = evaluate_recovery_decision(funds_ctx)
        assert funds_dec.recommended_action == RecoveryAction.WAIT_AND_RETRY

        # Expired card -> CHANGE_PAYMENT_METHOD
        exp_ctx = _build_recovery_context("expired_card")
        exp_dec = evaluate_recovery_decision(exp_ctx)
        assert exp_dec.recommended_action == RecoveryAction.CHANGE_PAYMENT_METHOD
