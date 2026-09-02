"""
Regression test suite for CodeRabbit review findings (Issues 1-9).

Verifies:
1. Intentional NO_ACTION safety decisions and prohibited action filtering during fallback.
2. Mutually exclusive --json and --markdown CLI options in decision and recovery benchmarks.
3. Baseline comparison never compares a candidate against itself as 'latest'.
4. Async worker execution in _run_async_or_sync when calling inside an active event loop.
5. Evaluator failure isolation for malformed pipeline outputs and invalid confidence values.
6. Relative and absolute custom directory handling in decision and recovery persistence.
7. Recovery scenario set integrity (uniqueness, equality, clear ValueError) in uplift comparison.
8. Enforcement of max_cost_per_recovered_dollar quality gate threshold and edge cases.
9. Schema validation for cost_per_action and success_action_rates in RecoveryScenario.
"""

import asyncio
import json
import shutil
import tempfile
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

import pytest
from pydantic import ValidationError

from app.context import (
    CustomerContext,
    CustomerRecoveryContext,
    CustomerRecoveryStatsContext,
    PaymentContext,
    RecoveryOpportunityContext,
)
from app.decision_engine import RecoveryAction, RecoveryDecision
from app.evaluation.agent_evaluation_provider import (
    EvaluationAgentLLMProvider,
    create_evaluation_agent_orchestrator,
)
from app.evaluation.decision_benchmark import run_decision_cli
from app.evaluation.decision_evaluator import (
    AgentRAGPipeline,
    DecisionEvaluator,
    _run_async_or_sync,
)
from app.evaluation.decision_persistence import (
    compare_decision_with_baseline,
    list_decision_reports,
    load_decision_report,
    load_latest_decision_report,
    save_decision_report,
)
from app.evaluation.decision_regression import compare_decision_runs
from app.evaluation.recovery_benchmark import run_recovery_cli
from app.evaluation.recovery_comparison import (
    calculate_cost_per_recovered_dollar,
    compute_recovery_strategy_uplift,
)
from app.evaluation.recovery_persistence import (
    list_recovery_reports,
    load_latest_recovery_report,
    load_recovery_report,
    save_recovery_report,
)
from app.evaluation.recovery_regression import (
    RecoveryQualityThresholds,
    compare_recovery_runs,
    evaluate_recovery_quality_gate,
)
from app.evaluation.recovery_schemas import (
    RecoveryBenchmarkReport,
    RecoveryScenario,
    SimulatedRecoveryOutcome,
)
from app.evaluation.schemas import (
    DecisionBenchmarkReport,
    DecisionGroundTruth,
    EvaluationCase,
)


def _dummy_context(
    failure_reason: str = "insufficient_funds",
    amount: float = 2000.0,
) -> CustomerRecoveryContext:
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
            amount=amount,
            currency="INR",
            payment_method="upi",
            status="failed",
            failure_reason=failure_reason,
            created_at=t_now,
        ),
        current_opportunity=RecoveryOpportunityContext(
            opportunity_id=uuid4(),
            status="failed",
            revenue_at_risk=amount,
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


def _dummy_case(query_id: Any = None) -> EvaluationCase:
    q_id = query_id if isinstance(query_id, UUID) else uuid4()
    return EvaluationCase(
        query_id=q_id,
        context=_dummy_context(),
        expected_case_ids=(),
        description="Dummy test evaluation case",
        decision_ground_truth=DecisionGroundTruth(
            expected_action=RecoveryAction.RETRY_PAYMENT,
            acceptable_actions=(
                RecoveryAction.RETRY_PAYMENT,
                RecoveryAction.WAIT_AND_RETRY,
            ),
            prohibited_actions=(RecoveryAction.NO_ACTION,),
            expected_policy_ids=(),
            rationale="Test rationale",
        ),
    )


def _make_eval_messages(
    failure_reason: str = "insufficient_funds",
    allowed_actions: Sequence[str] = ("retry_payment", "payment_link"),
    prohibited_actions: Sequence[str] = (),
) -> list[dict[str, str]]:
    payload = {
        "policy_envelope": {
            "allowed_actions": list(allowed_actions),
            "prohibited_actions": list(prohibited_actions),
        },
        "current_payment": {
            "failure_reason": failure_reason,
        },
    }
    return [
        {"role": "system", "content": "You are Revora recovery agent evaluator."},
        {"role": "user", "content": json.dumps(payload)},
    ]


def _dummy_decision_report(
    pipeline_name: str,
    exact_match: float = 0.9,
    report_id: str | None = None,
) -> DecisionBenchmarkReport:
    return DecisionBenchmarkReport(
        pipeline_name=pipeline_name,
        num_queries=1,
        aggregate_metrics={
            "exact_match_rate": exact_match,
            "acceptable_match_rate": 1.0,
            "safety_violation_rate": 0.0,
            "fallback_rate": 0.0,
            "mean_latency_ms": 15.0,
        },
        eval_results=(),
        dataset_name="test_dataset",
        report_id=report_id or f"test_{pipeline_name}_{uuid4().hex[:8]}",
    )


def _dummy_recovery_report(
    pipeline_name: str,
    gross_recovered: float = 10000.0,
    cost: float = 50.0,
    report_id: str | None = None,
    scenario_ids: tuple[str, ...] = ("scen_01", "scen_02"),
) -> RecoveryBenchmarkReport:
    n = len(scenario_ids)
    outcomes = tuple(
        SimulatedRecoveryOutcome(
            scenario_id=s_id,
            pipeline_name=pipeline_name,
            predicted_action=RecoveryAction.RETRY_PAYMENT,
            was_recovered=True,
            amount_attempted=gross_recovered / n,
            amount_recovered=gross_recovered / n,
            intervention_cost=cost / n,
            net_recovered=(gross_recovered - cost) / n,
            is_policy_violation=False,
            is_stopping_rule_violation=False,
            is_unnecessary_intervention=False,
        )
        for s_id in scenario_ids
    )
    return RecoveryBenchmarkReport(
        pipeline_name=pipeline_name,
        dataset_name="test_dataset",
        num_scenarios=n,
        total_attempted_revenue=gross_recovered,
        total_recoverable_revenue=gross_recovered,
        total_recovered_revenue=gross_recovered,
        recovery_rate=1.0,
        gross_recovered_amount=gross_recovered,
        total_intervention_cost=cost,
        net_recovered_amount=gross_recovered - cost,
        average_recovered_per_case=gross_recovered / n,
        average_net_per_case=(gross_recovered - cost) / n,
        policy_violation_rate=0.0,
        stopping_rule_violation_rate=0.0,
        unnecessary_intervention_rate=0.0,
        duplicate_action_rate=0.0,
        outcomes=outcomes,
        report_id=report_id or f"rec_{pipeline_name}_{uuid4().hex[:8]}",
    )


# ==============================================================================
# ISSUE 1: Intentional NO_ACTION Safety Decisions & Prohibited Action Fallback
# ==============================================================================


def test_issue1_fraud_scenario_chooses_no_action():
    provider = EvaluationAgentLLMProvider()
    messages = _make_eval_messages(
        failure_reason="suspected_fraud",
        allowed_actions=["retry_payment", "payment_link"],
        prohibited_actions=[],
    )
    rec = asyncio.run(provider.generate(messages))
    assert rec.recommended_action == RecoveryAction.NO_ACTION


def test_issue1_no_action_preserved_even_when_absent_from_allowed():
    provider = EvaluationAgentLLMProvider()
    # allowed_actions does NOT contain "no_action"
    messages = _make_eval_messages(
        failure_reason="stolen_card",
        allowed_actions=["retry_payment", "payment_link"],
        prohibited_actions=[],
    )
    rec = asyncio.run(provider.generate(messages))
    assert rec.recommended_action == RecoveryAction.NO_ACTION


def test_issue1_fallback_skips_prohibited_actions():
    provider = EvaluationAgentLLMProvider()
    # Soft decline would choose RETRY_PAYMENT, but retry is prohibited
    messages = _make_eval_messages(
        failure_reason="insufficient_funds",
        allowed_actions=["retry_payment", "payment_link"],
        prohibited_actions=["retry_payment"],
    )
    rec = asyncio.run(provider.generate(messages))
    assert rec.recommended_action == RecoveryAction.PAYMENT_LINK
    assert rec.recommended_action != RecoveryAction.RETRY_PAYMENT


def test_issue1_fallback_fails_closed_when_no_valid_candidate():
    provider = EvaluationAgentLLMProvider()
    # Both allowed actions are prohibited
    messages = _make_eval_messages(
        failure_reason="insufficient_funds",
        allowed_actions=["retry_payment", "payment_link"],
        prohibited_actions=["retry_payment", "payment_link"],
    )
    rec = asyncio.run(provider.generate(messages))
    assert rec.recommended_action == RecoveryAction.NO_ACTION


def test_issue1_invalid_action_strings_handled_safely():
    provider = EvaluationAgentLLMProvider()
    messages = _make_eval_messages(
        failure_reason="insufficient_funds",
        allowed_actions=["completely_invalid_action", "another_unknown"],
        prohibited_actions=[],
    )
    rec = asyncio.run(provider.generate(messages))
    assert rec.recommended_action == RecoveryAction.NO_ACTION


# ==============================================================================
# ISSUE 2: --json and --markdown Must Be Mutually Exclusive
# ==============================================================================


def test_issue2_decision_cli_mutually_exclusive(capsys):
    case = _dummy_case()
    # 1. JSON alone succeeds
    ret_json = run_decision_cli(
        ["-p", "deterministic_baseline", "--json", "--no-save", "-q"],
        evaluation_cases=[case],
    )
    assert ret_json == 0
    captured = capsys.readouterr()
    assert '"deterministic_baseline"' in captured.out
    assert "# Revora" not in captured.out

    # 2. Markdown alone succeeds
    ret_md = run_decision_cli(
        ["-p", "deterministic_baseline", "--markdown", "--no-save", "-q"],
        evaluation_cases=[case],
    )
    assert ret_md == 0
    captured = capsys.readouterr()
    assert "# " in captured.out

    # 3. Both together rejected by argparse
    with pytest.raises(SystemExit):
        run_decision_cli(
            ["-p", "deterministic_baseline", "--json", "--markdown", "--no-save"],
            evaluation_cases=[case],
        )


def test_issue2_recovery_cli_mutually_exclusive(capsys):
    scen = RecoveryScenario(
        scenario_id="scen_001",
        context=_dummy_context(),
        payment_amount=1500.0,
        failure_category="soft_decline",
        expected_recoverable_amount=1500.0,
    )
    # 1. JSON alone succeeds
    ret_json = run_recovery_cli(
        ["-p", "deterministic_baseline", "--json", "--no-save", "-q"],
        scenarios=[scen],
    )
    assert ret_json == 0
    captured = capsys.readouterr()
    assert '"deterministic_baseline"' in captured.out
    assert "# Revora" not in captured.out

    # 2. Markdown alone succeeds
    ret_md = run_recovery_cli(
        ["-p", "deterministic_baseline", "--markdown", "--no-save", "-q"],
        scenarios=[scen],
    )
    assert ret_md == 0
    captured = capsys.readouterr()
    assert "# " in captured.out

    # 3. Both together rejected by argparse
    with pytest.raises(SystemExit):
        run_recovery_cli(
            ["-p", "deterministic_baseline", "--json", "--markdown", "--no-save"],
            scenarios=[scen],
        )


# ==============================================================================
# ISSUE 3: Never Compare a Benchmark Against Itself as "latest"
# ==============================================================================


def test_issue3_decision_benchmark_never_compares_against_itself():
    with tempfile.TemporaryDirectory() as tmpdir:
        # Step 1: Persist baseline A as latest
        baseline_a = _dummy_decision_report(
            pipeline_name="deterministic_baseline",
            exact_match=0.95,
            report_id="decision_base_a",
        )
        save_decision_report(baseline_a, directory=tmpdir)

        # Step 2: Create candidate B that is intentionally worse
        candidate_b = _dummy_decision_report(
            pipeline_name="deterministic_baseline",
            exact_match=0.50,
            report_id="decision_cand_b",
        )
        # Even if candidate B is saved to disk
        save_decision_report(candidate_b, directory=tmpdir, overwrite=True)

        # Step 3: compare_decision_with_baseline must NOT compare B against B
        resolved_base = load_latest_decision_report(
            pipeline_name="deterministic_baseline",
            directory=tmpdir,
            exclude_report_id=candidate_b.report_id,
        )
        assert resolved_base is not None
        assert resolved_base.report_id == baseline_a.report_id
        assert resolved_base.report_id != candidate_b.report_id

        res = compare_decision_with_baseline(
            current_report=candidate_b,
            baseline_id_or_path=None,  # 'latest'
            directory=tmpdir,
        )
        # It must have compared against A (exact_match 0.95 -> 0.50 is regression)
        assert not res.passed
        assert "exact_match_rate" in res.regressed_metrics
        assert res.metric_deltas["exact_match_rate"] == pytest.approx(-0.45)

        # Direct self comparison must raise ValueError
        with pytest.raises(ValueError, match="cannot be compared against itself"):
            compare_decision_runs(
                current_report=candidate_b,
                baseline_report=candidate_b,
            )


def test_issue3_recovery_benchmark_never_compares_against_itself():
    with tempfile.TemporaryDirectory() as tmpdir:
        # Step 1: Persist baseline A as latest
        baseline_a = _dummy_recovery_report(
            pipeline_name="deterministic_baseline",
            gross_recovered=10000.0,
            cost=20.0,
            report_id="recovery_base_a",
        )
        save_recovery_report(baseline_a, directory=tmpdir)

        # Step 2: Create candidate B that is intentionally worse
        candidate_b = _dummy_recovery_report(
            pipeline_name="deterministic_baseline",
            gross_recovered=2000.0,
            cost=100.0,
            report_id="recovery_cand_b",
        )
        save_recovery_report(candidate_b, directory=tmpdir, overwrite=True)

        # Step 3: Exclude report ID from latest lookup
        resolved_base = load_latest_recovery_report(
            pipeline_name="deterministic_baseline",
            directory=tmpdir,
            exclude_report_id=candidate_b.report_id,
        )
        assert resolved_base is not None
        assert resolved_base.report_id == baseline_a.report_id
        assert resolved_base.report_id != candidate_b.report_id

        # Direct self comparison must raise ValueError
        with pytest.raises(ValueError, match="cannot be compared against itself"):
            compare_recovery_runs(
                current_report=candidate_b,
                baseline_report=candidate_b,
            )


# ==============================================================================
# ISSUE 4: _run_async_or_sync Inside an Existing Event Loop
# ==============================================================================


def test_issue4_sync_invocation_works():
    async def sample_coro():
        await asyncio.sleep(0.01)
        return 42

    res = _run_async_or_sync(sample_coro())
    assert res == 42


def test_issue4_invocation_from_inside_active_event_loop():
    async def inner_coro():
        await asyncio.sleep(0.01)
        return "nested_success"

    async def main_task():
        # Inside active event loop
        return _run_async_or_sync(inner_coro())

    res = asyncio.run(main_task())
    assert res == "nested_success"


def test_issue4_exception_propagates_from_worker_thread():
    async def failing_coro():
        await asyncio.sleep(0.01)
        raise ValueError("simulated_async_failure")

    async def main_task():
        return _run_async_or_sync(failing_coro())

    with pytest.raises(ValueError, match="simulated_async_failure"):
        asyncio.run(main_task())


def test_issue4_coroutine_not_awaited_twice():
    run_count = 0

    async def counting_coro():
        nonlocal run_count
        run_count += 1
        return run_count

    async def main_task():
        return _run_async_or_sync(counting_coro())

    res = asyncio.run(main_task())
    assert res == 1
    assert run_count == 1


def test_issue4_agent_rag_pipeline_evaluates_inside_active_event_loop():
    async def run_pipeline_eval():
        orch = create_evaluation_agent_orchestrator()
        pipeline = AgentRAGPipeline(agent_orchestrator=orch)
        case = _dummy_case()
        evaluator = DecisionEvaluator(evaluation_cases=[case])

        # evaluate() calls _run_async_or_sync inside evaluate_case
        return evaluator.evaluate(pipeline)

    report = asyncio.run(run_pipeline_eval())
    assert report.num_queries == 1
    assert len(report.results) == 1
    assert report.results[0].error is None


# ==============================================================================
# ISSUE 5: Output Parsing Failure Isolation
# ==============================================================================


def test_issue5_unsupported_output_type_isolated():
    class BrokenPipeline:
        name = "broken_pipe"

        def evaluate(self, context, historical_cases=None, policy_context=None):
            return "not_a_decision_object"

    case = _dummy_case()
    evaluator = DecisionEvaluator(evaluation_cases=[case])
    res = evaluator.evaluate_case(case=case, pipeline=BrokenPipeline())
    assert res.error is not None
    assert "Unsupported output type 'str'" in res.error
    assert res.metadata.get("status") == "error"
    assert res.predicted_action == RecoveryAction.NO_ACTION


def test_issue5_invalid_confidence_values_isolated():
    class RawOutputWithBadConfidence:
        recommended_action = RecoveryAction.RETRY_PAYMENT
        confidence = 1.5  # Invalid: > 1.0

    class BadConfidencePipeline:
        name = "bad_conf_pipe"

        def evaluate(self, context, historical_cases=None, policy_context=None):
            return RawOutputWithBadConfidence()

    case = _dummy_case()
    evaluator = DecisionEvaluator(evaluation_cases=[case])
    res = evaluator.evaluate_case(case=case, pipeline=BadConfidencePipeline())
    assert res.error is not None
    assert "Confidence 1.5 is outside valid range [0.0, 1.0]" in res.error
    assert res.metadata.get("status") == "error"


def test_issue5_subsequent_cases_still_execute():
    cases = [
        _dummy_case(uuid4()),
        _dummy_case(uuid4()),
        _dummy_case(uuid4()),
    ]

    call_index = 0

    class FlakyPipeline:
        name = "flaky_pipe"

        def evaluate(self, context, historical_cases=None, policy_context=None):
            nonlocal call_index
            call_index += 1
            if call_index == 2:
                raise RuntimeError("Case 2 crashed")
            return RecoveryDecision(
                recommended_action=RecoveryAction.RETRY_PAYMENT,
                confidence=0.9,
                reason="Regular soft decline",
                decision_basis={},
            )

    evaluator = DecisionEvaluator(evaluation_cases=cases)
    report = evaluator.evaluate(FlakyPipeline())
    assert report.num_queries == 3
    assert len(report.results) == 3
    assert report.results[0].error is None
    assert report.results[1].error is not None
    assert "Case 2 crashed" in report.results[1].error
    assert report.results[2].error is None


# ==============================================================================
# ISSUE 6: Relative Custom Output Directories
# ==============================================================================


def test_issue6_relative_custom_output_directory_decision():
    rel_dir = Path("temp_test_rel_decision")
    try:
        report = _dummy_decision_report("test_pipe", report_id="rel_dec_01")
        saved_path = save_decision_report(
            report, directory=rel_dir, report_id="rel_dec_01"
        )
        assert saved_path.exists()
        assert rel_dir.name in str(saved_path)

        # Load by ID using relative directory
        loaded = load_decision_report("rel_dec_01", directory=rel_dir)
        assert loaded.report_id == "rel_dec_01"

        # Load latest
        latest = load_latest_decision_report("test_pipe", directory=rel_dir)
        assert latest is not None
        assert latest.report_id == "rel_dec_01"

        # List
        files = list_decision_reports(directory=rel_dir)
        assert "rel_dec_01" in files
    finally:
        if rel_dir.exists():
            shutil.rmtree(rel_dir)


def test_issue6_relative_custom_output_directory_recovery():
    rel_dir = Path("temp_test_rel_recovery")
    try:
        report = _dummy_recovery_report("test_pipe", report_id="rel_rec_01")
        saved_path = save_recovery_report(
            report, directory=rel_dir, report_id="rel_rec_01"
        )
        assert saved_path.exists()

        # Load by ID using relative directory
        loaded = load_recovery_report("rel_rec_01", directory=rel_dir)
        assert loaded.report_id == "rel_rec_01"

        # Load latest
        latest = load_latest_recovery_report("test_pipe", directory=rel_dir)
        assert latest is not None
        assert latest.report_id == "rel_rec_01"

        # List
        files = list_recovery_reports(directory=rel_dir)
        assert "rel_rec_01" in files
    finally:
        if rel_dir.exists():
            shutil.rmtree(rel_dir)


# ==============================================================================
# ISSUE 7: Recovery Comparison Requires Identical Unique Scenario Sets
# ==============================================================================


def test_issue7_recovery_comparison_identical_matching_reports():
    cand = _dummy_recovery_report("cand", scenario_ids=("s1", "s2"))
    base = _dummy_recovery_report("base", scenario_ids=("s1", "s2"))
    uplift = compute_recovery_strategy_uplift(cand, base)
    assert uplift.candidate_pipeline == "cand"
    assert uplift.baseline_pipeline == "base"


def test_issue7_candidate_missing_scenario_rejected():
    cand = _dummy_recovery_report("cand", scenario_ids=("s1",))
    base = _dummy_recovery_report("base", scenario_ids=("s1", "s2"))
    with pytest.raises(ValueError, match="Scenario ID mismatch"):
        compute_recovery_strategy_uplift(cand, base)


def test_issue7_candidate_extra_scenario_rejected():
    cand = _dummy_recovery_report("cand", scenario_ids=("s1", "s2", "s3"))
    base = _dummy_recovery_report("base", scenario_ids=("s1", "s2"))
    with pytest.raises(ValueError, match="Scenario ID mismatch"):
        compute_recovery_strategy_uplift(cand, base)


def test_issue7_candidate_duplicate_scenario_id_rejected():
    cand = _dummy_recovery_report("cand", scenario_ids=("s1", "s1"))
    base = _dummy_recovery_report("base", scenario_ids=("s1", "s2"))
    with pytest.raises(ValueError, match="duplicate scenario IDs"):
        compute_recovery_strategy_uplift(cand, base)


def test_issue7_baseline_duplicate_scenario_id_rejected():
    cand = _dummy_recovery_report("cand", scenario_ids=("s1", "s2"))
    base = _dummy_recovery_report("base", scenario_ids=("s1", "s1"))
    with pytest.raises(ValueError, match="duplicate scenario IDs"):
        compute_recovery_strategy_uplift(cand, base)


# ==============================================================================
# ISSUE 8: max_cost_per_recovered_dollar Enforcement & CPRD Edge Cases
# ==============================================================================


def test_issue8_cprd_canonical_calculations():
    # 1. Gross > 0
    assert calculate_cost_per_recovered_dollar(1000.0, 10.0) == 0.01

    # 2. Gross == 0 and cost > 0 -> 1.0 (loss of intervention capital)
    assert calculate_cost_per_recovered_dollar(0.0, 10.0) == 1.0

    # 3. Gross == 0 and cost == 0 -> 0.0
    assert calculate_cost_per_recovered_dollar(0.0, 0.0) == 0.0


def test_issue8_max_cost_per_recovered_dollar_quality_gate():
    report = _dummy_recovery_report(
        "test_pipe",
        gross_recovered=1000.0,
        cost=100.0,  # CPRD = 0.10
    )

    # 1. Below threshold passes
    t_pass = RecoveryQualityThresholds(max_cost_per_recovered_dollar=0.20)
    res_pass = evaluate_recovery_quality_gate(report, thresholds=t_pass)
    assert res_pass.passed

    # 2. Equal threshold passes
    t_eq = RecoveryQualityThresholds(max_cost_per_recovered_dollar=0.10)
    res_eq = evaluate_recovery_quality_gate(report, thresholds=t_eq)
    assert res_eq.passed

    # 3. Above threshold fails
    t_fail = RecoveryQualityThresholds(max_cost_per_recovered_dollar=0.05)
    res_fail = evaluate_recovery_quality_gate(report, thresholds=t_fail)
    assert not res_fail.passed
    assert any("Cost Per Recovered Amount failure" in v for v in res_fail.violations)

    # 4. Zero recovered with positive cost fails finite threshold
    zero_rec_report = _dummy_recovery_report(
        "zero_rec_pipe",
        gross_recovered=0.0,
        cost=50.0,
    )
    res_zero = evaluate_recovery_quality_gate(zero_rec_report, thresholds=t_pass)
    assert not res_zero.passed


# ==============================================================================
# ISSUE 9: Schema Boundary Validation in RecoveryScenario
# ==============================================================================


def test_issue9_cost_per_action_validation():
    # Valid: 0.0 and positive
    scen_ok = RecoveryScenario(
        scenario_id="scen_valid",
        context=_dummy_context(),
        payment_amount=1000.0,
        failure_category="soft_decline",
        expected_recoverable_amount=1000.0,
        cost_per_action={"retry_payment": 0.0, "payment_link": 12.5},
    )
    assert scen_ok.cost_per_action["payment_link"] == 12.5

    # Invalid: negative cost
    with pytest.raises(ValidationError, match="cannot be negative"):
        RecoveryScenario(
            scenario_id="scen_bad",
            context=_dummy_context(),
            payment_amount=1000.0,
            failure_category="soft_decline",
            expected_recoverable_amount=1000.0,
            cost_per_action={"retry_payment": -2.5},
        )


def test_issue9_success_action_rates_validation():
    # Valid boundaries: 0.0, 1.0, 0.5
    scen_ok = RecoveryScenario(
        scenario_id="scen_rates_ok",
        context=_dummy_context(),
        payment_amount=1000.0,
        failure_category="soft_decline",
        expected_recoverable_amount=1000.0,
        success_action_rates={
            "retry_payment": 0.0,
            "wait_and_retry": 1.0,
            "payment_link": 0.5,
        },
    )
    assert scen_ok.success_action_rates["wait_and_retry"] == 1.0

    # Invalid: negative rate
    with pytest.raises(ValidationError, match="must be in range"):
        RecoveryScenario(
            scenario_id="scen_bad_neg",
            context=_dummy_context(),
            payment_amount=1000.0,
            failure_category="soft_decline",
            expected_recoverable_amount=1000.0,
            success_action_rates={"retry_payment": -0.05},
        )

    # Invalid: > 1.0
    with pytest.raises(ValidationError, match="must be in range"):
        RecoveryScenario(
            scenario_id="scen_bad_over",
            context=_dummy_context(),
            payment_amount=1000.0,
            failure_category="soft_decline",
            expected_recoverable_amount=1000.0,
            success_action_rates={"retry_payment": 1.05},
        )
