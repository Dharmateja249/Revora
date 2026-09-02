"""
Revora Deterministic Recovery Simulator.

Simulates financial recovery outcomes, intervention costs, stopping rule enforcement,
and policy constraint checks across synthetic recovery scenarios.
"""

import inspect
import time
from collections import defaultdict
from collections.abc import Sequence
from typing import Any

from app.agent.schemas import AgentDecisionResult
from app.decision_engine import RecoveryAction, RecoveryDecision
from app.evaluation.decision_evaluator import DecisionPipeline
from app.evaluation.recovery_schemas import (
    DEFAULT_ACTION_COSTS,
    RecoveryBenchmarkReport,
    RecoveryScenario,
    SimulatedRecoveryOutcome,
)


class RecoverySimulator:
    """
    Deterministic simulator evaluating recovery outcomes, costs, and compliance.
    """

    def __init__(self, action_costs: dict[RecoveryAction, float] | None = None) -> None:
        self.action_costs = (
            dict(action_costs)
            if action_costs is not None
            else dict(DEFAULT_ACTION_COSTS)
        )

    def simulate_scenario(
        self,
        pipeline: DecisionPipeline,
        scenario: RecoveryScenario,
    ) -> SimulatedRecoveryOutcome:
        """
        Execute pipeline decision and simulate the resulting financial recovery outcome.

        Args:
            pipeline: DecisionPipeline under test.
            scenario: Synthetic RecoveryScenario.

        Returns:
            SimulatedRecoveryOutcome instance.
        """
        start_time = time.perf_counter()
        raw_output: Any = None
        predicted_action: RecoveryAction = RecoveryAction.NO_ACTION
        confidence: float = 0.0
        error_msg: str | None = None
        applied_policy_ids: tuple[str, ...] = ()
        violated_policy_ids: tuple[str, ...] = ()

        try:
            raw_output = pipeline.evaluate(scenario.context)
            if inspect.isawaitable(raw_output):
                import asyncio

                try:
                    asyncio.get_running_loop()
                    import concurrent.futures

                    with concurrent.futures.ThreadPoolExecutor() as pool:
                        raw_output = pool.submit(asyncio.run, raw_output).result()
                except RuntimeError:
                    raw_output = asyncio.run(raw_output)

            if isinstance(raw_output, RecoveryDecision):
                predicted_action = raw_output.recommended_action
                confidence = float(raw_output.confidence)
                basis = getattr(raw_output, "decision_basis", {}) or {}
                applied_policy_ids = tuple(
                    str(x) for x in basis.get("applied_policy_ids", ())
                )
                violated_policy_ids = tuple(
                    str(x) for x in basis.get("violated_policy_ids", ())
                )
            elif isinstance(raw_output, AgentDecisionResult):
                predicted_action = raw_output.recommendation.recommended_action
                confidence = float(raw_output.recommendation.confidence)
                meta = getattr(raw_output, "metadata", {}) or {}
                applied_policy_ids = tuple(
                    str(x) for x in meta.get("applied_policy_ids", ())
                )
                violated_policy_ids = tuple(
                    str(x) for x in meta.get("violated_policy_ids", ())
                )
            elif isinstance(raw_output, RecoveryAction):
                predicted_action = raw_output
                confidence = 1.0
            else:
                error_msg = f"Unknown decision output type: {type(raw_output).__name__}"
        except Exception as exc:  # noqa: BLE001
            error_msg = f"{type(exc).__name__}: {exc}"
            predicted_action = RecoveryAction.NO_ACTION

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        # 1. Cost calculation
        action_cost_map = scenario.cost_per_action or {}
        cost = action_cost_map.get(
            predicted_action.value,
            self.action_costs.get(predicted_action, 0.0),
        )

        # 2. Stopping rules check
        is_stopping_rule_violation = (
            scenario.current_attempt_count >= scenario.max_allowed_attempts
            and predicted_action
            in (RecoveryAction.RETRY_PAYMENT, RecoveryAction.WAIT_AND_RETRY)
        )

        # 3. Policy violation check
        is_prohibited = predicted_action in scenario.prohibited_actions
        is_policy_violation = is_prohibited or bool(
            set(scenario.effective_policy_ids).intersection(violated_policy_ids)
        )

        # 4. Unnecessary intervention check
        is_unnecessary_intervention = (
            not scenario.is_recoverable
            and predicted_action
            in (
                RecoveryAction.RETRY_PAYMENT,
                RecoveryAction.WAIT_AND_RETRY,
                RecoveryAction.PAYMENT_LINK,
                RecoveryAction.CHANGE_PAYMENT_METHOD,
            )
        )

        # 5. Duplicate action check
        last_action = None
        if scenario.context.current_payment_attempts:
            last_action = scenario.context.current_payment_attempts[-1].action
        elif scenario.context.recovery_statistics.previously_failed_actions:
            last_action = (
                scenario.context.recovery_statistics.previously_failed_actions[-1]
            )

        is_duplicate = (
            last_action is not None
            and last_action.lower() == predicted_action.value.lower()
            and predicted_action
            in (
                RecoveryAction.RETRY_PAYMENT,
                RecoveryAction.PAYMENT_LINK,
            )
        )

        # 6. Recovery calculation
        amount_recovered = 0.0
        if (
            not is_stopping_rule_violation
            and not is_policy_violation
            and scenario.is_recoverable
            and predicted_action in scenario.allowed_actions
            and error_msg is None
        ):
            rate = scenario.success_action_rates.get(predicted_action.value, 1.0)
            amount_recovered = round(scenario.payment_amount * rate, 2)

        was_recovered = amount_recovered > 0.0
        net_recovered = round(amount_recovered - cost, 2)

        return SimulatedRecoveryOutcome(
            scenario_id=scenario.scenario_id,
            pipeline_name=pipeline.name,
            predicted_action=predicted_action,
            was_recovered=was_recovered,
            amount_attempted=scenario.payment_amount,
            amount_recovered=amount_recovered,
            intervention_cost=cost,
            net_recovered=net_recovered,
            is_policy_violation=is_policy_violation,
            is_stopping_rule_violation=is_stopping_rule_violation,
            is_unnecessary_intervention=is_unnecessary_intervention,
            is_duplicate_action=is_duplicate,
            violated_policy_ids=violated_policy_ids,
            applied_policy_ids=applied_policy_ids,
            confidence=confidence,
            latency_ms=elapsed_ms,
            error=error_msg,
        )

    def simulate_batch(
        self,
        pipeline: DecisionPipeline,
        scenarios: Sequence[RecoveryScenario],
        dataset_name: str = "synthetic_recovery_100",
    ) -> RecoveryBenchmarkReport:
        """
        Simulate an entire batch of RecoveryScenarios and compute aggregate financial metrics.

        Args:
            pipeline: DecisionPipeline to evaluate.
            scenarios: Sequence of RecoveryScenarios.
            dataset_name: Dataset name identifier.

        Returns:
            Immutable RecoveryBenchmarkReport.
        """
        if not scenarios:
            raise ValueError("scenarios cannot be empty.")

        outcomes: list[SimulatedRecoveryOutcome] = [
            self.simulate_scenario(pipeline, sc) for sc in scenarios
        ]

        total_attempted = sum(s.payment_amount for s in scenarios)
        total_recoverable = sum(s.expected_recoverable_amount for s in scenarios)
        total_recovered = sum(o.amount_recovered for o in outcomes)
        total_cost = sum(o.intervention_cost for o in outcomes)
        net_recovered = total_recovered - total_cost

        rec_rate = (
            float(total_recovered / total_recoverable)
            if total_recoverable > 0.0
            else 0.0
        )
        n = len(scenarios)

        # Failure category breakdown
        cat_stats: dict[str, dict[str, float]] = defaultdict(
            lambda: {
                "count": 0.0,
                "attempted": 0.0,
                "recoverable": 0.0,
                "recovered": 0.0,
                "cost": 0.0,
                "net": 0.0,
                "recovery_rate": 0.0,
            }
        )

        for sc, out in zip(scenarios, outcomes, strict=True):
            cat = sc.failure_category
            entry = cat_stats[cat]
            entry["count"] += 1.0
            entry["attempted"] += sc.payment_amount
            entry["recoverable"] += sc.expected_recoverable_amount
            entry["recovered"] += out.amount_recovered
            entry["cost"] += out.intervention_cost
            entry["net"] += out.net_recovered

        for cat, entry in cat_stats.items():
            if entry["recoverable"] > 0.0:
                entry["recovery_rate"] = round(
                    entry["recovered"] / entry["recoverable"], 4
                )
            else:
                entry["recovery_rate"] = 0.0
            for k in ("attempted", "recoverable", "recovered", "cost", "net"):
                entry[k] = round(entry[k], 2)

        return RecoveryBenchmarkReport(
            pipeline_name=pipeline.name,
            dataset_name=dataset_name,
            num_scenarios=n,
            total_attempted_revenue=round(total_attempted, 2),
            total_recoverable_revenue=round(total_recoverable, 2),
            total_recovered_revenue=round(total_recovered, 2),
            recovery_rate=round(rec_rate, 4),
            gross_recovered_amount=round(total_recovered, 2),
            total_intervention_cost=round(total_cost, 2),
            net_recovered_amount=round(net_recovered, 2),
            average_recovered_per_case=round(total_recovered / n, 2),
            average_net_per_case=round(net_recovered / n, 2),
            policy_violation_rate=round(
                sum(1 for o in outcomes if o.is_policy_violation) / n, 4
            ),
            stopping_rule_violation_rate=round(
                sum(1 for o in outcomes if o.is_stopping_rule_violation) / n, 4
            ),
            unnecessary_intervention_rate=round(
                sum(1 for o in outcomes if o.is_unnecessary_intervention) / n, 4
            ),
            duplicate_action_rate=round(
                sum(1 for o in outcomes if o.is_duplicate_action) / n, 4
            ),
            category_breakdown=dict(cat_stats),
            outcomes=tuple(outcomes),
        )
