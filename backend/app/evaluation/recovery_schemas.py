"""
Revora Recovery Outcome & Synthetic Batch Simulation Schemas.

Defines the domain contracts for synthetic recovery scenarios, simulated intervention
outcomes, and aggregate recovery benchmark financial reports.
"""

from collections.abc import Mapping
from datetime import datetime
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.context import CustomerRecoveryContext, utc_now
from app.decision_engine import RecoveryAction

DEFAULT_ACTION_COSTS: dict[RecoveryAction, float] = {
    RecoveryAction.RETRY_PAYMENT: 2.50,
    RecoveryAction.WAIT_AND_RETRY: 3.50,
    RecoveryAction.PAYMENT_LINK: 8.00,
    RecoveryAction.CHANGE_PAYMENT_METHOD: 15.00,
    RecoveryAction.NO_ACTION: 0.00,
}


class RecoveryScenario(BaseModel):
    """
    Represents an independent synthetic recovery case for financial outcome simulation.
    """

    model_config = ConfigDict(
        frozen=True,
        from_attributes=True,
        validate_default=True,
        arbitrary_types_allowed=True,
    )

    scenario_id: str
    context: CustomerRecoveryContext
    payment_amount: float = Field(ge=0.0)
    failure_category: str
    is_recoverable: bool = True
    expected_recoverable_amount: float = Field(ge=0.0)
    max_allowed_attempts: int = Field(default=4, ge=1)
    current_attempt_count: int = Field(default=1, ge=1)
    cost_per_action: Mapping[str, float] = Field(default_factory=dict)
    allowed_actions: tuple[RecoveryAction, ...] = Field(default_factory=tuple)
    prohibited_actions: tuple[RecoveryAction, ...] = Field(default_factory=tuple)
    effective_policy_ids: tuple[str, ...] = Field(default_factory=tuple)
    success_action_rates: Mapping[str, float] = Field(default_factory=dict)
    description: str = ""

    @field_validator("cost_per_action")
    @classmethod
    def validate_cost_per_action(cls, v: Mapping[str, float]) -> Mapping[str, float]:
        for action, cost in v.items():
            if cost < 0.0:
                raise ValueError(
                    f"Action cost for '{action}' cannot be negative, got {cost}"
                )
        return v

    @field_validator("success_action_rates")
    @classmethod
    def validate_success_action_rates(
        cls, v: Mapping[str, float]
    ) -> Mapping[str, float]:
        for action, rate in v.items():
            if rate < 0.0 or rate > 1.0:
                raise ValueError(
                    f"Success rate for '{action}' must be in range [0.0, 1.0], got {rate}"
                )
        return v


class SimulatedRecoveryOutcome(BaseModel):
    """
    Simulation outcome of applying a pipeline's decision to a RecoveryScenario.
    """

    model_config = ConfigDict(frozen=True, from_attributes=True, validate_default=True)

    scenario_id: str
    pipeline_name: str
    predicted_action: RecoveryAction
    was_recovered: bool
    amount_attempted: float = Field(ge=0.0)
    amount_recovered: float = Field(ge=0.0)
    intervention_cost: float = Field(ge=0.0)
    net_recovered: float
    is_policy_violation: bool = False
    is_stopping_rule_violation: bool = False
    is_unnecessary_intervention: bool = False
    is_duplicate_action: bool = False
    violated_policy_ids: tuple[str, ...] = Field(default_factory=tuple)
    applied_policy_ids: tuple[str, ...] = Field(default_factory=tuple)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    latency_ms: float = Field(default=0.0, ge=0.0)
    error: str | None = None


class RecoveryBenchmarkReport(BaseModel):
    """
    Aggregate recovery benchmark report evaluating financial efficiency, costs,
    and policy compliance across a synthetic batch.
    """

    model_config = ConfigDict(frozen=True, from_attributes=True, validate_default=True)

    pipeline_name: str
    dataset_name: str
    num_scenarios: int = Field(ge=0)
    total_attempted_revenue: float = Field(ge=0.0)
    total_recoverable_revenue: float = Field(ge=0.0)
    total_recovered_revenue: float = Field(ge=0.0)
    recovery_rate: float = Field(ge=0.0, le=1.0)
    gross_recovered_amount: float = Field(ge=0.0)
    total_intervention_cost: float = Field(ge=0.0)
    net_recovered_amount: float
    average_recovered_per_case: float
    average_net_per_case: float
    policy_violation_rate: float = Field(ge=0.0, le=1.0)
    stopping_rule_violation_rate: float = Field(ge=0.0, le=1.0)
    unnecessary_intervention_rate: float = Field(ge=0.0, le=1.0)
    duplicate_action_rate: float = Field(ge=0.0, le=1.0)
    category_breakdown: Mapping[str, Mapping[str, float]] = Field(default_factory=dict)
    outcomes: tuple[SimulatedRecoveryOutcome, ...] = Field(default_factory=tuple)
    evaluated_at: datetime = Field(default_factory=utc_now)
    evaluation_version: str = "1.0"
    report_id: str = Field(default_factory=lambda: f"recovery_{uuid4().hex[:12]}")
    metadata: Mapping[str, Any] = Field(default_factory=dict)
