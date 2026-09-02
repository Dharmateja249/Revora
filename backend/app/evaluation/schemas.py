"""
Revora Retrieval Evaluation Schemas and Data Contracts.

Defines immutable Pydantic v2 domain contracts for evaluating retrieval quality,
including EvaluationCase, GroundTruthJudgment, RetrievalEvalResult, and
RetrieverBenchmarkReport.
"""

import types
from collections.abc import Mapping
from datetime import datetime, timezone
from typing import Any
from uuid import UUID, uuid4

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_serializer,
    field_validator,
    model_validator,
)

from app.context import CustomerRecoveryContext
from app.decision_engine import RecoveryAction


def utc_now() -> datetime:
    """Return current UTC timestamp with timezone awareness."""
    return datetime.now(timezone.utc)


def _freeze_nested(value: Any) -> Any:
    """Recursively freeze dictionaries to MappingProxyType and sequences to tuples."""
    if isinstance(value, (dict, types.MappingProxyType)):
        return types.MappingProxyType({k: _freeze_nested(v) for k, v in value.items()})
    if isinstance(value, (list, set, tuple)):
        return tuple(_freeze_nested(v) for v in value)
    return value


def _unfreeze_for_serialization(val: Any) -> Any:
    """Recursively convert MappingProxyType and tuples to dicts and lists for serialization."""
    if isinstance(val, (dict, types.MappingProxyType)):
        return {k: _unfreeze_for_serialization(v) for k, v in val.items()}
    if isinstance(val, (list, tuple, set)):
        return [_unfreeze_for_serialization(v) for v in val]
    return val


# Relevance grading scale definitions
RELEVANCE_GRADE_DESCRIPTIONS: Mapping[int, str] = types.MappingProxyType(
    {
        3: "Highly relevant (same customer, same root failure, compatible rail, successfully recovered)",
        2: "Moderately relevant (same customer, related failure category or payment method)",
        1: "Marginally relevant (same customer, general historical transaction)",
        0: "Irrelevant (unrelated failure, incompatible rail, or uninformative historical case)",
    }
)


class GroundTruthJudgment(BaseModel):
    """
    Immutable representation of a relevance judgment for a single historical payment
    with respect to a specific evaluation query.
    """

    model_config = ConfigDict(
        frozen=True,
        from_attributes=True,
        validate_default=True,
        arbitrary_types_allowed=False,
    )

    payment_id: UUID
    relevance_grade: int = Field(ge=0, le=3)
    rationale: str | None = None

    @field_validator("relevance_grade", mode="before")
    @classmethod
    def _validate_grade(cls, v: Any) -> int:
        if isinstance(v, bool) or not isinstance(v, int):
            raise ValueError(  # noqa: TRY004
                f"relevance_grade must be an integer between 0 and 3, got: {v!r}"
            )
        if v < 0 or v > 3:
            raise ValueError(f"relevance_grade must be in [0, 3], got: {v}")
        return v


class DecisionGroundTruth(BaseModel):
    """
    Immutable oracle describing the expected business decision and
    policy envelope for an evaluation scenario.
    """

    model_config = ConfigDict(
        frozen=True,
        from_attributes=True,
        validate_default=True,
        arbitrary_types_allowed=False,
    )

    expected_action: RecoveryAction
    acceptable_actions: tuple[RecoveryAction, ...] = Field(default_factory=tuple)
    expected_policy_ids: tuple[str, ...] = Field(default_factory=tuple)
    prohibited_actions: tuple[RecoveryAction, ...] = Field(default_factory=tuple)
    rationale: str | None = None
    expected_reasoning_factors: tuple[str, ...] = Field(default_factory=tuple)

    @field_validator("expected_action", mode="before")
    @classmethod
    def _validate_expected_action(cls, v: Any) -> RecoveryAction:
        if isinstance(v, RecoveryAction):
            return v
        if isinstance(v, str):
            try:
                return RecoveryAction(v.strip().lower())
            except ValueError as exc:
                raise ValueError(
                    f"Invalid RecoveryAction: {v!r}. Expected one of {[a.value for a in RecoveryAction]}"
                ) from exc
        raise TypeError(
            f"expected_action must be RecoveryAction or str, got {type(v).__name__}"
        )

    @field_validator("acceptable_actions", "prohibited_actions", mode="before")
    @classmethod
    def _normalize_actions_tuple(cls, v: Any) -> tuple[RecoveryAction, ...]:
        if v is None:
            return ()
        if isinstance(v, (RecoveryAction, str)):
            v = (v,)
        if not isinstance(v, (list, tuple, set)):
            raise TypeError(
                f"Action collection must be a sequence, got {type(v).__name__}"
            )

        actions: list[RecoveryAction] = []
        seen: set[RecoveryAction] = set()
        for idx, item in enumerate(v):
            action: RecoveryAction
            if isinstance(item, RecoveryAction):
                action = item
            elif isinstance(item, str):
                try:
                    action = RecoveryAction(item.strip().lower())
                except ValueError as exc:
                    raise ValueError(
                        f"Invalid RecoveryAction at index {idx}: {item!r}"
                    ) from exc
            else:
                raise TypeError(
                    f"Item at index {idx} must be RecoveryAction or valid action string, got {type(item).__name__}"
                )
            if action not in seen:
                seen.add(action)
                actions.append(action)
        return tuple(actions)

    @field_validator("expected_policy_ids", "expected_reasoning_factors", mode="before")
    @classmethod
    def _normalize_string_tuples(cls, v: Any) -> tuple[str, ...]:
        if v is None:
            return ()
        if isinstance(v, str):
            v = (v,)
        if not isinstance(v, (list, tuple, set)):
            raise TypeError(f"Expected sequence of strings, got {type(v).__name__}")
        normalized: list[str] = []
        seen: set[str] = set()
        for idx, item in enumerate(v):
            if not isinstance(item, str):
                raise TypeError(
                    f"Item at index {idx} must be str, got {type(item).__name__}"
                )
            cleaned = item.strip()
            if cleaned and cleaned not in seen:
                seen.add(cleaned)
                normalized.append(cleaned)
        return tuple(normalized)

    @field_validator("rationale", mode="before")
    @classmethod
    def _normalize_rationale(cls, v: Any) -> str | None:
        if v is None:
            return None
        if not isinstance(v, str):
            raise TypeError(f"rationale must be a string, got {type(v).__name__}")
        clean = v.strip()
        return clean if clean else None

    @model_validator(mode="after")
    def _validate_decision_ground_truth_invariants(self) -> "DecisionGroundTruth":
        # Invariant 1: Ensure expected_action is present in acceptable_actions
        acceptable = list(self.acceptable_actions)
        if self.expected_action not in acceptable:
            acceptable.insert(0, self.expected_action)
        acceptable_tuple = tuple(acceptable)

        # Invariant 3: expected_action cannot be in prohibited_actions
        if self.expected_action in self.prohibited_actions:
            raise ValueError(
                f"Contradictory ground truth: expected_action '{self.expected_action.value}' "
                f"cannot be in prohibited_actions {[a.value for a in self.prohibited_actions]}."
            )

        # Invariant 4: acceptable_actions and prohibited_actions cannot overlap
        overlap = set(acceptable_tuple).intersection(set(self.prohibited_actions))
        if overlap:
            raise ValueError(
                f"Contradictory ground truth: acceptable_actions and prohibited_actions overlap "
                f"on actions: {[a.value for a in sorted(overlap, key=lambda x: x.value)]}."
            )

        if acceptable_tuple != self.acceptable_actions:
            object.__setattr__(self, "acceptable_actions", acceptable_tuple)

        return self

    @field_serializer("acceptable_actions", "prohibited_actions")
    def _serialize_action_tuples(
        self, v: tuple[RecoveryAction, ...], _info: Any
    ) -> list[str]:
        return [a.value for a in v]

    @field_serializer("expected_policy_ids", "expected_reasoning_factors")
    def _serialize_string_tuples(self, v: tuple[str, ...], _info: Any) -> list[str]:
        return list(v)


class EvaluationCase(BaseModel):
    """
    Immutable representation of a single retrieval and decision evaluation query scenario.

    Encapsulates the query CustomerRecoveryContext alongside its ground-truth
    relevance judgments, decision ground truth oracle, and scenario metadata.
    """

    model_config = ConfigDict(
        frozen=True,
        from_attributes=True,
        validate_default=True,
        arbitrary_types_allowed=True,
    )

    query_id: UUID
    context: CustomerRecoveryContext
    ground_truth: tuple[GroundTruthJudgment, ...] = Field(default_factory=tuple)
    decision_ground_truth: DecisionGroundTruth | None = None
    description: str | None = None
    created_at: datetime = Field(default_factory=utc_now)
    metadata: Mapping[str, Any] = Field(default_factory=dict)

    @field_validator("ground_truth", mode="before")
    @classmethod
    def _normalize_and_validate_ground_truth(
        cls, v: Any
    ) -> tuple[GroundTruthJudgment, ...]:
        if v is None:
            return ()
        if not isinstance(v, (list, tuple, set)):
            raise TypeError(
                f"ground_truth must be a sequence of GroundTruthJudgment instances, got {type(v).__name__}"
            )

        judgments: list[GroundTruthJudgment] = []
        seen_payment_ids: set[UUID] = set()

        for idx, item in enumerate(v):
            if isinstance(item, GroundTruthJudgment):
                judgment = item
            elif isinstance(item, dict):
                judgment = GroundTruthJudgment(**item)
            else:
                raise TypeError(
                    f"Item at index {idx} in ground_truth must be GroundTruthJudgment or dict, got {type(item).__name__}"
                )

            if judgment.payment_id in seen_payment_ids:
                raise ValueError(
                    f"Duplicate payment_id '{judgment.payment_id}' found in ground_truth judgments."
                )
            seen_payment_ids.add(judgment.payment_id)
            judgments.append(judgment)

        return tuple(judgments)

    @field_validator("decision_ground_truth", mode="before")
    @classmethod
    def _normalize_and_validate_decision_ground_truth(cls, v: Any) -> Any:
        if v is None:
            return None
        if isinstance(v, DecisionGroundTruth):
            return v
        if isinstance(v, dict):
            return DecisionGroundTruth(**v)
        raise TypeError(
            f"decision_ground_truth must be DecisionGroundTruth or dict, got {type(v).__name__}"
        )

    @field_validator("metadata", mode="before")
    @classmethod
    def _normalize_metadata(cls, v: Any) -> Any:
        return v if v is not None else {}

    @field_validator("metadata", mode="after")
    @classmethod
    def _freeze_metadata(cls, v: Any) -> Mapping[str, Any]:
        return _freeze_nested(v) if v is not None else types.MappingProxyType({})

    @field_serializer("metadata")
    def _serialize_metadata(self, v: Mapping[str, Any], _info: Any) -> dict[str, Any]:
        return _unfreeze_for_serialization(v)


class RetrievalEvalResult(BaseModel):
    """
    Immutable evaluation result for a single (query, retriever, K) evaluation tuple.

    Holds the retrieved payment IDs and computed ranking metrics.
    """

    model_config = ConfigDict(
        frozen=True,
        from_attributes=True,
        validate_default=True,
        arbitrary_types_allowed=True,
    )

    query_id: UUID
    retriever_name: str
    k: int = Field(gt=0)
    retrieved_payment_ids: tuple[UUID, ...] = Field(default_factory=tuple)
    precision_at_k: float = Field(ge=0.0, le=1.0)
    recall_at_k: float = Field(ge=0.0, le=1.0)
    reciprocal_rank: float = Field(ge=0.0, le=1.0)
    ndcg_at_k: float = Field(ge=0.0, le=1.0)
    latency_ms: float = Field(default=0.0, ge=0.0)
    metadata: Mapping[str, Any] = Field(default_factory=dict)

    @field_validator("retriever_name", mode="before")
    @classmethod
    def _validate_retriever_name(cls, v: Any) -> str:
        if not isinstance(v, str) or not v.strip():
            raise ValueError(f"retriever_name must be a non-empty string, got: {v!r}")
        return v.strip()

    @field_validator("k", mode="before")
    @classmethod
    def _validate_k(cls, v: Any) -> int:
        if isinstance(v, bool) or not isinstance(v, int):
            raise ValueError(f"k must be an integer, got: {v!r}")  # noqa: TRY004
        if v <= 0:
            raise ValueError(f"k must be a positive integer (> 0), got: {v}")
        return v

    @field_validator("retrieved_payment_ids", mode="before")
    @classmethod
    def _normalize_retrieved_payment_ids(cls, v: Any) -> tuple[UUID, ...]:
        if v is None:
            return ()
        if not isinstance(v, (list, tuple, set)):
            raise TypeError(
                f"retrieved_payment_ids must be a sequence of UUIDs, got {type(v).__name__}"
            )
        validated: list[UUID] = []
        for idx, item in enumerate(v):
            if isinstance(item, UUID):
                validated.append(item)
            elif isinstance(item, str):
                try:
                    validated.append(UUID(item.strip()))
                except Exception as exc:
                    raise ValueError(
                        f"Invalid UUID string at index {idx}: {item!r}"
                    ) from exc
            else:
                raise TypeError(
                    f"Item at index {idx} must be UUID or valid UUID string, got {type(item).__name__}"
                )
        return tuple(validated)

    @field_validator("metadata", mode="before")
    @classmethod
    def _normalize_metadata(cls, v: Any) -> Any:
        return v if v is not None else {}

    @field_validator("metadata", mode="after")
    @classmethod
    def _freeze_metadata(cls, v: Any) -> Mapping[str, Any]:
        return _freeze_nested(v) if v is not None else types.MappingProxyType({})

    @field_serializer("metadata")
    def _serialize_metadata(self, v: Mapping[str, Any], _info: Any) -> dict[str, Any]:
        return _unfreeze_for_serialization(v)


class DecisionEvalResult(BaseModel):
    """
    Immutable result of evaluating a recovery decision pipeline
    against independent decision ground truth.
    """

    model_config = ConfigDict(
        frozen=True,
        from_attributes=True,
        validate_default=True,
        arbitrary_types_allowed=False,
    )

    query_id: UUID
    pipeline_name: str
    predicted_action: RecoveryAction
    expected_action: RecoveryAction
    acceptable_actions: tuple[RecoveryAction, ...] = Field(default_factory=tuple)
    prohibited_actions: tuple[RecoveryAction, ...] = Field(default_factory=tuple)
    expected_policy_ids: tuple[str, ...] = Field(default_factory=tuple)
    is_exact_match: bool
    is_acceptable_match: bool
    confidence: float = Field(ge=0.0, le=1.0)
    policy_overridden: bool = False
    is_fallback: bool = False
    fallback_reason: str | None = None
    applied_policy_ids: tuple[str, ...] = Field(default_factory=tuple)
    violated_policy_ids: tuple[str, ...] = Field(default_factory=tuple)
    referenced_case_ids: tuple[str, ...] = Field(default_factory=tuple)
    key_factors: tuple[str, ...] = Field(default_factory=tuple)
    latency_ms: float = Field(default=0.0, ge=0.0)
    error: str | None = None
    metadata: Mapping[str, Any] = Field(default_factory=dict)

    @field_validator("pipeline_name", mode="before")
    @classmethod
    def _validate_pipeline_name(cls, v: Any) -> str:
        if not isinstance(v, str) or not v.strip():
            raise ValueError(f"pipeline_name must be a non-empty string, got: {v!r}")
        return v.strip()

    @field_validator("predicted_action", "expected_action", mode="before")
    @classmethod
    def _validate_action(cls, v: Any) -> RecoveryAction:
        if isinstance(v, RecoveryAction):
            return v
        if isinstance(v, str):
            try:
                return RecoveryAction(v.strip().lower())
            except ValueError as exc:
                raise ValueError(
                    f"Invalid RecoveryAction: {v!r}. Expected one of {[a.value for a in RecoveryAction]}"
                ) from exc
        raise TypeError(f"Action must be RecoveryAction or str, got {type(v).__name__}")

    @field_validator("acceptable_actions", "prohibited_actions", mode="before")
    @classmethod
    def _normalize_action_tuple(cls, v: Any) -> tuple[RecoveryAction, ...]:
        if v is None:
            return ()
        if isinstance(v, (RecoveryAction, str)):
            v = (v,)
        if not isinstance(v, (list, tuple, set)):
            raise TypeError(f"Expected sequence of actions, got {type(v).__name__}")
        actions: list[RecoveryAction] = []
        for idx, item in enumerate(v):
            if isinstance(item, RecoveryAction):
                actions.append(item)
            elif isinstance(item, str):
                try:
                    actions.append(RecoveryAction(item.strip().lower()))
                except ValueError as exc:
                    raise ValueError(
                        f"Invalid RecoveryAction at index {idx}: {item!r}"
                    ) from exc
            else:
                raise TypeError(
                    f"Item at index {idx} must be RecoveryAction or str, got {type(item).__name__}"
                )
        return tuple(actions)

    @field_validator(
        "expected_policy_ids",
        "applied_policy_ids",
        "violated_policy_ids",
        "referenced_case_ids",
        "key_factors",
        mode="before",
    )
    @classmethod
    def _normalize_string_tuples(cls, v: Any) -> tuple[str, ...]:
        if v is None:
            return ()
        if isinstance(v, str):
            v = (v,)
        if not isinstance(v, (list, tuple, set)):
            raise TypeError(f"Expected sequence of strings, got {type(v).__name__}")
        return tuple(str(x).strip() for x in v if str(x).strip())

    @field_validator("fallback_reason", "error", mode="before")
    @classmethod
    def _normalize_optional_strings(cls, v: Any) -> str | None:
        if v is None:
            return None
        if not isinstance(v, str):
            raise TypeError(f"Expected string, got {type(v).__name__}")
        clean = v.strip()
        return clean if clean else None

    @field_validator("metadata", mode="before")
    @classmethod
    def _normalize_metadata(cls, v: Any) -> Any:
        return v if v is not None else {}

    @field_validator("metadata", mode="after")
    @classmethod
    def _freeze_metadata(cls, v: Any) -> Mapping[str, Any]:
        return _freeze_nested(v) if v is not None else types.MappingProxyType({})

    @field_serializer("acceptable_actions", "prohibited_actions")
    def _serialize_action_tuples(
        self, v: tuple[RecoveryAction, ...], _info: Any
    ) -> list[str]:
        return [a.value for a in v]

    @field_serializer(
        "expected_policy_ids",
        "applied_policy_ids",
        "violated_policy_ids",
        "referenced_case_ids",
        "key_factors",
    )
    def _serialize_string_tuples(self, v: tuple[str, ...], _info: Any) -> list[str]:
        return list(v)

    @field_serializer("metadata")
    def _serialize_metadata(self, v: Mapping[str, Any], _info: Any) -> dict[str, Any]:
        return _unfreeze_for_serialization(v)


class DecisionBenchmarkReport(BaseModel):
    """
    Immutable aggregate benchmark report capturing the overall decision quality,
    safety compliance, policy adherence, and performance of a decision pipeline.
    """

    model_config = ConfigDict(
        frozen=True,
        from_attributes=True,
        validate_default=True,
        arbitrary_types_allowed=False,
    )

    pipeline_name: str
    dataset_name: str
    num_queries: int = Field(ge=0)
    results: tuple[DecisionEvalResult, ...] = Field(default_factory=tuple)
    aggregate_metrics: Mapping[str, float] = Field(default_factory=dict)
    evaluated_at: datetime = Field(default_factory=utc_now)
    evaluation_version: str = "1.0"
    report_id: str = Field(default_factory=lambda: f"decision_{uuid4().hex[:12]}")
    metadata: Mapping[str, Any] = Field(default_factory=dict)

    @field_validator("pipeline_name", mode="before")
    @classmethod
    def _validate_pipeline_name(cls, v: Any) -> str:
        if not isinstance(v, str) or not v.strip():
            raise ValueError(f"pipeline_name must be a non-empty string, got: {v!r}")
        return v.strip()

    @field_validator("dataset_name", "evaluation_version", mode="before")
    @classmethod
    def _validate_non_empty_strings(cls, v: Any) -> str:
        if not isinstance(v, str) or not v.strip():
            raise ValueError(f"String fields must be non-empty strings, got: {v!r}")
        return v.strip()

    @field_validator("num_queries", mode="before")
    @classmethod
    def _validate_num_queries(cls, v: Any) -> int:
        if isinstance(v, bool) or not isinstance(v, int):
            raise ValueError(f"num_queries must be an integer, got: {v!r}")  # noqa: TRY004
        if v < 0:
            raise ValueError(f"num_queries cannot be negative, got: {v}")
        return v

    @field_validator("results", mode="before")
    @classmethod
    def _normalize_results(cls, v: Any) -> tuple[DecisionEvalResult, ...]:
        if v is None:
            return ()
        if not isinstance(v, (list, tuple, set)):
            raise TypeError(
                f"results must be a sequence of DecisionEvalResult, got {type(v).__name__}"
            )
        res_list: list[DecisionEvalResult] = []
        for idx, item in enumerate(v):
            if isinstance(item, DecisionEvalResult):
                res_list.append(item)
            elif isinstance(item, dict):
                res_list.append(DecisionEvalResult(**item))
            else:
                raise TypeError(
                    f"Item at index {idx} in results must be DecisionEvalResult, got {type(item).__name__}"
                )
        return tuple(res_list)

    @field_validator("aggregate_metrics", "metadata", mode="before")
    @classmethod
    def _normalize_mappings(cls, v: Any) -> Any:
        return v if v is not None else {}

    @field_validator("aggregate_metrics", "metadata", mode="after")
    @classmethod
    def _freeze_mappings(cls, v: Any) -> Mapping[str, Any]:
        return _freeze_nested(v) if v is not None else types.MappingProxyType({})

    @field_serializer("aggregate_metrics", "metadata")
    def _serialize_mappings(self, v: Mapping[str, Any], _info: Any) -> dict[str, Any]:
        return _unfreeze_for_serialization(v)


class RetrieverBenchmarkReport(BaseModel):
    """
    Immutable aggregate benchmark report capturing the overall performance
    of a single retriever across an entire evaluation dataset.
    """

    model_config = ConfigDict(
        frozen=True,
        from_attributes=True,
        validate_default=True,
        arbitrary_types_allowed=True,
    )

    retriever_name: str
    dataset_name: str
    num_queries: int = Field(ge=0)
    k_values: tuple[int, ...] = Field(default=(1, 3, 5, 10))
    results: tuple[RetrievalEvalResult, ...] = Field(default_factory=tuple)
    aggregate_metrics: Mapping[str, float] = Field(default_factory=dict)
    evaluated_at: datetime = Field(default_factory=utc_now)
    metadata: Mapping[str, Any] = Field(default_factory=dict)

    @field_validator("retriever_name", mode="before")
    @classmethod
    def _validate_retriever_name(cls, v: Any) -> str:
        if not isinstance(v, str) or not v.strip():
            raise ValueError(f"retriever_name must be a non-empty string, got: {v!r}")
        return v.strip()

    @field_validator("dataset_name", mode="before")
    @classmethod
    def _validate_dataset_name(cls, v: Any) -> str:
        if not isinstance(v, str) or not v.strip():
            raise ValueError(f"dataset_name must be a non-empty string, got: {v!r}")
        return v.strip()

    @field_validator("num_queries", mode="before")
    @classmethod
    def _validate_num_queries(cls, v: Any) -> int:
        if isinstance(v, bool) or not isinstance(v, int):
            raise ValueError(f"num_queries must be an integer, got: {v!r}")  # noqa: TRY004
        if v < 0:
            raise ValueError(f"num_queries cannot be negative, got: {v}")
        return v

    @field_validator("k_values", mode="before")
    @classmethod
    def _normalize_k_values(cls, v: Any) -> tuple[int, ...]:
        if v is None:
            return (1, 3, 5, 10)
        if not isinstance(v, (list, tuple, set)):
            raise TypeError(
                f"k_values must be a sequence of positive integers, got {type(v).__name__}"
            )
        k_list: list[int] = []
        for idx, item in enumerate(v):
            if isinstance(item, bool) or not isinstance(item, int) or item <= 0:
                raise ValueError(
                    f"Item at index {idx} in k_values must be a positive integer (> 0), got: {item!r}"
                )
            k_list.append(item)
        return tuple(k_list)

    @field_validator("results", mode="before")
    @classmethod
    def _normalize_results(cls, v: Any) -> tuple[RetrievalEvalResult, ...]:
        if v is None:
            return ()
        if not isinstance(v, (list, tuple, set)):
            raise TypeError(
                f"results must be a sequence of RetrievalEvalResult, got {type(v).__name__}"
            )
        res_list: list[RetrievalEvalResult] = []
        for idx, item in enumerate(v):
            if isinstance(item, RetrievalEvalResult):
                res_list.append(item)
            elif isinstance(item, dict):
                res_list.append(RetrievalEvalResult(**item))
            else:
                raise TypeError(
                    f"Item at index {idx} in results must be RetrievalEvalResult, got {type(item).__name__}"
                )
        return tuple(res_list)

    @field_validator("aggregate_metrics", "metadata", mode="before")
    @classmethod
    def _normalize_mappings(cls, v: Any) -> Any:
        return v if v is not None else {}

    @field_validator("aggregate_metrics", "metadata", mode="after")
    @classmethod
    def _freeze_mappings(cls, v: Any) -> Mapping[str, Any]:
        return _freeze_nested(v) if v is not None else types.MappingProxyType({})

    @field_serializer("aggregate_metrics", "metadata")
    def _serialize_mappings(self, v: Mapping[str, Any], _info: Any) -> dict[str, Any]:
        return _unfreeze_for_serialization(v)


class MetricSnapshot(BaseModel):
    """Immutable representation of an individual named metric value."""

    model_config = ConfigDict(frozen=True, from_attributes=True, validate_default=True)

    metric_name: str
    value: float


class RetrieverEvaluationSummary(BaseModel):
    """
    Immutable evaluation summary for a single retriever across all queries and depths.
    """

    model_config = ConfigDict(
        frozen=True,
        from_attributes=True,
        validate_default=True,
        arbitrary_types_allowed=False,
    )

    retriever_name: str
    query_count: int = Field(ge=0)
    mrr: float = Field(ge=0.0, le=1.0)
    mean_latency_ms: float = Field(ge=0.0)
    precision_at_k: Mapping[int, float] = Field(default_factory=dict)
    recall_at_k: Mapping[int, float] = Field(default_factory=dict)
    ndcg_at_k: Mapping[int, float] = Field(default_factory=dict)
    metadata: Mapping[str, Any] = Field(default_factory=dict)

    @field_validator("retriever_name", mode="before")
    @classmethod
    def _validate_retriever_name(cls, v: Any) -> str:
        if not isinstance(v, str) or not v.strip():
            raise ValueError("retriever_name must be a non-empty string.")
        return v.strip()

    @field_validator("precision_at_k", "recall_at_k", "ndcg_at_k", mode="before")
    @classmethod
    def _validate_k_mappings(cls, v: Any) -> Any:
        if v is None:
            return {}
        if not isinstance(v, (dict, types.MappingProxyType)):
            raise TypeError(
                f"Per-K metric mapping must be a dict/Mapping, got {type(v).__name__}"
            )
        validated: dict[int, float] = {}
        for k, val in v.items():
            k_int = (
                int(k) if isinstance(k, (int, str)) and not isinstance(k, bool) else -1
            )
            if k_int <= 0:
                raise ValueError(f"K must be a positive integer, got: {k!r}")
            if (
                isinstance(val, bool)
                or not isinstance(val, (int, float))
                or not (0.0 <= float(val) <= 1.0)
            ):
                raise ValueError(
                    f"Metric value for K={k} must be in [0.0, 1.0], got {val!r}"
                )
            validated[k_int] = float(val)
        return validated

    @field_validator(
        "precision_at_k", "recall_at_k", "ndcg_at_k", "metadata", mode="after"
    )
    @classmethod
    def _freeze_mappings(cls, v: Any) -> Mapping[Any, Any]:
        return _freeze_nested(v) if v is not None else types.MappingProxyType({})

    @field_serializer("precision_at_k", "recall_at_k", "ndcg_at_k", "metadata")
    def _serialize_mappings(self, v: Mapping[Any, Any], _info: Any) -> dict[Any, Any]:
        return _unfreeze_for_serialization(v)


class EvaluationReport(BaseModel):
    """
    Immutable top-level report capturing a full benchmark execution over all retrievers.
    """

    model_config = ConfigDict(
        frozen=True,
        from_attributes=True,
        validate_default=True,
        arbitrary_types_allowed=False,
    )

    report_id: str
    created_at: datetime = Field(default_factory=utc_now)
    dataset_name: str
    dataset_version: str = "v1"
    query_count: int = Field(ge=0)
    configured_k_values: tuple[int, ...] = Field(default=(1, 3, 5, 10))
    retriever_summaries: Mapping[str, RetrieverEvaluationSummary] = Field(
        default_factory=dict
    )
    evaluation_version: str = "1.0"
    metadata: Mapping[str, Any] = Field(default_factory=dict)

    @field_validator(
        "report_id",
        "dataset_name",
        "dataset_version",
        "evaluation_version",
        mode="before",
    )
    @classmethod
    def _validate_non_empty_strings(cls, v: Any) -> str:
        if not isinstance(v, str) or not v.strip():
            raise ValueError("String identifier fields must be non-empty strings.")
        return v.strip()

    @field_validator("configured_k_values", mode="before")
    @classmethod
    def _normalize_k_values(cls, v: Any) -> tuple[int, ...]:
        if v is None:
            return (1, 3, 5, 10)
        if not isinstance(v, (list, tuple, set)):
            raise TypeError(
                f"configured_k_values must be a sequence, got {type(v).__name__}"
            )
        k_list: list[int] = []
        for item in v:
            if isinstance(item, bool) or not isinstance(item, int) or item <= 0:
                raise ValueError(
                    f"K values must be positive integers (> 0), got: {item!r}"
                )
            k_list.append(item)
        return tuple(k_list)

    @field_validator("retriever_summaries", mode="before")
    @classmethod
    def _normalize_summaries(cls, v: Any) -> Any:
        if v is None:
            return {}
        if not isinstance(v, (dict, types.MappingProxyType)):
            raise TypeError(
                f"retriever_summaries must be a dict/Mapping, got {type(v).__name__}"
            )
        summaries: dict[str, RetrieverEvaluationSummary] = {}
        for name, summary in v.items():
            if isinstance(summary, RetrieverEvaluationSummary):
                summaries[name] = summary
            elif isinstance(summary, dict):
                summaries[name] = RetrieverEvaluationSummary(**summary)
            else:
                raise TypeError(
                    f"Summary for '{name}' must be RetrieverEvaluationSummary or dict, got {type(summary).__name__}"
                )
        return summaries

    @field_validator("retriever_summaries", "metadata", mode="after")
    @classmethod
    def _freeze_mappings(cls, v: Any) -> Mapping[str, Any]:
        return _freeze_nested(v) if v is not None else types.MappingProxyType({})

    @field_serializer("retriever_summaries", "metadata")
    def _serialize_mappings(self, v: Mapping[str, Any], _info: Any) -> dict[str, Any]:
        return _unfreeze_for_serialization(v)


class RegressionCheck(BaseModel):
    """
    Immutable record of a single metric regression check.
    """

    model_config = ConfigDict(
        frozen=True,
        from_attributes=True,
        validate_default=True,
        arbitrary_types_allowed=False,
    )

    metric_name: str
    retriever_name: str
    baseline_value: float
    candidate_value: float
    delta: float
    relative_change: float
    threshold: float
    status: str = Field(pattern="^(PASS|WARN|FAIL)$")
    message: str = ""


class RegressionReport(BaseModel):
    """
    Immutable comprehensive regression report comparing two EvaluationReport runs.
    """

    model_config = ConfigDict(
        frozen=True,
        from_attributes=True,
        validate_default=True,
        arbitrary_types_allowed=False,
    )

    baseline_report_id: str
    candidate_report_id: str
    checks: tuple[RegressionCheck, ...] = Field(default_factory=tuple)
    overall_status: str = Field(pattern="^(PASS|WARN|FAIL)$")
    created_at: datetime = Field(default_factory=utc_now)
    metadata: Mapping[str, Any] = Field(default_factory=dict)

    @field_validator("checks", mode="before")
    @classmethod
    def _normalize_checks(cls, v: Any) -> tuple[RegressionCheck, ...]:
        if v is None:
            return ()
        return tuple(v) if isinstance(v, (list, tuple, set)) else (v,)

    @field_validator("metadata", mode="before")
    @classmethod
    def _normalize_metadata(cls, v: Any) -> Any:
        return v if v is not None else {}

    @field_validator("metadata", mode="after")
    @classmethod
    def _freeze_metadata(cls, v: Any) -> Mapping[str, Any]:
        return _freeze_nested(v) if v is not None else types.MappingProxyType({})


class EvaluationRegressionError(Exception):
    """Raised when an evaluation run violates quality regression assertions in CI."""
