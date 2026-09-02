"""
Revora Adaptive Recovery Agent Schemas.

Defines immutable Pydantic v2 data contracts for LLM prompt context construction,
structured LLM decision generation, and agent execution results.
"""

import types
from collections.abc import Mapping
from datetime import datetime, timezone
from typing import Any

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_serializer,
    field_validator,
    model_validator,
)

from app.decision_engine import RecoveryAction


def _freeze_nested(value: Any) -> Any:
    """Recursively freeze dictionaries to MappingProxyType and collections to tuples."""
    if isinstance(value, (dict, types.MappingProxyType)):
        return types.MappingProxyType({k: _freeze_nested(v) for k, v in value.items()})
    if isinstance(value, (list, set, tuple)):
        return tuple(_freeze_nested(v) for v in value)
    return value


def _unfreeze_for_serialization(val: Any) -> Any:
    """Recursively unfreeze MappingProxyType and tuples to dicts and lists for clean serialization."""
    if isinstance(val, (dict, types.MappingProxyType)):
        return {k: _unfreeze_for_serialization(v) for k, v in val.items()}
    if isinstance(val, (list, tuple, set)):
        return [_unfreeze_for_serialization(v) for v in val]
    return val


def utc_now() -> datetime:
    """Return current timezone-aware UTC datetime."""
    return datetime.now(timezone.utc)


class LLMRecoveryRecommendation(BaseModel):
    """
    Structured recommendation contract produced by the LLM reasoning layer.
    """

    model_config = ConfigDict(
        frozen=True,
        from_attributes=True,
        validate_default=True,
        arbitrary_types_allowed=False,
    )

    recommended_action: RecoveryAction = Field(
        ...,
        description="Candidate recovery action chosen from the supported RecoveryAction enum.",
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confidence score bounded between 0.0 and 1.0.",
    )
    reasoning: str = Field(
        ...,
        description="Structured explanation articulating why the recovery action was chosen.",
    )
    key_factors: tuple[str, ...] = Field(
        default_factory=tuple,
        description="Key supporting factors or evidence signals contributing to the decision.",
    )
    referenced_case_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        description="Historical case or payment IDs referenced by the LLM as supporting evidence.",
    )

    @field_validator("reasoning", mode="before")
    @classmethod
    def _validate_non_empty_reasoning(cls, v: Any) -> str:
        if not isinstance(v, str) or not v.strip():
            raise ValueError("reasoning is required and must be a non-empty string.")
        return v.strip()

    @field_validator("key_factors", "referenced_case_ids", mode="before")
    @classmethod
    def _normalize_string_tuples(cls, v: Any) -> tuple[str, ...]:
        if v is None:
            return ()
        if isinstance(v, str):
            return (v.strip(),) if v.strip() else ()
        if not isinstance(v, (list, tuple, set)):
            raise TypeError(f"Expected sequence of strings, got {type(v).__name__}")
        normalized = []
        for idx, item in enumerate(v):
            if item is None:
                continue
            str_item = str(item).strip()
            if str_item:
                normalized.append(str_item)
        return tuple(normalized)

    @field_serializer("key_factors", "referenced_case_ids")
    def _serialize_tuples(self, v: tuple[str, ...], _info: Any) -> list[str]:
        return list(v)


class AgentDecisionPromptContext(BaseModel):
    """
    Immutable representation of bounded, sanitized context supplied to the LLM.
    Strictly excludes customer PII, raw authentication secrets, and unrestricted ORM objects.
    """

    model_config = ConfigDict(
        frozen=True,
        from_attributes=True,
        validate_default=True,
        arbitrary_types_allowed=False,
    )

    current_payment: Mapping[str, Any] = Field(
        ...,
        description="Sanitized payment details (e.g., amount, currency, payment_method, failure_reason).",
    )
    customer_profile: Mapping[str, Any] = Field(
        ...,
        description="Anonymized customer recovery metrics and success statistics.",
    )
    attempt_budget: Mapping[str, Any] = Field(
        default_factory=dict,
        description="Attempt budget tracking current attempt, maximum allowed attempts, and remaining attempts.",
    )
    recovery_attempt_history: tuple[Mapping[str, Any], ...] = Field(
        default_factory=tuple,
        description="Chronological sequence of prior attempts on the active payment.",
    )
    historical_cases: tuple[Mapping[str, Any], ...] = Field(
        default_factory=tuple,
        description="Top-K retrieved similar historical recovery cases without PII.",
    )
    allowed_actions: tuple[str, ...] = Field(
        default_factory=tuple,
        description="List of action names permitted under active policy rules.",
    )
    prohibited_actions: tuple[str, ...] = Field(
        default_factory=tuple,
        description="List of action names strictly prohibited by active policy rules.",
    )
    mandatory_fallback: str | None = Field(
        default=None,
        description="Mandatory fallback action string if applicable.",
    )
    policy_constraints: tuple[str, ...] = Field(
        default_factory=tuple,
        description="Human-readable policy descriptions governing this recovery scenario.",
    )

    @field_validator(
        "current_payment", "customer_profile", "attempt_budget", mode="before"
    )
    @classmethod
    def _validate_dict_fields(cls, v: Any) -> Any:
        if v is None:
            return {}
        if not isinstance(v, (dict, types.MappingProxyType)):
            raise TypeError(f"Expected mapping, got {type(v).__name__}")
        return v

    @field_validator(
        "current_payment", "customer_profile", "attempt_budget", mode="after"
    )
    @classmethod
    def _freeze_dict_fields(cls, v: Any) -> Mapping[str, Any]:
        return _freeze_nested(v) if v is not None else types.MappingProxyType({})

    @field_validator("recovery_attempt_history", "historical_cases", mode="before")
    @classmethod
    def _normalize_list_of_mappings(cls, v: Any) -> Any:
        if v is None:
            return ()
        if not isinstance(v, (list, tuple, set)):
            raise TypeError(f"Expected sequence of mappings, got {type(v).__name__}")
        return v

    @field_validator("recovery_attempt_history", "historical_cases", mode="after")
    @classmethod
    def _freeze_list_of_mappings(cls, v: Any) -> tuple[Mapping[str, Any], ...]:
        return _freeze_nested(v) if v is not None else ()

    @field_validator(
        "allowed_actions", "prohibited_actions", "policy_constraints", mode="before"
    )
    @classmethod
    def _normalize_string_tuples(cls, v: Any) -> tuple[str, ...]:
        if v is None:
            return ()
        if not isinstance(v, (list, tuple, set)):
            raise TypeError(f"Expected sequence of strings, got {type(v).__name__}")
        return tuple(str(x).strip() for x in v if str(x).strip())

    @field_validator("mandatory_fallback", mode="before")
    @classmethod
    def _normalize_mandatory_fallback(cls, v: Any) -> str | None:
        if v is None:
            return None
        clean = str(v).strip()
        return clean if clean else None

    @field_serializer("current_payment", "customer_profile", "attempt_budget")
    def _serialize_mappings(self, v: Mapping[str, Any], _info: Any) -> dict[str, Any]:
        return _unfreeze_for_serialization(v)

    @field_serializer("recovery_attempt_history", "historical_cases")
    def _serialize_mapping_tuples(
        self, v: tuple[Mapping[str, Any], ...], _info: Any
    ) -> list[dict[str, Any]]:
        return _unfreeze_for_serialization(v)

    @field_serializer("allowed_actions", "prohibited_actions", "policy_constraints")
    def _serialize_string_tuples(self, v: tuple[str, ...], _info: Any) -> list[str]:
        return list(v)


class AgentDecisionResult(BaseModel):
    """
    Immutable outcome contract representing the result of the Adaptive Recovery Agent evaluation.
    Captures whether the LLM was utilized, provider metadata, and fallback diagnostics.
    """

    model_config = ConfigDict(
        frozen=True,
        from_attributes=True,
        validate_default=True,
        arbitrary_types_allowed=False,
    )

    recommendation: LLMRecoveryRecommendation = Field(
        ...,
        description="Structured candidate recommendation generated by the agent or fallback engine.",
    )
    agent_used: bool = Field(
        default=True,
        description="True if recommendation was generated by an LLM; False if produced by deterministic fallback.",
    )
    provider: str | None = Field(
        default=None,
        description="LLM provider name (e.g., 'openai', 'anthropic', 'mock_llm').",
    )
    model_name: str | None = Field(
        default=None,
        description="Specific model identifier used for decision generation.",
    )
    is_fallback: bool = Field(
        default=False,
        description="True if deterministic fallback was triggered due to LLM timeout, error, or unparseable response.",
    )
    fallback_reason: str | None = Field(
        default=None,
        description="Diagnostic explanation for why fallback occurred, if applicable.",
    )
    latency_ms: float | None = Field(
        default=None,
        ge=0.0,
        description="Execution latency in milliseconds.",
    )
    evaluated_at: datetime = Field(
        default_factory=utc_now,
        description="UTC timestamp when the decision was generated.",
    )
    metadata: Mapping[str, Any] = Field(
        default_factory=dict,
        description="Additional execution metadata and token metrics.",
    )

    @field_validator("fallback_reason", "provider", "model_name", mode="before")
    @classmethod
    def _normalize_optional_strings(cls, v: Any) -> str | None:
        if v is None:
            return None
        clean = str(v).strip()
        return clean if clean else None

    @field_validator("evaluated_at", mode="after")
    @classmethod
    def _ensure_utc_evaluated_at(cls, v: datetime) -> datetime:
        if v.tzinfo is None or v.tzinfo.utcoffset(v) is None:
            raise ValueError("evaluated_at must be a timezone-aware datetime.")
        return v.astimezone(timezone.utc)

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

    @model_validator(mode="after")
    def _validate_fallback_state_consistency(self) -> "AgentDecisionResult":
        if self.agent_used and self.is_fallback:
            raise ValueError(
                "Inconsistent state: agent_used cannot be True when is_fallback is True."
            )
        if self.is_fallback and not self.fallback_reason:
            raise ValueError(
                "Inconsistent state: fallback_reason is required and cannot be empty when is_fallback is True."
            )
        if not self.is_fallback and not self.agent_used:
            raise ValueError(
                "Inconsistent state: agent_used must be True when is_fallback is False."
            )
        return self
