"""
Revora Payment Provider Policy Schemas and Data Contracts.

Defines immutable Pydantic v2 domain models for structured payment policies,
policy contexts, and deterministic policy validation results.
"""

from enum import Enum
import types
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple, Union

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


class PolicyType(str, Enum):
    """
    Classification of recovery policy constraints.
    """

    SAFETY = "safety"
    PROVIDER_CONSTRAINT = "provider_constraint"
    BUSINESS_RULE = "business_rule"


class PolicyRule(BaseModel):
    """
    Immutable representation of an individual recovery policy rule.
    """

    model_config = ConfigDict(
        frozen=True,
        from_attributes=True,
        validate_default=True,
        arbitrary_types_allowed=False,
    )

    policy_id: str
    provider: str
    version: str
    policy_type: PolicyType
    description: str
    applicable_failure_reasons: Tuple[str, ...] = Field(default_factory=tuple)
    applicable_payment_methods: Tuple[str, ...] = Field(default_factory=tuple)
    allowed_actions: Tuple[RecoveryAction, ...] = Field(default_factory=tuple)
    prohibited_actions: Tuple[RecoveryAction, ...] = Field(default_factory=tuple)
    mandatory_fallback: Optional[RecoveryAction] = None
    priority: int = Field(default=100, ge=0)
    metadata: Mapping[str, Any] = Field(default_factory=dict)

    @field_validator("policy_id", "provider", "version", "description", mode="before")
    @classmethod
    def _validate_non_empty_strings(cls, v: Any) -> str:
        if not isinstance(v, str) or not v.strip():
            raise ValueError("String identifier and description fields must be non-empty strings.")
        return v.strip()

    @field_validator("applicable_failure_reasons", "applicable_payment_methods", mode="before")
    @classmethod
    def _normalize_string_tuples(cls, v: Any) -> Tuple[str, ...]:
        if v is None:
            return ()
        if not isinstance(v, (list, tuple, set)):
            raise TypeError(f"Expected sequence of strings, got {type(v).__name__}")
        normalized = []
        for idx, item in enumerate(v):
            if not isinstance(item, str) or not item.strip():
                raise ValueError(f"Item at index {idx} must be a non-empty string, got: {item!r}")
            normalized.append(item.strip().lower())
        return tuple(normalized)

    @field_validator("allowed_actions", "prohibited_actions", mode="before")
    @classmethod
    def _normalize_action_tuples(cls, v: Any) -> Tuple[RecoveryAction, ...]:
        if v is None:
            return ()
        if not isinstance(v, (list, tuple, set)):
            raise TypeError(f"Expected sequence of RecoveryAction items, got {type(v).__name__}")
        actions = []
        for idx, item in enumerate(v):
            if isinstance(item, RecoveryAction):
                actions.append(item)
            elif isinstance(item, str):
                try:
                    actions.append(RecoveryAction(item.strip().lower()))
                except ValueError as exc:
                    raise ValueError(f"Invalid RecoveryAction string at index {idx}: {item!r}") from exc
            else:
                raise TypeError(f"Item at index {idx} must be a RecoveryAction or valid enum string.")
        return tuple(actions)

    @field_validator("mandatory_fallback", mode="before")
    @classmethod
    def _normalize_mandatory_fallback(cls, v: Any) -> Optional[RecoveryAction]:
        if v is None:
            return None
        if isinstance(v, RecoveryAction):
            return v
        if isinstance(v, str):
            return RecoveryAction(v.strip().lower())
        raise TypeError(f"mandatory_fallback must be a RecoveryAction or string, got {type(v).__name__}")

    @field_validator("metadata", mode="before")
    @classmethod
    def _normalize_metadata(cls, v: Any) -> Any:
        return v if v is not None else {}

    @field_validator("metadata", mode="after")
    @classmethod
    def _freeze_metadata(cls, v: Any) -> Mapping[str, Any]:
        return _freeze_nested(v) if v is not None else types.MappingProxyType({})

    @field_serializer("metadata")
    def _serialize_metadata(self, v: Mapping[str, Any], _info: Any) -> Dict[str, Any]:
        return _unfreeze_for_serialization(v)

    @model_validator(mode="after")
    def _validate_internal_consistency(self) -> "PolicyRule":
        # Check intersection between allowed and prohibited actions
        overlap = set(self.allowed_actions).intersection(set(self.prohibited_actions))
        if overlap:
            overlap_names = [a.value for a in overlap]
            raise ValueError(
                f"Policy '{self.policy_id}' is internally inconsistent: actions {overlap_names} "
                f"cannot be both allowed and prohibited in the same rule."
            )
        if self.mandatory_fallback is not None and self.mandatory_fallback in self.prohibited_actions:
            raise ValueError(
                f"Policy '{self.policy_id}' has mandatory_fallback '{self.mandatory_fallback.value}' "
                f"which is listed in prohibited_actions."
            )
        return self


class RecoveryPolicyContext(BaseModel):
    """
    Immutable resolved policy envelope governing the current recovery decision.
    """

    model_config = ConfigDict(
        frozen=True,
        from_attributes=True,
        validate_default=True,
        arbitrary_types_allowed=False,
    )

    provider: str
    policy_version: str
    applicable_rules: Tuple[PolicyRule, ...] = Field(default_factory=tuple)
    allowed_actions: Tuple[RecoveryAction, ...] = Field(default_factory=tuple)
    prohibited_actions: Tuple[RecoveryAction, ...] = Field(default_factory=tuple)
    mandatory_fallback_action: Optional[RecoveryAction] = None
    metadata: Mapping[str, Any] = Field(default_factory=dict)

    @field_validator("provider", "policy_version", mode="before")
    @classmethod
    def _validate_non_empty_strings(cls, v: Any) -> str:
        if not isinstance(v, str) or not v.strip():
            raise ValueError("provider and policy_version must be non-empty strings.")
        return v.strip()

    @field_validator("applicable_rules", mode="before")
    @classmethod
    def _normalize_rules(cls, v: Any) -> Tuple[PolicyRule, ...]:
        if v is None:
            return ()
        if not isinstance(v, (list, tuple, set)):
            raise TypeError(f"applicable_rules must be a sequence of PolicyRule, got {type(v).__name__}")
        rules = []
        for idx, item in enumerate(v):
            if isinstance(item, PolicyRule):
                rules.append(item)
            elif isinstance(item, dict):
                rules.append(PolicyRule(**item))
            else:
                raise TypeError(f"Item at index {idx} must be a PolicyRule or dict, got {type(item).__name__}")
        return tuple(rules)

    @field_validator("allowed_actions", "prohibited_actions", mode="before")
    @classmethod
    def _normalize_actions(cls, v: Any) -> Tuple[RecoveryAction, ...]:
        if v is None:
            return ()
        if not isinstance(v, (list, tuple, set)):
            raise TypeError(f"Expected sequence of RecoveryAction items, got {type(v).__name__}")
        actions = []
        for idx, item in enumerate(v):
            if isinstance(item, RecoveryAction):
                actions.append(item)
            elif isinstance(item, str):
                actions.append(RecoveryAction(item.strip().lower()))
            else:
                raise TypeError(f"Item at index {idx} must be a RecoveryAction or string.")
        return tuple(actions)

    @field_validator("metadata", mode="before")
    @classmethod
    def _normalize_metadata(cls, v: Any) -> Any:
        return v if v is not None else {}

    @field_validator("metadata", mode="after")
    @classmethod
    def _freeze_metadata(cls, v: Any) -> Mapping[str, Any]:
        return _freeze_nested(v) if v is not None else types.MappingProxyType({})

    @field_serializer("metadata")
    def _serialize_metadata(self, v: Mapping[str, Any], _info: Any) -> Dict[str, Any]:
        return _unfreeze_for_serialization(v)


class PolicyValidationResult(BaseModel):
    """
    Immutable evaluation result produced by PolicyValidator after inspecting a candidate action.
    """

    model_config = ConfigDict(
        frozen=True,
        from_attributes=True,
        validate_default=True,
        arbitrary_types_allowed=False,
    )

    is_valid: bool
    candidate_action: RecoveryAction
    effective_action: RecoveryAction
    was_overridden: bool
    violated_policy_ids: Tuple[str, ...] = Field(default_factory=tuple)
    applied_policy_ids: Tuple[str, ...] = Field(default_factory=tuple)
    explanation: str
    metadata: Mapping[str, Any] = Field(default_factory=dict)

    @field_validator("violated_policy_ids", "applied_policy_ids", mode="before")
    @classmethod
    def _normalize_string_tuples(cls, v: Any) -> Tuple[str, ...]:
        if v is None:
            return ()
        if not isinstance(v, (list, tuple, set)):
            raise TypeError(f"Expected sequence of strings, got {type(v).__name__}")
        return tuple(str(x).strip() for x in v if str(x).strip())

    @field_validator("explanation", mode="before")
    @classmethod
    def _validate_explanation(cls, v: Any) -> str:
        return str(v).strip() if v is not None else ""

    @field_validator("metadata", mode="before")
    @classmethod
    def _normalize_metadata(cls, v: Any) -> Any:
        return v if v is not None else {}

    @field_validator("metadata", mode="after")
    @classmethod
    def _freeze_metadata(cls, v: Any) -> Mapping[str, Any]:
        return _freeze_nested(v) if v is not None else types.MappingProxyType({})

    @field_serializer("metadata")
    def _serialize_metadata(self, v: Mapping[str, Any], _info: Any) -> Dict[str, Any]:
        return _unfreeze_for_serialization(v)
