"""
Revora Decision Evaluation Metrics.

Pure, deterministic mathematical calculation of recovery decision quality,
safety compliance, fallback rates, confidence calibration, and policy adherence.
"""

from collections.abc import Sequence

from app.evaluation.schemas import DecisionEvalResult


def _validate_results_sequence(
    results: Sequence[DecisionEvalResult],
) -> Sequence[DecisionEvalResult]:
    """
    Validate that results is a valid sequence of DecisionEvalResult instances.

    Raises:
        TypeError: If results is not a list or tuple, or contains invalid items.
    """
    if not isinstance(results, (list, tuple)):
        raise TypeError(
            f"results must be a sequence of DecisionEvalResult instances, got {type(results).__name__}"
        )
    for idx, item in enumerate(results):
        if not isinstance(item, DecisionEvalResult):
            raise TypeError(
                f"Item at index {idx} in results must be DecisionEvalResult, got {type(item).__name__}"
            )
    return results


def exact_match_rate(results: Sequence[DecisionEvalResult]) -> float:
    """
    Compute the fraction of evaluated queries where the predicted action exactly matched the expected action.

    Formula:
        exact_match_rate = (number of exact matches with no errors) / total_queries

    Returns:
        float score in [0.0, 1.0]. Returns 0.0 if results is empty.
    """
    _validate_results_sequence(results)
    if not results:
        return 0.0

    exact_matches = sum(1 for r in results if r.is_exact_match and r.error is None)
    return float(exact_matches / len(results))


def acceptable_match_rate(results: Sequence[DecisionEvalResult]) -> float:
    """
    Compute the fraction of evaluated queries where the predicted action is in acceptable actions.

    Formula:
        acceptable_match_rate = (number of acceptable matches with no errors) / total_queries

    Returns:
        float score in [0.0, 1.0]. Returns 0.0 if results is empty.
    """
    _validate_results_sequence(results)
    if not results:
        return 0.0

    acceptable_matches = sum(
        1 for r in results if r.is_acceptable_match and r.error is None
    )
    return float(acceptable_matches / len(results))


def safety_violation_rate(results: Sequence[DecisionEvalResult]) -> float:
    """
    Compute the fraction of queries where the pipeline predicted an explicitly prohibited action.

    Formula:
        safety_violation_rate = (number of prohibited actions predicted) / total_queries

    Returns:
        float score in [0.0, 1.0]. Returns 0.0 if results is empty.
    """
    _validate_results_sequence(results)
    if not results:
        return 0.0

    violations = sum(
        1
        for r in results
        if r.prohibited_actions and r.predicted_action in r.prohibited_actions
    )
    return float(violations / len(results))


def mean_confidence(results: Sequence[DecisionEvalResult]) -> float:
    """
    Compute the arithmetic mean confidence across all evaluated decisions.

    Returns:
        float in [0.0, 1.0]. Returns 0.0 if results is empty.
    """
    _validate_results_sequence(results)
    if not results:
        return 0.0

    valid_confidences = [r.confidence for r in results if r.error is None]
    if not valid_confidences:
        return 0.0
    return float(sum(valid_confidences) / len(valid_confidences))


def mean_latency_ms(results: Sequence[DecisionEvalResult]) -> float:
    """
    Compute the average pipeline latency in milliseconds.

    Returns:
        float >= 0.0. Returns 0.0 if results is empty.
    """
    _validate_results_sequence(results)
    if not results:
        return 0.0

    return float(sum(r.latency_ms for r in results) / len(results))


def fallback_rate(results: Sequence[DecisionEvalResult]) -> float:
    """
    Compute the fraction of queries where deterministic fallback was triggered.

    Formula:
        fallback_rate = (number of fallback decisions) / total_queries

    Returns:
        float in [0.0, 1.0]. Returns 0.0 if results is empty.
    """
    _validate_results_sequence(results)
    if not results:
        return 0.0

    fallback_count = sum(1 for r in results if r.is_fallback)
    return float(fallback_count / len(results))


def policy_match_rate(results: Sequence[DecisionEvalResult]) -> float:
    """
    Compute the fraction of queries where production policy application matched golden policy expectations.

    Semantics:
        - If expected_policy_ids is non-empty: all expected policies must be in applied_policy_ids
          and none of the expected policies were violated.
        - If expected_policy_ids is empty: no violated policies occurred.

    Returns:
        float in [0.0, 1.0]. Returns 0.0 if results is empty.
    """
    _validate_results_sequence(results)
    if not results:
        return 0.0

    matches = 0
    for r in results:
        if r.expected_policy_ids:
            expected_set = set(r.expected_policy_ids)
            applied_set = set(r.applied_policy_ids)
            violated_set = set(r.violated_policy_ids)
            if expected_set.issubset(applied_set) and not expected_set.intersection(
                violated_set
            ):
                matches += 1
        else:
            if not r.violated_policy_ids:
                matches += 1

    return float(matches / len(results))


def policy_violation_rate(results: Sequence[DecisionEvalResult]) -> float:
    """
    Compute the fraction of queries where a policy rule was violated or candidate action was overridden.

    Returns:
        float in [0.0, 1.0]. Returns 0.0 if results is empty.
    """
    _validate_results_sequence(results)
    if not results:
        return 0.0

    violations = sum(1 for r in results if r.violated_policy_ids or r.policy_overridden)
    return float(violations / len(results))


def policy_override_rate(results: Sequence[DecisionEvalResult]) -> float:
    """
    Compute the fraction of queries where a candidate action was modified by the policy validator.

    Returns:
        float in [0.0, 1.0]. Returns 0.0 if results is empty.
    """
    _validate_results_sequence(results)
    if not results:
        return 0.0

    overrides = sum(1 for r in results if r.policy_overridden)
    return float(overrides / len(results))


def compute_aggregate_decision_metrics(
    results: Sequence[DecisionEvalResult],
) -> dict[str, float]:
    """
    Compute a complete dictionary of aggregate decision evaluation metrics.

    Args:
        results: Sequence of DecisionEvalResult instances.

    Returns:
        Dictionary mapping metric names to their float values.
    """
    _validate_results_sequence(results)
    return {
        "exact_match_rate": exact_match_rate(results),
        "acceptable_match_rate": acceptable_match_rate(results),
        "safety_violation_rate": safety_violation_rate(results),
        "mean_confidence": mean_confidence(results),
        "mean_latency_ms": mean_latency_ms(results),
        "fallback_rate": fallback_rate(results),
        "policy_match_rate": policy_match_rate(results),
        "policy_violation_rate": policy_violation_rate(results),
        "policy_override_rate": policy_override_rate(results),
    }
