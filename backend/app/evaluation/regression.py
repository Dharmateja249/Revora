"""
Revora Retrieval Evaluation Regression Detection.

Provides quality and latency regression analysis, threshold enforcement,
severity classification (INFO, WARNING, CRITICAL), per-query diagnostics,
and CI assertion helpers.
"""

from enum import Enum
import types
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple, Union
from uuid import UUID

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_serializer,
    field_validator,
)

from app.evaluation.schemas import (
    EvaluationReport,
    EvaluationRegressionError,
    RegressionCheck,
    RegressionReport,
    RetrievalEvalResult,
    RetrieverBenchmarkReport,
    _freeze_nested,
    _unfreeze_for_serialization,
)


class RegressionSeverity(str, Enum):
    """Severity levels for detected retrieval quality and latency regressions."""

    INFO = "INFO"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"


class RegressionThresholds(BaseModel):
    """
    Immutable configuration defining quality floors and maximum acceptable degradation ratios.
    """

    model_config = ConfigDict(
        frozen=True,
        from_attributes=True,
        validate_default=True,
        arbitrary_types_allowed=False,
    )

    # Maximum allowed absolute drop in quality metrics (e.g. 0.02 = 2 percentage points)
    max_quality_drop: float = Field(default=0.02, ge=0.0, le=1.0)
    # Maximum allowed relative increase in latency (e.g. 0.10 = 10% increase)
    max_latency_increase_ratio: float = Field(default=0.10, ge=0.0)

    # Hard quality floors (optional)
    mrr_min: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    precision_at_k_min: Mapping[int, float] = Field(default_factory=dict)
    recall_at_k_min: Mapping[int, float] = Field(default_factory=dict)
    ndcg_at_k_min: Mapping[int, float] = Field(default_factory=dict)
    max_latency_ms: Optional[float] = Field(default=None, ge=0.0)

    # Relative quality drop tolerances
    max_relative_quality_drop: float = Field(default=0.05, ge=0.0, le=1.0)
    critical_relative_quality_drop: float = Field(default=0.15, ge=0.0, le=1.0)
    max_relative_latency_increase: float = Field(default=0.50, ge=0.0)
    critical_relative_latency_increase: float = Field(default=1.50, ge=0.0)

    @field_validator(
        "precision_at_k_min",
        "recall_at_k_min",
        "ndcg_at_k_min",
        mode="before",
    )
    @classmethod
    def _validate_per_k_thresholds(cls, v: Any) -> Any:
        if v is None:
            return {}
        if not isinstance(v, (dict, types.MappingProxyType)):
            raise TypeError(f"Per-K threshold must be a mapping of int -> float, got {type(v).__name__}")
        validated: Dict[int, float] = {}
        for k, val in v.items():
            k_int = int(k) if isinstance(k, (int, str)) and not isinstance(k, bool) else -1
            if k_int <= 0:
                raise ValueError(f"K value must be positive integer (> 0), got {k!r}")
            if isinstance(val, bool) or not isinstance(val, (int, float)) or not (0.0 <= float(val) <= 1.0):
                raise ValueError(f"Metric threshold for K={k} must be in [0.0, 1.0], got {val!r}")
            validated[k_int] = float(val)
        return validated

    @field_validator(
        "precision_at_k_min",
        "recall_at_k_min",
        "ndcg_at_k_min",
        mode="after",
    )
    @classmethod
    def _freeze_mappings(cls, v: Any) -> Mapping[int, float]:
        return _freeze_nested(v) if v is not None else types.MappingProxyType({})


# Alias for backward compatibility
EvaluationThresholds = RegressionThresholds


class RegressionFinding(BaseModel):
    """
    Immutable representation of an individual detected metric regression.
    """

    model_config = ConfigDict(
        frozen=True,
        from_attributes=True,
        validate_default=True,
        arbitrary_types_allowed=False,
    )

    retriever_name: str
    metric: str
    baseline_value: float
    candidate_value: float
    absolute_delta: float
    relative_delta_percent: float
    severity: RegressionSeverity
    message: str


class QueryRegressionDiagnostic(BaseModel):
    """
    Diagnostic identifying a specific evaluation case that suffered significant degradation.
    """

    model_config = ConfigDict(
        frozen=True,
        from_attributes=True,
        validate_default=True,
        arbitrary_types_allowed=False,
    )

    query_id: UUID
    retriever_name: str
    metric_name: str
    baseline_value: float
    candidate_value: float
    delta: float
    description: Optional[str] = None
    details: str = ""


class RegressionAnalysis(BaseModel):
    """
    Comprehensive regression analysis result for a single retriever or full benchmark comparison.
    """

    model_config = ConfigDict(
        frozen=True,
        from_attributes=True,
        validate_default=True,
        arbitrary_types_allowed=False,
    )

    retriever_name: str
    status: str = Field(pattern="^(PASS|WARN|FAIL)$")
    findings: Tuple[RegressionFinding, ...] = Field(default_factory=tuple)
    query_diagnostics: Tuple[QueryRegressionDiagnostic, ...] = Field(default_factory=tuple)
    has_critical_regression: bool = False
    metadata: Mapping[str, Any] = Field(default_factory=dict)

    @field_validator("findings", mode="before")
    @classmethod
    def _normalize_findings(cls, v: Any) -> Tuple[RegressionFinding, ...]:
        if v is None:
            return ()
        return tuple(v) if isinstance(v, (list, tuple, set)) else (v,)

    @field_validator("query_diagnostics", mode="before")
    @classmethod
    def _normalize_diagnostics(cls, v: Any) -> Tuple[QueryRegressionDiagnostic, ...]:
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

    @field_serializer("metadata")
    def _serialize_metadata(self, v: Mapping[str, Any], _info: Any) -> Dict[str, Any]:
        return _unfreeze_for_serialization(v)


def _compute_deltas(baseline: float, candidate: float) -> Tuple[float, float]:
    """Calculate absolute and relative percentage deltas safely."""
    abs_delta = candidate - baseline
    if abs(baseline) < 1e-12:
        rel_pct = 0.0 if abs(candidate) < 1e-12 else (100.0 if candidate > baseline else -100.0)
    else:
        rel_pct = (abs_delta / baseline) * 100.0
    return abs_delta, rel_pct


def find_worst_query_regressions(
    baseline_results: Sequence[RetrievalEvalResult],
    candidate_results: Sequence[RetrievalEvalResult],
    metric: str = "ndcg_at_3",
    limit: int = 10,
) -> Tuple[QueryRegressionDiagnostic, ...]:
    """
    Identify and rank individual evaluation queries that suffered the greatest regression.

    Args:
        baseline_results: Sequence of RetrievalEvalResult from baseline run.
        candidate_results: Sequence of RetrievalEvalResult from candidate run.
        metric: Metric name to inspect (e.g. 'ndcg_at_3', 'precision_at_1', 'mrr').
        limit: Maximum number of top degraded queries to return.

    Returns:
        Tuple of QueryRegressionDiagnostic ordered by magnitude of degradation descending.
    """
    if limit <= 0:
        return ()

    # Index baseline by (query_id, k, retriever_name)
    base_map: Dict[Tuple[UUID, int, str], RetrievalEvalResult] = {
        (r.query_id, r.k, r.retriever_name): r for r in baseline_results
    }

    # Extract target K from metric if present
    is_mrr = "mrr" in metric.lower() or "reciprocal" in metric.lower()
    target_k = 3
    for part in metric.split("_"):
        if part.isdigit():
            target_k = int(part)

    diagnostics: List[QueryRegressionDiagnostic] = []
    seen_mrr_queries: Set[Tuple[UUID, str]] = set()

    for cand_res in candidate_results:
        if is_mrr:
            mrr_key = (cand_res.query_id, cand_res.retriever_name)
            if mrr_key in seen_mrr_queries:
                continue
        elif cand_res.k != target_k:
            continue

        key = (cand_res.query_id, cand_res.k, cand_res.retriever_name)
        if key not in base_map:
            continue
        base_res = base_map[key]

        # Extract metric values
        if "ndcg" in metric.lower():
            b_val, c_val = base_res.ndcg_at_k, cand_res.ndcg_at_k
        elif "precision" in metric.lower():
            b_val, c_val = base_res.precision_at_k, cand_res.precision_at_k
        elif "recall" in metric.lower():
            b_val, c_val = base_res.recall_at_k, cand_res.recall_at_k
        elif is_mrr:
            b_val, c_val = base_res.reciprocal_rank, cand_res.reciprocal_rank
        else:
            b_val, c_val = base_res.ndcg_at_k, cand_res.ndcg_at_k

        delta = c_val - b_val
        if delta < -1e-6:  # Strict degradation
            if is_mrr:
                seen_mrr_queries.add((cand_res.query_id, cand_res.retriever_name))

            diagnostics.append(
                QueryRegressionDiagnostic(
                    query_id=cand_res.query_id,
                    retriever_name=cand_res.retriever_name,
                    metric_name=metric,
                    baseline_value=b_val,
                    candidate_value=c_val,
                    delta=delta,
                    description=cand_res.metadata.get("case_description"),
                    details=f"{metric} dropped by {abs(delta):.3f} (Baseline: {b_val:.3f} -> Candidate: {c_val:.3f})",
                )
            )

    # Sort deterministically: worst drop first (delta ascending), tie-break on query_id
    diagnostics.sort(key=lambda d: (d.delta, str(d.query_id)))
    return tuple(diagnostics[:limit])


def detect_regressions(
    baseline: RetrieverBenchmarkReport,
    candidate: RetrieverBenchmarkReport,
    thresholds: Optional[RegressionThresholds] = None,
) -> RegressionAnalysis:
    """
    Compare a candidate retriever benchmark against a baseline benchmark and identify regressions.
    """
    if not isinstance(baseline, RetrieverBenchmarkReport):
        raise TypeError(f"baseline must be RetrieverBenchmarkReport, got {type(baseline).__name__}")
    if not isinstance(candidate, RetrieverBenchmarkReport):
        raise TypeError(f"candidate must be RetrieverBenchmarkReport, got {type(candidate).__name__}")

    retriever_name = candidate.retriever_name
    cfg = thresholds or RegressionThresholds()

    findings: List[RegressionFinding] = []

    # Quality Metrics Comparison
    quality_metric_keys = ["mrr"]
    for k in candidate.k_values:
        quality_metric_keys.extend([
            f"mean_precision_at_{k}",
            f"mean_recall_at_{k}",
            f"mean_ndcg_at_{k}",
        ])

    for m_key in quality_metric_keys:
        base_val = baseline.aggregate_metrics.get(m_key, 0.0)
        cand_val = candidate.aggregate_metrics.get(m_key, 0.0)
        abs_delta, rel_pct = _compute_deltas(base_val, cand_val)

        # 1. Quality Floor Check
        floor_violated = False
        floor_limit: Optional[float] = None

        if m_key == "mrr" and cfg.mrr_min is not None:
            floor_limit = cfg.mrr_min
            if cand_val < cfg.mrr_min:
                floor_violated = True
        elif m_key.startswith("mean_precision_at_"):
            k = int(m_key.replace("mean_precision_at_", ""))
            if k in cfg.precision_at_k_min:
                floor_limit = cfg.precision_at_k_min[k]
                if cand_val < floor_limit:
                    floor_violated = True
        elif m_key.startswith("mean_recall_at_"):
            k = int(m_key.replace("mean_recall_at_", ""))
            if k in cfg.recall_at_k_min:
                floor_limit = cfg.recall_at_k_min[k]
                if cand_val < floor_limit:
                    floor_violated = True
        elif m_key.startswith("mean_ndcg_at_"):
            k = int(m_key.replace("mean_ndcg_at_", ""))
            if k in cfg.ndcg_at_k_min:
                floor_limit = cfg.ndcg_at_k_min[k]
                if cand_val < floor_limit:
                    floor_violated = True

        if floor_violated and floor_limit is not None:
            findings.append(
                RegressionFinding(
                    retriever_name=retriever_name,
                    metric=m_key,
                    baseline_value=base_val,
                    candidate_value=cand_val,
                    absolute_delta=abs_delta,
                    relative_delta_percent=rel_pct,
                    severity=RegressionSeverity.CRITICAL,
                    message=f"{m_key} ({cand_val:.4f}) dropped below configured floor ({floor_limit:.4f}).",
                )
            )
        elif abs_delta < -1e-6:
            abs_drop = abs(abs_delta)
            rel_drop = abs_drop / max(base_val, 1e-12)

            if abs_drop > cfg.max_quality_drop or rel_drop >= cfg.critical_relative_quality_drop:
                findings.append(
                    RegressionFinding(
                        retriever_name=retriever_name,
                        metric=m_key,
                        baseline_value=base_val,
                        candidate_value=cand_val,
                        absolute_delta=abs_delta,
                        relative_delta_percent=rel_pct,
                        severity=RegressionSeverity.CRITICAL,
                        message=f"{m_key} dropped by {abs_drop:.4f} ({rel_drop * 100.0:.2f}%), exceeding tolerance.",
                    )
                )
            elif rel_drop >= cfg.max_relative_quality_drop:
                findings.append(
                    RegressionFinding(
                        retriever_name=retriever_name,
                        metric=m_key,
                        baseline_value=base_val,
                        candidate_value=cand_val,
                        absolute_delta=abs_delta,
                        relative_delta_percent=rel_pct,
                        severity=RegressionSeverity.WARNING,
                        message=f"{m_key} degraded by {rel_drop * 100.0:.2f}% (from {base_val:.4f} to {cand_val:.4f}).",
                    )
                )

    # 2. Latency Comparison
    base_lat = baseline.aggregate_metrics.get("mean_latency_ms", 0.0)
    cand_lat = candidate.aggregate_metrics.get("mean_latency_ms", 0.0)
    lat_delta, lat_rel_pct = _compute_deltas(base_lat, cand_lat)

    if cfg.max_latency_ms is not None and cand_lat > cfg.max_latency_ms:
        findings.append(
            RegressionFinding(
                retriever_name=retriever_name,
                metric="mean_latency_ms",
                baseline_value=base_lat,
                candidate_value=cand_lat,
                absolute_delta=lat_delta,
                relative_delta_percent=lat_rel_pct,
                severity=RegressionSeverity.CRITICAL,
                message=f"Latency ({cand_lat:.2f} ms) exceeded maximum threshold ({cfg.max_latency_ms:.2f} ms).",
            )
        )
    elif lat_delta > 0.05:
        rel_lat_inc = lat_delta / max(base_lat, 1e-12)
        if rel_lat_inc > cfg.max_latency_increase_ratio or rel_lat_inc >= cfg.critical_relative_latency_increase:
            findings.append(
                RegressionFinding(
                    retriever_name=retriever_name,
                    metric="mean_latency_ms",
                    baseline_value=base_lat,
                    candidate_value=cand_lat,
                    absolute_delta=lat_delta,
                    relative_delta_percent=lat_rel_pct,
                    severity=RegressionSeverity.CRITICAL,
                    message=f"Latency increased by {rel_lat_inc * 100.0:.1f}% (+{lat_delta:.2f} ms), exceeding {cfg.max_latency_increase_ratio * 100:.0f}% threshold.",
                )
            )

    # 3. Query Diagnostics
    diagnostics = find_worst_query_regressions(
        baseline_results=baseline.results,
        candidate_results=candidate.results,
        metric="ndcg_at_3",
        limit=10,
    )

    has_critical = any(f.severity == RegressionSeverity.CRITICAL for f in findings)
    has_warning = any(f.severity == RegressionSeverity.WARNING for f in findings)

    status = "FAIL" if has_critical else ("WARN" if has_warning else "PASS")

    return RegressionAnalysis(
        retriever_name=retriever_name,
        status=status,
        findings=tuple(findings),
        query_diagnostics=diagnostics,
        has_critical_regression=has_critical,
        metadata={"findings_count": len(findings), "diagnostics_count": len(diagnostics)},
    )


def compare_reports(
    baseline: EvaluationReport,
    candidate: EvaluationReport,
    thresholds: Optional[RegressionThresholds] = None,
    baseline_results: Optional[Sequence[RetrievalEvalResult]] = None,
    candidate_results: Optional[Sequence[RetrievalEvalResult]] = None,
) -> RegressionReport:
    """
    Compare a candidate EvaluationReport against a baseline EvaluationReport and generate a RegressionReport.

    Validates dataset compatibility before comparing metrics.
    """
    if not isinstance(baseline, EvaluationReport):
        raise TypeError(f"baseline must be EvaluationReport, got {type(baseline).__name__}")
    if not isinstance(candidate, EvaluationReport):
        raise TypeError(f"candidate must be EvaluationReport, got {type(candidate).__name__}")

    # Validate dataset compatibility
    if baseline.dataset_name != candidate.dataset_name:
        raise ValueError(
            f"Dataset name mismatch: baseline '{baseline.dataset_name}' vs candidate '{candidate.dataset_name}'"
        )
    if baseline.dataset_version != candidate.dataset_version:
        raise ValueError(
            f"Dataset version mismatch: baseline '{baseline.dataset_version}' vs candidate '{candidate.dataset_version}'"
        )
    if baseline.query_count != candidate.query_count:
        raise ValueError(
            f"Query count mismatch: baseline {baseline.query_count} vs candidate {candidate.query_count}"
        )
    if set(baseline.configured_k_values) != set(candidate.configured_k_values):
        raise ValueError(
            f"K values mismatch: baseline {baseline.configured_k_values} vs candidate {candidate.configured_k_values}"
        )

    cfg = thresholds or RegressionThresholds()
    checks: List[RegressionCheck] = []

    # Compare each retriever present in both reports
    for ret_name, c_sum in candidate.retriever_summaries.items():
        if ret_name not in baseline.retriever_summaries:
            continue
        b_sum = baseline.retriever_summaries[ret_name]

        # 1. MRR Check
        mrr_delta, mrr_rel = _compute_deltas(b_sum.mrr, c_sum.mrr)
        mrr_status = "PASS"
        mrr_msg = ""

        if cfg.mrr_min is not None and c_sum.mrr < cfg.mrr_min:
            mrr_status = "FAIL"
            mrr_msg = f"MRR ({c_sum.mrr:.4f}) dropped below floor threshold ({cfg.mrr_min:.4f})."
        elif mrr_delta < -cfg.max_quality_drop:
            mrr_status = "FAIL"
            mrr_msg = f"MRR dropped by {abs(mrr_delta):.4f}, exceeding tolerance ({cfg.max_quality_drop:.4f})."
        elif mrr_delta < 0:
            mrr_status = "WARN"
            mrr_msg = f"MRR dropped slightly by {abs(mrr_delta):.4f}."

        checks.append(
            RegressionCheck(
                metric_name="mrr",
                retriever_name=ret_name,
                baseline_value=b_sum.mrr,
                candidate_value=c_sum.mrr,
                delta=mrr_delta,
                relative_change=mrr_rel,
                threshold=cfg.max_quality_drop,
                status=mrr_status,
                message=mrr_msg,
            )
        )

        # 2. Latency Check
        lat_delta, lat_rel = _compute_deltas(b_sum.mean_latency_ms, c_sum.mean_latency_ms)
        lat_status = "PASS"
        lat_msg = ""
        rel_lat_ratio = lat_delta / max(b_sum.mean_latency_ms, 1e-12)

        if cfg.max_latency_ms is not None and c_sum.mean_latency_ms > cfg.max_latency_ms:
            lat_status = "FAIL"
            lat_msg = f"Latency ({c_sum.mean_latency_ms:.2f} ms) exceeded max limit ({cfg.max_latency_ms:.2f} ms)."
        elif rel_lat_ratio > cfg.max_latency_increase_ratio and lat_delta > 0.05:
            lat_status = "FAIL"
            lat_msg = f"Latency increased by {rel_lat_ratio * 100:.1f}%, exceeding {cfg.max_latency_increase_ratio * 100:.0f}% threshold."
        elif lat_delta > 0.05:
            lat_status = "WARN"
            lat_msg = f"Latency increased slightly by {lat_delta:.2f} ms."

        checks.append(
            RegressionCheck(
                metric_name="mean_latency_ms",
                retriever_name=ret_name,
                baseline_value=b_sum.mean_latency_ms,
                candidate_value=c_sum.mean_latency_ms,
                delta=lat_delta,
                relative_change=lat_rel,
                threshold=cfg.max_latency_increase_ratio,
                status=lat_status,
                message=lat_msg,
            )
        )

        # 3. Precision, Recall, NDCG per K
        for k in candidate.configured_k_values:
            for metric_type, b_map, c_map, min_cfg in (
                ("precision_at_k", b_sum.precision_at_k, c_sum.precision_at_k, cfg.precision_at_k_min),
                ("recall_at_k", b_sum.recall_at_k, c_sum.recall_at_k, cfg.recall_at_k_min),
                ("ndcg_at_k", b_sum.ndcg_at_k, c_sum.ndcg_at_k, cfg.ndcg_at_k_min),
            ):
                bv = b_map.get(k, 0.0)
                cv = c_map.get(k, 0.0)
                d, rel = _compute_deltas(bv, cv)
                st = "PASS"
                msg = ""

                floor = min_cfg.get(k)
                if floor is not None and cv < floor:
                    st = "FAIL"
                    msg = f"{metric_type}@{k} ({cv:.4f}) dropped below floor ({floor:.4f})."
                elif d < -cfg.max_quality_drop:
                    st = "FAIL"
                    msg = f"{metric_type}@{k} dropped by {abs(d):.4f}, exceeding tolerance."
                elif d < 0:
                    st = "WARN"
                    msg = f"{metric_type}@{k} dropped slightly by {abs(d):.4f}."

                checks.append(
                    RegressionCheck(
                        metric_name=f"{metric_type}_{k}",
                        retriever_name=ret_name,
                        baseline_value=bv,
                        candidate_value=cv,
                        delta=d,
                        relative_change=rel,
                        threshold=cfg.max_quality_drop,
                        status=st,
                        message=msg,
                    )
                )

    has_fail = any(c.status == "FAIL" for c in checks)
    has_warn = any(c.status == "WARN" for c in checks)
    overall = "FAIL" if has_fail else ("WARN" if has_warn else "PASS")

    return RegressionReport(
        baseline_report_id=baseline.report_id,
        candidate_report_id=candidate.report_id,
        checks=tuple(checks),
        overall_status=overall,
        metadata={
            "dataset_name": candidate.dataset_name,
            "checks_count": len(checks),
        },
    )


def assert_no_regressions(
    report: Union[RegressionReport, RegressionAnalysis],
) -> None:
    """
    Assert that the evaluation regression check passed without critical failures.

    Raises:
        EvaluationRegressionError: If any metric regression or threshold violation caused overall_status == 'FAIL'.
    """
    status = report.overall_status if hasattr(report, "overall_status") else report.status
    if status == "FAIL":
        failures: List[str] = []
        if isinstance(report, RegressionReport):
            failures = [f"[{c.retriever_name}] {c.metric_name}: {c.message}" for c in report.checks if c.status == "FAIL"]
        elif isinstance(report, RegressionAnalysis):
            failures = [f"[{f.retriever_name}] {f.metric}: {f.message}" for f in report.findings if f.severity == RegressionSeverity.CRITICAL]

        error_msg = f"Evaluation Regression Detected (Status: FAIL)!\n" + "\n".join(f"  - {fail}" for fail in failures)
        raise EvaluationRegressionError(error_msg)


def compare_benchmark_runs(
    baseline_reports: Mapping[str, RetrieverBenchmarkReport],
    candidate_reports: Mapping[str, RetrieverBenchmarkReport],
    thresholds: Optional[RegressionThresholds] = None,
) -> Dict[str, RegressionAnalysis]:
    """
    Compare multiple candidate retriever benchmark reports against corresponding baselines.
    """
    analyses: Dict[str, RegressionAnalysis] = {}
    for name, cand_report in candidate_reports.items():
        if name in baseline_reports:
            base_report = baseline_reports[name]
            analyses[name] = detect_regressions(
                baseline=base_report,
                candidate=cand_report,
                thresholds=thresholds,
            )
        else:
            analyses[name] = detect_regressions(
                baseline=cand_report,
                candidate=cand_report,
                thresholds=thresholds,
            )
    return analyses
