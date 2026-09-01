"""
Revora Retrieval Evaluation Package.

Provides clean, immutable data contracts for evaluating retrieval quality
(Precision@K, Recall@K, MRR, NDCG@K, latency) across deterministic, semantic,
and hybrid historical retrieval backends.
"""

from app.evaluation.decision_evaluator import (
    AgentRAGPipeline,
    DecisionEvaluator,
    DecisionPipeline,
    DeterministicBaselinePipeline,
    DeterministicRAGPipeline,
    extract_historical_cases_from_context,
)
from app.evaluation.decision_metrics import (
    acceptable_match_rate,
    compute_aggregate_decision_metrics,
    exact_match_rate,
    fallback_rate,
    mean_confidence,
    mean_latency_ms,
    policy_match_rate,
    policy_override_rate,
    policy_violation_rate,
    safety_violation_rate,
)
from app.evaluation.decision_persistence import (
    compare_decision_with_baseline,
    get_decision_evaluation_directory,
    list_decision_reports,
    load_decision_report,
    load_latest_decision_report,
    save_decision_report,
)
from app.evaluation.decision_regression import (
    DecisionMetricCheck,
    DecisionQualityGateResult,
    DecisionQualityThresholds,
    DecisionRegressionComparisonResult,
    assert_decision_quality_gate,
    compare_decision_runs,
    evaluate_decision_quality_gate,
    format_quality_gate_terminal_summary,
)
from app.evaluation.decision_reporting import (
    analyze_decision_failures,
    compare_decision_pipelines,
    generate_decision_comparison_markdown,
    generate_decision_json_report,
    generate_decision_markdown_report,
    save_decision_benchmark_artifacts,
)
from app.evaluation.evaluator import (
    RetrievalEvaluator,
    RetrieverProtocol,
)
from app.evaluation.metrics import (
    mean_reciprocal_rank,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)
from app.evaluation.persistence import (
    list_reports,
    load_latest_report,
    load_report,
    save_report,
)
from app.evaluation.recovery_comparison import (
    RecoveryLeaderboardEntry,
    RecoveryStrategyUplift,
    calculate_cost_per_recovered_dollar,
    calculate_recovery_roi,
    compare_category_recovery_performance,
    compute_recovery_strategy_uplift,
    generate_recovery_leaderboard,
)
from app.evaluation.recovery_persistence import (
    get_recovery_evaluation_directory,
    list_recovery_reports,
    load_latest_recovery_report,
    load_recovery_report,
    save_recovery_report,
)
from app.evaluation.recovery_regression import (
    RecoveryMetricCheck,
    RecoveryQualityGateResult,
    RecoveryQualityThresholds,
    RecoveryRegressionComparisonResult,
    assert_recovery_quality_gate,
    compare_recovery_runs,
    evaluate_recovery_quality_gate,
    format_recovery_quality_gate_terminal_summary,
)
from app.evaluation.recovery_reporting import (
    compare_recovery_pipelines,
    generate_recovery_comparison_markdown,
    generate_recovery_json_report,
    generate_recovery_markdown_report,
    save_recovery_benchmark_artifacts,
)
from app.evaluation.recovery_schemas import (
    DEFAULT_ACTION_COSTS,
    RecoveryBenchmarkReport,
    RecoveryScenario,
    SimulatedRecoveryOutcome,
)
from app.evaluation.recovery_simulator import (
    RecoverySimulator,
)
from app.evaluation.regression import (
    EvaluationThresholds,
    QueryRegressionDiagnostic,
    RegressionAnalysis,
    RegressionFinding,
    RegressionSeverity,
    RegressionThresholds,
    assert_no_regressions,
    compare_benchmark_runs,
    compare_reports,
    detect_regressions,
    find_worst_query_regressions,
)
from app.evaluation.reporting import (
    create_evaluation_report,
    generate_json_report,
    generate_markdown_report,
    save_benchmark_artifacts,
)
from app.evaluation.schemas import (
    DecisionBenchmarkReport,
    DecisionEvalResult,
    DecisionGroundTruth,
    EvaluationCase,
    EvaluationRegressionError,
    EvaluationReport,
    GroundTruthJudgment,
    MetricSnapshot,
    RegressionCheck,
    RegressionReport,
    RetrievalEvalResult,
    RetrieverBenchmarkReport,
    RetrieverEvaluationSummary,
)

__all__ = [
    "DEFAULT_ACTION_COSTS",
    "AgentRAGPipeline",
    "DecisionBenchmarkReport",
    "DecisionEvalResult",
    "DecisionEvaluator",
    "DecisionGroundTruth",
    "DecisionMetricCheck",
    "DecisionPipeline",
    "DecisionQualityGateResult",
    "DecisionQualityThresholds",
    "DecisionRegressionComparisonResult",
    "DeterministicBaselinePipeline",
    "DeterministicRAGPipeline",
    "EvaluationCase",
    "EvaluationRegressionError",
    "EvaluationReport",
    "EvaluationThresholds",
    "GroundTruthJudgment",
    "MetricSnapshot",
    "QueryRegressionDiagnostic",
    "RecoveryBenchmarkReport",
    "RecoveryLeaderboardEntry",
    "RecoveryMetricCheck",
    "RecoveryQualityGateResult",
    "RecoveryQualityThresholds",
    "RecoveryRegressionComparisonResult",
    "RecoveryScenario",
    "RecoverySimulator",
    "RecoveryStrategyUplift",
    "RegressionAnalysis",
    "RegressionCheck",
    "RegressionFinding",
    "RegressionReport",
    "RegressionSeverity",
    "RegressionThresholds",
    "RetrievalEvalResult",
    "RetrievalEvaluator",
    "RetrieverBenchmarkReport",
    "RetrieverEvaluationSummary",
    "RetrieverProtocol",
    "SimulatedRecoveryOutcome",
    "acceptable_match_rate",
    "analyze_decision_failures",
    "assert_decision_quality_gate",
    "assert_no_regressions",
    "assert_recovery_quality_gate",
    "calculate_cost_per_recovered_dollar",
    "calculate_recovery_roi",
    "compare_benchmark_runs",
    "compare_benchmarks",
    "compare_category_recovery_performance",
    "compare_decision_pipelines",
    "compare_decision_runs",
    "compare_decision_with_baseline",
    "compare_recovery_pipelines",
    "compare_recovery_runs",
    "compare_reports",
    "compute_aggregate_decision_metrics",
    "compute_recovery_strategy_uplift",
    "create_evaluation_report",
    "detect_regressions",
    "evaluate_decision_quality_gate",
    "evaluate_recovery_quality_gate",
    "exact_match_rate",
    "extract_historical_cases_from_context",
    "fallback_rate",
    "find_worst_query_regressions",
    "format_comparison_markdown",
    "format_decision_benchmark_terminal_summary",
    "format_quality_gate_terminal_summary",
    "format_recovery_benchmark_terminal_summary",
    "format_recovery_quality_gate_terminal_summary",
    "generate_decision_comparison_markdown",
    "generate_decision_json_report",
    "generate_decision_markdown_report",
    "generate_json_report",
    "generate_markdown_report",
    "generate_recovery_comparison_markdown",
    "generate_recovery_json_report",
    "generate_recovery_leaderboard",
    "generate_recovery_markdown_report",
    "get_decision_evaluation_directory",
    "get_recovery_evaluation_directory",
    "list_decision_reports",
    "list_recovery_reports",
    "list_reports",
    "load_decision_report",
    "load_latest_decision_report",
    "load_latest_recovery_report",
    "load_latest_report",
    "load_recovery_report",
    "load_report",
    "mean_confidence",
    "mean_latency_ms",
    "mean_reciprocal_rank",
    "ndcg_at_k",
    "policy_match_rate",
    "policy_override_rate",
    "policy_violation_rate",
    "populate_benchmark_vector_index",
    "precision_at_k",
    "recall_at_k",
    "resolve_decision_pipeline",
    "run_benchmark",
    "run_decision_benchmark",
    "run_decision_cli",
    "run_recovery_benchmark",
    "run_recovery_cli",
    "safety_violation_rate",
    "save_benchmark_artifacts",
    "save_decision_benchmark_artifacts",
    "save_decision_report",
    "save_recovery_benchmark_artifacts",
    "save_recovery_report",
    "save_report",
]


def __getattr__(name: str):
    if name in (
        "compare_benchmarks",
        "format_comparison_markdown",
        "populate_benchmark_vector_index",
        "run_benchmark",
    ):
        import app.evaluation.benchmark as _bench

        return getattr(_bench, name)
    if name in (
        "format_decision_benchmark_terminal_summary",
        "resolve_decision_pipeline",
        "run_decision_benchmark",
        "run_decision_cli",
    ):
        import app.evaluation.decision_benchmark as _dbench

        return getattr(_dbench, name)
    if name in (
        "format_recovery_benchmark_terminal_summary",
        "run_recovery_benchmark",
        "run_recovery_cli",
    ):
        import app.evaluation.recovery_benchmark as _rbench

        return getattr(_rbench, name)
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
