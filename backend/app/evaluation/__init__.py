"""
Revora Retrieval Evaluation Package.

Provides clean, immutable data contracts for evaluating retrieval quality
(Precision@K, Recall@K, MRR, NDCG@K, latency) across deterministic, semantic,
and hybrid historical retrieval backends.
"""

from app.evaluation.benchmark import (
    compare_benchmarks,
    format_comparison_markdown,
    populate_benchmark_vector_index,
    run_benchmark,
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
    EvaluationCase,
    EvaluationReport,
    EvaluationRegressionError,
    GroundTruthJudgment,
    MetricSnapshot,
    RegressionCheck,
    RegressionReport,
    RetrievalEvalResult,
    RetrieverBenchmarkReport,
    RetrieverEvaluationSummary,
)

__all__ = [
    "EvaluationCase",
    "GroundTruthJudgment",
    "RetrievalEvalResult",
    "RetrieverBenchmarkReport",
    "MetricSnapshot",
    "RetrieverEvaluationSummary",
    "EvaluationReport",
    "RegressionCheck",
    "RegressionReport",
    "EvaluationRegressionError",
    "RetrievalEvaluator",
    "RetrieverProtocol",
    "precision_at_k",
    "recall_at_k",
    "mean_reciprocal_rank",
    "ndcg_at_k",
    "run_benchmark",
    "compare_benchmarks",
    "format_comparison_markdown",
    "populate_benchmark_vector_index",
    "EvaluationThresholds",
    "RegressionThresholds",
    "RegressionSeverity",
    "RegressionFinding",
    "QueryRegressionDiagnostic",
    "RegressionAnalysis",
    "detect_regressions",
    "compare_benchmark_runs",
    "compare_reports",
    "find_worst_query_regressions",
    "assert_no_regressions",
    "generate_json_report",
    "generate_markdown_report",
    "create_evaluation_report",
    "save_benchmark_artifacts",
    "save_report",
    "load_report",
    "load_latest_report",
    "list_reports",
]
