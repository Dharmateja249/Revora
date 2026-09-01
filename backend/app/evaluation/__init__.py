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
    "EvaluationCase",
    "EvaluationRegressionError",
    "EvaluationReport",
    "EvaluationThresholds",
    "GroundTruthJudgment",
    "MetricSnapshot",
    "QueryRegressionDiagnostic",
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
    "assert_no_regressions",
    "compare_benchmark_runs",
    "compare_benchmarks",
    "compare_reports",
    "create_evaluation_report",
    "detect_regressions",
    "find_worst_query_regressions",
    "format_comparison_markdown",
    "generate_json_report",
    "generate_markdown_report",
    "list_reports",
    "load_latest_report",
    "load_report",
    "mean_reciprocal_rank",
    "ndcg_at_k",
    "populate_benchmark_vector_index",
    "precision_at_k",
    "recall_at_k",
    "run_benchmark",
    "save_benchmark_artifacts",
    "save_report",
]
