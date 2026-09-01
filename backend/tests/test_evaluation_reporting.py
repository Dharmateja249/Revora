"""
Revora Retrieval Evaluation Reporting Unit Tests.

Validates JSON serialization, Markdown report formatting, executive summaries,
empty benchmark handling, and artifact persistence.
"""

import json
from pathlib import Path
from uuid import uuid4

from app.evaluation.regression import (
    EvaluationThresholds,
    detect_regressions,
)
from app.evaluation.reporting import (
    generate_json_report,
    generate_markdown_report,
    save_benchmark_artifacts,
)
from app.evaluation.schemas import (
    RetrievalEvalResult,
    RetrieverBenchmarkReport,
)


def _make_dummy_report(
    retriever_name: str = "TestRetriever",
    mrr: float = 0.95,
    p1: float = 0.90,
    r1: float = 0.40,
    ndcg1: float = 0.90,
    latency_ms: float = 1.5,
    k_values=(1, 3),
) -> RetrieverBenchmarkReport:
    qid = uuid4()
    results = [
        RetrievalEvalResult(
            query_id=qid,
            retriever_name=retriever_name,
            k=1,
            retrieved_payment_ids=(uuid4(),),
            precision_at_k=p1,
            recall_at_k=r1,
            reciprocal_rank=mrr,
            ndcg_at_k=ndcg1,
            latency_ms=latency_ms,
        ),
        RetrievalEvalResult(
            query_id=qid,
            retriever_name=retriever_name,
            k=3,
            retrieved_payment_ids=(uuid4(), uuid4()),
            precision_at_k=0.80,
            recall_at_k=0.90,
            reciprocal_rank=mrr,
            ndcg_at_k=0.85,
            latency_ms=latency_ms,
        ),
    ]

    return RetrieverBenchmarkReport(
        retriever_name=retriever_name,
        dataset_name="golden_test_v1",
        num_queries=1,
        k_values=k_values,
        results=tuple(results),
        aggregate_metrics={
            "mrr": mrr,
            "mean_latency_ms": latency_ms,
            "mean_precision_at_1": p1,
            "mean_recall_at_1": r1,
            "mean_ndcg_at_1": ndcg1,
            "mean_precision_at_3": 0.80,
            "mean_recall_at_3": 0.90,
            "mean_ndcg_at_3": 0.85,
        },
    )


# ============================================================================
# 1. JSON Report Tests
# ============================================================================


def test_generate_json_report_structure_and_serialization():
    rep1 = _make_dummy_report("RetrieverA", mrr=0.98, latency_ms=0.5)
    rep2 = _make_dummy_report("RetrieverB", mrr=0.90, latency_ms=2.0)
    reports = {"RetrieverA": rep1, "RetrieverB": rep2}

    json_str = generate_json_report(reports)
    assert isinstance(json_str, str)

    parsed = json.loads(json_str)
    assert "benchmark" in parsed
    assert parsed["benchmark"]["dataset_name"] == "golden_test_v1"
    assert parsed["benchmark"]["query_count"] == 1
    assert parsed["benchmark"]["k_values"] == [1, 3]
    assert parsed["benchmark"]["retrievers_count"] == 2

    assert "retrievers" in parsed
    assert "RetrieverA" in parsed["retrievers"]
    assert "RetrieverB" in parsed["retrievers"]
    assert parsed["retrievers"]["RetrieverA"]["mrr"] == 0.98
    assert parsed["retrievers"]["RetrieverA"]["mean_latency_ms"] == 0.5


def test_generate_json_report_with_regressions():
    rep_base = _make_dummy_report("RetrieverA", mrr=0.98)
    rep_cand = _make_dummy_report("RetrieverA", mrr=0.85)

    reg = detect_regressions(
        baseline=rep_base,
        candidate=rep_cand,
        thresholds=EvaluationThresholds(mrr_min=0.90),
    )

    json_str = generate_json_report(
        reports={"RetrieverA": rep_cand},
        regressions={"RetrieverA": reg},
    )
    parsed = json.loads(json_str)

    ret_entry = parsed["retrievers"]["RetrieverA"]
    assert "regression_analysis" in ret_entry
    assert ret_entry["regression_analysis"]["status"] == "FAIL"
    assert ret_entry["regression_analysis"]["has_critical_regression"] is True
    assert len(ret_entry["regression_analysis"]["findings"]) > 0


def test_generate_json_report_empty():
    json_str = generate_json_report({})
    parsed = json.loads(json_str)
    assert parsed == {"benchmark": {}, "retrievers": {}}


def test_generate_json_report_determinism():
    rep = _make_dummy_report("DeterministicRetriever")
    reports = {"DeterministicRetriever": rep}

    s1 = generate_json_report(reports)
    s2 = generate_json_report(reports)
    assert s1 == s2


# ============================================================================
# 2. Markdown Report Tests
# ============================================================================


def test_generate_markdown_report_comprehensive():
    rep_det = _make_dummy_report(
        "DeterministicHistoricalRetriever", mrr=0.98, latency_ms=0.05
    )
    rep_sem = _make_dummy_report(
        "SemanticHistoricalRetriever", mrr=0.92, latency_ms=1.10
    )
    rep_hyb = _make_dummy_report("HybridHistoricalRetriever", mrr=0.98, latency_ms=1.25)
    reports = {
        "DeterministicHistoricalRetriever": rep_det,
        "SemanticHistoricalRetriever": rep_sem,
        "HybridHistoricalRetriever": rep_hyb,
    }

    reg_det = detect_regressions(rep_det, rep_det)
    reg_sem = detect_regressions(rep_sem, rep_sem)
    reg_hyb = detect_regressions(rep_hyb, rep_hyb)
    regressions = {
        "DeterministicHistoricalRetriever": reg_det,
        "SemanticHistoricalRetriever": reg_sem,
        "HybridHistoricalRetriever": reg_hyb,
    }

    md = generate_markdown_report(reports=reports, regressions=regressions)

    assert "# Revora Retrieval Evaluation Report" in md
    assert "## 1. Benchmark Overview" in md
    assert "## 2. Executive Summary" in md
    assert "## 3. Regression Status" in md
    assert "## 4. Comprehensive Metrics Comparison" in md
    assert "## 5. Automated Engineering Findings" in md

    # Verify retriever names appear in tables
    assert "DeterministicHistoricalRetriever" in md
    assert "SemanticHistoricalRetriever" in md
    assert "HybridHistoricalRetriever" in md

    # Verify status badges
    assert "🟢 PASS" in md


def test_generate_markdown_report_empty():
    md = generate_markdown_report({})
    assert "No benchmark reports available" in md


# ============================================================================
# 3. Artifact Persistence Tests
# ============================================================================


def test_save_benchmark_artifacts(tmp_path: Path):
    rep = _make_dummy_report("RetrieverA", mrr=0.95)
    reports = {"RetrieverA": rep}

    artifacts = save_benchmark_artifacts(reports=reports, output_dir=tmp_path)

    assert "json" in artifacts
    assert "markdown" in artifacts

    json_file = artifacts["json"]
    md_file = artifacts["markdown"]

    assert json_file.exists()
    assert md_file.exists()

    # Verify JSON content is valid
    data = json.loads(json_file.read_text(encoding="utf-8"))
    assert "RetrieverA" in data["retrievers"]

    # Verify Markdown content is present
    md_text = md_file.read_text(encoding="utf-8")
    assert "Revora Retrieval Evaluation Report" in md_text


def test_app_evaluation_isolated_import():
    """Verify app.evaluation and all submodules can be imported cleanly without tests on PYTHONPATH."""
    import subprocess
    import sys
    from pathlib import Path

    backend_dir = str(Path(__file__).resolve().parent.parent)
    code = f"import sys; sys.path.insert(0, r'{backend_dir}'); sys.path = [p for p in sys.path if 'tests' not in p]; import app.evaluation; assert hasattr(app.evaluation, 'run_benchmark')"
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, f"Import failed: {result.stderr}"
