"""
Revora Decision Evaluation Persistence & Historical Tracking Tests.

Comprehensive testing of decision benchmark JSON storage, retrieval, atomic updates,
latest pointer management, disk baseline comparisons, and CLI regression enforcement.
"""

import json
from pathlib import Path
from uuid import uuid4

import pytest

from app.decision_engine import RecoveryAction
from app.evaluation.decision_benchmark import run_decision_cli
from app.evaluation.decision_persistence import (
    compare_decision_with_baseline,
    list_decision_reports,
    load_decision_report,
    load_latest_decision_report,
    save_decision_report,
)
from app.evaluation.schemas import DecisionBenchmarkReport, DecisionEvalResult
from tests.fixtures.retrieval_golden_dataset import get_golden_evaluation_cases


def _make_benchmark_report(
    pipeline_name="deterministic_rag",
    metrics=None,
) -> DecisionBenchmarkReport:
    m = metrics or {
        "exact_match_rate": 0.90,
        "acceptable_match_rate": 0.96,
        "safety_violation_rate": 0.0,
        "policy_match_rate": 0.92,
        "policy_violation_rate": 0.02,
        "policy_override_rate": 0.02,
        "fallback_rate": 0.04,
        "mean_confidence": 0.88,
        "mean_latency_ms": 40.0,
    }
    dummy_result = DecisionEvalResult(
        query_id=uuid4(),
        pipeline_name=pipeline_name,
        predicted_action=RecoveryAction.RETRY_PAYMENT,
        expected_action=RecoveryAction.RETRY_PAYMENT,
        acceptable_actions=(RecoveryAction.RETRY_PAYMENT,),
        is_exact_match=True,
        is_acceptable_match=True,
        confidence=0.9,
    )
    return DecisionBenchmarkReport(
        pipeline_name=pipeline_name,
        dataset_name="golden_dataset_50",
        num_queries=1,
        results=(dummy_result,),
        aggregate_metrics=m,
    )


# =============================================================================
# Persistence & Roundtrip Tests
# =============================================================================


def test_save_and_load_decision_report(tmp_path: Path):
    report = _make_benchmark_report()
    saved_path = save_decision_report(
        report=report,
        directory=tmp_path,
        report_id="rag_run_1",
    )

    assert saved_path.exists()
    assert saved_path.name == "rag_run_1.json"

    loaded = load_decision_report("rag_run_1", directory=tmp_path)
    assert isinstance(loaded, DecisionBenchmarkReport)
    assert loaded.pipeline_name == report.pipeline_name
    assert loaded.dataset_name == report.dataset_name
    assert loaded.num_queries == report.num_queries
    assert loaded.aggregate_metrics == report.aggregate_metrics


def test_save_decision_report_duplicate_prevention(tmp_path: Path):
    report = _make_benchmark_report()
    save_decision_report(report, directory=tmp_path, report_id="dup_test")

    with pytest.raises(FileExistsError, match="already exists"):
        save_decision_report(
            report, directory=tmp_path, report_id="dup_test", overwrite=False
        )

    # Overwrite=True succeeds
    overwrite_path = save_decision_report(
        report, directory=tmp_path, report_id="dup_test", overwrite=True
    )
    assert overwrite_path.exists()


def test_save_decision_report_invalid_type_raises_type_error(tmp_path: Path):
    with pytest.raises(TypeError, match="must be DecisionBenchmarkReport"):
        save_decision_report("invalid", directory=tmp_path)  # type: ignore[arg-type]


# =============================================================================
# Latest Pointer & List Reports Tests
# =============================================================================


def test_load_latest_decision_report(tmp_path: Path):
    assert load_latest_decision_report(directory=tmp_path) is None

    r1 = _make_benchmark_report(pipeline_name="p1")
    r2 = _make_benchmark_report(pipeline_name="p2")

    save_decision_report(r1, directory=tmp_path, report_id="p1_run")
    save_decision_report(r2, directory=tmp_path, report_id="p2_run")

    latest_p1 = load_latest_decision_report(pipeline_name="p1", directory=tmp_path)
    assert latest_p1 is not None
    assert latest_p1.pipeline_name == "p1"

    latest_p2 = load_latest_decision_report(pipeline_name="p2", directory=tmp_path)
    assert latest_p2 is not None
    assert latest_p2.pipeline_name == "p2"


def test_list_decision_reports(tmp_path: Path):
    assert list_decision_reports(directory=tmp_path) == []

    r = _make_benchmark_report()
    save_decision_report(r, directory=tmp_path, report_id="report_b")
    save_decision_report(r, directory=tmp_path, report_id="report_a")

    reports = list_decision_reports(directory=tmp_path)
    assert reports == ["report_a", "report_b"]


# =============================================================================
# Error Handling Tests
# =============================================================================


def test_load_nonexistent_report_raises_file_not_found(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        load_decision_report("nonexistent_id", directory=tmp_path)


def test_load_corrupted_report_raises_value_error(tmp_path: Path):
    corrupt_file = tmp_path / "corrupt.json"
    corrupt_file.write_text("{invalid json", encoding="utf-8")

    with pytest.raises(ValueError, match="Corrupted JSON"):
        load_decision_report("corrupt", directory=tmp_path)


def test_load_invalid_schema_report_raises_value_error(tmp_path: Path):
    invalid_file = tmp_path / "invalid_schema.json"
    invalid_file.write_text(json.dumps({"pipeline_name": 12345}), encoding="utf-8")

    with pytest.raises(ValueError, match="Invalid decision benchmark report schema"):
        load_decision_report("invalid_schema", directory=tmp_path)


# =============================================================================
# Disk Baseline Comparison Tests
# =============================================================================


def test_compare_decision_with_baseline_success(tmp_path: Path):
    baseline = _make_benchmark_report(
        pipeline_name="deterministic_rag",
        metrics={
            "exact_match_rate": 0.88,
            "safety_violation_rate": 0.0,
            "mean_latency_ms": 50.0,
        },
    )
    save_decision_report(baseline, directory=tmp_path, report_id="rag_baseline")

    current = _make_benchmark_report(
        pipeline_name="deterministic_rag",
        metrics={
            "exact_match_rate": 0.90,
            "safety_violation_rate": 0.0,
            "mean_latency_ms": 48.0,
        },
    )

    comp = compare_decision_with_baseline(
        current_report=current,
        baseline_id_or_path="rag_baseline",
        directory=tmp_path,
    )

    assert comp.passed is True
    assert "exact_match_rate" in comp.improved_metrics


def test_compare_decision_with_baseline_missing_raises_file_not_found(tmp_path: Path):
    current = _make_benchmark_report()
    with pytest.raises(FileNotFoundError, match="No baseline report found on disk"):
        compare_decision_with_baseline(current_report=current, directory=tmp_path)


# =============================================================================
# CLI Baseline Comparison Tests
# =============================================================================


def test_run_decision_cli_compare_baseline_success(tmp_path: Path, capsys):
    cases = get_golden_evaluation_cases()[:3]
    # 1. Run and save baseline
    exit_code_1 = run_decision_cli(
        args=[
            "-p",
            "deterministic_baseline",
            "--output-dir",
            str(tmp_path),
        ],
        evaluation_cases=cases,
    )
    assert exit_code_1 == 0

    # 2. Run current and compare against latest baseline on disk
    exit_code_2 = run_decision_cli(
        args=[
            "-p",
            "deterministic_baseline",
            "--compare-baseline",
            "latest",
            "--assert-no-regressions",
            "--output-dir",
            str(tmp_path),
            "--no-save",
        ],
        evaluation_cases=cases,
    )
    assert exit_code_2 == 0
    captured = capsys.readouterr()
    assert "Baseline Comparison for 'deterministic_baseline': PASS" in captured.out


def test_run_decision_cli_assert_no_regressions_without_baseline_fails(capsys):
    exit_code = run_decision_cli(
        args=["--assert-no-regressions", "--no-save"],
        evaluation_cases=get_golden_evaluation_cases()[:1],
    )
    assert exit_code == 1
    captured = capsys.readouterr()
    assert "requires --compare-baseline" in captured.err
