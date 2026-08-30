"""
Revora Retrieval Evaluation Persistence Tests.

Validates atomic file persistence, report loading, listing, error handling on corrupted data,
and overwriting safeguards.
"""

from pathlib import Path
import pytest

from app.evaluation.persistence import (
    list_reports,
    load_latest_report,
    load_report,
    save_report,
)
from app.evaluation.schemas import (
    EvaluationReport,
    RetrieverEvaluationSummary,
)


def _make_report(report_id: str = "report_test_001") -> EvaluationReport:
    summary = RetrieverEvaluationSummary(
        retriever_name="HistoricalRetriever",
        query_count=50,
        mrr=0.98,
        mean_latency_ms=0.05,
        precision_at_k={1: 0.98, 3: 0.88},
        recall_at_k={1: 0.37, 3: 0.97},
        ndcg_at_k={1: 0.98, 3: 0.97},
    )
    return EvaluationReport(
        report_id=report_id,
        dataset_name="retrieval_golden_dataset",
        dataset_version="v1",
        query_count=50,
        configured_k_values=(1, 3),
        retriever_summaries={"HistoricalRetriever": summary},
    )


def test_save_and_load_report(tmp_path: Path):
    report = _make_report("eval_run_abc")
    saved_path = save_report(report, directory=tmp_path)

    assert saved_path.exists()
    assert saved_path.name == "eval_run_abc.json"

    loaded = load_report("eval_run_abc", directory=tmp_path)
    assert loaded.report_id == "eval_run_abc"
    assert loaded.query_count == 50
    assert "HistoricalRetriever" in loaded.retriever_summaries
    assert loaded.retriever_summaries["HistoricalRetriever"].mrr == 0.98


def test_save_report_duplicate_prevention(tmp_path: Path):
    report = _make_report("duplicate_test")
    save_report(report, directory=tmp_path)

    # Second save with overwrite=False must raise FileExistsError
    with pytest.raises(FileExistsError):
        save_report(report, directory=tmp_path, overwrite=False)

    # Save with overwrite=True should succeed
    save_report(report, directory=tmp_path, overwrite=True)


def test_load_latest_report(tmp_path: Path):
    r1 = _make_report("report_1")
    r2 = _make_report("report_2")

    save_report(r1, directory=tmp_path)
    save_report(r2, directory=tmp_path, overwrite=True)

    latest = load_latest_report(directory=tmp_path)
    assert latest is not None
    assert latest.report_id == "report_2"


def test_list_reports(tmp_path: Path):
    r1 = _make_report("alpha_report")
    r2 = _make_report("beta_report")

    save_report(r1, directory=tmp_path)
    save_report(r2, directory=tmp_path)

    reports = list_reports(directory=tmp_path)
    assert reports == ["alpha_report", "beta_report"]


def test_load_nonexistent_report_raises_filenotfound(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        load_report("nonexistent_id", directory=tmp_path)


def test_load_corrupted_report_raises_valueerror(tmp_path: Path):
    bad_file = tmp_path / "corrupted.json"
    bad_file.write_text("{ this is corrupted json", encoding="utf-8")

    with pytest.raises(ValueError, match="Corrupted JSON"):
        load_report("corrupted", directory=tmp_path)
