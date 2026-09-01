"""
Revora Decision Evaluation Persistence & Historical Tracking.

Provides filesystem-based JSON storage, retrieval, atomic write operations,
and historical baseline tracking for DecisionBenchmarkReport artifacts.
"""

import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.evaluation.decision_regression import (
    DecisionQualityThresholds,
    DecisionRegressionComparisonResult,
    compare_decision_runs,
)
from app.evaluation.persistence import get_evaluation_directory
from app.evaluation.schemas import DecisionBenchmarkReport


def get_decision_evaluation_directory(
    custom_dir: str | Path | None = None,
) -> Path:
    """Resolve and ensure the target directory for persisting decision evaluation reports."""
    if custom_dir is not None:
        path = Path(custom_dir)
    else:
        path = get_evaluation_directory() / "decisions"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _atomic_write_json(file_path: Path, data: dict[str, Any]) -> None:
    """Atomically write JSON payload to disk using a temporary file in the same directory."""
    parent_dir = file_path.parent
    parent_dir.mkdir(parents=True, exist_ok=True)

    json_content = json.dumps(data, indent=2, sort_keys=True)

    temp_file = tempfile.NamedTemporaryFile(  # noqa: SIM115
        mode="w",
        encoding="utf-8",
        dir=parent_dir,
        delete=False,
        suffix=".tmp",
    )
    try:
        temp_file.write(json_content)
        temp_file.flush()
        os.fsync(temp_file.fileno())
        temp_file.close()
        os.replace(temp_file.name, file_path)
    except Exception:
        if os.path.exists(temp_file.name):
            os.remove(temp_file.name)
        raise


def save_decision_report(
    report: DecisionBenchmarkReport,
    directory: str | Path | None = None,
    report_id: str | None = None,
    overwrite: bool = False,
) -> Path:
    """
    Persist a DecisionBenchmarkReport to disk atomically as JSON.

    Args:
        report: DecisionBenchmarkReport instance to persist.
        directory: Target storage directory.
        report_id: Optional unique report identifier.
        overwrite: Whether to overwrite an existing report with the same ID.

    Returns:
        Path to the saved JSON report file.

    Raises:
        TypeError: If report is not a DecisionBenchmarkReport.
        FileExistsError: If report already exists and overwrite is False.
    """
    if not isinstance(report, DecisionBenchmarkReport):
        raise TypeError(
            f"report must be DecisionBenchmarkReport, got {type(report).__name__}"
        )

    target_dir = get_decision_evaluation_directory(directory)
    timestamp_str = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    clean_id = report_id or f"decision_{report.pipeline_name}_{timestamp_str}"
    if not clean_id.endswith(".json"):
        clean_id = f"{clean_id}.json"

    file_path = target_dir / clean_id
    if file_path.exists() and not overwrite:
        raise FileExistsError(
            f"Decision report '{clean_id}' already exists at {file_path}. Use overwrite=True to replace."
        )

    payload = report.model_dump(mode="json")
    _atomic_write_json(file_path, payload)

    # Maintain latest report for this pipeline
    latest_path = target_dir / f"{report.pipeline_name}_latest.json"
    _atomic_write_json(latest_path, payload)

    return file_path


def load_decision_report(
    report_id_or_path: str | Path,
    directory: str | Path | None = None,
) -> DecisionBenchmarkReport:
    """
    Load a DecisionBenchmarkReport by ID or filepath.

    Args:
        report_id_or_path: String ID, filename, or Path to JSON report.
        directory: Optional directory containing reports.

    Returns:
        Reconstituted DecisionBenchmarkReport instance.

    Raises:
        FileNotFoundError: If the report file does not exist.
        ValueError: If file contains corrupted JSON or invalid schema.
    """
    target_dir = get_decision_evaluation_directory(directory)
    path_obj = Path(report_id_or_path)

    if path_obj.is_absolute() and path_obj.exists():
        file_path = path_obj
    else:
        name = str(report_id_or_path).strip()
        if not name.endswith(".json"):
            name = f"{name}.json"
        file_path = target_dir / name

    if not file_path.exists():
        raise FileNotFoundError(f"Decision benchmark report not found at: {file_path}")

    try:
        raw_text = file_path.read_text(encoding="utf-8")
        data = json.loads(raw_text)
        return DecisionBenchmarkReport.model_validate(data)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Corrupted JSON in report file {file_path}: {exc}") from exc
    except Exception as exc:
        raise ValueError(
            f"Invalid decision benchmark report schema in {file_path}: {exc}"
        ) from exc


def load_latest_decision_report(
    pipeline_name: str | None = None,
    directory: str | Path | None = None,
) -> DecisionBenchmarkReport | None:
    """
    Load the most recently saved DecisionBenchmarkReport for a pipeline (or across all pipelines).

    Args:
        pipeline_name: Optional pipeline name filter (e.g. 'deterministic_rag').
        directory: Optional directory containing reports.

    Returns:
        DecisionBenchmarkReport if found, else None.
    """
    target_dir = get_decision_evaluation_directory(directory)
    if not target_dir.exists():
        return None

    if pipeline_name:
        latest_file = target_dir / f"{pipeline_name}_latest.json"
        if latest_file.exists():
            try:
                return load_decision_report(latest_file)
            except (ValueError, OSError, Exception):  # noqa: BLE001, S110
                pass

        matching_files = sorted(
            [
                p
                for p in target_dir.glob(f"*{pipeline_name}*.json")
                if not p.name.endswith("_latest.json")
            ],
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if matching_files:
            return load_decision_report(matching_files[0], directory=target_dir)
        return None

    # Load any newest report
    all_files = sorted(
        [p for p in target_dir.glob("*.json") if not p.name.endswith("_latest.json")],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not all_files:
        return None

    return load_decision_report(all_files[0])


def list_decision_reports(
    directory: str | Path | None = None,
) -> list[str]:
    """
    List all persisted decision benchmark report identifiers in deterministic alphabetical order.

    Args:
        directory: Optional directory containing reports.

    Returns:
        Sorted list of report base stems.
    """
    target_dir = get_decision_evaluation_directory(directory)
    if not target_dir.exists():
        return []

    return sorted(
        [
            p.stem
            for p in target_dir.glob("*.json")
            if not p.name.endswith("_latest.json")
        ]
    )


def compare_decision_with_baseline(
    current_report: DecisionBenchmarkReport,
    baseline_id_or_path: str | Path | None = None,
    directory: str | Path | None = None,
    thresholds: DecisionQualityThresholds | None = None,
) -> DecisionRegressionComparisonResult:
    """
    Compare a current DecisionBenchmarkReport against a baseline loaded from disk.

    Args:
        current_report: Current benchmark report.
        baseline_id_or_path: Path or ID of baseline (defaults to latest report for this pipeline).
        directory: Storage directory for reports.
        thresholds: Optional quality thresholds defining maximum allowable degradation.

    Returns:
        DecisionRegressionComparisonResult.

    Raises:
        FileNotFoundError: If baseline cannot be found on disk.
    """
    if baseline_id_or_path is not None:
        baseline_rep = load_decision_report(baseline_id_or_path, directory=directory)
    else:
        baseline_rep = load_latest_decision_report(
            pipeline_name=current_report.pipeline_name,
            directory=directory,
        )
        if baseline_rep is None:
            raise FileNotFoundError(
                f"No baseline report found on disk for pipeline '{current_report.pipeline_name}'."
            )

    return compare_decision_runs(
        current_report=current_report,
        baseline_report=baseline_rep,
        thresholds=thresholds,
    )
