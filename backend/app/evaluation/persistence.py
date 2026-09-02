"""
Revora Retrieval Evaluation Persistence.

Provides filesystem-based JSON storage, retrieval, atomic write operations,
and historical benchmark tracking for EvaluationReport artifacts.
"""

import json
import os
import tempfile
from pathlib import Path
from typing import Any

from app.evaluation.schemas import EvaluationReport

DEFAULT_EVALUATION_DIR = (
    Path(__file__).resolve().parent.parent.parent / "evaluation_results"
)


def get_evaluation_directory(custom_dir: str | Path | None = None) -> Path:
    """Resolve and ensure the target directory for persisting evaluation reports."""
    path = Path(custom_dir) if custom_dir is not None else DEFAULT_EVALUATION_DIR
    path.mkdir(parents=True, exist_ok=True)
    return path


def _atomic_write_json(file_path: Path, data: dict[str, Any]) -> None:
    """Atomically write JSON payload to file using a temporary file in the same directory."""
    parent_dir = file_path.parent
    parent_dir.mkdir(parents=True, exist_ok=True)

    json_content = json.dumps(data, indent=2, sort_keys=True)

    # Create temporary file in the same filesystem/directory to ensure atomic rename.
    # On Windows, atomic os.replace requires closing the file handle first.
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
        # Atomic replace
        os.replace(temp_file.name, file_path)
    except Exception:
        try:
            temp_file.close()
        except Exception:
            pass
        if os.path.exists(temp_file.name):
            try:
                os.remove(temp_file.name)
            except Exception:
                pass
        raise


def save_report(
    report: EvaluationReport,
    directory: str | Path | None = None,
    overwrite: bool = False,
) -> Path:
    """
    Persist an EvaluationReport to disk atomically as JSON.

    Args:
        report: EvaluationReport instance to persist.
        directory: Target storage directory (defaults to backend/evaluation_results).
        overwrite: Whether to allow overwriting an existing report with the same report_id.

    Returns:
        Path to the saved report file.

    Raises:
        FileExistsError: If report with report_id already exists and overwrite is False.
    """
    if not isinstance(report, EvaluationReport):
        raise TypeError(f"report must be EvaluationReport, got {type(report).__name__}")

    target_dir = get_evaluation_directory(directory)
    file_path = target_dir / f"{report.report_id}.json"

    if file_path.exists() and not overwrite:
        raise FileExistsError(
            f"Evaluation report with id '{report.report_id}' already exists at {file_path}. Use overwrite=True to replace."
        )

    # Serialize using Pydantic model_dump
    payload = report.model_dump(mode="json")
    _atomic_write_json(file_path, payload)

    # Update latest.json symlink/copy atomically
    latest_path = target_dir / "latest.json"
    _atomic_write_json(latest_path, payload)

    return file_path


def load_report(
    report_id: str,
    directory: str | Path | None = None,
) -> EvaluationReport:
    """
    Load an EvaluationReport by its report_id.

    Args:
        report_id: Non-empty identifier of the report to load.
        directory: Storage directory containing reports.

    Returns:
        Reconstituted EvaluationReport instance.

    Raises:
        FileNotFoundError: If the report file does not exist.
        ValueError: If file is corrupted or contains invalid schema data.
    """
    if not isinstance(report_id, str) or not report_id.strip():
        raise ValueError("report_id must be a non-empty string.")

    target_dir = get_evaluation_directory(directory)
    clean_id = report_id.strip()
    if not clean_id.endswith(".json"):
        clean_id = f"{clean_id}.json"

    file_path = target_dir / clean_id
    if not file_path.exists():
        raise FileNotFoundError(
            f"Evaluation report '{report_id}' not found at {file_path}"
        )

    try:
        raw_text = file_path.read_text(encoding="utf-8")
        data = json.loads(raw_text)
        return EvaluationReport.model_validate(data)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Corrupted JSON in report file {file_path}: {exc}") from exc
    except Exception as exc:
        raise ValueError(
            f"Invalid evaluation report schema in {file_path}: {exc}"
        ) from exc


def load_latest_report(
    directory: str | Path | None = None,
) -> EvaluationReport | None:
    """
    Load the most recently persisted EvaluationReport (latest.json).

    Returns:
        EvaluationReport if found, or None if no reports have been persisted.
    """
    target_dir = get_evaluation_directory(directory)
    latest_path = target_dir / "latest.json"

    if latest_path.exists():
        try:
            return load_report("latest", directory=target_dir)
        except (ValueError, OSError, Exception):  # noqa: BLE001, S110
            pass

    # Fallback to sorting all report JSONs by modification time
    report_files = sorted(
        [p for p in target_dir.glob("*.json") if p.name != "latest.json"],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not report_files:
        return None

    return load_report(report_files[0].stem, directory=target_dir)


def list_reports(
    directory: str | Path | None = None,
) -> list[str]:
    """
    List all persisted report IDs in deterministic alphabetical order.
    """
    target_dir = get_evaluation_directory(directory)
    if not target_dir.exists():
        return []

    report_ids = [p.stem for p in target_dir.glob("*.json") if p.name != "latest.json"]
    return sorted(report_ids)
