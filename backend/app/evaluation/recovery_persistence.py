"""
Revora Recovery Outcome Benchmark Persistence.

Provides filesystem-based JSON storage, retrieval, atomic updates, and latest-run tracking
for RecoveryBenchmarkReport artifacts.
"""

import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from app.evaluation.persistence import get_evaluation_directory
from app.evaluation.recovery_schemas import RecoveryBenchmarkReport


def get_recovery_evaluation_directory(
    custom_dir: str | Path | None = None,
) -> Path:
    """Resolve and ensure the target directory for persisting recovery outcome reports."""
    if custom_dir is not None:
        path = Path(custom_dir).resolve()
    else:
        path = (get_evaluation_directory() / "recovery_outcomes").resolve()
    path.mkdir(parents=True, exist_ok=True)
    return path


def _atomic_write_json(file_path: Path, data: dict[str, Any]) -> None:
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


def save_recovery_report(
    report: RecoveryBenchmarkReport,
    directory: str | Path | None = None,
    report_id: str | None = None,
    overwrite: bool = False,
) -> Path:
    """
    Persist a RecoveryBenchmarkReport to disk atomically as JSON.

    Args:
        report: RecoveryBenchmarkReport instance to persist.
        directory: Target directory.
        report_id: Optional unique report identifier.
        overwrite: Whether to overwrite existing file.

    Returns:
        Path to the saved JSON report file.
    """
    if not isinstance(report, RecoveryBenchmarkReport):
        raise TypeError(
            f"report must be RecoveryBenchmarkReport, got {type(report).__name__}"
        )

    target_dir = get_recovery_evaluation_directory(directory)
    timestamp_str = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    clean_id = (
        report_id
        or f"recovery_{report.pipeline_name}_{timestamp_str}_{uuid4().hex[:6]}"
    )
    if not clean_id.endswith(".json"):
        clean_id = f"{clean_id}.json"

    file_path = target_dir / clean_id
    if file_path.exists() and not overwrite:
        raise FileExistsError(
            f"Recovery report '{clean_id}' already exists at {file_path}. Use overwrite=True to replace."
        )

    payload = report.model_dump(mode="json")
    _atomic_write_json(file_path, payload)

    # Maintain latest report pointer
    latest_path = target_dir / f"{report.pipeline_name}_latest.json"
    _atomic_write_json(latest_path, payload)

    return file_path


def load_recovery_report(
    report_id_or_path: str | Path,
    directory: str | Path | None = None,
) -> RecoveryBenchmarkReport:
    """
    Load a RecoveryBenchmarkReport by ID or filepath.

    Args:
        report_id_or_path: String ID, filename, or Path to JSON report.
        directory: Optional directory containing reports.

    Returns:
        Reconstituted RecoveryBenchmarkReport instance.
    """
    target_dir = get_recovery_evaluation_directory(directory)
    path_obj = Path(report_id_or_path)

    if path_obj.is_file():
        file_path = path_obj.resolve()
    elif (target_dir / path_obj).is_file():
        file_path = (target_dir / path_obj).resolve()
    else:
        name = str(report_id_or_path).strip()
        if not name.endswith(".json"):
            name = f"{name}.json"
        file_path = (target_dir / name).resolve()

    if not file_path.exists():
        raise FileNotFoundError(f"Recovery benchmark report not found at: {file_path}")

    try:
        raw_text = file_path.read_text(encoding="utf-8")
        data = json.loads(raw_text)
        return RecoveryBenchmarkReport.model_validate(data)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Corrupted JSON in report file {file_path}: {exc}") from exc
    except Exception as exc:
        raise ValueError(
            f"Invalid recovery benchmark report schema in {file_path}: {exc}"
        ) from exc


def load_latest_recovery_report(
    pipeline_name: str | None = None,
    directory: str | Path | None = None,
    exclude_report_id: str | None = None,
) -> RecoveryBenchmarkReport | None:
    """
    Load the most recently saved RecoveryBenchmarkReport for a pipeline (or across all pipelines).
    """
    target_dir = get_recovery_evaluation_directory(directory)
    if not target_dir.exists():
        return None

    if pipeline_name:
        latest_file = target_dir / f"{pipeline_name}_latest.json"
        if latest_file.exists():
            try:
                loaded = load_recovery_report(latest_file, directory=target_dir)
                if exclude_report_id is None or loaded.report_id != exclude_report_id:
                    return loaded
            except (ValueError, OSError, Exception):  # noqa: BLE001, S110
                pass

        matching = sorted(
            [
                p
                for p in target_dir.glob("*.json")
                if not p.name.endswith("_latest.json")
            ],
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        for candidate_file in matching:
            try:
                loaded = load_recovery_report(candidate_file, directory=target_dir)
                if loaded.pipeline_name != pipeline_name:
                    continue
                if exclude_report_id is None or loaded.report_id != exclude_report_id:
                    return loaded
            except (ValueError, OSError):
                continue
        return None

    all_files = sorted(
        [p for p in target_dir.glob("*.json") if not p.name.endswith("_latest.json")],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    for candidate_file in all_files:
        try:
            loaded = load_recovery_report(candidate_file, directory=target_dir)
            if exclude_report_id is None or loaded.report_id != exclude_report_id:
                return loaded
        except (ValueError, OSError):
            continue

    return None


def list_recovery_reports(
    directory: str | Path | None = None,
) -> list[str]:
    """List all saved recovery report identifiers in alphabetical order."""
    target_dir = get_recovery_evaluation_directory(directory)
    if not target_dir.exists():
        return []

    return sorted(
        [
            p.stem
            for p in target_dir.glob("*.json")
            if not p.name.endswith("_latest.json")
        ]
    )
