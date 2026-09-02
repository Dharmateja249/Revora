"""
Revora Decision Benchmark Runner & CLI Tests.

Comprehensive testing of reproducible decision benchmark execution,
pipeline resolution, artifact generation, CLI integration, and terminal summaries.
"""

import json
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from app.agent.orchestrator import AgentOrchestrator
from app.agent.schemas import AgentDecisionResult, LLMRecoveryRecommendation
from app.decision_engine import RecoveryAction
from app.evaluation.decision_benchmark import (
    format_decision_benchmark_terminal_summary,
    resolve_decision_pipeline,
    run_decision_benchmark,
    run_decision_cli,
)
from app.evaluation.decision_evaluator import (
    AgentRAGPipeline,
    DeterministicBaselinePipeline,
    DeterministicRAGPipeline,
)
from app.evaluation.schemas import DecisionBenchmarkReport
from tests.fixtures.retrieval_golden_dataset import get_golden_evaluation_cases


@pytest.fixture(autouse=True)
def _reset_settings_cache():
    from app.config import get_settings

    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


# =============================================================================
# Pipeline Resolution Tests
# =============================================================================


def test_resolve_decision_pipeline_named_instances():
    p1 = resolve_decision_pipeline("deterministic_baseline")
    assert isinstance(p1, DeterministicBaselinePipeline)
    assert p1.name == "deterministic_baseline"

    p2 = resolve_decision_pipeline("deterministic_rag")
    assert isinstance(p2, DeterministicRAGPipeline)
    assert p2.name == "deterministic_rag"

    p3 = resolve_decision_pipeline("baseline")
    assert isinstance(p3, DeterministicBaselinePipeline)

    p4 = resolve_decision_pipeline("rag")
    assert isinstance(p4, DeterministicRAGPipeline)


def test_resolve_decision_pipeline_agent_orchestrator():
    mock_orch = AsyncMock(spec=AgentOrchestrator)
    p = resolve_decision_pipeline("agent_rag", agent_orchestrator=mock_orch)
    assert isinstance(p, AgentRAGPipeline)
    assert p.name == "agent_rag"


def test_resolve_decision_pipeline_missing_agent_orchestrator_raises_value_error():
    with pytest.raises(ValueError, match="requires an active AgentOrchestrator"):
        resolve_decision_pipeline(
            "agent_rag",
            agent_orchestrator=None,
            allow_default_evaluation_orchestrator=False,
        )


def test_resolve_decision_pipeline_auto_instantiates_evaluation_orchestrator():
    p = resolve_decision_pipeline("agent_rag")
    assert isinstance(p, AgentRAGPipeline)
    assert p.name == "agent_rag"


def test_resolve_decision_pipeline_unknown_name_raises_value_error():
    with pytest.raises(ValueError, match="Unknown decision pipeline identifier"):
        resolve_decision_pipeline("unsupported_pipeline_xyz")


# =============================================================================
# Benchmark Runner Execution & Artifact Tests
# =============================================================================


def test_run_decision_benchmark_single_pipeline(tmp_path: Path):
    cases = get_golden_evaluation_cases()[:5]
    reports = run_decision_benchmark(
        evaluation_cases=cases,
        pipelines=["deterministic_baseline"],
        output_dir=tmp_path,
        save_artifacts=True,
    )

    assert len(reports) == 1
    assert "deterministic_baseline" in reports
    rep = reports["deterministic_baseline"]
    assert isinstance(rep, DecisionBenchmarkReport)
    assert rep.num_queries == 5

    # Verify saved artifacts on disk
    json_files = list(tmp_path.glob("decision_benchmark_*.json"))
    md_files = list(tmp_path.glob("decision_benchmark_*.md"))
    assert len(json_files) == 1
    assert len(md_files) == 1

    content = json.loads(json_files[0].read_text(encoding="utf-8"))
    assert content["pipeline_name"] == "deterministic_baseline"
    assert content["num_queries"] == 5


def test_run_decision_benchmark_multiple_pipelines(tmp_path: Path):
    cases = get_golden_evaluation_cases()[:3]
    reports = run_decision_benchmark(
        evaluation_cases=cases,
        pipelines=["deterministic_baseline", "deterministic_rag"],
        output_dir=tmp_path,
        save_artifacts=True,
    )

    assert len(reports) == 2
    assert "deterministic_baseline" in reports
    assert "deterministic_rag" in reports

    # Check comparative report artifact was saved
    comp_json = tmp_path / "decision_pipeline_comparison.json"
    comp_md = tmp_path / "decision_pipeline_comparison.md"
    assert comp_json.exists()
    assert comp_md.exists()

    parsed_comp = json.loads(comp_json.read_text(encoding="utf-8"))
    assert "deterministic_baseline" in parsed_comp
    assert "deterministic_rag" in parsed_comp


def test_run_decision_benchmark_no_save(tmp_path: Path):
    cases = get_golden_evaluation_cases()[:2]
    reports = run_decision_benchmark(
        evaluation_cases=cases,
        pipelines=["deterministic_baseline"],
        output_dir=tmp_path,
        save_artifacts=False,
    )

    assert len(reports) == 1
    # Verify no files were written to disk
    assert list(tmp_path.glob("*")) == []


def test_run_decision_benchmark_with_agent_orchestrator(tmp_path: Path):
    cases = get_golden_evaluation_cases()[:2]
    mock_orch = AsyncMock(spec=AgentOrchestrator)
    mock_orch.decide.return_value = AgentDecisionResult(
        recommendation=LLMRecoveryRecommendation(
            recommended_action=RecoveryAction.RETRY_PAYMENT,
            confidence=0.9,
            reasoning="Precedent match",
        ),
        agent_used=True,
        provider="mock",
        model_name="mock-model",
        is_fallback=False,
        latency_ms=75.0,
        metadata={},
    )

    reports = run_decision_benchmark(
        evaluation_cases=cases,
        pipelines=["agent_rag"],
        agent_orchestrator=mock_orch,
        output_dir=tmp_path,
        save_artifacts=False,
    )

    assert "agent_rag" in reports
    assert reports["agent_rag"].num_queries == 2


# =============================================================================
# Terminal Summary Tests
# =============================================================================


def test_format_decision_benchmark_terminal_summary():
    cases = get_golden_evaluation_cases()[:2]
    reports = run_decision_benchmark(
        evaluation_cases=cases,
        pipelines=["deterministic_baseline"],
        save_artifacts=False,
    )

    summary_text = format_decision_benchmark_terminal_summary(reports)

    assert "REVORA DECISION BENCHMARK SUMMARY" in summary_text
    assert "deterministic_baseline" in summary_text
    assert "Exact Match" in summary_text
    assert "Acceptable" in summary_text
    assert "Safety Viol" in summary_text
    assert "Latency (ms)" in summary_text


def test_format_decision_benchmark_terminal_summary_empty():
    assert (
        format_decision_benchmark_terminal_summary({})
        == "No decision benchmark reports to display."
    )


# =============================================================================
# CLI Entrypoint Tests
# =============================================================================


def test_run_decision_cli_standard_execution(tmp_path: Path, capsys):
    cases = get_golden_evaluation_cases()[:3]
    exit_code = run_decision_cli(
        args=[
            "--pipeline",
            "deterministic_baseline",
            "--output-dir",
            str(tmp_path),
        ],
        evaluation_cases=cases,
    )

    assert exit_code == 0
    captured = capsys.readouterr()
    assert "REVORA DECISION BENCHMARK SUMMARY" in captured.out
    assert len(list(tmp_path.glob("decision_benchmark_*.json"))) == 1


def test_run_decision_cli_json_and_markdown_flags(tmp_path: Path, capsys):
    cases = get_golden_evaluation_cases()[:2]
    exit_code = run_decision_cli(
        args=[
            "-p",
            "deterministic_baseline",
            "--json",
            "--no-save",
        ],
        evaluation_cases=cases,
    )

    assert exit_code == 0
    captured = capsys.readouterr()
    parsed = json.loads(captured.out)
    assert "deterministic_baseline" in parsed
    assert parsed["deterministic_baseline"]["pipeline_name"] == "deterministic_baseline"


def test_run_decision_cli_invalid_pipeline_returns_exit_code_1(capsys):
    exit_code = run_decision_cli(
        args=["-p", "invalid_pipeline_name", "--no-save"],
        evaluation_cases=get_golden_evaluation_cases()[:1],
    )

    assert exit_code == 1
    captured = capsys.readouterr()
    assert "Error running decision benchmark" in captured.err


def test_resolve_decision_pipeline_openai_provider_with_key(monkeypatch):
    """Verify resolve_decision_pipeline constructs OpenAILLMProvider when requested with key."""
    from app.agent.openai_provider import OpenAILLMProvider

    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-mock-key-12345")
    pipe = resolve_decision_pipeline("agent_rag", llm_provider="openai")

    assert isinstance(pipe, AgentRAGPipeline)
    assert pipe.name == "agent_rag"
    assert isinstance(pipe._orchestrator.provider, OpenAILLMProvider)


def test_resolve_decision_pipeline_openai_agent_rag_name_with_key(monkeypatch):
    """Verify resolve_decision_pipeline supports openai_agent_rag name."""
    from app.agent.openai_provider import OpenAILLMProvider

    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-mock-key-12345")
    pipe = resolve_decision_pipeline("openai_agent_rag")

    assert isinstance(pipe, AgentRAGPipeline)
    assert pipe.name == "openai_agent_rag"
    assert isinstance(pipe._orchestrator.provider, OpenAILLMProvider)


def test_resolve_decision_pipeline_unsupported_llm_provider_raises_value_error():
    """Verify resolve_decision_pipeline raises ValueError for unsupported llm_provider."""
    with pytest.raises(ValueError, match="Unsupported LLM provider: 'opneai'"):
        resolve_decision_pipeline("agent_rag", llm_provider="opneai")


def test_resolve_decision_pipeline_openai_provider_missing_key_fails(monkeypatch):
    """Verify resolve_decision_pipeline raises ValueError if OpenAI key is missing."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    from app.config import get_settings

    settings = get_settings()
    monkeypatch.setattr(settings, "OPENAI_API_KEY", None)

    with pytest.raises(ValueError, match="OpenAI API key must be provided"):
        resolve_decision_pipeline("agent_rag", llm_provider="openai")


def test_resolve_decision_pipeline_mock_default():
    """Verify default mock provider uses EvaluationAgentLLMProvider."""
    from app.evaluation.agent_evaluation_provider import EvaluationAgentLLMProvider

    pipe = resolve_decision_pipeline("agent_rag", llm_provider="mock")

    assert isinstance(pipe, AgentRAGPipeline)
    assert isinstance(pipe._orchestrator.provider, EvaluationAgentLLMProvider)


def test_resolve_decision_pipeline_huggingface_provider_with_token(monkeypatch):
    """Verify resolve_decision_pipeline constructs HuggingFaceLLMProvider when requested with token."""
    from app.agent.huggingface_provider import HuggingFaceLLMProvider

    monkeypatch.setenv("HF_TOKEN", "hf_test_mock_token_12345")
    pipe = resolve_decision_pipeline("agent_rag", llm_provider="huggingface")

    assert isinstance(pipe, AgentRAGPipeline)
    assert pipe.name == "agent_rag"
    assert isinstance(pipe._orchestrator.provider, HuggingFaceLLMProvider)


def test_resolve_decision_pipeline_huggingface_provider_missing_token_fails(monkeypatch):
    """Verify resolve_decision_pipeline raises ValueError if HF_TOKEN is missing."""
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    from app.config import get_settings

    settings = get_settings()
    monkeypatch.setattr(settings, "HF_TOKEN", None)

    with pytest.raises(ValueError, match="Hugging Face API token must be provided"):
        resolve_decision_pipeline("agent_rag", llm_provider="huggingface")


def test_run_decision_cli_with_llm_provider_mock(capsys):
    """Verify running decision CLI with --llm-provider mock succeeds offline."""
    cases = get_golden_evaluation_cases()[:2]
    exit_code = run_decision_cli(
        args=[
            "-p",
            "agent_rag",
            "--llm-provider",
            "mock",
            "--no-save",
            "--quiet",
        ],
        evaluation_cases=cases,
    )

    assert exit_code == 0


def test_run_decision_cli_with_llm_provider_openai_missing_key_fails(
    monkeypatch, capsys
):
    """Verify running decision CLI with --llm-provider openai fails cleanly when key missing."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    from app.config import get_settings

    settings = get_settings()
    monkeypatch.setattr(settings, "OPENAI_API_KEY", None)

    cases = get_golden_evaluation_cases()[:1]
    exit_code = run_decision_cli(
        args=[
            "-p",
            "agent_rag",
            "--llm-provider",
            "openai",
            "--no-save",
        ],
        evaluation_cases=cases,
    )

    assert exit_code == 1
    captured = capsys.readouterr()
    assert "OpenAI API key must be provided" in captured.err
