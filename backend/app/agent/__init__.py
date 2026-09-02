"""
Revora Adaptive Recovery Agent Package.

Provides domain contracts, prompt context builders, LLM providers, and agent
decision orchestrators for adaptive failed payment recovery.
"""

from app.agent.context_builder import AgentContextBuilder
from app.agent.factory import create_llm_provider
from app.agent.openai_provider import (
    OpenAILLMProvider,
    RealLLMProvider,
)
from app.agent.orchestrator import AgentOrchestrator
from app.agent.prompts import (
    REVORA_AGENT_SYSTEM_PROMPT,
    build_agent_messages,
)
from app.agent.provider import (
    LLMAuthenticationError,
    LLMConnectionError,
    LLMProvider,
    LLMProviderConfigurationError,
    LLMProviderError,
    LLMRateLimitError,
    LLMResponseValidationError,
    LLMTimeoutError,
    MockLLMProvider,
)
from app.agent.schemas import (
    AgentDecisionPromptContext,
    AgentDecisionResult,
    LLMRecoveryRecommendation,
)

__all__ = [
    "REVORA_AGENT_SYSTEM_PROMPT",
    "AgentContextBuilder",
    "AgentDecisionPromptContext",
    "AgentDecisionResult",
    "AgentOrchestrator",
    "LLMAuthenticationError",
    "LLMConnectionError",
    "LLMProvider",
    "LLMProviderConfigurationError",
    "LLMProviderError",
    "LLMRateLimitError",
    "LLMRecoveryRecommendation",
    "LLMResponseValidationError",
    "LLMTimeoutError",
    "MockLLMProvider",
    "OpenAILLMProvider",
    "RealLLMProvider",
    "build_agent_messages",
    "create_llm_provider",
]
