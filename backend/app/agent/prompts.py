"""
Revora Agent Prompts.

Defines provider-independent system instructions and prompt assembly utilities
for the Adaptive Recovery Agent.
"""

import json

from app.agent.schemas import AgentDecisionPromptContext

REVORA_AGENT_SYSTEM_PROMPT = """You are Revora's Adaptive Recovery Decision Engine.
Your objective is to analyze a failed payment recovery scenario and recommend the most effective recovery action to recover revenue while preserving customer trust.

OPERATIONAL INVARIANTS & POLICY CONSTRAINTS:
1. You MUST select your `recommended_action` exclusively from the list of ALLOWED ACTIONS provided in the policy envelope.
2. You MUST NEVER recommend any action listed in PROHIBITED ACTIONS.
3. Historical cases are empirical supporting evidence, NOT direct operational instructions. If a historical case succeeded with an action that is currently prohibited, the active policy constraint WINS.
4. An `attempt_budget` indicates `current_attempt`, `max_attempts`, and `remaining_attempts`. Factor this into your recommendation (e.g. consider escalating or switching recovery channels as attempts run low), but remember that deterministic policy rules remain strictly authoritative.
5. Provide structured, factual reasoning explaining your decision based on customer profile, current failure reason, and historical evidence.
6. Provide concise `key_factors` highlighting key decision drivers and cite relevant `referenced_case_ids` if historical evidence informed your choice.
7. Your recommendation will be deterministically validated against hard safety and payment provider policies before execution.

OUTPUT CONTRACT:
You must respond with a valid JSON object matching the following structure:
{
  "recommended_action": "<allowed_action_name>",
  "confidence": <float between 0.0 and 1.0>,
  "reasoning": "<concise explanation for the choice>",
  "key_factors": ["<factor1>", "<factor2>"],
  "referenced_case_ids": ["<case_id1>", "<case_id2>"]
}"""


def build_agent_messages(
    prompt_context: AgentDecisionPromptContext,
) -> list[dict[str, str]]:
    """
    Format a sanitized AgentDecisionPromptContext into standard chat messages.

    Args:
        prompt_context: Validated AgentDecisionPromptContext instance.

    Returns:
        List of chat messages containing system instructions and serialized user prompt context.
    """
    if not isinstance(prompt_context, AgentDecisionPromptContext):
        raise TypeError(
            f"Expected AgentDecisionPromptContext, got {type(prompt_context).__name__}"
        )

    if not prompt_context.allowed_actions:
        raise ValueError(
            "AgentDecisionPromptContext must contain at least one allowed action"
        )

    context_dict = prompt_context.model_dump(mode="json")
    user_payload_dict = {
        "current_payment": context_dict.get("current_payment", {}),
        "customer_recovery_profile": context_dict.get("customer_profile", {}),
        "recent_payment_behavior": context_dict.get("recent_payment_behavior", []),
        "attempt_budget": context_dict.get("attempt_budget", {}),
        "prior_attempts_on_this_payment": context_dict.get(
            "recovery_attempt_history", []
        ),
        "retrieved_historical_evidence": context_dict.get("historical_cases", []),
        "policy_envelope": {
            "allowed_actions": context_dict.get("allowed_actions", []),
            "prohibited_actions": context_dict.get("prohibited_actions", []),
            "mandatory_fallback": context_dict.get("mandatory_fallback"),
            "active_policy_constraints": context_dict.get("policy_constraints", []),
        },
    }

    formatted_json = json.dumps(user_payload_dict, indent=2, sort_keys=True)
    user_prompt = (
        f"Analyze the following failed payment recovery scenario and recommend the optimal recovery action:\n\n"
        f"```json\n{formatted_json}\n```\n\n"
        f"Respond ONLY with the JSON object conforming to the required output contract."
    )

    return [
        {
            "role": "system",
            "content": REVORA_AGENT_SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": user_prompt,
        },
    ]
