"""LLM initialization for the personal agent."""

from langchain_openai import ChatOpenAI
from core import load_config


def get_llm():
    """Get LLM instance for standalone use (e.g., memory tidying)."""
    config = load_config()
    llm_config = config.get("llm", {})

    api_key = llm_config.get("api_key")
    model = llm_config.get("model")
    base_url = llm_config.get("base_url")
    reasoning_effort = llm_config.get("reasoning_effort", "high")
    thinking_type = llm_config.get("thinking_type", "disabled")

    if not api_key:
        print("Warning: api_key not found in config.yaml.")

    return ChatOpenAI(
        model=model,
        api_key=api_key,
        base_url=base_url,
        reasoning_effort=reasoning_effort,
        extra_body={"thinking": {"type": thinking_type}}
    )
