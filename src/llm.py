"""LLM initialization for the personal agent."""

from langchain_openai import ChatOpenAI
from core import load_config


def get_llm():
    """Get LLM instance for standalone use (e.g., memory tidying)."""
    config = load_config()
    llm_config = config.get("llm", {})

    return ChatOpenAI(
        model=llm_config.get("model"),
        api_key=llm_config.get("api_key"),
        base_url=llm_config.get("base_url"),
    )
