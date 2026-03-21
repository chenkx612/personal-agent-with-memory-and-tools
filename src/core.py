"""Core agent initialization logic."""

from langchain_openai import ChatOpenAI

from graph import build_agent_graph
from config import load_config
from tools import (
    get_environment_context,
    search_memory,
    get_memory,
    update_user_memory,
    web_search,
    add_note,
    search_notes,
    get_note,
)


def get_agent_executor():
    """Create and return the agent executor.

    Returns a compiled StateGraph with:
    - ReAct pattern (agent ↔ tools loop)
    - Memory tools for long-term user memory
    - MemorySaver checkpointer for conversation persistence
    """
    config = load_config()
    llm_config = config.get("llm", {})

    api_key = llm_config.get("api_key")
    model = llm_config.get("model")
    base_url = llm_config.get("base_url")
    temperature = llm_config.get("temperature", 0.7)

    if not api_key:
        print("Warning: api_key not found in config.yaml.")

    llm = ChatOpenAI(
        model=model,
        api_key=api_key,
        base_url=base_url,
        temperature=temperature,
    )

    tools = [
        get_environment_context,
        update_user_memory,
        search_memory,
        get_memory,
        web_search,
        add_note,
        search_notes,
        get_note,
    ]

    return build_agent_graph(
        llm=llm,
        tools=tools,
        system_prompt=config["system_prompt"],
    )
