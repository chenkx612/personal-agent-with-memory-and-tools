"""Core agent initialization logic."""

import os
from dotenv import load_dotenv
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

# Load environment variables
load_dotenv()


def get_agent_executor():
    """Create and return the agent executor.

    Returns a compiled StateGraph with:
    - ReAct pattern (agent ↔ tools loop)
    - Memory tools for long-term user memory
    - MemorySaver checkpointer for conversation persistence
    """
    config = load_config()

    api_key = os.getenv("API_KEY")
    model = config.get("llm_model") or os.getenv("MODEL")
    base_url = os.getenv("BASE_URL")

    if not api_key:
        print("Warning: API_KEY not found in environment variables.")

    llm_kwargs = {
        "model": model,
        "api_key": api_key,
        "base_url": base_url,
    }
    if "llm_temperature" in config:
        llm_kwargs["temperature"] = config["llm_temperature"]

    llm = ChatOpenAI(**llm_kwargs)

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
