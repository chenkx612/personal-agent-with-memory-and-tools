"""Core agent initialization logic."""

import os
import sqlite3
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver

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


def get_agent_executor(checkpointer=None):
    """Create and return the agent executor and checkpointer.

    Returns a compiled StateGraph with:
    - ReAct pattern (agent ↔ tools loop)
    - Memory tools for long-term user memory
    - SqliteSaver checkpointer for conversation persistence

    Args:
        checkpointer: Optional custom checkpointer. If None, creates SqliteSaver
            with default database path (data/checkpoints.db).

    Returns:
        Tuple of (agent_executor, checkpointer)
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

    if checkpointer is None:
        db_path = os.path.join("data", "checkpoints.db")
        os.makedirs("data", exist_ok=True)
        conn = sqlite3.connect(db_path, check_same_thread=False)
        checkpointer = SqliteSaver(conn)

    agent = build_agent_graph(
        llm=llm,
        tools=tools,
        system_prompt=config["system_prompt"],
        checkpointer=checkpointer,
    )

    return agent, checkpointer
