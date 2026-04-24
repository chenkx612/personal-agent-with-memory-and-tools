"""Core agent initialization logic."""

import os
import sqlite3
import yaml
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver

from graph import build_agent_graph


def load_config():
    """Load configuration from .config.yaml.

    Returns:
        dict: Configuration dictionary with all settings.

    Raises:
        FileNotFoundError: If .config.yaml is not found.
    """
    config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".config.yaml")

    if not os.path.exists(config_path):
        raise FileNotFoundError(
            f"配置文件不存在: {config_path}\n"
            "请复制 .config.yaml.template 为 .config.yaml 并填入你的配置"
        )

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}

    return config
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
from llm import get_llm


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

    llm = get_llm()

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

        max_sessions = config.get("checkpoint_max_sessions", 10)
        if max_sessions and max_sessions > 0:
            cur = conn.cursor()
            cur.execute("""
                DELETE FROM checkpoints
                WHERE thread_id NOT IN (
                    SELECT thread_id FROM (
                        SELECT thread_id, MAX(rowid) as latest
                        FROM checkpoints
                        GROUP BY thread_id
                        ORDER BY latest DESC
                        LIMIT ?
                    )
                )
            """, (max_sessions,))
            conn.commit()
            conn.execute("VACUUM")

    agent = build_agent_graph(
        llm=llm,
        tools=tools,
        system_prompt=config["system_prompt"],
        checkpointer=checkpointer,
    )

    return agent, checkpointer
