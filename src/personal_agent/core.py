import os
import yaml

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from .graph import build_agent_graph
from .tools import (
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

# Default system prompt (fallback if config.yaml not found)
_DEFAULT_SYSTEM_PROMPT = """你是我（用户）的专属个人秘书。

你的核心性格：
- 优雅：言谈举止得体，令人愉悦，保持专业风度。
- 聪明：反应敏捷，逻辑清晰，能提供高质量的见解。
- 干练：办事效率高，不拖泥带水，专业可靠。
- 温柔：态度友善，耐心细致，充满人文关怀。
"""


def load_config():
    """Load configuration from config.yaml.

    Returns:
        dict: Configuration dictionary with system_prompt and other settings.
    """
    config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "config.yaml")
    config = {}

    if os.path.exists(config_path):
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f) or {}
        except Exception as e:
            print(f"Warning: Failed to load config.yaml: {e}")

    # Set defaults
    if "system_prompt" not in config:
        config["system_prompt"] = _DEFAULT_SYSTEM_PROMPT

    return config


def get_llm():
    """Get LLM instance for standalone use (e.g., memory tidying)."""
    api_key = os.getenv("API_KEY")
    model = os.getenv("MODEL")
    base_url = os.getenv("BASE_URL")

    return ChatOpenAI(
        model=model,
        api_key=api_key,
        base_url=base_url,
    )


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
