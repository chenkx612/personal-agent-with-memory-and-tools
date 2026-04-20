"""Configuration loading for the personal agent."""

import os
import yaml


# Default system prompt (fallback if config.yaml not found)
_DEFAULT_SYSTEM_PROMPT = """你是我（用户）的专属个人秘书。

你的核心性格：
- 优雅：言谈举止得体，令人愉悦，保持专业风度。
- 聪明：反应敏捷，逻辑清晰，能提供高质量的见解。
- 干练：办事效率高，不拖泥带水，专业可靠。
- 温柔：态度友善，耐心细致，充满人文关怀。
"""

# Default configuration
_DEFAULT_CONFIG = {
    "llm": {
        "api_key": None,
        "model": "deepseek-chat",
        "base_url": "https://api.deepseek.com",
        "temperature": 0.7,
    },
    "stream_output": True,
    "system_prompt": _DEFAULT_SYSTEM_PROMPT,
    "checkpoint_max_sessions": 10,
}


def load_config():
    """Load configuration from config.yaml.

    Returns:
        dict: Configuration dictionary with all settings.
    """
    config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".config.yaml")
    config = {}

    if os.path.exists(config_path):
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f) or {}
        except Exception as e:
            print(f"Warning: Failed to load config.yaml: {e}")

    # Merge with defaults
    merged = _deep_merge(_DEFAULT_CONFIG, config)

    # Set HuggingFace endpoint if configured
    if merged.get("hf_endpoint"):
        os.environ["HF_ENDPOINT"] = merged["hf_endpoint"]

    return merged


def _deep_merge(default, override):
    """Deep merge two dictionaries."""
    result = default.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result
