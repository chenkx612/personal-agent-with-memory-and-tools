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


def load_config():
    """Load configuration from config.yaml.

    Returns:
        dict: Configuration dictionary with system_prompt and other settings.
    """
    config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config.yaml")
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
