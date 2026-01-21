import json
import os
from datetime import datetime
from typing import Optional
from langchain_core.tools import tool

# Memory file path
# Use absolute path relative to project root
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MEMORY_FILE = os.path.join(BASE_DIR, "user_memory.json")

def _load_memory():
    if not os.path.exists(MEMORY_FILE):
        return {}
    try:
        with open(MEMORY_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}

def _save_memory(memory):
    with open(MEMORY_FILE, "w", encoding="utf-8") as f:
        json.dump(memory, f, ensure_ascii=False, indent=2)

@tool
def get_current_time():
    """Get the current time and date."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

@tool
def get_weather(location: str):
    """Get the current weather for a specific location.
    
    Args:
        location: The name of the city or region (e.g., "Beijing", "New York").
    """
    # Mock implementation for demonstration
    # In a real app, this would call a weather API
    return f"The weather in {location} is currently sunny, 25°C."

@tool
def update_user_memory(key: str, value: str):
    """Update or add a piece of information about the user in long-term memory.
    
    Args:
        key: The category or key of the information (e.g., "name", "preference", "hobby").
        value: The detailed information to store.
    """
    memory = _load_memory()
    memory[key] = value
    _save_memory(memory)
    return f"Successfully updated memory: {key} = {value}"

@tool
def get_user_memory(key: Optional[str] = None):
    """Retrieve information from the user's long-term memory.
    
    Args:
        key: The specific key to retrieve. If None, returns all memory.
    """
    memory = _load_memory()
    if key:
        return memory.get(key, "Information not found.")
    return json.dumps(memory, ensure_ascii=False)
