"""Tools package for the personal agent.

This package contains all the tools available to the agent, organized by category.
"""

from tools.base import (
    BASE_DIR,
    MEMORY_FILE,
    MEMORY_FAISS_DIR,
    NOTES_FILE,
    NOTES_FAISS_DIR,
    _get_embeddings,
)
from tools.memory import (
    _load_memory,
    _save_memory,
    _get_vectorstore,
    search_memory,
    update_user_memory,
    get_memory,
)
from tools.notes import (
    _load_notes,
    _save_notes,
    _get_notes_vectorstore,
    _notes_vectorstore_cache,
    add_note,
    search_notes,
    get_note,
)
from tools.environment import (
    get_environment_context,
)
from tools.web import (
    web_search,
)

__all__ = [
    # Base
    "BASE_DIR",
    "MEMORY_FILE",
    "MEMORY_FAISS_DIR",
    "NOTES_FILE",
    "NOTES_FAISS_DIR",
    "_get_embeddings",
    # Memory
    "_load_memory",
    "_save_memory",
    "_get_vectorstore",
    "search_memory",
    "update_user_memory",
    "get_memory",
    # Notes
    "_load_notes",
    "_save_notes",
    "_get_notes_vectorstore",
    "_notes_vectorstore_cache",
    "add_note",
    "search_notes",
    "get_note",
    # Environment
    "get_environment_context",
    # Web
    "web_search",
]
