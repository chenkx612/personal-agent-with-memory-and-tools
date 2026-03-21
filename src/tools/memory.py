"""Memory-related tools for user profile and preferences."""

import os
import json
from langchain_core.tools import tool
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from tools.base import (
    MEMORY_FILE,
    MEMORY_FAISS_DIR,
    MEMORY_FAISS_MTIME_FILE,
    _get_embeddings,
)

# Global cache for vector store
_vectorstore_cache = None
_last_memory_mtime = 0


def _load_memory():
    """Load memory from JSON file."""
    if not os.path.exists(MEMORY_FILE):
        return {}
    try:
        with open(MEMORY_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _save_memory(memory):
    """Save memory to JSON file."""
    with open(MEMORY_FILE, "w", encoding="utf-8") as f:
        json.dump(memory, f, ensure_ascii=False, indent=2)


def _save_vectorstore(vectorstore, memory_mtime):
    """Save FAISS index to disk along with memory mtime."""
    if not os.path.exists(MEMORY_FAISS_DIR):
        os.makedirs(MEMORY_FAISS_DIR, exist_ok=True)
    vectorstore.save_local(MEMORY_FAISS_DIR)
    with open(MEMORY_FAISS_MTIME_FILE, "w") as f:
        f.write(str(memory_mtime))


def _load_vectorstore():
    """Load FAISS index from disk if it exists and is up-to-date."""
    if not os.path.exists(MEMORY_FAISS_DIR) or not os.path.exists(MEMORY_FAISS_MTIME_FILE):
        return None

    if not os.path.exists(MEMORY_FILE):
        return None

    # Check if index is up-to-date
    current_mtime = os.path.getmtime(MEMORY_FILE)
    try:
        with open(MEMORY_FAISS_MTIME_FILE, "r") as f:
            saved_mtime = float(f.read().strip())
        if saved_mtime != current_mtime:
            return None
    except Exception:
        return None

    # Load the index
    try:
        embeddings = _get_embeddings()
        return FAISS.load_local(MEMORY_FAISS_DIR, embeddings, allow_dangerous_deserialization=True)
    except Exception:
        return None


def _get_vectorstore():
    """Get memory vector store: memory cache → disk → full build."""
    global _vectorstore_cache, _last_memory_mtime

    if not os.path.exists(MEMORY_FILE):
        return None

    current_mtime = os.path.getmtime(MEMORY_FILE)

    # If memory cache is valid, return it
    if _vectorstore_cache is not None and current_mtime == _last_memory_mtime:
        return _vectorstore_cache

    # Try to load from disk first
    loaded_vectorstore = _load_vectorstore()
    if loaded_vectorstore is not None:
        _vectorstore_cache = loaded_vectorstore
        _last_memory_mtime = current_mtime
        return _vectorstore_cache

    # Rebuild index from scratch
    memory = _load_memory()
    if not memory:
        return None

    documents = []
    for key, value in memory.items():
        doc = Document(
            page_content=f"{key}: {value}",
            metadata={"key": key}
        )
        documents.append(doc)

    if not documents:
        return None

    embeddings = _get_embeddings()
    _vectorstore_cache = FAISS.from_documents(documents, embeddings)
    _last_memory_mtime = current_mtime

    # Save to disk for future runs
    _save_vectorstore(_vectorstore_cache, current_mtime)

    return _vectorstore_cache


@tool
def search_memory(query: str, k: int = 3):
    """Search the user's long-term memory for relevant information (user profile/preferences only).

    Use this tool when you need to recall stable attributes about the user (preferences, habits,
    personal info). Do NOT use this to find notes or recorded content — use search_notes instead.

    Args:
        query: The search query (e.g., "user's favorite food", "birthday").
        k: Number of results to return (default: 3).
    """
    vectorstore = _get_vectorstore()
    if not vectorstore:
        return "Memory is empty."

    docs = vectorstore.similarity_search(query, k=k)
    if not docs:
        return "No relevant information found in memory."

    PREVIEW_LEN = 20
    results = []
    for doc in docs:
        key = doc.metadata["key"]
        value = doc.page_content[len(key) + 2:]  # strip "key: " prefix
        if len(value) <= PREVIEW_LEN:
            results.append(f"{key}: {value}")
        else:
            results.append(f"{key}: {value[:PREVIEW_LEN]}...  (truncated, use get_memory to read full value)")

    return "\n".join(results)


@tool
def update_user_memory(key: str, value: str, overwrite_confirmed: bool = False):
    """Update or add a stable attribute about the user in long-term memory (user profile only).

    仅用于存储用户的稳定属性和偏好（如姓名、生日、饮食偏好、兴趣爱好）。
    若用户想记录具体内容、想法、事件或文章要点，请使用 add_note 工具。

    格式规范：
    - key: 使用简洁的分类标签，如"姓名"、"生日"、"职业"、"饮食偏好"、"兴趣爱好"
    - value: 使用简洁的陈述句或列表，避免冗长解释
    - 同类信息用同一个 key，更新时整合已有内容，避免重复 key

    更新已有 key 的流程：
    1. 首次调用（不设 overwrite_confirmed）时，若 key 已存在，会返回现有内容
    2. 将现有内容与新信息合并后，再次调用并设 overwrite_confirmed=True

    示例：
    - 好：key="饮食偏好", value="不吃香菜；喜欢辣；偏好清淡"
    - 差：key="用户不喜欢吃香菜", value="用户在2024年1月告诉我..."

    Args:
        key: The category or key of the information.
        value: The information to store.
        overwrite_confirmed: Must be True when updating an existing key. Set this only after
            reading the existing value and merging it with the new information.
    """
    memory = _load_memory()
    if key in memory and not overwrite_confirmed:
        existing = memory[key]
        return (
            f"⚠️ Key '{key}' already exists with content:\n{existing}\n\n"
            f"Please merge the above with your new information, then call again with "
            f"the merged value and overwrite_confirmed=True."
        )
    memory[key] = value
    _save_memory(memory)
    return f"Successfully updated memory: {key} = {value}"


@tool
def get_memory(keys: list[str]):
    """Retrieve full values for one or more memory keys.

    Use this after search_memory when a result is marked as truncated and you need the complete value.

    Args:
        keys: List of memory keys to retrieve (e.g., ["饮食偏好", "健康状况"]).
    """
    memory = _load_memory()
    results = []
    for key in keys:
        if key in memory:
            results.append(f"{key}: {memory[key]}")
        else:
            results.append(f"{key}: (not found)")
    return "\n".join(results)
