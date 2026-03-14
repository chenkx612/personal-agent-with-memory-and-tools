import json
import os
import logging
from datetime import datetime
from langchain_core.tools import tool
from langchain_community.vectorstores import FAISS
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

# Suppress HuggingFace transformers warnings
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)

# Suppress std/stderr output from sentence_transformers
import sys
import io
import contextlib

@contextlib.contextmanager
def suppress_stdout_stderr():
    """Context manager to suppress stdout and stderr."""
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    try:
        sys.stdout = io.StringIO()
        sys.stderr = io.StringIO()
        yield
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr

# Memory file path
# Use absolute path relative to project root
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MEMORY_FILE = os.path.join(BASE_DIR, "data", "user_memory.json")
FAISS_INDEX_DIR = os.path.join(BASE_DIR, "data", "faiss_index")
FAISS_INDEX_MTIME_FILE = os.path.join(FAISS_INDEX_DIR, "memory_mtime.txt")

# Global cache for vector store and embeddings
_vectorstore_cache = None
_embeddings_cache = None
_last_memory_mtime = 0

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

def _get_embeddings():
    global _embeddings_cache
    if _embeddings_cache is None:
        # Use a small, fast local model
        with suppress_stdout_stderr():
            _embeddings_cache = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return _embeddings_cache

def _save_vectorstore(vectorstore, memory_mtime):
    """Save FAISS index to disk along with memory mtime."""
    if not os.path.exists(FAISS_INDEX_DIR):
        os.makedirs(FAISS_INDEX_DIR, exist_ok=True)
    vectorstore.save_local(FAISS_INDEX_DIR)
    with open(FAISS_INDEX_MTIME_FILE, "w") as f:
        f.write(str(memory_mtime))

def _load_vectorstore():
    """Load FAISS index from disk if it exists and is up-to-date."""
    if not os.path.exists(FAISS_INDEX_DIR) or not os.path.exists(FAISS_INDEX_MTIME_FILE):
        return None

    if not os.path.exists(MEMORY_FILE):
        return None

    # Check if index is up-to-date
    current_mtime = os.path.getmtime(MEMORY_FILE)
    try:
        with open(FAISS_INDEX_MTIME_FILE, "r") as f:
            saved_mtime = float(f.read().strip())
        if saved_mtime != current_mtime:
            return None
    except Exception:
        return None

    # Load the index
    try:
        embeddings = _get_embeddings()
        return FAISS.load_local(FAISS_INDEX_DIR, embeddings, allow_dangerous_deserialization=True)
    except Exception:
        return None

def _get_vectorstore():
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
    """Search the user's long-term memory for relevant information.

    Use this tool when you need to recall specific details about the user that might be stored in memory,
    rather than guessing or asking the user again.

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

    results = []
    for doc in docs:
        results.append(doc.page_content)

    return "\n".join(results)

@tool
def get_current_time():
    """Get the current time and date."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def _clear_faiss_index():
    """Clear the saved FAISS index (call when memory is updated)."""
    if os.path.exists(FAISS_INDEX_MTIME_FILE):
        try:
            os.remove(FAISS_INDEX_MTIME_FILE)
        except Exception:
            pass

@tool
def update_user_memory(key: str, value: str):
    """Update or add a piece of information about the user in long-term memory.

    格式规范：
    - key: 使用简洁的分类标签，如"姓名"、"生日"、"职业"、"饮食偏好"、"兴趣爱好"
    - value: 使用简洁的陈述句或列表，避免冗长解释
    - 同类信息用同一个 key，更新时整合已有内容，避免重复 key

    示例：
    - 好：key="饮食偏好", value="不吃香菜；喜欢辣；偏好清淡"
    - 差：key="用户不喜欢吃香菜", value="用户在2024年1月告诉我..."

    Args:
        key: The category or key of the information.
        value: The information to store.
    """
    memory = _load_memory()
    memory[key] = value
    _save_memory(memory)
    _clear_faiss_index()  # Clear index so it will be rebuilt next time
    return f"Successfully updated memory: {key} = {value}"


@tool
def web_search(query: str):
    """Search the web for up-to-date information.

    Use this tool when you need to find:
    - Current news or recent events
    - Real-time information (e.g., stock prices, weather if not available via other tools)
    - Facts that might have changed recently
    - Information not in the user's memory

    Args:
        query: The search query string.
    """
    search = DuckDuckGoSearchRun()
    return search.run(query)
