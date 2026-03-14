import json
import os
from datetime import datetime
from langchain_core.tools import tool
from langchain_community.vectorstores import FAISS
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

# Memory file path
# Use absolute path relative to project root
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MEMORY_FILE = os.path.join(BASE_DIR, "data", "user_memory.json")

# Global cache for vector store
_vectorstore_cache = None
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

def _get_vectorstore():
    global _vectorstore_cache, _last_memory_mtime

    if not os.path.exists(MEMORY_FILE):
        return None

    current_mtime = os.path.getmtime(MEMORY_FILE)

    # If cache is valid, return it
    if _vectorstore_cache is not None and current_mtime == _last_memory_mtime:
        return _vectorstore_cache

    # Rebuild index
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

    # Use a small, fast local model
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    _vectorstore_cache = FAISS.from_documents(documents, embeddings)
    _last_memory_mtime = current_mtime

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
