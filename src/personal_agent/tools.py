import json
import os
import logging
import uuid
import urllib.request
import urllib.error
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

NOTES_FILE = os.path.join(BASE_DIR, "data", "notes.json")
NOTES_FAISS_DIR = os.path.join(BASE_DIR, "data", "notes_faiss_index")

# Global cache for vector store and embeddings
_vectorstore_cache = None
_embeddings_cache = None
_last_memory_mtime = 0

_notes_vectorstore_cache = None

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

_location_cache = None

def _get_location():
    """Get current location via IP geolocation. Cached for the session."""
    global _location_cache
    if _location_cache is not None:
        return _location_cache
    try:
        req = urllib.request.Request(
            "http://ipinfo.io/json",
            headers={"User-Agent": "personal-agent/1.0"}
        )
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read().decode())
        city = data.get("city", "")
        region = data.get("region", "")
        country = data.get("country", "")
        loc = data.get("loc", "")  # "lat,lon"
        location_str = ", ".join(filter(None, [city, region, country]))
        lat, lon = None, None
        if loc and "," in loc:
            parts = loc.split(",")
            lat, lon = float(parts[0]), float(parts[1])
        _location_cache = {"location": location_str, "lat": lat, "lon": lon}
        return _location_cache
    except Exception:
        return None


def _get_weather(lat, lon):
    """Get current weather from Open-Meteo API (no API key required)."""
    try:
        url = (
            f"https://api.open-meteo.com/v1/forecast"
            f"?latitude={lat}&longitude={lon}"
            f"&current_weather=true"
            f"&hourly=relative_humidity_2m"
            f"&forecast_days=1"
        )
        req = urllib.request.Request(url, headers={"User-Agent": "personal-agent/1.0"})
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read().decode())
        cw = data.get("current_weather", {})
        temp = cw.get("temperature")
        wind = cw.get("windspeed")
        wmo = cw.get("weathercode", 0)
        # Pick humidity at first hourly slot
        humidity = None
        hourly = data.get("hourly", {})
        rh_list = hourly.get("relative_humidity_2m", [])
        if rh_list:
            humidity = rh_list[0]
        # Map WMO weather code to description
        if wmo == 0:
            desc = "晴"
        elif wmo in (1, 2, 3):
            desc = "多云"
        elif wmo in range(45, 50):
            desc = "雾"
        elif wmo in range(51, 68):
            desc = "雨"
        elif wmo in range(71, 78):
            desc = "雪"
        elif wmo in range(80, 83):
            desc = "阵雨"
        elif wmo in range(95, 100):
            desc = "雷暴"
        else:
            desc = f"天气代码{wmo}"
        parts = []
        if temp is not None:
            parts.append(f"{temp}°C")
        parts.append(desc)
        if wind is not None:
            parts.append(f"风速 {wind} km/h")
        if humidity is not None:
            parts.append(f"湿度 {humidity}%")
        return ", ".join(parts)
    except Exception:
        return None


_WEEKDAY_ZH = ["星期一", "星期二", "星期三", "星期四", "星期五", "星期六", "星期日"]


@tool
def get_environment_context():
    """Get current date, time, location and weather information.

    Returns structured environment context including:
    - Current date (with weekday) and time
    - User's approximate location based on IP
    - Current weather conditions
    """
    now = datetime.now()
    weekday = _WEEKDAY_ZH[now.weekday()]
    date_str = now.strftime("%Y-%m-%d")
    time_str = now.strftime("%H:%M:%S")

    lines = [
        f"日期: {date_str}（{weekday}）",
        f"时间: {time_str}",
    ]

    loc_info = _get_location()
    if loc_info and loc_info.get("location"):
        lines.append(f"位置: {loc_info['location']}")
        if loc_info.get("lat") is not None and loc_info.get("lon") is not None:
            weather = _get_weather(loc_info["lat"], loc_info["lon"])
            if weather:
                lines.append(f"天气: {weather}")

    return "\n".join(lines)


@tool
def update_user_memory(key: str, value: str):
    """Update or add a stable attribute about the user in long-term memory (user profile only).

    仅用于存储用户的稳定属性和偏好（如姓名、生日、饮食偏好、兴趣爱好）。
    若用户想记录具体内容、想法、事件或文章要点，请使用 add_note 工具。

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


def _load_notes():
    if not os.path.exists(NOTES_FILE):
        return {}
    try:
        with open(NOTES_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}

def _save_notes(notes):
    with open(NOTES_FILE, "w", encoding="utf-8") as f:
        json.dump(notes, f, ensure_ascii=False, indent=2)

def _get_notes_vectorstore():
    """Get notes vector store: memory cache → disk → full build."""
    global _notes_vectorstore_cache

    if _notes_vectorstore_cache is not None:
        return _notes_vectorstore_cache

    # Try to load from disk
    if os.path.exists(NOTES_FAISS_DIR):
        try:
            embeddings = _get_embeddings()
            _notes_vectorstore_cache = FAISS.load_local(NOTES_FAISS_DIR, embeddings, allow_dangerous_deserialization=True)
            return _notes_vectorstore_cache
        except Exception:
            pass

    # Full build as fallback (first run or index missing)
    notes = _load_notes()
    if not notes:
        return None

    documents = []
    for note_id, note in notes.items():
        doc = Document(
            page_content=f"{note['title']}\n{note['content']}",
            metadata={"note_id": note_id, "title": note["title"],
                      "tags": note["tags"], "created_at": note["created_at"]}
        )
        documents.append(doc)

    if not documents:
        return None

    embeddings = _get_embeddings()
    _notes_vectorstore_cache = FAISS.from_documents(documents, embeddings)

    os.makedirs(NOTES_FAISS_DIR, exist_ok=True)
    _notes_vectorstore_cache.save_local(NOTES_FAISS_DIR)

    return _notes_vectorstore_cache


@tool
def add_note(title: str, content: str, tags: str = ""):
    """添加一条新笔记到笔记本。

    当用户想要记录某件事、某个想法、某次对话要点时使用此工具。
    笔记与用户画像（user_memory）不同：笔记是用户主动记录的内容，支持后续搜索和回顾。

    Args:
        title: 笔记标题，简洁概括内容。
        content: 笔记正文，详细记录内容。
        tags: 标签，逗号分隔（可选），如"论文,研究"。
    """
    note_id = str(uuid.uuid4())[:8]
    created_at = datetime.now().strftime("%Y-%m-%d")
    notes = _load_notes()
    notes[note_id] = {
        "title": title,
        "content": content,
        "tags": tags,
        "created_at": created_at,
    }
    _save_notes(notes)

    # Incrementally add to FAISS index
    doc = Document(
        page_content=f"{title}\n{content}",
        metadata={"note_id": note_id, "title": title,
                  "tags": tags, "created_at": created_at}
    )
    vectorstore = _get_notes_vectorstore()
    if vectorstore is not None:
        vectorstore.add_documents([doc])
    else:
        global _notes_vectorstore_cache
        embeddings = _get_embeddings()
        _notes_vectorstore_cache = FAISS.from_documents([doc], embeddings)
        vectorstore = _notes_vectorstore_cache

    os.makedirs(NOTES_FAISS_DIR, exist_ok=True)
    vectorstore.save_local(NOTES_FAISS_DIR)

    return f"笔记已保存，id: {note_id}，标题：{title}"


@tool
def search_notes(query: str, k: int = 5):
    """搜索笔记，返回相关笔记的摘要列表。

    当用户想查找之前记录的笔记时使用此工具。返回摘要列表（不含全文），
    若需要阅读完整内容，请使用 get_note 工具并传入 note_id。

    Args:
        query: 搜索关键词或描述。
        k: 返回结果数量（默认5）。
    """
    vectorstore = _get_notes_vectorstore()
    if not vectorstore:
        return "笔记本为空。"

    docs = vectorstore.similarity_search(query, k=k)
    if not docs:
        return "未找到相关笔记。"

    lines = []
    for doc in docs:
        m = doc.metadata
        preview = doc.page_content.split("\n", 1)[-1][:60]
        lines.append(f'[{m["note_id"]}] "{m["title"]}" ({m["created_at"]}) tags: {m["tags"]} - "{preview}..."')
    return "\n".join(lines)


@tool
def get_note(note_id: str):
    """按 id 读取笔记全文。

    使用 search_notes 获取 note_id 后，调用此工具读取完整笔记内容。

    Args:
        note_id: 笔记的唯一 id（8位字符串）。
    """
    notes = _load_notes()
    note = notes.get(note_id)
    if not note:
        return f"未找到 id 为 {note_id} 的笔记。"
    return (
        f"标题：{note['title']}\n"
        f"日期：{note['created_at']}\n"
        f"标签：{note['tags']}\n"
        f"正文：\n{note['content']}"
    )


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
