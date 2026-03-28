"""Notes-related tools for recording and searching user notes."""

import os
import json
import uuid
from datetime import datetime
from langchain_core.tools import tool
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from tools.base import (
    NOTES_FILE,
    NOTES_FAISS_DIR,
    _get_embeddings,
)

# Global cache for notes vector store
_notes_vectorstore_cache = None


def _load_notes():
    """Load notes from JSON file."""
    if not os.path.exists(NOTES_FILE):
        return {}
    try:
        with open(NOTES_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _save_notes(notes):
    """Save notes to JSON file."""
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

    当用户说"记一下 XXX"（或类似表达），且内容是用户的想法、感慨、思考、辩论、会议要点等需要回顾的内容时使用。
    笔记是用户主动记录的内容，支持后续搜索和回顾。

    不适用于：用户的稳定属性和偏好（如饮食偏好、作息习惯）——这些请使用 update_user_memory 工具。

    Args:
        title: 笔记标题，简洁概括内容。
        content: 笔记正文，详细记录内容（使用第一人称，以你自己的口吻来写）。
        tags: 标签，逗号分隔（可选）。
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
