"""Base utilities and shared functions for all tools."""

import os
import sys
import io
import logging
import contextlib
from langchain_huggingface import HuggingFaceEmbeddings

# Suppress HuggingFace transformers warnings
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)


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


# Base directory path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Memory file paths
MEMORY_FILE = os.path.join(BASE_DIR, "data", "user_memory.json")
MEMORY_FAISS_DIR = os.path.join(BASE_DIR, "data", "memory_faiss_index")
MEMORY_FAISS_MTIME_FILE = os.path.join(MEMORY_FAISS_DIR, "memory_mtime.txt")

# Notes file paths
NOTES_FILE = os.path.join(BASE_DIR, "data", "notes.json")
NOTES_FAISS_DIR = os.path.join(BASE_DIR, "data", "notes_faiss_index")

# Global cache for embeddings
_embeddings_cache = None


def _get_embeddings():
    """Get cached embeddings model."""
    global _embeddings_cache
    if _embeddings_cache is None:
        with suppress_stdout_stderr():
            _embeddings_cache = HuggingFaceEmbeddings(model_name="BAAI/bge-small-zh-v1.5")
    return _embeddings_cache
