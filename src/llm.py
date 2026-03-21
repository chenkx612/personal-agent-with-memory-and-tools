"""LLM initialization for the personal agent."""

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

# Load environment variables
load_dotenv()


def get_llm():
    """Get LLM instance for standalone use (e.g., memory tidying)."""
    api_key = os.getenv("API_KEY")
    model = os.getenv("MODEL")
    base_url = os.getenv("BASE_URL")

    return ChatOpenAI(
        model=model,
        api_key=api_key,
        base_url=base_url,
    )
