"""Web-related tools for searching the internet."""

from langchain_core.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun


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
