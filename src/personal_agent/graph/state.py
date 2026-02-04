"""Agent state definition for LangGraph."""

from typing import Annotated, Sequence

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict


class AgentState(TypedDict):
    """Agent state with message history.

    Uses the add_messages reducer to properly handle message accumulation,
    including deduplication and updates.

    Future extension points:
    - Add `context` field for cross-node shared context
    - Add `current_agent` field for multi-agent routing
    - Add `interrupt_data` field for human-in-the-loop support
    """

    messages: Annotated[Sequence[BaseMessage], add_messages]
