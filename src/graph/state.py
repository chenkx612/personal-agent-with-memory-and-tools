"""Agent state definition for LangGraph."""

from typing import Annotated, Sequence, Optional

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
    """

    messages: Annotated[Sequence[BaseMessage], add_messages]
    pending_tool_approval: Optional[dict]  # 待用户确认的工具调用: {tool_call_id, name, args, original_message}
