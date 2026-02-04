"""Node functions for the agent graph."""

from langchain_core.messages import SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.prebuilt import ToolNode

from .state import AgentState


def create_agent_node(llm, system_prompt: str):
    """Create an agent node that calls the LLM with system prompt.

    Args:
        llm: The language model to use.
        system_prompt: System prompt to prepend to messages.

    Returns:
        A function that processes agent state and returns updated messages.
    """

    def agent_node(state: AgentState, config: RunnableConfig) -> dict:
        """Call LLM to generate response or tool calls."""
        messages = state["messages"]

        # Prepend system message if not already present
        if not messages or not isinstance(messages[0], SystemMessage):
            messages = [SystemMessage(content=system_prompt)] + list(messages)

        response = llm.invoke(messages, config)
        return {"messages": [response]}

    return agent_node


def create_tool_node(tools: list):
    """Create a tool node using LangGraph's built-in ToolNode.

    Args:
        tools: List of tool functions.

    Returns:
        A ToolNode instance that executes tool calls.
    """
    return ToolNode(tools)
