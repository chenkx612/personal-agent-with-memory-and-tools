"""Graph builder for the ReAct agent."""

from typing import Literal

from langchain_core.language_models import BaseChatModel
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from graph.nodes import create_agent_node, create_tool_node
from graph.state import AgentState

# 需要用户审批的工具列表
TOOLS_REQUIRING_APPROVAL = {"add_note"}


def should_continue(state: AgentState) -> Literal["tools", "__end__"]:
    """Determine whether to continue with tool execution or end.

    Args:
        state: Current agent state.

    Returns:
        "tools" if the last message has tool calls, "__end__" otherwise.
    """
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    return END


def build_agent_graph(
    llm: BaseChatModel,
    tools: list,
    system_prompt: str,
    checkpointer=None,
):
    """Build the agent graph with ReAct pattern and human-in-the-loop approval.

    Graph structure:
        START → agent ←→ tools
                  ↓
                 END

    Args:
        llm: Language model with tool binding support.
        tools: List of tool functions to make available.
        system_prompt: System prompt for the agent.
        checkpointer: Optional checkpointer for conversation persistence.
            Defaults to MemorySaver if not provided.

    Returns:
        Compiled StateGraph ready for execution.
    """
    if checkpointer is None:
        checkpointer = MemorySaver()

    # Bind tools to the LLM
    llm_with_tools = llm.bind_tools(tools)

    # Create nodes
    agent_node = create_agent_node(llm_with_tools, system_prompt)
    tool_node = create_tool_node(tools)

    # Build the graph
    builder = StateGraph(AgentState)

    # Add nodes
    builder.add_node("agent", agent_node)
    builder.add_node("tools", tool_node)

    # Add edges
    builder.add_edge(START, "agent")
    builder.add_conditional_edges(
        "agent",
        should_continue,
        {"tools": "tools", END: END},
    )
    builder.add_edge("tools", "agent")

    # 在 tools 节点前设置中断点，用于需要用户审批的工具
    # 这样当有工具调用时，会在执行前中断，我们在 CLI 层判断是否需要审批
    graph = builder.compile(
        checkpointer=checkpointer,
        interrupt_before=["tools"],  # 在执行工具前中断
    )

    return graph
