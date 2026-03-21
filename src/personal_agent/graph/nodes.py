"""Node functions for the agent graph."""

from langchain_core.messages import SystemMessage, ToolMessage, AIMessage
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


def check_pending_approval_node(state: AgentState) -> dict:
    """处理待审批节点 - 检查是否有待审批的工具调用.

    这个节点用于在恢复执行时，将用户批准/修改后的工具调用放回消息重新放入待执行状态。
    """
    pending = state.get("pending_tool_approval")
    if not pending:
        return {}

    # 根据用户选择后的处理会在 CLI 层完成，这里只需清理 pending 状态
    return {"pending_tool_approval": None}


def create_approval_result_node(state: AgentState, approval_result: dict) -> dict:
    """创建审批结果节点 - 根据用户选择生成相应的 ToolMessage.

    Args:
        approval_result: 用户审批结果，包含:
            - action: "approve" | "modify" | "reject"
            - tool_call_id: 工具调用ID
            - tool_name: 工具名称
            - args: (modify时是修改后的参数
            - message: 给agent的回复消息
    """
    action = approval_result["action"]
    tool_call_id = approval_result["tool_call_id"]
    tool_name = approval_result["tool_name"]

    if action == "reject":
        # 用户拒绝，返回一个说明被拒绝的 ToolMessage
        content = approval_result.get("message", "用户拒绝执行此操作")
        tool_msg = ToolMessage(content=content, name=tool_name, tool_call_id=tool_call_id)
        return {"messages": [tool_msg], "pending_tool_approval": None}

    elif action == "approve" or action == "modify":
        # 批准或修改后的处理会在 CLI 层执行实际工具调用
        # 这里只返回结果
        content = approval_result.get("message", "操作已完成")
        tool_msg = ToolMessage(content=content, name=tool_name, tool_call_id=tool_call_id)
        return {"messages": [tool_msg], "pending_tool_approval": None}

    return {"pending_tool_approval": None}
