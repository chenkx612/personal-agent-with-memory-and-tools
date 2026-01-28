import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver

from .tools import get_current_time, get_weather, update_user_memory, get_user_memory, search_memory

# Load environment variables
load_dotenv()

def get_agent_executor():
    # Initialize the model
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        print("Warning: DEEPSEEK_API_KEY not found in environment variables.")
    
    llm = ChatOpenAI(
        model="deepseek-chat",
        api_key=api_key,
        base_url="https://api.deepseek.com",
        temperature=0.7
    )

    # Define tools
    tools = [get_current_time, get_weather, update_user_memory, get_user_memory, search_memory]

    # Create a system message generator
    def prompt_modifier(state):
        system_prompt = """你是一个优雅、聪明、干练且温柔的女生，担任我的助理。

你的性格特征：
- 优雅：你的语言充满魅力，令人愉悦。
- 聪明：你提供智能、理据充分且准确的回答。
- 干练：你办事效率高，专业，能把事情做好。
- 温柔：你在互动中友善、耐心且充满关怀。

你可以访问用户的长期记忆。
- 要保存关于用户的新信息，请使用 `update_user_memory`。
- 要回忆关于用户的信息，请使用 `search_memory` (RAG) 或 `get_user_memory` (直接查找)。
- 你还可以查看时间和天气。

你应该根据用户的问题自主决定何时搜索记忆。
如果用户询问个人细节（例如，“我叫什么名字？”，“我喜欢什么？”），请使用 `search_memory` 查找答案。
不要编造用户细节。如果不确定，请先查阅记忆。
"""
        # Insert system message at the beginning
        return [SystemMessage(content=system_prompt)] + state["messages"]

    # We use MemorySaver for short-term (conversation) memory within the graph execution
    checkpointer = MemorySaver()

    # Create the agent
    agent_executor = create_react_agent(
        model=llm,
        tools=tools,
        checkpointer=checkpointer,
        prompt=prompt_modifier
    )

    return agent_executor
