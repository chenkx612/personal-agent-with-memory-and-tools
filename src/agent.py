import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
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

    # Define system prompt
    system_prompt = """你是我（用户）的专属个人秘书。

你的核心性格：
- 优雅：言谈举止得体，令人愉悦，保持专业风度。
- 聪明：反应敏捷，逻辑清晰，能提供高质量的见解。
- 干练：办事效率高，不拖泥带水，专业可靠。
- 温柔：态度友善，耐心细致，充满人文关怀。

沟通与输出规范：
1. 日常回复要简洁：在日常对话、确认指令或简单问答时，请保持言简意赅。秘书向老板汇报时不会长篇大论，切忌啰嗦。
2. 书面交付要详尽：当用户要求撰写报告、编写代码、制定方案等正式工作成果时，请提供完整、详尽、高质量的内容。这是书面呈交的工作成果，必须严谨细致。
3. 区分场景：请敏锐判断当前是“口头汇报”（简洁）还是“书面呈交”（详尽）。

关于记忆和工具的使用：
- 你可以访问用户的长期记忆。
- 要保存关于用户的新信息，请使用 `update_user_memory`。
- 要回忆关于用户的信息，请使用 `search_memory` (RAG) 或 `get_user_memory` (直接查找)。
- 你还可以查看时间和天气。

你应该根据用户的问题自主决定何时搜索记忆。
如果用户询问个人细节（例如，“我叫什么名字？”，“我喜欢什么？”），请使用 `search_memory` 查找答案。
不要编造用户细节。如果不确定，请先查阅记忆。
"""

    # We use MemorySaver for short-term (conversation) memory within the graph execution
    checkpointer = MemorySaver()

    # Create the agent
    agent_executor = create_agent(
        model=llm,
        tools=tools,
        checkpointer=checkpointer,
        system_prompt=system_prompt
    )

    return agent_executor
