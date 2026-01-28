import os
import json
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver

from .tools import get_current_time, get_weather, update_user_memory, get_user_memory, search_memory, MEMORY_FILE

# Load environment variables
load_dotenv()

def get_agent_executor():
    # Initialize the model
    # Note: User needs to set DEEPSEEK_API_KEY in .env
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        print("Warning: DEEPSEEK_API_KEY not found in environment variables.")
        # We might fail later if actual call happens
    
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
        system_prompt = """You are a helpful personal AI assistant.

You have access to a long-term memory of the user.
- To save new information about the user, use `update_user_memory`.
- To recall information about the user, use `search_memory` (RAG) or `get_user_memory` (direct lookup).
- You can also check time and weather.

You should autonomously decide when to search memory based on the user's query.
If the user asks about personal details (e.g., "what is my name?", "what do I like?"), use `search_memory` to find the answer.
Do NOT hallucinate user details. If you are not sure, check the memory.
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
