import os
import json
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver

from .tools import get_current_time, get_weather, update_user_memory, get_user_memory, MEMORY_FILE

# Load environment variables
load_dotenv()

def _load_memory_content():
    if not os.path.exists(MEMORY_FILE):
        return "{}"
    try:
        with open(MEMORY_FILE, "r", encoding="utf-8") as f:
            return f.read()
    except Exception:
        return "{}"

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
    tools = [get_current_time, get_weather, update_user_memory, get_user_memory]

    # Create a system message generator that includes current memory
    # Since create_react_agent uses a fixed state_modifier or messages, 
    # we can pass a function to state_modifier if using latest langgraph, 
    # or just pre-inject.
    # For simplicity with create_react_agent, we will define a simple prompt.
    
    # However, to support dynamic memory injection, we might need to wrap the agent 
    # or just rely on the tool usage. 
    # Let's make the system prompt generic, but encourage checking memory.
    # Actually, a better way is to define a `state_modifier` that prepends the system message with memory.
    
    def prompt_modifier(state):
        memory_content = _load_memory_content()
        system_prompt = f"""You are a helpful personal AI assistant.
        
Current User Memory (JSON):
{memory_content}

You have access to tools to manage this memory (update_user_memory) and get external info (time, weather).
When the user tells you something about themselves, save it to memory.
When answering, use the information from memory if relevant.
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
        state_modifier=prompt_modifier
    )

    return agent_executor
