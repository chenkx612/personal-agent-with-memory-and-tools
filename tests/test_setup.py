import os
import sys
# Add the project root to sys.path to allow importing from src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import json
from langchain_core.messages import AIMessage, HumanMessage, BaseMessage
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.language_models import BaseChatModel
from langchain_core.outputs import ChatResult, ChatGeneration
from typing import List, Optional, Any
from src.tools import get_current_time, get_weather, update_user_memory, get_user_memory, MEMORY_FILE

class MockChatModel(BaseChatModel):
    responses: List[BaseMessage]
    i: int = 0
    
    def _generate(self, messages: List[BaseMessage], stop: Optional[List[str]] = None, run_manager: Any = None, **kwargs: Any) -> ChatResult:
        response = self.responses[self.i]
        self.i = (self.i + 1) % len(self.responses)
        return ChatResult(generations=[ChatGeneration(message=response)])
    
    @property
    def _llm_type(self) -> str:
        return "mock"
    
    def bind_tools(self, tools: Any, **kwargs: Any) -> "MockChatModel":
        return self

# Mock tools logic validation
def test_tools():
    print("Testing tools...")
    
    # Test Memory Tools
    # Clear memory file first
    if os.path.exists(MEMORY_FILE):
        os.remove(MEMORY_FILE)
    
    update_user_memory.invoke({"key": "name", "value": "Alice"})
    res = get_user_memory.invoke({"key": "name"})
    assert "Alice" in res
    print("Memory tools passed.")
    
    # Test Time
    time = get_current_time.invoke({})
    assert time is not None
    print("Time tool passed.")
    
    # Test Weather
    weather = get_weather.invoke({"location": "Beijing"})
    assert "Beijing" in weather
    print("Weather tool passed.")

def test_agent_graph():
    print("\nTesting agent graph construction...")
    
    # Mock LLM
    responses = [
        AIMessage(content="I will save your name.", tool_calls=[
            {"name": "update_user_memory", "args": {"key": "name", "value": "Bob"}, "id": "call1"}
        ]),
        AIMessage(content="I have updated your memory."),
    ]
    llm = MockChatModel(responses=responses)
    
    tools = [get_current_time, get_weather, update_user_memory, get_user_memory]
    checkpointer = MemorySaver()
    
    agent = create_react_agent(model=llm, tools=tools, checkpointer=checkpointer)
    
    # Test invocation
    config = {"configurable": {"thread_id": "test_thread"}}
    input_msg = {"messages": [("user", "My name is Bob")]}
    
    # Run
    final_state = agent.invoke(input_msg, config)
    
    # Verify memory was updated by tool
    # mem = get_user_memory("name") # This is a tool, not callable directly without invoke
    
    print("Agent execution finished.")
    # Check if tool was actually called (which updates the file)
    # The first response had a tool call. The agent should have executed it.
    
    # Reload memory from file to verify
    with open(MEMORY_FILE, "r") as f:
        data = json.load(f)
    
    if data.get("name") == "Bob":
        print("Agent successfully called tool and updated memory.")
    else:
        print(f"Agent failed to update memory. Current data: {data}")

if __name__ == "__main__":
    test_tools()
    test_agent_graph()
