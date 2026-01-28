import os
# Set environment variable for HuggingFace mirror to resolve connection issues
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import streamlit as st
import uuid
from src.agent import get_agent_executor
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

# Page config
st.set_page_config(page_title="Personal Asistant", page_icon="👩")
st.title("👩 Personal Asistant")

# Initialize session state for thread_id
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

# Initialize agent (cached resource)
@st.cache_resource
def load_agent():
    return get_agent_executor()

try:
    agent = load_agent()
except Exception as e:
    st.error(f"Error initializing agent: {e}")
    st.stop()

config = {"configurable": {"thread_id": st.session_state.thread_id}}

# Sidebar for session info
with st.sidebar:
    st.header("Session Info")
    st.text(f"ID: {st.session_state.thread_id}")
    if st.button("Clear History"):
        # Generate a new thread ID to effectively clear history
        st.session_state.thread_id = str(uuid.uuid4())
        st.rerun()

# Load existing history from agent state
try:
    snapshot = agent.get_state(config)
    if snapshot.values and "messages" in snapshot.values:
        messages = snapshot.values["messages"]
    else:
        messages = []
except Exception:
    messages = []

# Display chat history
for msg in messages:
    if isinstance(msg, HumanMessage):
        with st.chat_message("user"):
            st.write(msg.content)
    elif isinstance(msg, AIMessage):
        # Only display non-empty AI messages (skip empty tool calls if any)
        if msg.content:
            with st.chat_message("assistant"):
                st.write(msg.content)
    elif isinstance(msg, ToolMessage):
        # Optionally display tool outputs (maybe in an expander)
        with st.chat_message("assistant"):
            with st.expander(f"Tool: {msg.name}"):
                st.code(msg.content)

# Chat input
if prompt := st.chat_input("What can I do for you?"):
    # Display user message immediately
    with st.chat_message("user"):
        st.write(prompt)

    # Process with agent
    with st.chat_message("assistant"):
        # Create status container for internal steps (thoughts, tool calls)
        status = st.status("Thinking...", expanded=True)
        # Create placeholder for final response
        message_placeholder = st.empty()
        full_response = ""
        
        # Determine start index for new messages
        # 'messages' variable contains history loaded before this run
        start_idx = len(messages)
        processed_idx = start_idx

        # Run the agent stream
        events = agent.stream(
            {"messages": [("user", prompt)]},
            config,
            stream_mode="values"
        )
        
        for event in events:
            if "messages" in event:
                current_messages = event["messages"]
                
                # Process only new messages
                if len(current_messages) > processed_idx:
                    for i in range(processed_idx, len(current_messages)):
                        msg = current_messages[i]
                        
                        # Skip the user prompt that started this turn
                        if isinstance(msg, HumanMessage) and msg.content == prompt:
                            continue
                            
                        # Visualize AI Tool Calls
                        if isinstance(msg, AIMessage) and msg.tool_calls:
                            status.write("🛠️ **Calling Tools**")
                            for tool_call in msg.tool_calls:
                                status.text(f"Tool: {tool_call['name']}")
                                status.json(tool_call['args'])
                        
                        # Visualize Tool Outputs
                        elif isinstance(msg, ToolMessage):
                            status.write(f"✅ **Tool Result ({msg.name})**")
                            # Truncate long outputs for display
                            content = str(msg.content)
                            if len(content) > 500:
                                content = content[:500] + "... (truncated)"
                            status.code(content)
                            
                        # Handle AI Response (Thought or Final Answer)
                        elif isinstance(msg, AIMessage) and msg.content:
                            # If it has tool calls, the content is likely a thought
                            if msg.tool_calls:
                                status.markdown(f"**Thought:** {msg.content}")
                            else:
                                # Final response
                                full_response = msg.content
                                message_placeholder.markdown(full_response)
                    
                    # Update processed index
                    processed_idx = len(current_messages)
        
        # Close status
        status.update(label="Finished", state="complete", expanded=False)
                
    # Rerun to update the history properly (though we just displayed it)
    # Actually, since we use agent state, the next run will fetch it.
    # But we don't strictly need to rerun if we displayed it.
    # However, to sync up the 'messages' variable for next interaction:
    # st.rerun() 
