import streamlit as st
import uuid
import os
from src.agent import get_agent_executor
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

# Page config
st.set_page_config(page_title="Personal Agent", page_icon="🤖")

st.title("🤖 Personal Agent")

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
        message_placeholder = st.empty()
        full_response = ""
        
        # We use a container to show tool executions if we want
        # For now, let's just stream the final response or updates
        
        # Run the agent stream
        # stream_mode="values" returns the full list of messages at each step
        # This is tricky for streaming tokens.
        # "updates" mode might be better if we want to see what changed.
        # But "values" is safer to get the whole state.
        
        # Let's try to just wait for the final response for v1, 
        # or use a spinner.
        with st.spinner("Thinking..."):
            events = agent.stream(
                {"messages": [("user", prompt)]},
                config,
                stream_mode="values"
            )
            
            last_msg = None
            for event in events:
                if "messages" in event:
                    messages = event["messages"]
                    if messages:
                        last_msg = messages[-1]
                        
                        # If it's an AI message, update the placeholder
                        if isinstance(last_msg, AIMessage) and last_msg.content:
                            full_response = last_msg.content
                            message_placeholder.markdown(full_response)
                        
                        # If it's a Tool execution, we could show it, but it might clutter
                        # The stream yields state *after* the step.
                        
            # Ensure final state is captured
            if not full_response and last_msg and isinstance(last_msg, AIMessage):
                full_response = last_msg.content
                message_placeholder.markdown(full_response)
                
    # Rerun to update the history properly (though we just displayed it)
    # Actually, since we use agent state, the next run will fetch it.
    # But we don't strictly need to rerun if we displayed it.
    # However, to sync up the 'messages' variable for next interaction:
    # st.rerun() 
