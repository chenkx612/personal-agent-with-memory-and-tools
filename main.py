import uuid
from .agent import get_agent_executor

def main():
    print("Initializing Personal Agent...")
    try:
        agent = get_agent_executor()
    except Exception as e:
        print(f"Error initializing agent: {e}")
        return

    # Use a fixed thread_id for this session to maintain conversation history
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    print(f"Session ID: {thread_id}")
    print("You can start chatting. Type 'quit' or 'exit' to end.")
    print("-----------------------------------------------------")

    while True:
        try:
            user_input = input("User: ")
            if user_input.lower() in ["quit", "exit"]:
                break
            
            # Run the agent
            # We stream the events to see steps
            events = agent.stream(
                {"messages": [("user", user_input)]},
                config,
                stream_mode="values"
            )

            # Print the final response
            for event in events:
                if "messages" in event:
                    last_msg = event["messages"][-1]
                    # Only print AI messages to avoid duplicating user input in output
                    # and ensure we show the final answer
                    # In 'values' mode, it emits the full state. 
                    pass
            
            # To get the final response more cleanly:
            # The loop above iterates through state updates.
            # We want the last message from the AI.
            snapshot = agent.get_state(config)
            if snapshot.values and "messages" in snapshot.values:
                last_message = snapshot.values["messages"][-1]
                if last_message.type == "ai":
                    print(f"Agent: {last_message.content}")
                elif last_message.type == "tool":
                    # If the last thing was a tool output, the agent hasn't responded yet? 
                    # create_react_agent usually ends with an AI message.
                    # If stream stopped, it should be done.
                    pass
            
        except Exception as e:
            print(f"An error occurred: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()
