import os
# Set environment variable for HuggingFace mirror to resolve connection issues
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import uuid
import json
from prompt_toolkit import PromptSession
from prompt_toolkit.history import InMemoryHistory
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.key_binding import KeyBindings
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.text import Text
from rich.live import Live
from langchain_core.messages import AIMessageChunk, ToolMessage
from personal_agent.core import get_agent_executor

console = Console()


def format_tool_call(tool_call: dict) -> Panel:
    """Format a tool call for display."""
    name = tool_call.get("name", "unknown")
    args = tool_call.get("args", {})
    args_str = json.dumps(args, ensure_ascii=False, indent=2)
    content = Text()
    content.append(f"📤 调用工具: ", style="bold yellow")
    content.append(f"{name}\n", style="bold cyan")
    content.append(f"参数:\n{args_str}", style="dim")
    return Panel(content, border_style="yellow", expand=False)


def format_tool_result(tool_name: str, result: str) -> Panel:
    """Format a tool result for display."""
    # Truncate long results
    display_result = result if len(result) < 500 else result[:500] + "..."
    content = Text()
    content.append(f"📥 工具返回: ", style="bold green")
    content.append(f"{tool_name}\n", style="bold cyan")
    content.append(display_result, style="dim")
    return Panel(content, border_style="green", expand=False)


def stream_agent_response(agent, user_input: str, config: dict):
    """Stream agent response with tool calls and thinking visible."""

    # Track state for streaming
    current_content = ""
    # Use index as key since id may be None in subsequent chunks
    pending_tool_calls = {}  # index -> {id, name, args}
    printed_tool_calls = set()  # Track by index

    console.print("[bold blue]Agent:[/bold blue]")

    # Use Live for real-time markdown rendering
    with Live(Markdown(""), console=console, refresh_per_second=10, vertical_overflow="visible") as live:
        for event in agent.stream(
            {"messages": [("user", user_input)]},
            config,
            stream_mode="messages"
        ):
            msg, metadata = event

            # Handle AI message chunks (streaming text)
            if isinstance(msg, AIMessageChunk):
                # Stream content tokens
                if msg.content:
                    current_content += msg.content
                    live.update(Markdown(current_content))

                # Handle tool calls - use index as the key
                if msg.tool_call_chunks:
                    for chunk in msg.tool_call_chunks:
                        idx = chunk.get("index", 0)

                        if idx not in pending_tool_calls:
                            pending_tool_calls[idx] = {
                                "id": chunk.get("id"),
                                "name": chunk.get("name") or "",
                                "args": chunk.get("args") or ""
                            }
                        else:
                            # Update id if provided
                            if chunk.get("id"):
                                pending_tool_calls[idx]["id"] = chunk["id"]
                            # Update name if provided
                            if chunk.get("name"):
                                pending_tool_calls[idx]["name"] = chunk["name"]
                            # Accumulate args string
                            if chunk.get("args"):
                                pending_tool_calls[idx]["args"] += chunk["args"]

            # Handle tool messages (results)
            elif isinstance(msg, ToolMessage):
                # Stop live update temporarily to print tool info
                live.stop()

                # Print any pending tool calls first
                for idx, tc in pending_tool_calls.items():
                    if idx not in printed_tool_calls and tc.get("name"):
                        try:
                            args = json.loads(tc["args"]) if tc["args"] else {}
                        except json.JSONDecodeError:
                            args = {"raw": tc["args"]}
                        console.print(format_tool_call({"name": tc["name"], "args": args}))
                        printed_tool_calls.add(idx)

                # Print tool result
                tool_name = msg.name or "unknown"
                console.print(format_tool_result(tool_name, str(msg.content)))
                # Reset for potential next round of tool calls
                pending_tool_calls.clear()
                printed_tool_calls.clear()
                current_content = ""
                console.print("[bold blue]Agent:[/bold blue]")

                # Restart live update for next response
                live.start()


def main():
    console.print("[bold]Initializing Personal Agent...[/bold]")
    try:
        agent = get_agent_executor()
    except Exception as e:
        console.print(f"[red]Error initializing agent: {e}[/red]")
        return

    # Use a fixed thread_id for this session to maintain conversation history
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    # Initialize PromptSession for better input handling (fixes backspace issues)
    session = PromptSession(history=InMemoryHistory())

    # Define custom key bindings
    bindings = KeyBindings()

    @bindings.add('enter')
    def _(event):
        """Bind Enter to submit the buffer."""
        event.current_buffer.validate_and_handle()

    @bindings.add('escape', 'enter')
    def _(event):
        """Bind Meta+Enter (Option+Enter) to insert newline."""
        event.current_buffer.insert_text('\n')

    console.print(f"[dim]Session ID: {thread_id}[/dim]")
    console.print("[dim]输入 'quit' 或 'exit' 退出[/dim]")
    console.print("[dim]─" * 50 + "[/dim]")

    while True:
        try:
            user_input = session.prompt(
                "User: ",
                multiline=True,
                key_bindings=bindings,
                bottom_toolbar=HTML(" <b>[Enter]</b> 发送  <b>[Option+Enter]</b> 换行 ")
            )
            if user_input.lower().strip() in ["quit", "exit"]:
                console.print("[dim]再见！[/dim]")
                break

            if not user_input.strip():
                continue

            # Stream the agent response
            stream_agent_response(agent, user_input, config)
            print()  # Extra line between exchanges

        except (KeyboardInterrupt, EOFError):
            console.print("\n[dim]再见！[/dim]")
            break
        except Exception as e:
            console.print(f"[red]发生错误: {e}[/red]")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()
