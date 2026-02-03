import os
# Set environment variable for HuggingFace mirror to resolve connection issues
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import uuid
import json
import tempfile
import subprocess
from prompt_toolkit import PromptSession
from prompt_toolkit.history import InMemoryHistory
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.key_binding import KeyBindings
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.text import Text
from rich.live import Live
from rich.prompt import Prompt
from langchain_core.messages import AIMessageChunk, ToolMessage
from personal_agent.core import get_agent_executor, get_llm
from personal_agent.tools import _load_memory, _save_memory

console = Console()

TIDY_PROMPT = """你是一个记忆整理助手。请分析以下用户记忆数据，进行整理优化。

当前记忆（JSON格式）：
```json
{memory_json}
```

整理规则：
1. 合并相似或重复的条目（如"喜欢的食物"和"饮食偏好"应合并）
2. 精简冗长的描述，保留核心信息
3. 使用统一的 key 命名风格（简洁的中文标签）
4. 删除过时或矛盾的信息（保留更具体/更新的）
5. 确保每个 key 语义明确，value 简洁有条理

请直接输出整理后的 JSON，不要有其他解释。输出格式：
```json
{{整理后的记忆}}
```"""


def edit_memory_json(memory: dict) -> dict | None:
    """在系统编辑器中编辑记忆 JSON，返回编辑后的字典，失败或取消返回 None。"""
    editor = os.environ.get("EDITOR", "vim")

    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".json",
        delete=False,
        encoding="utf-8"
    ) as f:
        json.dump(memory, f, ensure_ascii=False, indent=2)
        temp_path = f.name

    try:
        # 打开编辑器
        console.print(f"[dim]使用 {editor} 编辑... (保存退出后生效)[/dim]")
        result = subprocess.run([editor, temp_path])

        if result.returncode != 0:
            console.print(f"[red]编辑器退出异常 (code {result.returncode})[/red]")
            return None

        # 读取编辑后的内容
        with open(temp_path, "r", encoding="utf-8") as f:
            edited_content = f.read()

        edited_memory = json.loads(edited_content)
        return edited_memory

    except json.JSONDecodeError as e:
        console.print(f"[red]JSON 解析错误: {e}[/red]")
        return None
    except Exception as e:
        console.print(f"[red]编辑失败: {e}[/red]")
        return None
    finally:
        # 清理临时文件
        try:
            os.unlink(temp_path)
        except OSError:
            pass


def tidy_memory() -> bool:
    """整理用户记忆，返回是否成功修改。"""
    memory = _load_memory()

    if not memory:
        console.print("[yellow]记忆为空，无需整理。[/yellow]")
        return False

    console.print("[bold]正在分析记忆...[/bold]")

    # 显示当前记忆
    console.print("\n[bold cyan]当前记忆：[/bold cyan]")
    console.print(Panel(
        json.dumps(memory, ensure_ascii=False, indent=2),
        border_style="dim"
    ))

    # 调用 LLM 整理
    llm = get_llm()
    prompt = TIDY_PROMPT.format(memory_json=json.dumps(memory, ensure_ascii=False, indent=2))

    console.print("\n[bold]LLM 整理中...[/bold]")

    try:
        response = llm.invoke(prompt)
        content = response.content

        # 处理 content 可能是 list 的情况
        if isinstance(content, list):
            text_parts = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    text_parts.append(item.get("text", ""))
                elif isinstance(item, str):
                    text_parts.append(item)
            content = "".join(text_parts)

        # 提取 JSON
        import re
        json_match = re.search(r'```json\s*([\s\S]*?)\s*```', content)
        if json_match:
            tidied_json = json_match.group(1)
        else:
            # 尝试直接解析整个响应
            tidied_json = content.strip()

        tidied_memory = json.loads(tidied_json)

        # 显示整理后的记忆
        console.print("\n[bold green]整理后：[/bold green]")
        console.print(Panel(
            json.dumps(tidied_memory, ensure_ascii=False, indent=2),
            border_style="green"
        ))

        # 统计变化
        old_keys = set(memory.keys())
        new_keys = set(tidied_memory.keys())
        removed = old_keys - new_keys
        added = new_keys - old_keys

        console.print(f"\n[dim]变化: {len(old_keys)} → {len(new_keys)} 条记忆[/dim]")
        if removed:
            console.print(f"[red]移除: {', '.join(removed)}[/red]")
        if added:
            console.print(f"[green]新增: {', '.join(added)}[/green]")

        # 确认
        while True:
            console.print("\n[dim]y=应用 n=取消 e=手动编辑[/dim]")
            confirm = Prompt.ask(
                "[bold yellow]操作[/bold yellow]",
                choices=["y", "n", "e"],
                default="n"
            )

            if confirm == 'y':
                _save_memory(tidied_memory)
                console.print("[bold green]✓ 记忆已更新[/bold green]")
                return True
            elif confirm == 'e':
                # 手动编辑
                edited = edit_memory_json(tidied_memory)
                if edited is not None:
                    tidied_memory = edited
                    console.print("\n[bold green]编辑后：[/bold green]")
                    console.print(Panel(
                        json.dumps(tidied_memory, ensure_ascii=False, indent=2),
                        border_style="green"
                    ))
                    # 继续循环，再次询问
                    continue
                else:
                    console.print("[yellow]编辑已取消或无变化[/yellow]")
                    continue
            else:
                console.print("[dim]已取消[/dim]")
                return False

    except json.JSONDecodeError as e:
        console.print(f"[red]LLM 返回的 JSON 无法解析: {e}[/red]")
        console.print(f"[dim]原始响应: {content[:500]}...[/dim]")
        return False
    except Exception as e:
        console.print(f"[red]整理失败: {e}[/red]")
        return False


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
                    # Handle both string and list content formats
                    if isinstance(msg.content, str):
                        current_content += msg.content
                    elif isinstance(msg.content, list):
                        for item in msg.content:
                            if isinstance(item, dict) and item.get("type") == "text":
                                current_content += item.get("text", "")
                            elif isinstance(item, str):
                                current_content += item
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
    console.print("[dim]命令: /tidy 整理记忆 | /exit 退出[/dim]")
    console.print("[dim]─" * 50 + "[/dim]")

    while True:
        try:
            user_input = session.prompt(
                "User: ",
                multiline=True,
                key_bindings=bindings,
                bottom_toolbar=HTML(" <b>[Enter]</b> 发送  <b>[Option+Enter]</b> 换行 ")
            )
            stripped_input = user_input.strip()

            if stripped_input == "/exit":
                console.print("[dim]再见！[/dim]")
                break

            if stripped_input == "/tidy":
                tidy_memory()
                continue

            if not stripped_input:
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
