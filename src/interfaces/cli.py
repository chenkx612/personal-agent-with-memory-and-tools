import os
from config import load_config

# 加载配置
config = load_config()

# 读取输出模式配置，默认为流式输出
STREAM_OUTPUT = config.get("stream_output", True)

import uuid
import json
import tempfile
import subprocess
from prompt_toolkit import PromptSession
from prompt_toolkit.history import InMemoryHistory
from prompt_toolkit.key_binding import KeyBindings
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.text import Text
from rich.live import Live
from rich.prompt import Prompt, Confirm
from langchain_core.messages import AIMessageChunk, ToolMessage
from core import get_agent_executor
from llm import get_llm
from tools import (
    _load_memory,
    _save_memory,
    _load_notes,
    _save_notes,
    _get_notes_vectorstore,
    _notes_vectorstore_cache,
    _get_embeddings,
    NOTES_FAISS_DIR,
    get_environment_context,
    search_memory,
    get_memory,
    update_user_memory,
    web_search,
    add_note,
    search_notes,
    get_note,
)
from graph.builder import TOOLS_REQUIRING_APPROVAL
from langchain_core.documents import Document
import datetime

console = Console()


def normalize_llm_content(content) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text_parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                text_parts.append(item.get("text", ""))
            elif isinstance(item, str):
                text_parts.append(item)
        return "".join(text_parts)
    return str(content)


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
        content = normalize_llm_content(response.content)

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


def view_notes_menu():
    """笔记浏览子菜单：列出笔记并可选择查看详情。"""
    while True:
        notes = _load_notes()

        if not notes:
            console.print("[yellow]笔记本为空。[/yellow]")
            return

        # 按日期倒序排列
        sorted_notes = sorted(
            notes.items(),
            key=lambda x: x[1]["created_at"],
            reverse=True
        )

        # 显示笔记列表
        console.print("\n[bold cyan]📔 笔记列表[/bold cyan]")
        console.print("[dim]输入序号查看笔记，q 返回主菜单[/dim]\n")

        for idx, (note_id, note) in enumerate(sorted_notes, 1):
            title = note["title"]
            date = note["created_at"]
            tags = note["tags"]
            content_preview = note["content"][:50].replace("\n", " ")

            tag_str = f" [dim]| tags: {tags}[/dim]" if tags else ""
            console.print(f"  [bold]{idx:2d}[/bold]. [{date}] {title}{tag_str}")
            console.print(f"       [dim]{content_preview}...[/dim]\n")

        # 等待用户输入
        choice = Prompt.ask("[bold yellow]选择[/bold yellow]", default="q")

        if choice.lower() in ("q", "quit", "exit"):
            return

        # 尝试解析序号
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(sorted_notes):
                note_id, note = sorted_notes[idx]
                show_note_detail(note_id, note)
            else:
                console.print("[red]序号超出范围[/red]")
        except ValueError:
            console.print("[red]请输入有效序号或 q[/red]")


def show_note_detail(note_id: str, note: dict):
    """显示单条笔记详情。"""
    console.print()
    console.print(Panel(
        Markdown(f"# {note['title']}\n\n{note['content']}"),
        title=f"[bold blue]{note['title']}[/bold blue]",
        subtitle=f"[dim]{note['created_at']} | id: {note_id}[/dim]",
        border_style="blue"
    ))
    if note["tags"]:
        console.print(f"[dim]标签: {note['tags']}[/dim]")
    console.print()
    Prompt.ask("[dim]按 Enter 返回列表[/dim]")


def edit_note_content(title: str, content: str, tags: str) -> tuple[str, str, str] | None:
    """在系统编辑器中编辑笔记内容，返回编辑后的 (title, content, tags)，失败或取消返回 None。"""
    editor = os.environ.get("EDITOR", "vim")

    # 准备编辑内容
    edit_content = f"""# 标题 (第一行是标题，后面的 # 开头的行是注释)
{title}

# 标签 (多个标签用逗号分隔)
{tags}

# 内容 (从这里开始是笔记正文)
{content}
"""

    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".md",
        delete=False,
        encoding="utf-8"
    ) as f:
        f.write(edit_content)
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
            lines = f.readlines()

        # 解析编辑后的内容
        new_title = ""
        new_tags = ""
        new_content_parts = []
        state = "title"  # title -> tags -> content

        for line in lines:
            stripped = line.rstrip("\n")
            if state == "title":
                if stripped.startswith("#"):
                    continue
                new_title = stripped
                state = "tags"
            elif state == "tags":
                if stripped.startswith("#"):
                    continue
                new_tags = stripped
                state = "content"
            elif state == "content":
                if stripped.startswith("# 内容"):
                    continue
                new_content_parts.append(line)

        # 移除内容开头可能的空行
        while new_content_parts and new_content_parts[0].strip() == "":
            new_content_parts.pop(0)

        new_content = "".join(new_content_parts).rstrip("\n")

        return (new_title, new_content, new_tags)

    except Exception as e:
        console.print(f"[red]编辑失败: {e}[/red]")
        return None
    finally:
        # 清理临时文件
        try:
            os.unlink(temp_path)
        except OSError:
            pass


def handle_add_note_approval(tool_call: dict) -> dict:
    """处理 add_note 工具的用户审批.

    Args:
        tool_call: 工具调用字典，包含 id, name, args

    Returns:
        审批结果字典，包含 action, tool_call_id, tool_name, args, message
    """
    args = tool_call.get("args", {})
    title = args.get("title", "")
    content = args.get("content", "")
    tags = args.get("tags", "")

    console.print()
    console.print(Panel(
        Text.assemble(
            ("📝 Agent 想要记录一条笔记\n\n", "bold yellow"),
            ("标题: ", "bold cyan"), (f"{title}\n\n"),
            ("内容: ", "bold cyan"), (f"{content}\n\n"),
            ("标签: ", "bold cyan"), (f"{tags}" if tags else "无"),
        ),
        title="[bold]待确认操作[/bold]",
        border_style="yellow"
    ))

    while True:
        console.print("\n[dim]y=确认保存 | n=拒绝 | e=编辑修改[/dim]")
        choice = Prompt.ask(
            "[bold yellow]是否保存这条笔记？[/bold yellow]",
            choices=["y", "n", "e"],
            default="y"
        )

        if choice == "y":
            # 直接执行 add_note
            result_msg = _execute_add_note(title, content, tags)
            return {
                "action": "approve",
                "tool_call_id": tool_call.get("id"),
                "tool_name": tool_call.get("name"),
                "args": args,
                "message": result_msg
            }

        elif choice == "n":
            return {
                "action": "reject",
                "tool_call_id": tool_call.get("id"),
                "tool_name": tool_call.get("name"),
                "message": "用户拒绝保存这条笔记"
            }

        elif choice == "e":
            # 使用 vim 编辑
            edited = edit_note_content(title, content, tags)
            if edited is not None:
                new_title, new_content, new_tags = edited

                # 显示修改后的内容再次确认
                console.print()
                console.print(Panel(
                    Text.assemble(
                        ("修改后的笔记：\n\n", "bold green"),
                        ("标题: ", "bold cyan"), (f"{new_title}\n\n"),
                        ("内容: ", "bold cyan"), (f"{new_content}\n\n"),
                        ("标签: ", "bold cyan"), (f"{new_tags}" if new_tags else "无"),
                    ),
                    title="[bold]确认修改[/bold]",
                    border_style="green"
                ))

                if Confirm.ask("[bold yellow]确认保存修改后的笔记？[/bold yellow]", default=True):
                    result_msg = _execute_add_note(new_title, new_content, new_tags)
                    return {
                        "action": "modify",
                        "tool_call_id": tool_call.get("id"),
                        "tool_name": tool_call.get("name"),
                        "args": {"title": new_title, "content": new_content, "tags": new_tags},
                        "message": result_msg
                    }
            else:
                console.print("[yellow]编辑已取消或无变化[/yellow]")


def _execute_add_note(title: str, content: str, tags: str = "") -> str:
    """直接执行 add_note 操作.

    Returns:
        成功消息字符串
    """
    note_id = str(uuid.uuid4())[:8]
    created_at = datetime.datetime.now().strftime("%Y-%m-%d")
    notes = _load_notes()
    notes[note_id] = {
        "title": title,
        "content": content,
        "tags": tags,
        "created_at": created_at,
    }
    _save_notes(notes)

    # 增量更新 FAISS 索引
    doc = Document(
        page_content=f"{title}\n{content}",
        metadata={"note_id": note_id, "title": title,
                  "tags": tags, "created_at": created_at}
    )
    vectorstore = _get_notes_vectorstore()
    if vectorstore is not None:
        vectorstore.add_documents([doc])
    else:
        embeddings = _get_embeddings()
        from langchain_community.vectorstores import FAISS
        _notes_vectorstore_cache = FAISS.from_documents([doc], embeddings)
        vectorstore = _notes_vectorstore_cache

    os.makedirs(NOTES_FAISS_DIR, exist_ok=True)
    vectorstore.save_local(NOTES_FAISS_DIR)

    return f"笔记已保存，id: {note_id}，标题：{title}"


def check_and_handle_interrupt(agent, config: dict) -> bool:
    """检查是否有中断状态，并处理工具审批.

    Args:
        agent: agent 执行器
        config: 配置字典

    Returns:
        True if 需要继续执行，False 表示结束或无中断
    """
    from langgraph.prebuilt import ToolNode

    # 获取当前状态
    state = agent.get_state(config)
    if not state.next:
        return False

    # 检查是否在 tools 节点前中断（待审批状态）
    if "tools" in state.next:
        messages = state.values.get("messages", [])
        if not messages:
            return False

        last_msg = messages[-1]
        if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
            # 检查是否有需要审批的工具
            has_approval_tool = False
            for tc in last_msg.tool_calls:
                if tc.get("name") in TOOLS_REQUIRING_APPROVAL:
                    has_approval_tool = True
                    break

            if has_approval_tool:
                # 有需要审批的工具，走审批流程
                approval_results = []
                normal_tool_calls = []

                for tc in last_msg.tool_calls:
                    tool_name = tc.get("name")
                    if tool_name in TOOLS_REQUIRING_APPROVAL:
                        if tool_name == "add_note":
                            result = handle_add_note_approval(tc)
                            approval_results.append(result)
                    else:
                        normal_tool_calls.append(tc)

                # 构建结果消息
                new_messages = []

                # 添加审批工具的结果
                for res in approval_results:
                    tool_msg = ToolMessage(
                        content=res["message"],
                        name=res["tool_name"],
                        tool_call_id=res["tool_call_id"]
                    )
                    new_messages.append(tool_msg)
                    # 显示工具结果
                    console.print(format_tool_result(res["tool_name"], res["message"]))

                # 如果有普通工具，单独执行
                if normal_tool_calls:
                    # 创建只包含普通工具的消息
                    from copy import deepcopy
                    temp_last_msg = deepcopy(last_msg)
                    temp_last_msg.tool_calls = normal_tool_calls

                    temp_state = {
                        "messages": messages[:-1] + [temp_last_msg]
                    }

                    # 工具列表
                    tools = [
                        get_environment_context,
                        update_user_memory,
                        search_memory,
                        get_memory,
                        web_search,
                        add_note,
                        search_notes,
                        get_note,
                    ]

                    # 显示普通工具调用
                    for tc in normal_tool_calls:
                        console.print(format_tool_call(tc))

                    # 执行普通工具
                    tool_node = ToolNode(tools)
                    try:
                        result = tool_node.invoke(temp_state, config)
                        for msg in result["messages"]:
                            new_messages.append(msg)
                            # 显示工具结果
                            console.print(format_tool_result(msg.name, str(msg.content)))
                    except Exception as e:
                        console.print(f"[red]执行工具出错: {e}[/red]")

                # 更新状态
                if new_messages:
                    agent.update_state(
                        config,
                        {"messages": new_messages},
                        as_node="tools"
                    )

                return True
            else:
                # 没有需要审批的工具，直接继续执行
                return True

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


def stream_agent_response(agent, user_input: str, config: dict) -> str:
    """Stream agent response with tool calls and thinking visible.

    Returns the final response content for potential copying.
    """
    console.print("[bold blue]Agent:[/bold blue]")

    # 第一步：发送用户输入
    final_output = ""
    current_input = {"messages": [("user", user_input)]}

    while True:
        # Track state for streaming
        current_content = ""
        final_content = ""
        pending_tool_calls = {}
        printed_tool_calls = set()

        # Use Live for real-time markdown rendering
        with Live(Markdown(""), console=console, refresh_per_second=10, vertical_overflow="visible") as live:
            try:
                # 遍历 stream
                for event in agent.stream(
                    current_input,
                    config,
                    stream_mode="messages"
                ):
                    msg, metadata = event

                    # Handle AI message chunks (streaming text)
                    if isinstance(msg, AIMessageChunk):
                        if msg.content:
                            current_content += normalize_llm_content(msg.content)
                            live.update(Markdown(current_content))

                        # Handle tool calls
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
                                    if chunk.get("id"):
                                        pending_tool_calls[idx]["id"] = chunk["id"]
                                    if chunk.get("name"):
                                        pending_tool_calls[idx]["name"] = chunk["name"]
                                    if chunk.get("args"):
                                        pending_tool_calls[idx]["args"] += chunk["args"]

                    # Handle tool messages (results)
                    elif isinstance(msg, ToolMessage):
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

                        pending_tool_calls.clear()
                        printed_tool_calls.clear()
                        if current_content:
                            final_content = current_content
                        current_content = ""
                        console.print("[bold blue]Agent:[/bold blue]")

                        live.start()
            except Exception:
                pass

        # 保存输出
        iteration_output = current_content if current_content else final_content
        if iteration_output:
            final_output = iteration_output

        # 检查中断
        if check_and_handle_interrupt(agent, config):
            current_input = None  # 继续执行
            continue

        break

    return final_output


def blocking_agent_response(agent, user_input: str, config: dict) -> str:
    """Non-streaming agent response with tool calls visible.

    Returns the final response content for potential copying.
    """
    console.print("[bold blue]Agent:[/bold blue]")

    final_content = ""
    current_input = {"messages": [("user", user_input)]}

    while True:
        response = agent.invoke(current_input, config)

        # 提取所有消息，显示工具调用信息
        messages = response.get("messages", [])

        for msg in messages:
            # 显示工具调用
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                for tc in msg.tool_calls:
                    console.print(format_tool_call({
                        "name": tc.get("name", "unknown"),
                        "args": tc.get("args", {})
                    }))

            # 显示工具返回结果
            if isinstance(msg, ToolMessage):
                tool_name = msg.name or "unknown"
                console.print(format_tool_result(tool_name, str(msg.content)))

            # 获取最终 AI 响应内容
            if hasattr(msg, "content") and msg.content and hasattr(msg, "type") and msg.type == "ai":
                content = normalize_llm_content(msg.content)
                if content:
                    final_content = content

        # 检查中断
        if check_and_handle_interrupt(agent, config):
            current_input = None  # 继续执行
            continue

        break

    # 渲染最终响应
    if final_content:
        console.print(Markdown(final_content))

    return final_content


def copy_to_clipboard(text: str) -> bool:
    """Copy text to system clipboard. Returns True on success."""
    import sys
    try:
        if sys.platform == "darwin":
            subprocess.run(["pbcopy"], input=text.encode("utf-8"), check=True)
        elif sys.platform == "win32":
            subprocess.run(["clip"], input=text.encode("utf-8"), check=True)
        else:
            # Linux - try xclip first, then xsel
            try:
                subprocess.run(
                    ["xclip", "-selection", "clipboard"],
                    input=text.encode("utf-8"),
                    check=True
                )
            except FileNotFoundError:
                subprocess.run(
                    ["xsel", "--clipboard", "--input"],
                    input=text.encode("utf-8"),
                    check=True
                )
        return True
    except Exception as e:
        console.print(f"[red]复制失败: {e}[/red]")
        return False


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
    session = PromptSession(
        history=InMemoryHistory(),
    )

    # Define custom key bindings
    bindings = KeyBindings()

    @bindings.add('enter')
    def _(event):
        """Bind Enter to submit the buffer."""
        event.current_buffer.validate_and_handle()

    output_mode = "流式" if STREAM_OUTPUT else "阻塞"
    console.print(f"[dim]Session ID: {thread_id}[/dim]")
    console.print(f"[dim]输出模式: {output_mode}[/dim]")
    console.print("[dim]命令: /notes 浏览笔记 | /tidy 整理记忆 | /clear 清空上下文 | /copy 复制上轮输出 | /exit 退出[/dim]")
    console.print("[dim]─" * 50 + "[/dim]")

    last_response = ""
    use_prompt_toolkit = True

    while True:
        try:
            if use_prompt_toolkit:
                try:
                    user_input = session.prompt(
                        "User: ",
                        multiline=False,
                        key_bindings=bindings,
                        wrap_lines=False,
                    )
                except Exception as e:
                    console.print(f"[yellow]警告: 输入组件出错，切换到兼容模式 ({e})[/yellow]")
                    console.print("[dim]提示: 兼容模式下使用上下方向键浏览历史输入[/dim]")
                    use_prompt_toolkit = False
                    user_input = input("User: ")
            else:
                user_input = input("User: ")

            stripped_input = user_input.strip()

            if stripped_input == "/exit":
                console.print("[dim]再见！[/dim]")
                break

            if stripped_input == "/notes":
                view_notes_menu()
                continue

            if stripped_input == "/tidy":
                tidy_memory()
                continue

            if stripped_input == "/clear":
                thread_id = str(uuid.uuid4())
                config = {"configurable": {"thread_id": thread_id}}
                console.print("[green]✓ 上下文已清空，新对话已开始[/green]")
                console.print(f"[dim]Session ID: {thread_id}[/dim]")
                continue

            if stripped_input == "/copy":
                if last_response:
                    if copy_to_clipboard(last_response):
                        console.print("[green]✓ 已复制到剪贴板[/green]")
                else:
                    console.print("[yellow]没有可复制的内容[/yellow]")
                continue

            if not stripped_input:
                continue

            if STREAM_OUTPUT:
                last_response = stream_agent_response(agent, user_input, config)
            else:
                last_response = blocking_agent_response(agent, user_input, config)
            print()

        except (KeyboardInterrupt, EOFError):
            console.print("\n[dim]再见！[/dim]")
            break
        except Exception as e:
            console.print(f"[red]发生错误: {e}[/red]")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()
