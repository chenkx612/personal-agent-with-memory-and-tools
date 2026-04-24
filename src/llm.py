"""LLM initialization for the personal agent."""

from langchain_core.messages import AIMessage, AIMessageChunk
from langchain_openai import ChatOpenAI
from core import load_config


class ReasoningChatOpenAI(ChatOpenAI):
    """ChatOpenAI 扩展：支持带思维链 (reasoning_content) 的模型（如 DeepSeek-Reasoner）。

    1. 流式 chunk 解析：把 delta['reasoning_content'] 透传到
       AIMessageChunk.additional_kwargs['reasoning_content']，便于 UI 实时展示。
    2. 请求 payload：把历史 AIMessage.additional_kwargs['reasoning_content'] 作为顶层字段
       回传给服务端 —— DeepSeek 文档要求两个 user 消息之间发生过工具调用时，
       assistant 的 reasoning_content 必须原样回传，否则会返回 400。
    """

    def _convert_chunk_to_generation_chunk(self, chunk, default_chunk_class, base_generation_info):
        generation_chunk = super()._convert_chunk_to_generation_chunk(
            chunk, default_chunk_class, base_generation_info
        )
        if generation_chunk is None:
            return generation_chunk

        choices = chunk.get("choices") or chunk.get("chunk", {}).get("choices", [])
        if not choices:
            return generation_chunk
        delta = choices[0].get("delta") or {}
        reasoning_content = delta.get("reasoning_content")
        if reasoning_content and isinstance(generation_chunk.message, AIMessageChunk):
            generation_chunk.message.additional_kwargs["reasoning_content"] = reasoning_content
        return generation_chunk

    def _get_request_payload(self, input_, *, stop=None, **kwargs):
        payload = super()._get_request_payload(input_, stop=stop, **kwargs)

        # responses api 路径不处理，只补齐 chat/completions
        if "messages" not in payload:
            return payload

        # 按顺序把原始 AIMessage 的 reasoning_content 回写到 assistant dict 顶层
        original_messages = self._convert_input(input_).to_messages()
        ai_iter = iter(m for m in original_messages if isinstance(m, AIMessage))
        for msg_dict in payload["messages"]:
            if msg_dict.get("role") != "assistant":
                continue
            try:
                original = next(ai_iter)
            except StopIteration:
                break
            rc = original.additional_kwargs.get("reasoning_content") if original.additional_kwargs else None
            if rc:
                msg_dict["reasoning_content"] = rc
        return payload


def get_llm():
    """Get LLM instance for standalone use (e.g., memory tidying)."""
    config = load_config()
    llm_config = config.get("llm", {})

    api_key = llm_config.get("api_key")
    model = llm_config.get("model")
    base_url = llm_config.get("base_url")
    reasoning_effort = llm_config.get("reasoning_effort", "high")
    thinking_type = llm_config.get("thinking_type", "disabled")

    if not api_key:
        print("Warning: api_key not found in config.yaml.")

    return ReasoningChatOpenAI(
        model=model,
        api_key=api_key,
        base_url=base_url,
        reasoning_effort=reasoning_effort,
        extra_body={"thinking": {"type": thinking_type}}
    )
