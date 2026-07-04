import json
from typing import Any, Dict, Generator, List, Optional, Tuple

from click import UsageError

from ..config import cfg
from .base import ToolCallDelta


class AnthropicProvider:
    """Adapts ShellGPT's OpenAI-shaped internals to the Anthropic SDK.

    All translation happens here at the call boundary; handlers stay
    OpenAI-shaped.
    """

    def __init__(self) -> None:
        from anthropic import Anthropic

        api_key = cfg.get("ANTHROPIC_API_KEY")
        base_url_cfg = cfg.get("API_BASE_URL")
        self._client = Anthropic(
            api_key=api_key,
            base_url=None if base_url_cfg == "default" else base_url_cfg,
            timeout=float(cfg.get("REQUEST_TIMEOUT")),
        )

    # --- translation helpers ---

    def translate_tools(
        self, functions: Optional[List[Dict[str, Any]]]
    ) -> Optional[List[Dict[str, Any]]]:
        if not functions:
            return None
        tools: List[Dict[str, Any]] = []
        for fn in functions:
            spec = fn.get("function", fn)
            tools.append(
                {
                    "name": spec["name"],
                    "description": spec.get("description", ""),
                    "input_schema": spec.get(
                        "parameters",
                        {"type": "object", "properties": {}, "required": []},
                    ),
                }
            )
        return tools

    def split_system_message(
        self, messages: List[Dict[str, Any]]
    ) -> Tuple[Optional[str], List[Dict[str, Any]]]:
        if messages and messages[0].get("role") == "system":
            return messages[0]["content"], messages[1:]
        return None, messages

    def convert_history_tools(
        self, messages: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Convert any OpenAI-shaped tool messages in prior turns to
        Anthropic tool_use / tool_result content blocks.

        Needed when a chat session resumed from cache contains tool
        exchanges that were stored in OpenAI shape.
        """
        converted: List[Dict[str, Any]] = []
        for msg in messages:
            role = msg.get("role")
            if role == "tool":
                converted.append(
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": msg.get("tool_call_id", ""),
                                "content": msg.get("content", ""),
                            }
                        ],
                    }
                )
            elif role == "assistant" and msg.get("tool_calls"):
                blocks: List[Dict[str, Any]] = []
                for tc in msg["tool_calls"]:
                    fn = tc.get("function", {})
                    try:
                        parsed_input = json.loads(fn.get("arguments", "{}"))
                    except json.JSONDecodeError:
                        parsed_input = {}
                    blocks.append(
                        {
                            "type": "tool_use",
                            "id": tc.get("id", ""),
                            "name": fn.get("name", ""),
                            "input": parsed_input,
                        }
                    )
                converted.append({"role": "assistant", "content": blocks})
            else:
                converted.append(msg)
        return converted

    # --- Provider interface ---

    def get_completion(
        self,
        *,
        model: str,
        temperature: float,
        top_p: float,
        messages: List[Dict[str, Any]],
        functions: Optional[List[Dict[str, Any]]] = None,
    ) -> Any:
        system_text, rest = self.split_system_message(messages)
        anthropic_messages = self.convert_history_tools(rest)
        kwargs: Dict[str, Any] = {
            "model": model,
            "max_tokens": int(cfg.get("ANTHROPIC_MAX_TOKENS")),
            "messages": anthropic_messages,
            "temperature": temperature,
            "top_p": top_p,
        }
        if system_text is not None:
            kwargs["system"] = system_text
        tools = self.translate_tools(functions)
        if tools:
            kwargs["tools"] = tools

        # Return the raw MessageStreamManager; the handler owns iteration
        # via parse_stream (mirrors how OpenAIProvider returns a chunk
        # stream for the handler to walk).
        return self._client.messages.stream(**kwargs)

    def parse_stream(
        self, response: Any
    ) -> Generator[Tuple[Optional[str], Optional[ToolCallDelta]], None, None]:
        """Iterate an Anthropic MessageStream, yielding (text, tool_delta).

        Anthropic streaming events:
          - content_block_start.content_block  -> opens a block (text or tool_use)
          - content_block_delta.delta          -> text_delta or input_json_delta
          - content_block_stop                 -> block complete
          - message_delta.delta.stop_reason    -> "tool_use" or "end_turn"
          - message_stop                       -> stream end

        Tool-call input JSON arrives as input_json_delta.partial_json chunks
        we concatenate, then surface as a finished ToolCallDelta when the
        block closes and stop_reason is tool_use.
        """
        # active_tool_input[block_index] accumulates the partial_json for a
        # tool_use block, keyed by block index.
        tool_inputs: Dict[int, str] = {}
        tool_meta: Dict[int, Dict[str, str]] = {}
        stop_reason: Optional[str] = None

        # Anthropic's stream() returns a MessageStreamManager context
        # manager; the raw iterable may also be passed directly.
        stream = (
            response.__enter__() if hasattr(response, "__enter__") else response
        )
        try:
            for event in stream:
                etype = getattr(event, "type", None)

                if etype == "content_block_start":
                    block = event.content_block
                    if getattr(block, "type", None) == "tool_use":
                        tool_meta[event.index] = {
                            "id": getattr(block, "id", ""),
                            "name": getattr(block, "name", ""),
                        }
                        tool_inputs[event.index] = ""

                elif etype == "content_block_delta":
                    delta = event.delta
                    dtype = getattr(delta, "type", None)
                    if dtype == "text_delta":
                        yield getattr(delta, "text", ""), None
                    elif dtype == "input_json_delta":
                        tool_inputs[event.index] += getattr(
                            delta, "partial_json", ""
                        )

                elif etype == "content_block_stop":
                    # Block closed; nothing to emit yet (we wait for the
                    # message-level stop_reason to confirm tool_use).
                    pass

                elif etype == "message_delta":
                    delta = event.delta
                    stop_reason = getattr(delta, "stop_reason", stop_reason)

                elif etype == "message_stop":
                    break

            if stop_reason == "tool_use":
                # Surface the first tool call only — matches today's
                # one-tool-per-turn behavior.
                for idx in sorted(tool_inputs.keys()):
                    meta = tool_meta.get(idx, {})
                    yield None, ToolCallDelta(
                        id=meta.get("id", ""),
                        name=meta.get("name", ""),
                        arguments_json=tool_inputs[idx],
                        finish=True,
                    )
                    return
        finally:
            if hasattr(response, "__exit__"):
                response.__exit__(None, None, None)

    def render_tool_call_assistant(
        self, tool_call_id: str, name: str, arguments: str
    ) -> Dict[str, Any]:
        try:
            parsed_input = json.loads(arguments) if arguments else {}
        except json.JSONDecodeError:
            parsed_input = {}
        return {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "id": tool_call_id,
                    "name": name,
                    "input": parsed_input,
                }
            ],
        }

    def render_tool_result(
        self, tool_call_id: str, result: str
    ) -> Dict[str, Any]:
        return {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": tool_call_id,
                    "content": result,
                }
            ],
        }
