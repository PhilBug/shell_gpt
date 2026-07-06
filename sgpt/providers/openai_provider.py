from typing import Any, Dict, Generator, List, Optional, Tuple

from ..config import cfg
from .base import ToolCallDelta

base_url = cfg.get("API_BASE_URL")
use_litellm = cfg.get("USE_LITELLM") == "true"

if use_litellm:
    import litellm  # type: ignore

    _litellm_completion = litellm.completion
    litellm.suppress_debug_info = True
    _openai_client = None
else:
    from openai import OpenAI

    additional_kwargs = {
        "timeout": int(cfg.get("REQUEST_TIMEOUT")),
        "api_key": cfg.get("OPENAI_API_KEY"),
        "base_url": None if base_url == "default" else base_url,
    }
    _openai_client = OpenAI(**additional_kwargs)  # type: ignore
    _litellm_completion = None


class OpenAIProvider:
    """Wraps the OpenAI SDK (or LiteLLM) behind the Provider interface.

    This is a lift-and-shift of the wiring that previously lived at module
    scope in sgpt/handlers/handler.py.
    """

    use_litellm = use_litellm

    def get_completion(
        self,
        *,
        model: str,
        temperature: float,
        top_p: float,
        messages: List[Dict[str, Any]],
        functions: Optional[List[Dict[str, Any]]] = None,
    ) -> Generator[str, None, None]:
        kwargs: Dict[str, Any] = {
            "model": model,
            "temperature": temperature,
            "top_p": top_p,
            "messages": messages,
            "stream": True,
        }
        if use_litellm:
            kwargs["timeout"] = int(cfg.get("REQUEST_TIMEOUT"))
            kwargs["api_key"] = cfg.get("OPENAI_API_KEY")
            kwargs["base_url"] = None if base_url == "default" else base_url
            if functions:
                kwargs["tool_choice"] = "auto"
                kwargs["tools"] = functions
                kwargs["parallel_tool_calls"] = False
            return _litellm_completion(**kwargs)  # type: ignore

        if functions:
            kwargs["tool_choice"] = "auto"
            kwargs["tools"] = functions
            kwargs["parallel_tool_calls"] = False
        return _openai_client.chat.completions.create(**kwargs)  # type: ignore

    def parse_stream(
        self, response: Any
    ) -> Generator[Tuple[Optional[str], Optional[ToolCallDelta]], None, None]:
        tool_call_id = ""
        name = ""
        arguments = ""

        for chunk in response:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta

            # LiteLLM uses dict instead of Pydantic object like OpenAI does.
            tool_calls = delta.get("tool_calls") if use_litellm else delta.tool_calls
            if tool_calls:
                for tool_call in tool_calls:
                    if use_litellm:
                        tool_call_id = tool_call.get("id") or tool_call_id
                        name = tool_call.get("function", {}).get("name") or name
                        arguments += tool_call.get("function", {}).get("arguments", "")
                    else:
                        tool_call_id = tool_call.id or tool_call_id
                        name = tool_call.function.name or name
                        arguments += tool_call.function.arguments or ""

            if chunk.choices[0].finish_reason == "tool_calls":
                yield None, ToolCallDelta(
                    id=tool_call_id,
                    name=name,
                    arguments_json=arguments,
                    finish=True,
                )
                return

            content = delta.get("content") if use_litellm else delta.content
            if content:
                yield content, None

    def render_tool_call_assistant(
        self, tool_call_id: str, name: str, arguments: str
    ) -> Dict[str, Any]:
        return {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": tool_call_id,
                    "type": "function",
                    "function": {"name": name, "arguments": arguments},
                }
            ],
        }

    def render_tool_result(self, tool_call_id: str, result: str) -> Dict[str, Any]:
        return {"role": "tool", "content": result, "tool_call_id": tool_call_id}
