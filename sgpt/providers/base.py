from dataclasses import dataclass
from typing import Any, Dict, Generator, List, Optional, Protocol, Tuple


@dataclass
class ToolCallDelta:
    """Provider-neutral representation of a streaming tool call."""

    id: str
    name: str
    arguments_json: str
    finish: bool


class Provider(Protocol):
    """Each provider translates between ShellGPT's OpenAI-shaped internals
    and the SDK it wraps. Handlers talk to providers exclusively through
    this interface."""

    def get_completion(
        self,
        *,
        model: str,
        temperature: float,
        top_p: float,
        messages: List[Dict[str, Any]],
        functions: Optional[List[Dict[str, Any]]] = None,
    ) -> Generator[str, None, None]:
        ...

    def parse_stream(
        self, response: Any
    ) -> Generator[
        Tuple[Optional[str], Optional[ToolCallDelta]], None, None
    ]:
        ...

    def render_tool_call_assistant(
        self, tool_call_id: str, name: str, arguments: str
    ) -> Dict[str, Any]:
        ...

    def render_tool_result(
        self, tool_call_id: str, result: str
    ) -> Dict[str, Any]:
        ...
