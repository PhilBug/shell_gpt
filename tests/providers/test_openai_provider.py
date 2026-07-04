from datetime import datetime
from typing import List

from openai.types.chat.chat_completion_chunk import (
    ChatCompletionChunk,
)
from openai.types.chat.chat_completion_chunk import (
    Choice as StreamChoice,
)
from openai.types.chat.chat_completion_chunk import (
    ChoiceDelta,
    ChoiceDeltaToolCall,
    ChoiceDeltaToolCallFunction,
)

from sgpt.config import cfg
from sgpt.providers.base import ToolCallDelta
from sgpt.providers.openai_provider import OpenAIProvider


def _chunk(content=None, finish_reason=None, tool_calls=None):
    return ChatCompletionChunk(
        id="foo",
        model=cfg.get("DEFAULT_MODEL"),
        object="chat.completion.chunk",
        choices=[
            StreamChoice(
                index=0,
                finish_reason=finish_reason,
                delta=ChoiceDelta(
                    content=content,
                    role="assistant",
                    tool_calls=tool_calls,
                ),
            )
        ],
        created=int(datetime.now().timestamp()),
    )


def _tool_call_delta(id=None, name=None, arguments=None):
    """Build a minimal OpenAI tool-call delta like the SDK returns.

    Uses real SDK pydantic objects so the surrounding ChatCompletionChunk
    validates. Only the supplied fields are populated, mirroring how the
    OpenAI streaming API streams a tool call in pieces (id+name first,
    then argument fragments).
    """
    return [
        ChoiceDeltaToolCall(
            index=0,
            id=id,
            function=ChoiceDeltaToolCallFunction(name=name, arguments=arguments),
            type="function",
        )
    ]


def test_parse_stream_text_only():
    provider = OpenAIProvider()
    response = [_chunk("Hello"), _chunk(" world"), _chunk(finish_reason="stop")]
    out: List[str] = []
    tool_calls: List[ToolCallDelta] = []
    for text, tool in provider.parse_stream(response):
        if text:
            out.append(text)
        if tool:
            tool_calls.append(tool)
    assert "".join(out) == "Hello world"
    assert tool_calls == []


def test_parse_stream_tool_call_finishes():
    provider = OpenAIProvider()
    response = [
        _chunk(tool_calls=_tool_call_delta(id="call_1", name="execute_shell")),
        _chunk(tool_calls=_tool_call_delta(arguments='{"shell')),
        _chunk(tool_calls=_tool_call_delta(arguments='_command": "ls"}')),
        _chunk(finish_reason="tool_calls"),
    ]
    texts: List[str] = []
    tools: List[ToolCallDelta] = []
    for text, tool in provider.parse_stream(response):
        if text:
            texts.append(text)
        if tool:
            tools.append(tool)
    assert texts == []
    assert len(tools) == 1
    assert tools[0].id == "call_1"
    assert tools[0].name == "execute_shell"
    assert tools[0].arguments_json == '{"shell_command": "ls"}'
    assert tools[0].finish is True
