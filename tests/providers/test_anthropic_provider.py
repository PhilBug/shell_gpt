import json
from typing import Any, Dict, List
from unittest.mock import MagicMock

import pytest

from sgpt.providers.anthropic_provider import AnthropicProvider
from sgpt.providers.base import ToolCallDelta


def test_translate_tools_openai_to_anthropic():
    provider = AnthropicProvider()
    openai_tools = [
        {
            "type": "function",
            "function": {
                "name": "execute_shell_command",
                "description": "Runs a shell command.",
                "parameters": {
                    "type": "object",
                    "properties": {"shell_command": {"type": "string"}},
                    "required": ["shell_command"],
                },
            },
        }
    ]
    result = provider.translate_tools(openai_tools)
    assert result == [
        {
            "name": "execute_shell_command",
            "description": "Runs a shell command.",
            "input_schema": {
                "type": "object",
                "properties": {"shell_command": {"type": "string"}},
                "required": ["shell_command"],
            },
        }
    ]


def test_translate_tools_none():
    provider = AnthropicProvider()
    assert provider.translate_tools(None) is None


def test_split_system_message_extracts_system():
    provider = AnthropicProvider()
    messages = [
        {"role": "system", "content": "You are ShellGPT"},
        {"role": "user", "content": "hi"},
    ]
    system, rest = provider.split_system_message(messages)
    assert system == "You are ShellGPT"
    assert rest == [{"role": "user", "content": "hi"}]


def test_split_system_message_absent():
    provider = AnthropicProvider()
    messages = [{"role": "user", "content": "hi"}]
    system, rest = provider.split_system_message(messages)
    assert system is None
    assert rest == messages


def test_get_completion_builds_anthropic_call(monkeypatch):
    provider = AnthropicProvider()
    captured: Dict[str, Any] = {}

    class FakeStream:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return None

        def __iter__(self):
            return iter([])

    def fake_stream(**kwargs):
        captured.update(kwargs)
        return FakeStream()

    monkeypatch.setattr(provider._client.messages, "stream", fake_stream)
    monkeypatch.setattr(provider, "parse_stream", lambda resp: iter([]))

    messages = [
        {"role": "system", "content": "You are ShellGPT"},
        {"role": "user", "content": "hi"},
    ]
    list(
        provider.get_completion(
            model="claude-sonnet-4-5-20250929",
            temperature=0.0,
            top_p=1.0,
            messages=messages,
            functions=None,
        )
    )

    assert captured["model"] == "claude-sonnet-4-5-20250929"
    assert captured["system"] == "You are ShellGPT"
    assert captured["messages"] == [{"role": "user", "content": "hi"}]
    assert captured["max_tokens"] == 4096
    assert captured["temperature"] == 0.0
    assert captured["top_p"] == 1.0
    assert "tools" not in captured


def test_get_completion_passes_tools(monkeypatch):
    provider = AnthropicProvider()

    class FakeStream:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return None

        def __iter__(self):
            return iter([])

    captured: Dict[str, Any] = {}

    def fake_stream(**kwargs):
        captured.update(kwargs)
        return FakeStream()

    monkeypatch.setattr(provider._client.messages, "stream", fake_stream)
    monkeypatch.setattr(provider, "parse_stream", lambda resp: iter([]))

    openai_tools = [
        {
            "type": "function",
            "function": {
                "name": "execute_shell_command",
                "description": "run",
                "parameters": {
                    "type": "object",
                    "properties": {"shell_command": {"type": "string"}},
                    "required": ["shell_command"],
                },
            },
        }
    ]
    list(
        provider.get_completion(
            model="claude-sonnet-4-5-20250929",
            temperature=0.0,
            top_p=1.0,
            messages=[{"role": "user", "content": "ls"}],
            functions=openai_tools,
        )
    )

    assert captured["tools"] == [
        {
            "name": "execute_shell_command",
            "description": "run",
            "input_schema": {
                "type": "object",
                "properties": {"shell_command": {"type": "string"}},
                "required": ["shell_command"],
            },
        }
    ]


def test_parse_stream_text_and_tool_use():
    """parse_stream yields text deltas then a finished ToolCallDelta on
    stop_reason == tool_use."""

    provider = AnthropicProvider()

    class FakeBlock:
        def __init__(self, btype, **kw):
            self.type = btype
            for k, v in kw.items():
                setattr(self, k, v)

    class FakeEvent:
        def __init__(self, etype, **kw):
            self.type = etype
            for k, v in kw.items():
                setattr(self, k, v)

    tool_use_block = FakeBlock(
        "tool_use",
        id="toolu_1",
        name="execute_shell_command",
        input={"shell_command": "ls"},
    )

    events = [
        FakeEvent("content_block_start", index=0, content_block=FakeBlock("text", text="")),
        FakeEvent("content_block_delta", index=0, delta=MagicMock(text="Runni", type="text_delta")),
        FakeEvent("content_block_delta", index=0, delta=MagicMock(text="ng.", type="text_delta")),
        FakeEvent("content_block_stop", index=0),
        FakeEvent("content_block_start", index=1, content_block=tool_use_block),
        FakeEvent("content_block_delta", index=1, delta=MagicMock(partial_json='{"shell', type="input_json_delta")),
        FakeEvent("content_block_delta", index=1, delta=MagicMock(partial_json='_command": "ls"}', type="input_json_delta")),
        FakeEvent("content_block_stop", index=1),
        FakeEvent("message_delta", delta=MagicMock(stop_reason="tool_use"), usage=MagicMock()),
        FakeEvent("message_stop"),
    ]

    texts: List[str] = []
    tools: List[ToolCallDelta] = []
    for text, tool in provider.parse_stream(iter(events)):
        if text:
            texts.append(text)
        if tool:
            tools.append(tool)

    assert "".join(texts) == "Running."
    assert len(tools) == 1
    assert tools[0].id == "toolu_1"
    assert tools[0].name == "execute_shell_command"
    assert tools[0].arguments_json == '{"shell_command": "ls"}'
    assert tools[0].finish is True
