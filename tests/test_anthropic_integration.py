from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from sgpt import config, main
from sgpt.role import DefaultRoles, SystemRole
from sgpt.providers.anthropic_provider import AnthropicProvider

import typer

runner = CliRunner()
app = typer.Typer()
app.command()(main)


class FakeStream:
    """Stand-in for an Anthropic MessageStream over prebuilt events."""

    def __init__(self, events: List[Any]):
        self._events = events

    def __enter__(self):
        return iter(self._events)

    def __exit__(self, *a):
        return None


class FakeEvent:
    def __init__(self, etype, **kw):
        self.type = etype
        for k, v in kw.items():
            setattr(self, k, v)


class FakeDelta:
    def __init__(self, **kw):
        self.type = kw.pop("type")
        for k, v in kw.items():
            setattr(self, k, v)


class FakeBlock:
    def __init__(self, btype, **kw):
        self.type = btype
        for k, v in kw.items():
            setattr(self, k, v)


def _text_events(text: str) -> List[Any]:
    return [
        FakeEvent(
            "content_block_start",
            index=0,
            content_block=FakeBlock("text", text=""),
        ),
        FakeEvent(
            "content_block_delta",
            index=0,
            delta=FakeDelta(type="text_delta", text=text),
        ),
        FakeEvent("content_block_stop", index=0),
        FakeEvent(
            "message_delta",
            delta=FakeDelta(type="message_delta", stop_reason="end_turn"),
            usage=MagicMock(),
        ),
        FakeEvent("message_stop"),
    ]


@pytest.fixture
def anthropic_provider(monkeypatch):
    """Patch PROVIDER and install a fake Anthropic client on the provider."""
    monkeypatch.setenv("PROVIDER", "anthropic")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setitem(config.DEFAULT_CONFIG, "PROVIDER", "anthropic")
    monkeypatch.setitem(config.DEFAULT_CONFIG, "ANTHROPIC_API_KEY", "test-key")

    # Build a real provider, then swap its client for a fake.
    provider = AnthropicProvider()
    provider._client = MagicMock()
    provider._client.messages.stream = MagicMock(
        side_effect=lambda **kw: FakeStream(_text_events("Prague"))
    )
    monkeypatch.setattr("sgpt.handlers.handler.provider", provider)
    monkeypatch.setattr(
        "sgpt.handlers.handler.completion", provider.get_completion
    )
    return provider


def test_anthropic_text_completion(anthropic_provider, monkeypatch):
    # Don't read .sgptrc provider; force the handler-level patch above.
    captured: Dict[str, Any] = {}

    def spy_stream(**kwargs):
        captured.update(kwargs)
        return FakeStream(_text_events("Prague"))

    anthropic_provider._client.messages.stream = spy_stream

    args = ["capital of the Czech Republic?", "--no-cache", "--no-functions"]
    result = runner.invoke(app, args)

    assert result.exit_code == 0, result.output
    assert "Prague" in result.output
    # system message must be split out to the top-level system param
    assert captured["system"].startswith("You are")
    assert all(m["role"] != "system" for m in captured["messages"])
    assert captured["max_tokens"] == 4096
