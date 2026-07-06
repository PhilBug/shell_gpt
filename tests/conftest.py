import os

# The pre-existing test suite (test_default, test_code, test_shell, ...)
# patches sgpt.handlers.handler.completion with OpenAI-shaped
# ChatCompletionChunk mocks. That only works when the handler is wired to
# the OpenAI provider at import time. On a dev machine whose ~/.sgptrc
# sets PROVIDER=anthropic, the handler would otherwise build an
# AnthropicProvider whose parse_stream can't consume those mocks.
# Force PROVIDER=openai before any test module imports sgpt.handlers, so
# the suite is deterministic regardless of the developer's local config.
# (CI runners have no ~/.sgptrc, so this only affects local runs.)
# Anthropic-specific tests construct AnthropicProvider() directly and
# monkeypatch the handler, so they don't depend on this default.
os.environ["PROVIDER"] = "openai"
os.environ.setdefault("OPENAI_API_KEY", "test-key")

import pytest  # noqa: E402  (must run AFTER the env setup above)


@pytest.fixture(autouse=True)
def mock_os_name(monkeypatch):
    monkeypatch.setattr(os, "name", "test")
