import pytest


@pytest.fixture(autouse=True)
def _anthropic_api_key(monkeypatch):
    """AnthropicProvider.__init__ reads ANTHROPIC_API_KEY via cfg.get,
    which raises UsageError on an empty value. Provide a test key so any
    provider test that constructs AnthropicProvider() works offline."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
