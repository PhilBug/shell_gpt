import importlib

import sgpt.config as config_module


def _reload_with_env(monkeypatch, env: dict):
    """Reload sgpt.config with a controlled environment so DEFAULT_CONFIG
    reflects the literal source-of-truth defaults, not the test runner's
    ambient env vars (e.g. a real ANTHROPIC_API_KEY in the shell)."""
    for key in (
        "PROVIDER",
        "ANTHROPIC_API_KEY",
        "ANTHROPIC_MAX_TOKENS",
        "ANTHROPIC_BASE_URL",
        "ANTHROPIC_THINKING",
    ):
        monkeypatch.delenv(key, raising=False)
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    return importlib.reload(config_module)


def test_provider_default_is_openai(monkeypatch):
    mod = _reload_with_env(monkeypatch, {})
    assert mod.DEFAULT_CONFIG["PROVIDER"] == "openai"


def test_anthropic_api_key_default_empty(monkeypatch):
    mod = _reload_with_env(monkeypatch, {})
    assert mod.DEFAULT_CONFIG["ANTHROPIC_API_KEY"] == ""


def test_anthropic_max_tokens_default(monkeypatch):
    mod = _reload_with_env(monkeypatch, {})
    assert mod.DEFAULT_CONFIG["ANTHROPIC_MAX_TOKENS"] == "4096"


def test_anthropic_base_url_default(monkeypatch):
    mod = _reload_with_env(monkeypatch, {})
    assert mod.DEFAULT_CONFIG["ANTHROPIC_BASE_URL"] == "default"


def test_anthropic_thinking_default(monkeypatch):
    mod = _reload_with_env(monkeypatch, {})
    assert mod.DEFAULT_CONFIG["ANTHROPIC_THINKING"] == "adaptive"
