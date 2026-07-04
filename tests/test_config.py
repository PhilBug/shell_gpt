from sgpt.config import DEFAULT_CONFIG


def test_provider_default_is_openai():
    assert DEFAULT_CONFIG["PROVIDER"] == "openai"


def test_anthropic_api_key_default_empty():
    assert DEFAULT_CONFIG["ANTHROPIC_API_KEY"] == ""


def test_anthropic_max_tokens_default():
    assert DEFAULT_CONFIG["ANTHROPIC_MAX_TOKENS"] == "4096"
