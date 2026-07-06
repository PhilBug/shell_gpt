import pytest
from click import UsageError

from sgpt.providers import get_provider
from sgpt.providers.anthropic_provider import AnthropicProvider
from sgpt.providers.base import ToolCallDelta
from sgpt.providers.openai_provider import OpenAIProvider


def test_tool_call_delta_defaults():
    d = ToolCallDelta(id="t1", name="foo", arguments_json='{"a": 1}', finish=True)
    assert d.id == "t1"
    assert d.name == "foo"
    assert d.arguments_json == '{"a": 1}'
    assert d.finish is True


def test_get_provider_openai():
    assert isinstance(get_provider("openai"), OpenAIProvider)


def test_get_provider_anthropic():
    assert isinstance(get_provider("anthropic"), AnthropicProvider)


def test_get_provider_litellm_redirects_to_use_litellm():
    with pytest.raises(UsageError) as exc:
        get_provider("litellm")
    assert "USE_LITELLM" in str(exc.value)


def test_get_provider_unknown_raises():
    with pytest.raises(UsageError) as exc:
        get_provider("bogus")
    assert "bogus" in str(exc.value)
