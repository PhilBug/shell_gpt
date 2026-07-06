from click import UsageError

from .base import Provider, ToolCallDelta

__all__ = ["get_provider", "Provider", "ToolCallDelta"]


def get_provider(name: str) -> Provider:
    """Resolve a PROVIDER config value to a provider instance.

    Lazy imports keep the unused SDK out of memory for users of the other
    provider.
    """
    if name == "openai":
        from .openai_provider import OpenAIProvider

        return OpenAIProvider()
    if name == "anthropic":
        from .anthropic_provider import AnthropicProvider

        return AnthropicProvider()
    if name == "litellm":
        raise UsageError(
            "PROVIDER=litellm is not supported. Set USE_LITELLM=true instead."
        )
    raise UsageError(f"Unknown PROVIDER '{name}'. Valid values: openai, anthropic.")
