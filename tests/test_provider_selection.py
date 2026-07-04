import subprocess
import sys

from sgpt.handlers import handler
from sgpt.providers.openai_provider import OpenAIProvider


def test_default_provider_is_openai():
    """When PROVIDER is unset/openai, handler must use OpenAIProvider."""
    assert isinstance(handler.provider, OpenAIProvider)


def test_anthropic_sdk_not_imported_for_openai_provider():
    """Selecting the OpenAI provider must not import the Anthropic SDK.

    Run in a fresh interpreter (via subprocess) so the assertion is not
    polluted by other tests in the session that legitimately import the
    anthropic module. get_provider() uses lazy imports to keep the unused
    SDK out of memory.
    """
    code = (
        "import os, sys; "
        "os.environ['PROVIDER'] = 'openai'; "
        "import sgpt.handlers.handler; "  # triggers provider construction
        "sys.exit(0 if 'anthropic' not in sys.modules else 1)"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True)
    assert result.returncode == 0, (
        f"anthropic SDK was imported under PROVIDER=openai.\n"
        f"stdout: {result.stdout.decode()}\n"
        f"stderr: {result.stderr.decode()}"
    )
