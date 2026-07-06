import subprocess
import sys


def _run_in_clean_env(code: str, provider: str) -> subprocess.CompletedProcess:
    """Run code in a subprocess that overrides PROVIDER, so the live
    ~/.sgptrc (which may set PROVIDER=anthropic on the dev machine) does
    not pollute the assertion."""
    full = (
        "import os, sys; "
        f"os.environ['PROVIDER'] = {provider!r}; "
        "os.environ.setdefault('OPENAI_API_KEY', 'test-key'); "
        "os.environ.setdefault('ANTHROPIC_API_KEY', 'test-key'); "
        f"{code}"
    )
    return subprocess.run([sys.executable, "-c", full], capture_output=True)


def test_default_provider_is_openai():
    """When PROVIDER=openai, handler must use OpenAIProvider."""
    code = (
        "from sgpt.handlers import handler; "
        "from sgpt.providers.openai_provider import OpenAIProvider; "
        "sys.exit(0 if isinstance(handler.provider, OpenAIProvider) else 1)"
    )
    result = _run_in_clean_env(code, "openai")
    assert result.returncode == 0, (
        f"expected OpenAIProvider under PROVIDER=openai.\n"
        f"stdout: {result.stdout.decode()}\n"
        f"stderr: {result.stderr.decode()}"
    )


def test_anthropic_sdk_not_imported_for_openai_provider():
    """Selecting the OpenAI provider must not import the Anthropic SDK.

    Run in a fresh interpreter (via subprocess) so the assertion is not
    polluted by other tests in the session that legitimately import the
    anthropic module. get_provider() uses lazy imports to keep the unused
    SDK out of memory.
    """
    code = (
        "import sgpt.handlers.handler; "  # triggers provider construction
        "sys.exit(0 if 'anthropic' not in sys.modules else 1)"
    )
    result = _run_in_clean_env(code, "openai")
    assert result.returncode == 0, (
        f"anthropic SDK was imported under PROVIDER=openai.\n"
        f"stdout: {result.stdout.decode()}\n"
        f"stderr: {result.stderr.decode()}"
    )
