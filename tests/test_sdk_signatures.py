"""
Our calls against the installed SDKs' real signatures. The backend tests mock the SDK clients, so
they cannot see a parameter the SDK removed or renamed; these tests can. Each one is skipped when
that SDK is not installed.
"""

import contextlib
import importlib
import inspect
import sys

import pytest


@contextlib.contextmanager
def real(module_name):
    """The installed package, even if another test module put a mock in sys.modules. The SDKs
    import submodules lazily, so the real package stays in place for the whole block."""

    def ours(k):
        return k == module_name or k.startswith(module_name + ".")

    saved = {k: v for k, v in sys.modules.items() if ours(k)}
    for k in saved:
        del sys.modules[k]
    try:
        try:
            module = importlib.import_module(module_name)
        except ImportError:
            pytest.skip(f"{module_name} not installed")
        yield module  # an ImportError in the test body is a failure, not a skip
    finally:
        for k in [k for k in sys.modules if ours(k)]:
            del sys.modules[k]
        sys.modules.update(saved)


def params(fn):
    return set(inspect.signature(fn).parameters)


def test_anthropic_structured_output_call():
    with real("anthropic") as anthropic:
        client = anthropic.Anthropic(api_key="x")
        assert {
            "model",
            "system",
            "messages",
            "max_tokens",
            "output_config",
            "extra_body",
        } <= params(client.messages.create)
        async_client = anthropic.AsyncAnthropic(api_key="x")
        assert {"model", "system", "messages", "max_tokens", "output_format"} <= params(
            async_client.messages.parse
        )


def test_openai_structured_output_call():
    with real("openai") as openai:
        for client in (openai.OpenAI(api_key="x"), openai.AsyncOpenAI(api_key="x")):
            assert {"model", "messages", "response_format"} <= params(
                client.chat.completions.parse
            )


def test_bedrock_converse_call():
    with real("boto3") as boto3:
        client = boto3.client("bedrock-runtime", region_name="us-east-1")
        shape = client.meta.service_model.operation_model("Converse").input_shape
    assert {"modelId", "messages", "system", "inferenceConfig", "outputConfig"} <= set(
        shape.members
    )
