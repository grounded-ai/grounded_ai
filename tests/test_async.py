"""
Tests for AsyncEvaluator and the kwarg forwarding fix.
"""

import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Pre-mock provider SDKs so tests run without real credentials
mock_openai = MagicMock()
mock_openai.__spec__ = MagicMock()
sys.modules["openai"] = mock_openai

mock_anthropic = MagicMock()
mock_anthropic.__spec__ = MagicMock()
sys.modules["anthropic"] = mock_anthropic

from grounded_ai import AsyncEvaluator, EvaluationInput, EvaluationOutput, Evaluator  # noqa: E402
from grounded_ai.backends.anthropic import AnthropicBackend  # noqa: E402
from grounded_ai.backends.openai import OpenAIBackend  # noqa: E402
from grounded_ai.schemas import EvaluationError  # noqa: E402


# === AsyncEvaluator — OpenAI ===


class TestAsyncEvaluatorOpenAI:
    @pytest.fixture
    def mock_async_client(self):
        client = MagicMock()
        client.beta.chat.completions.parse = AsyncMock()
        return client

    @pytest.mark.asyncio
    async def test_evaluate_success(self, mock_async_client):
        mock_message = MagicMock()
        mock_message.refusal = None
        mock_message.parsed = EvaluationOutput(
            score=0.1, label="faithful", confidence=0.95, reasoning="Looks good."
        )
        mock_async_client.beta.chat.completions.parse.return_value.choices = [
            MagicMock(message=mock_message)
        ]

        backend = OpenAIBackend(model_name="gpt-4o", async_client=mock_async_client)
        evaluator = AsyncEvaluator.__new__(AsyncEvaluator)
        evaluator.backend = backend

        result = await evaluator.evaluate(response="Paris is the capital of France.")

        assert isinstance(result, EvaluationOutput)
        assert result.label == "faithful"
        mock_async_client.beta.chat.completions.parse.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_evaluate_refusal(self, mock_async_client):
        mock_message = MagicMock()
        mock_message.refusal = "Content policy violation."
        mock_message.parsed = None
        mock_async_client.beta.chat.completions.parse.return_value.choices = [
            MagicMock(message=mock_message)
        ]

        backend = OpenAIBackend(model_name="gpt-4o", async_client=mock_async_client)
        evaluator = AsyncEvaluator.__new__(AsyncEvaluator)
        evaluator.backend = backend

        result = await evaluator.evaluate(response="bad content")

        assert isinstance(result, EvaluationError)
        assert result.error_code == "MODEL_REFUSAL"

    @pytest.mark.asyncio
    async def test_evaluate_kwargs_forwarded(self, mock_async_client):
        mock_message = MagicMock()
        mock_message.refusal = None
        mock_message.parsed = EvaluationOutput(
            score=0.0, label="ok", confidence=1.0, reasoning="ok"
        )
        mock_async_client.beta.chat.completions.parse.return_value.choices = [
            MagicMock(message=mock_message)
        ]

        backend = OpenAIBackend(model_name="gpt-4o", async_client=mock_async_client)
        evaluator = AsyncEvaluator.__new__(AsyncEvaluator)
        evaluator.backend = backend

        await evaluator.evaluate(response="test", temperature=0.2)

        _, kwargs = mock_async_client.beta.chat.completions.parse.call_args
        assert kwargs["temperature"] == 0.2

    @pytest.mark.asyncio
    async def test_evaluate_error_handling(self, mock_async_client):
        err = Exception("Rate limit exceeded")
        err.status_code = 429
        mock_async_client.beta.chat.completions.parse.side_effect = err

        backend = OpenAIBackend(model_name="gpt-4o", async_client=mock_async_client)
        evaluator = AsyncEvaluator.__new__(AsyncEvaluator)
        evaluator.backend = backend

        result = await evaluator.evaluate(response="test")

        assert isinstance(result, EvaluationError)
        assert result.error_code == "429"


# === AsyncEvaluator — Anthropic ===


class TestAsyncEvaluatorAnthropic:
    @pytest.fixture
    def mock_async_client(self):
        client = MagicMock()
        client.beta.messages.parse = AsyncMock()
        return client

    @pytest.mark.asyncio
    async def test_evaluate_success(self, mock_async_client):
        mock_response = MagicMock()
        mock_response.parsed_output = EvaluationOutput(
            score=0.9, label="toxic", confidence=0.98, reasoning="Contains hate speech."
        )
        mock_async_client.beta.messages.parse.return_value = mock_response

        backend = AnthropicBackend(
            model_name="claude-haiku-4-5-20251001", async_client=mock_async_client
        )
        evaluator = AsyncEvaluator.__new__(AsyncEvaluator)
        evaluator.backend = backend

        result = await evaluator.evaluate(response="Some text to evaluate.")

        assert isinstance(result, EvaluationOutput)
        assert result.label == "toxic"
        mock_async_client.beta.messages.parse.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_evaluate_passes_output_format_directly(self, mock_async_client):
        """beta.messages.parse receives the Pydantic class directly — no schema surgery."""
        mock_response = MagicMock()
        mock_response.parsed_output = EvaluationOutput(
            score=0.0, label="ok", confidence=1.0, reasoning="ok"
        )
        mock_async_client.beta.messages.parse.return_value = mock_response

        backend = AnthropicBackend(
            model_name="claude-haiku-4-5-20251001", async_client=mock_async_client
        )
        evaluator = AsyncEvaluator.__new__(AsyncEvaluator)
        evaluator.backend = backend

        await evaluator.evaluate(response="test")

        _, kwargs = mock_async_client.beta.messages.parse.call_args
        assert kwargs["output_format"] is EvaluationOutput

    @pytest.mark.asyncio
    async def test_evaluate_empty_response(self, mock_async_client):
        mock_response = MagicMock()
        mock_response.parsed_output = None
        mock_async_client.beta.messages.parse.return_value = mock_response

        backend = AnthropicBackend(
            model_name="claude-haiku-4-5-20251001", async_client=mock_async_client
        )
        evaluator = AsyncEvaluator.__new__(AsyncEvaluator)
        evaluator.backend = backend

        result = await evaluator.evaluate(response="test")

        assert isinstance(result, EvaluationError)
        assert result.error_code == "EMPTY_RESPONSE"

    @pytest.mark.asyncio
    async def test_evaluate_error_handling(self, mock_async_client):
        err = Exception("Service unavailable")
        err.status_code = 503
        mock_async_client.beta.messages.parse.side_effect = err

        backend = AnthropicBackend(
            model_name="claude-haiku-4-5-20251001", async_client=mock_async_client
        )
        evaluator = AsyncEvaluator.__new__(AsyncEvaluator)
        evaluator.backend = backend

        result = await evaluator.evaluate(response="test")

        assert isinstance(result, EvaluationError)
        assert result.error_code == "503"


# === AsyncEvaluator factory routing ===


class TestAsyncEvaluatorFactory:
    def test_inherits_same_backend_routing(self):
        """AsyncEvaluator should route to the same backends as Evaluator."""
        with patch("grounded_ai.backends.openai.OpenAI"):
            evaluator = AsyncEvaluator("openai/gpt-4o")
            assert isinstance(evaluator.backend, OpenAIBackend)

    def test_evaluate_is_a_coroutine(self):
        """AsyncEvaluator.evaluate must be a coroutine function."""
        import asyncio
        with patch("grounded_ai.backends.openai.OpenAI"):
            evaluator = AsyncEvaluator("openai/gpt-4o")
        assert asyncio.iscoroutinefunction(evaluator.evaluate)

    def test_sync_evaluator_evaluate_is_not_a_coroutine(self):
        """Sanity check: Evaluator.evaluate must NOT be a coroutine function."""
        import asyncio
        with patch("grounded_ai.backends.openai.OpenAI"):
            evaluator = Evaluator("openai/gpt-4o")
        assert not asyncio.iscoroutinefunction(evaluator.evaluate)


# === Kwarg forwarding fix ===


class TestKwargForwarding:
    def test_backend_kwargs_forwarded(self):
        """Runtime kwargs (temperature, max_tokens) must reach the backend."""
        with patch("grounded_ai.backends.openai.OpenAI"):
            evaluator = Evaluator("openai/gpt-4o")

        captured = {}

        def fake_call(input_data, output_schema, **kwargs):
            captured.update(kwargs)
            return EvaluationOutput(score=0.0, label="ok", confidence=1.0, reasoning="ok")

        evaluator.backend._call_backend = fake_call
        evaluator.evaluate(response="test text", temperature=0.5, max_tokens=512)

        assert captured.get("temperature") == 0.5
        assert captured.get("max_tokens") == 512

    def test_input_kwargs_not_leaked_to_backend(self):
        """response/query/context/base_template must build EvaluationInput, not leak to backend."""
        with patch("grounded_ai.backends.openai.OpenAI"):
            evaluator = Evaluator("openai/gpt-4o")

        captured = {}

        def fake_call(input_data, output_schema, **kwargs):
            captured.update(kwargs)
            return EvaluationOutput(score=0.0, label="ok", confidence=1.0, reasoning="ok")

        evaluator.backend._call_backend = fake_call
        evaluator.evaluate(response="text", query="question?", temperature=0.3)

        assert "response" not in captured
        assert "query" not in captured
        assert captured.get("temperature") == 0.3

    def test_string_input_with_backend_kwargs(self):
        """String shorthand + backend kwargs should both work correctly."""
        with patch("grounded_ai.backends.openai.OpenAI"):
            evaluator = Evaluator("openai/gpt-4o")

        captured_input = {}
        captured_kwargs = {}

        def fake_call(input_data, output_schema, **kwargs):
            captured_input["data"] = input_data
            captured_kwargs.update(kwargs)
            return EvaluationOutput(score=0.0, label="ok", confidence=1.0, reasoning="ok")

        evaluator.backend._call_backend = fake_call
        evaluator.evaluate("my response text", temperature=0.7)

        assert captured_input["data"].response == "my response text"
        assert captured_kwargs.get("temperature") == 0.7
