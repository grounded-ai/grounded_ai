import sys
from unittest.mock import MagicMock, patch

import pytest

# Pre-mock openai module to ensure tests run even if package isn't installed
mock_openai_module = MagicMock()
mock_openai_module.__spec__ = MagicMock()
sys.modules["openai"] = mock_openai_module

from grounded_ai import Evaluator  # noqa: E402
from grounded_ai.backends.openai import OpenAIBackend  # noqa: E402
from grounded_ai.schemas import (  # noqa: E402
    EvaluationError,
    EvaluationInput,
    EvaluationOutput,
)


class TestOpenAIBackend:
    @pytest.fixture
    def mock_client(self):
        """Mocks the OpenAI client and the beta.chat.completions.parse method."""
        client = MagicMock()
        mock_completion = MagicMock()

        # Setup the chain: client.beta.chat.completions.parse(...)
        client.beta.chat.completions.parse.return_value = mock_completion

        return client

    def test_factory_routing(self):
        """Test that 'openai/*' model strings route to OpenAIBackend."""
        with patch("grounded_ai.backends.openai.OpenAI"):
            evaluator = Evaluator("openai/gpt-4o")
            assert isinstance(evaluator.backend, OpenAIBackend)
            assert evaluator.backend.model_name == "gpt-4o"

    def test_evaluate_success(self, mock_client):
        """Test a successful evaluation with structured output."""
        # Setup the mock response
        mock_message = MagicMock()
        mock_message.refusal = None
        mock_message.parsed = EvaluationOutput(
            score=0.9,
            label="faithful",
            confidence=1.0,
            reasoning="The text accurately reflects the context.",
        )

        # Connect message to completion.choices[0]
        mock_completion = mock_client.beta.chat.completions.parse.return_value
        mock_completion.choices = [MagicMock(message=mock_message)]

        # Initialize backend directly with injected client
        backend = OpenAIBackend(model_name="gpt-4o", client=mock_client)

        result = backend.evaluate(
            EvaluationInput(
                response="Paris is the capital.",
                query="What is the capital?",
                context="Paris is the capital of France.",
            )
        )

        # Assertions
        assert isinstance(result, EvaluationOutput)
        assert result.score == 0.9
        assert result.label == "faithful"
        assert result.reasoning == "The text accurately reflects the context."

        # Verify call arguments
        # We expect messages to be constructed from inputs
        args, kwargs = mock_client.beta.chat.completions.parse.call_args
        assert kwargs["model"] == "gpt-4o"
        assert kwargs["response_format"] == EvaluationOutput
        assert len(kwargs["messages"]) == 2  # System + User

    def test_evaluate_refusal(self, mock_client):
        """Test handling of model refusal."""
        # Setup mock to simulate a refusal
        mock_message = MagicMock()
        mock_message.refusal = "I cannot evaluate this content due to safety policies."
        mock_message.parsed = None

        mock_completion = mock_client.beta.chat.completions.parse.return_value
        mock_completion.choices = [MagicMock(message=mock_message)]

        backend = OpenAIBackend(model_name="gpt-4o", client=mock_client)

        # Expect EvaluationError (backend returns error object, doesn't raise)
        result = backend.evaluate(EvaluationInput(response="Unsafe content"))

        assert isinstance(result, EvaluationError)
        assert result.error_code == "MODEL_REFUSAL"
        assert "policies" in result.message or "refused" in result.message

    def test_custom_system_prompt(self, mock_client):
        """Test custom system prompt pass-through."""
        mock_message = MagicMock()
        mock_message.refusal = None
        mock_message.parsed = EvaluationOutput(
            score=0.1, label="ok", confidence=1.0, reasoning="ok"
        )

        mock_completion = mock_client.beta.chat.completions.parse.return_value
        mock_completion.choices = [MagicMock(message=mock_message)]

        backend = OpenAIBackend(
            model_name="gpt-4o",
            client=mock_client,
            system_prompt="You are a poetic evaluator.",
        )

        backend.evaluate(EvaluationInput(response="rose is red"))

        args, kwargs = mock_client.beta.chat.completions.parse.call_args
        messages = kwargs["messages"]
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == "You are a poetic evaluator."

    def test_custom_input_output_schemas(self, mock_client):
        """Test overriding default schemas with custom Pydantic models."""
        from pydantic import BaseModel

        class CustomInput(BaseModel):
            review_text: str

        class CustomOutput(BaseModel):
            sentiment_score: float
            is_positive: bool

        # Mock response to match custom schema
        mock_message = MagicMock()
        mock_message.refusal = None
        mock_message.parsed = CustomOutput(sentiment_score=0.9, is_positive=True)

        mock_completion = mock_client.beta.chat.completions.parse.return_value
        mock_completion.choices = [MagicMock(message=mock_message)]

        backend = OpenAIBackend(model_name="gpt-4o", client=mock_client)

        # Test evaluate with custom input and output schema
        input_data = CustomInput(review_text="Great job!")
        result = backend.evaluate(input_data, output_schema=CustomOutput)

        # Verify result type
        assert isinstance(result, CustomOutput)
        assert result.sentiment_score == 0.9
        assert result.is_positive is True

        # Verify passed to API
        args, kwargs = mock_client.beta.chat.completions.parse.call_args
        assert kwargs["response_format"] == CustomOutput
        # OpenAI backend calls .model_dump() on custom inputs if they lack formatted_prompt
        messages = kwargs["messages"]
        assert str(input_data.model_dump()) in messages[1]["content"]

    def test_kwargs_passthrough(self, mock_client):
        """Test that kwargs (e.g. temperature) are passed to the API."""
        mock_message = MagicMock()
        mock_message.refusal = None
        mock_message.parsed = EvaluationOutput(
            score=0.5, label="ok", confidence=1.0, reasoning="ok"
        )

        mock_completion = mock_client.beta.chat.completions.parse.return_value
        mock_completion.choices = [MagicMock(message=mock_message)]

        backend = OpenAIBackend(model_name="gpt-4o", client=mock_client)

        # Override temperature in evaluate call
        backend.evaluate(EvaluationInput(response="test"), temperature=0.7)

        args, kwargs = mock_client.beta.chat.completions.parse.call_args
        assert kwargs["temperature"] == 0.7
