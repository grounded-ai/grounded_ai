import sys
import json
import pytest
from unittest.mock import MagicMock, patch

# Pre-mock anthropic module
mock_anthropic_module = MagicMock()
sys.modules["anthropic"] = mock_anthropic_module

from grounded_ai import Evaluator, EvaluationOutput, EvaluationError, EvaluationInput
from grounded_ai.backends.anthropic import AnthropicBackend

class TestAnthropicBackend:

    @pytest.fixture
    def mock_client(self):
        """Mocks the Anthropic client and messages.create method."""
        client = MagicMock()
        mock_response = MagicMock()
        # Mock the beta path: client.beta.messages.create
        client.beta.messages.create.return_value = mock_response
        return client

    def test_factory_routing(self):
        """Test that 'anthropic/*' routes correctly."""
        with patch("grounded_ai.backends.anthropic.Anthropic") as MockAnthropic:
            evaluator = Evaluator("anthropic/claude-haiku-4-5-20251001")
            assert isinstance(evaluator.backend, AnthropicBackend)
            assert evaluator.backend.model_name == "claude-haiku-4-5-20251001"

    def test_evaluate_success_and_schema_patching(self, mock_client):
        """Test successful evaluation and verify schema patching logic."""
        # Setup mock response
        mock_content = MagicMock()
        mock_content.response = json.dumps({
            "score": 0.85,
            "label": "faithful",
            "confidence": 0.9,
            "reasoning": "Good match."
        })
        
        mock_client.beta.messages.create.return_value.content = [mock_content]

        backend = AnthropicBackend(model_name="claude-3", client=mock_client)
        
        result = backend.evaluate(EvaluationInput(
            response="Paris is capital.",
            query="Capital?",
            conresponse="Paris is capital of France."
        ))

        # 1. Assert Result
        assert isinstance(result, EvaluationOutput)
        assert result.score == 0.85
        assert result.label == "faithful"

        # 2. Assert Message Creation Call (Beta Check)
        args, kwargs = mock_client.beta.messages.create.call_args
        
        # Check Beta Header
        assert "betas" in kwargs
        assert "structured-outputs-2025-11-13" in kwargs["betas"]
        
        # Check Schema Patching in output_format
        output_format = kwargs["output_format"]
        assert output_format["type"] == "json_schema"
        schema = output_format["schema"]
        
        # Verify strict schema modifications
        def check_strictness(s):
            if s.get("type") == "object":
                assert s.get("additionalProperties") is False
            # Check for stripped keywords
            assert "minimum" not in s
            assert "maximum" not in s
            assert "title" not in s
            
            if "properties" in s:
                for prop in s["properties"].values():
                    check_strictness(prop)

        check_strictness(schema)

    def test_evaluate_api_error_handling(self, mock_client):
        """Test that backend catches API errors and returns EvaluationError."""
        # Setup mock to raise exception
        mock_err = Exception("Credit limit reached")
        mock_err.status_code = 402 # Fake status code
        mock_client.beta.messages.create.side_effect = mock_err

        backend = AnthropicBackend(model_name="claude-3", client=mock_client)
        
        result = backend.evaluate(EvaluationInput(response="fail"))
        
        assert isinstance(result, EvaluationError)
        assert result.error_code == "402"
        assert result.message == "Credit limit reached"
