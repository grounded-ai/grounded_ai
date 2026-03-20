import json
from unittest.mock import MagicMock, patch

import pytest
from pydantic import BaseModel

from grounded_ai.backends.bedrock import BedrockBackend
from grounded_ai.schemas import EvaluationInput, EvaluationOutput


# --- Mock Boto3 ---
@pytest.fixture
def mock_boto3():
    with patch("grounded_ai.backends.bedrock.boto3") as mock:
        yield mock


class TestBedrockBackend:
    def test_initialization(self, mock_boto3):
        """Test proper initialization of Bedrock client."""
        backend = BedrockBackend(
            model_id="anthropic.claude-v2", region_name="us-east-1"
        )

        mock_boto3.client.assert_called_with("bedrock-runtime", region_name="us-east-1")
        assert backend.model_id == "anthropic.claude-v2"

    def test_evaluate_structured_output(self, mock_boto3):
        """Test evaluate() calls converse with correct parameters and parses response."""
        # Setup Mock
        mock_client = MagicMock()
        mock_boto3.client.return_value = mock_client

        # Configure successful response
        expected_output = EvaluationOutput(
            score=0.9, label="pass", confidence=0.95, reasoning="Looks good"
        )

        # Converse response structure
        mock_response = {
            "output": {
                "message": {"content": [{"text": expected_output.model_dump_json()}]}
            }
        }
        mock_client.converse.return_value = mock_response

        # Init Backend
        backend = BedrockBackend(model_id="test-model")

        # Run Evaluate
        result = backend.evaluate(
            input_data=EvaluationInput(response="Test content"),
            output_schema=EvaluationOutput,
        )

        assert isinstance(result, EvaluationOutput)
        assert result.score == 0.9
        assert result.label == "pass"

        # Verify Call Arguments
        call_kwargs = mock_client.converse.call_args[1]
        assert call_kwargs["modelId"] == "test-model"
        assert "messages" in call_kwargs
        assert (
            "outputConfig" in call_kwargs
            or "additionalModelRequestFields" in call_kwargs
        )

        # Verify strict schema enforcement
        # If outputConfig is passed directly (as we implemented):
        if "outputConfig" in call_kwargs:
            schema_str = call_kwargs["outputConfig"]["textFormat"]["structure"][
                "jsonSchema"
            ]["schema"]
            schema = json.loads(schema_str)
            assert schema["additionalProperties"] is False

    def test_schema_strictness_recursion(self, mock_boto3):
        """Test that additionalProperties=False is added recursively."""
        mock_client = MagicMock()
        mock_boto3.client.return_value = mock_client

        class Child(BaseModel):
            name: str

        class Parent(BaseModel):
            child: Child

        backend = BedrockBackend(model_id="test-model")

        # Mock successful response to avoid error during parse (not important for this test but good for hygiene)
        mock_client.converse.return_value = {
            "output": {"message": {"content": [{"text": "{}"}]}}
        }

        try:
            backend.evaluate(
                input_data=EvaluationInput(response="x"), output_schema=Parent
            )
        except Exception:
            pass  # We only care about the call args

        call_kwargs = mock_client.converse.call_args[1]
        schema_str = call_kwargs["outputConfig"]["textFormat"]["structure"][
            "jsonSchema"
        ]["schema"]
        schema = json.loads(schema_str)

        # Check recursion
        assert schema.get("additionalProperties") is False

        # Check definitions ($defs)
        if "$defs" in schema:
            child_schema = schema["$defs"]["Child"]
            assert child_schema.get("additionalProperties") is False
        elif "definitions" in schema:
            child_schema = schema["definitions"]["Child"]
            assert child_schema.get("additionalProperties") is False

    def test_system_prompt_cleanup(self, mock_boto3):
        """Test that system_prompt is NOT passed in additionalModelRequestFields."""
        mock_client = MagicMock()
        mock_boto3.client.return_value = mock_client
        mock_client.converse.return_value = {
            "output": {"message": {"content": [{"text": "{}"}]}}
        }

        backend = BedrockBackend(
            model_id="test-model", system_prompt="You are a judge."
        )

        # Pass extra kwargs which usually end up in additionalModelRequestFields
        backend.evaluate(
            input_data=EvaluationInput(response="x"),
            # some extra arg
            temperature=0.5,
        )

        call_kwargs = mock_client.converse.call_args[1]

        # Check system prompt in 'system' arg
        assert call_kwargs["system"] == [{"text": "You are a judge."}]

        # Check additionalModelRequestFields
        if "additionalModelRequestFields" in call_kwargs:
            extras = call_kwargs["additionalModelRequestFields"]
            assert "system_prompt" not in extras
            # temperature might be captured in inferenceConfig if we implemented logic for it
            # In our implementation, we pop temperature into inferenceConfig, so it shouldn't be in additional fields either
