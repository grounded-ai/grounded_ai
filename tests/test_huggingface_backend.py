import importlib.util

import pytest

from grounded_ai import Evaluator

HAS_DEPS = (
    importlib.util.find_spec("transformers") is not None
    and importlib.util.find_spec("torch") is not None
)


@pytest.mark.skipif(not HAS_DEPS, reason="transformers or torch not installed")
class TestHuggingFaceBackend:
    def test_smollm_integration(self):
        """
        Test the HF backend with a real (small) model: HuggingFaceTB/SmolLM-135M-Instruct.
        This verifies the pipeline download and basic inference loop.
        """
        # Initialize with the specific model
        # Using "hf/" prefix to route to HuggingFaceBackend
        evaluator = Evaluator(
            "hf/HuggingFaceTB/SmolLM-135M-Instruct",
            task="text-generation",
            device="cpu",
        )

        # Verify backend type
        from grounded_ai.backends.huggingface import HuggingFaceBackend

        assert isinstance(evaluator.backend, HuggingFaceBackend)
        assert evaluator.backend.model_id == "HuggingFaceTB/SmolLM-135M-Instruct"

        # Run a simple evaluation
        result = evaluator.evaluate(
            response="The sky is blue.",
            query="What color is the sky?",
            context="The sky appears blue to the human eye.",
        )

        # Basic assertions on the output
        assert result.label == "generated_text"
        assert len(result.reasoning) > 0
        print(f"Tiny model succeeded! Reasoning: {result.reasoning}")

    def test_custom_system_prompt_mocks(self):
        """Test system prompt injection with mocks."""
        from unittest.mock import MagicMock, patch

        from grounded_ai.schemas import EvaluationInput

        # Mock pipeline
        mock_pipeline = MagicMock()
        # transformers pipeline with messages returns list of dicts with 'generated_text' being the conversation list
        mock_pipeline.return_value = [
            {
                "generated_text": [
                    {"role": "system", "content": "Be concise."},
                    {"role": "user", "content": "Evaluate: User says hi."},
                    {"role": "assistant", "content": "mock output"},
                ]
            }
        ]

        with patch(
            "grounded_ai.backends.huggingface.pipeline", return_value=mock_pipeline
        ):
            from grounded_ai.backends.huggingface import HuggingFaceBackend

            backend = HuggingFaceBackend(
                model_id="hf/gpt2", system_prompt="Be concise."
            )

            backend.evaluate(EvaluationInput(response="User says hi."))

            # Verify prompt
            call_args = mock_pipeline.call_args
            assert call_args is not None
            messages_arg = call_args[0][0]
            assert isinstance(messages_arg, list)
            assert messages_arg[0] == {"role": "system", "content": "Be concise."}
            assert messages_arg[1]["role"] == "user"
            assert "User says hi" in messages_arg[1]["content"]

    def test_custom_input_output_schemas(self):
        """Test overriding schemas for HF backend."""
        from unittest.mock import MagicMock, patch

        from pydantic import BaseModel

        class CustomInput(BaseModel):
            raw_text: str

        # Define a schema that is compatible with what HF backend returns (score, label, confidence, reasoning)
        # but maybe allows extra fields or validation.
        class CustomOutput(BaseModel):
            score: float
            label: str
            confidence: float
            reasoning: str

            # We can't easily change the keys the backend injects without refactoring the backend,
            # but we can verify the custom class is used.
            @property
            def is_generated(self):
                return self.label == "generated_text"

        mock_pipeline = MagicMock()
        mock_pipeline.return_value = [{"generated_text": "Processed summary"}]

        with patch(
            "grounded_ai.backends.huggingface.pipeline", return_value=mock_pipeline
        ):
            from grounded_ai.backends.huggingface import HuggingFaceBackend

            backend = HuggingFaceBackend(model_id="hf/t5")

            input_data = CustomInput(raw_text="Long text")
            result = backend.evaluate(input_data, output_schema=CustomOutput)

            assert isinstance(result, CustomOutput)
            assert result.reasoning == "Processed summary"
            assert result.is_generated is True
