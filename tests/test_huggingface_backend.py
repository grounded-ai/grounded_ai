import pytest
from grounded_ai import Evaluator
from grounded_ai.schemas import EvaluationInput, EvaluationOutput

try:
    import transformers
    import torch
    HAS_DEPS = True
except ImportError:
    HAS_DEPS = False

@pytest.mark.skipif(not HAS_DEPS, reason="transformers or torch not installed")
class TestHuggingFaceBackend:

    def test_smollm_integration(self):
        """
        Test the HF backend with a real (small) model: HuggingFaceTB/SmolLM-360M.
        This verifies the pipeline download and basic inference loop.
        """
        # Initialize with the specific model
        # Using "hf/" prefix to route to HuggingFaceBackend
        evaluator = Evaluator("hf/HuggingFaceTB/SmolLM-360M", task="text-generation", device="cpu")
        
        # Verify backend type
        from grounded_ai.backends.huggingface import HuggingFaceBackend
        assert isinstance(evaluator.backend, HuggingFaceBackend)
        assert evaluator.backend.model_id == "HuggingFaceTB/SmolLM-360M"

        # Run a simple evaluation
        result = evaluator.evaluate(
            response="The sky is blue.",
            query="What color is the sky?",
            conresponse="The sky appears blue to the human eye."
        )

        # Basic assertions on the output
        assert isinstance(result, EvaluationOutput)
        assert result.label == "generated_text"
        assert len(result.reasoning) > 0
        print(f"Tiny model succeeded! Reasoning: {result.reasoning}")
