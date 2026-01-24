import sys
import pytest
from unittest.mock import MagicMock, patch

# Pre-mock dependencies before imports
# We use a fixture or module-level patch, but for sys.modules logic it's safest to do it 
# at the top level or via conftest. For this file, top level is fine to ensure clean load.
sys.modules["peft"] = MagicMock()
sys.modules["transformers"] = MagicMock()
sys.modules["torch"] = MagicMock()

from grounded_ai import Evaluator
from grounded_ai.schemas import EvaluationInput, EvaluationOutput
from grounded_ai.backends.grounded_ai_slm.backend import GroundedAISLMBackend

class TestSLMBackend:
    
    @pytest.fixture
    def mock_deps(self):
        """Fixture to patch the internal backend imports/calls if needed again."""
        with patch("grounded_ai.backends.grounded_ai_slm.backend.pipeline") as mock_pipeline, \
             patch("grounded_ai.backends.grounded_ai_slm.backend.AutoModelForCausalLM") as mock_automodel, \
             patch("grounded_ai.backends.grounded_ai_slm.backend.AutoTokenizer") as mock_tokenizer, \
             patch("grounded_ai.backends.grounded_ai_slm.backend.PeftModel") as mock_peft, \
             patch("grounded_ai.backends.grounded_ai_slm.backend.PeftConfig") as mock_config:
            
            # Setup default behavior for pipeline
            mock_generator = MagicMock()
            # Simulation of model output for "Accurate" -> Faithful
            mock_output = [{"generated_text": [
                {"role": "user", "content": "..."}, 
                {"role": "assistant", "content": "<rating>Accurate</rating><reasoning>Matches context.</reasoning>"}
            ]}]
            mock_generator.return_value = mock_output
            mock_pipeline.return_value = mock_generator
            
            yield {
                "pipeline": mock_pipeline, 
                "automodel": mock_automodel,
                "tokenizer": mock_tokenizer,
                "peft": mock_peft
            }

    def test_factory_routing(self, mock_deps):
        """Test that the factory routes 'grounded-ai/*' to SLM backend."""
        evaluator = Evaluator("grounded-ai/hallucination-v1")
        assert isinstance(evaluator.backend, GroundedAISLMBackend)
        assert evaluator.backend.task == "hallucination"

    def test_hallucination_flow(self, mock_deps):
        """Test a full hallucination evaluation flow."""
        evaluator = Evaluator("grounded-ai/hallucination-v1")
        
        result = evaluator.evaluate(
            text="London is the capital.", # This maps to RESPONSE in hallucination
            query="What is the capital?",
            reference="London is the capital of UK."
        )
        
        assert isinstance(result, EvaluationOutput)
        assert result.score == 0.0 # Accurate -> 0.0 hallucination score
        assert result.label == "faithful"
        assert result.reasoning == "Matches context."

    def test_toxicity_flow(self, mock_deps):
        """Test toxicity flow logic."""
        # Setup mock to return 'Toxic'
        mock_deps["pipeline"].return_value.return_value = [{"generated_text": [
            {"role": "assistant", "content": "<rating>Toxic</rating><reasoning>Rude language.</reasoning>"}
        ]}]

        evaluator = Evaluator("grounded-ai/toxic-judge-v1")
        assert evaluator.backend.task == "toxicity"
        
        result = evaluator.evaluate(text="You are stupid.")
        
        assert result.score == 1.0 # Toxic -> 1.0
        assert result.label == "toxic"

    def test_rag_flow(self, mock_deps):
        """Test RAG flow logic."""
        # Setup mock to return 'Relevant'
        mock_deps["pipeline"].return_value.return_value = [{"generated_text": [
            {"role": "assistant", "content": "<rating>Relevant</rating><reasoning>Direct answer.</reasoning>"}
        ]}]

        evaluator = Evaluator("grounded-ai/rag-relevance-v1")
        assert evaluator.backend.task == "rag"
        
        result = evaluator.evaluate(
            query="Question?",
            text="Document content." # 'text' is the reference text in RAG relevance inputs
        )
        
        assert result.score == 1.0 # Relevant -> 1.0
        assert result.label == "relevant"
