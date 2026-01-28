import pytest
import json
from unittest.mock import MagicMock, patch
from grounded_ai import Evaluator
from grounded_ai.schemas import EvaluationInput, EvaluationOutput

# We mock dependencies so we don't need real torch installed for unit tests to pass logic checks
class TestHuggingFaceBackend:
    @pytest.fixture
    def mock_hf_components(self):
        with patch("grounded_ai.backends.huggingface.AutoTokenizer") as mock_tok, \
             patch("grounded_ai.backends.huggingface.AutoModelForCausalLM") as mock_model, \
             patch("grounded_ai.backends.huggingface.pipeline") as mock_pipe, \
             patch("grounded_ai.backends.huggingface.torch") as mock_torch:
            
            # Setup Tokenizer
            tokenizer = MagicMock()
            tokenizer.eos_token_id = 99
            tokenizer.pad_token_id = 99
            tokenizer.encode.return_value = [1, 2, 3] # dummy tokens
            # Default decode returns valid JSON
            tokenizer.decode.return_value = '{"score": 0.9, "label": "safe", "reasoning": "No issues.", "confidence": 1.0}'
            mock_tok.from_pretrained.return_value = tokenizer
            
            # Setup Model
            model = MagicMock()
            # generate returns dummy tensor
            model.generate.return_value = [[101]] 
            mock_model.from_pretrained.return_value = model
            
            # Setup Torch
            mock_tensor = MagicMock()
            mock_tensor.shape = [1, 10]
            mock_torch.LongTensor.return_value = mock_tensor
            
            yield {
                "tokenizer": tokenizer,
                "model": mock_model, # Return the patched CLASS helper, not the instance
                "pipeline": mock_pipe,
                "torch": mock_torch
            }

    def test_generation_flow_initialization(self, mock_hf_components):
        """Test that generation task initializes AutoModel, not pipeline."""
        evaluator = Evaluator("hf/test-model", task="text-generation", device="cpu")
        
        # Should call AutoModel, not pipeline
        assert mock_hf_components["model"].from_pretrained.called
        mock_hf_components["pipeline"].assert_not_called()
        
    def test_classification_flow_initialization(self, mock_hf_components):
        """Test that classification task initializes pipeline."""
        evaluator = Evaluator("hf/test-classifier", task="text-classification", device="cpu")
        
        # Should call pipeline, not AutoModel
        mock_hf_components["pipeline"].assert_called()
        # Note: AutoModel might not be called if pipeline handles it internally, strictly checking init logic

    def test_generation_execution_and_parsing(self, mock_hf_components):
        """Test full generation flow including prompt formatting and JSON parsing."""
        evaluator = Evaluator("hf/test-model", task="text-generation")
        
        # Execute
        result = evaluator.evaluate(response="Test input")
        
        # Assertions
        assert isinstance(result, EvaluationOutput)
        assert result.score == 0.9
        assert result.label == "safe"
        assert result.reasoning == "No issues."
        
        # Verify generate called with logits processor
        # Access the instance returned by from_pretrained
        model_instance = mock_hf_components["model"].from_pretrained.return_value
        args, kwargs = model_instance.generate.call_args
        assert "logits_processor" in kwargs
        assert len(kwargs["logits_processor"]) > 0

    def test_generation_invalid_json_fallback(self, mock_hf_components):
        """Test behavior when model outputs garbage."""
        # Setup tokenizer to return garbage
        mock_hf_components["tokenizer"].decode.return_value = "This is not JSON."
        
        evaluator = Evaluator("hf/test-model", task="text-generation")
        result = evaluator.evaluate(response="Crash me")
        
        # Should catch exception and return error object (based on backend implementation)
        assert result.label == "error"
        assert result.score == 0.0
        assert "Failed to parse" in result.reasoning

    def test_classification_execution(self, mock_hf_components):
        """Test standard classification flow (legacy support)."""
        # Setup pipeline mock return
        mock_pipeline_instance = mock_hf_components["pipeline"].return_value
        mock_pipeline_instance.return_value = [{"label": "POSITIVE", "score": 0.99}]
        
        evaluator = Evaluator("hf/classifier", task="text-classification")
        result = evaluator.evaluate(response="I love this.")
        
        assert result.label == "POSITIVE"
        assert result.score == 0.99
        assert "Classified as POSITIVE" in result.reasoning
