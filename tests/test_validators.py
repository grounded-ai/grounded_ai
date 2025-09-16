import json
import pytest
from pydantic import ValidationError
from grounded_ai.validators.toxic_data import ToxicityData
from grounded_ai.validators.rag_data import RagData
from grounded_ai.validators.hallucination_data import HallucinationData


class TestToxicityValidator:
    """Test cases for ToxicityData validator."""
    
    def test_toxicity_validator_valid_data(self):
        """Test that toxicity validator works with valid data."""
        data = [
            "You ugly idiot",
            "I love you",
        ]
        
        validated_input = ToxicityData(instances=data)
        result = validated_input.model_dump_json()
        
        expected = {
            "instances": [
                {"text": "You ugly idiot"},
                {"text": "I love you"}
            ]
        }
        
        assert json.loads(result) == expected
    
    def test_toxicity_validator_empty_data(self):
        """Test validation with empty instances list."""
        with pytest.raises(ValidationError, match=r".*List should have at least 1 item.*"):
            ToxicityData(instances=[])
    
    def test_toxicity_validator_single_item(self):
        """Test toxicity validator with single item."""
        data = ["This is a test message"]
        validated_input = ToxicityData(instances=data)
        result = validated_input.model_dump_json()
        
        expected = {
            "instances": [
                {"text": "This is a test message"}
            ]
        }
        assert json.loads(result) == expected
    
    def test_toxicity_validator_invalid_data_type(self):
        """Test that toxicity validator raises error with invalid data type."""
        with pytest.raises(ValueError):
            ToxicityData(instances=["valid", 123])  # Mixed types


class TestRagValidator:
    """Test cases for RagData validator."""
    
    def test_rag_validator_valid_data(self):
        """Test that RAG validator works with valid data."""
        data = [
            ("What is the capital of France?", "Paris is the capital of France."),
            ("What is the largest planet in our solar system?", "Jupiter is the largest planet in our solar system."),
            ("What is the capital of Germany?", "Berlin is the capital of Austria."),
            ("What is the largest country in the world by land area?", "Rhode Island is the largest country in the world by land area."),
        ]
        
        validated_input = RagData(instances=data)
        result = validated_input.model_dump_json()
        
        expected = {
            "instances": [
                {
                    "query": "What is the capital of France?", 
                    "context": "Paris is the capital of France."
                },
                {
                    "query": "What is the largest planet in our solar system?", 
                    "context": "Jupiter is the largest planet in our solar system."
                },
                {
                    "query": "What is the capital of Germany?", 
                    "context": "Berlin is the capital of Austria."
                },
                {
                    "query": "What is the largest country in the world by land area?", 
                    "context": "Rhode Island is the largest country in the world by land area."
                }
            ]
        }
        
        assert json.loads(result) == expected
    
    def test_rag_validator_empty_data(self):
        """Test validation with empty instances list."""
        with pytest.raises(ValidationError, match=r".*List should have at least 1 item.*"):
            RagData(instances=[])
    
    def test_rag_validator_single_item(self):
        """Test RAG validator with single item."""
        data = [("Query text", "Context text")]
        validated_input = RagData(instances=data)
        result = validated_input.model_dump_json()
        
        expected = {
            "instances": [
                {"query": "Query text", "context": "Context text"}
            ]
        }
        assert json.loads(result) == expected
    
    def test_rag_validator_invalid_tuple_length(self):
        """Test that RAG validator raises error with wrong tuple length."""
        with pytest.raises(ValueError):
            RagData(instances=[("only_one_item",)])  # Single item tuple
    
    def test_rag_validator_invalid_data_type(self):
        """Test that RAG validator raises error with invalid data type."""
        with pytest.raises(ValueError):
            RagData(instances=["not_a_tuple"])  # String instead of tuple


class TestHallucinationValidator:
    """Test cases for HallucinationData validator."""
    
    def test_hallucination_validator_without_reference(self):
        """Test hallucination validator with data without references."""
        data = [
            ('Based on the following <context>Walrus are the largest mammal</context> answer the question <query> What is the best PC?</query>', 'The best PC is the mac'),
            ('What is the color of an apple', "Apples are usually red or green"),
        ]
        
        validated_input = HallucinationData(instances=data)
        result = validated_input.model_dump_json()
        
        expected = {
            "instances": [
                {
                    "query": 'Based on the following <context>Walrus are the largest mammal</context> answer the question <query> What is the best PC?</query>',
                    "response": 'The best PC is the mac',
                    "reference": ""
                },
                {
                    "query": 'What is the color of an apple',
                    "response": "Apples are usually red or green",
                    "reference": ""
                }
            ]
        }
        
        assert json.loads(result) == expected
    
    def test_hallucination_validator_with_reference(self):
        """Test hallucination validator with data including references."""
        references = [
            "The chicken crossed the road to get to the other side",
            "The apple mac has the best hardware",
            "The cat is hungry"
        ]
        queries = [
            "Why did the chicken cross the road?",
            "What computer has the best screen?",
            "What pet does the context reference?"
        ]
        responses = [
            "To get to the other side",  # Grounded answer
            "Apple mac has the best screen",  # Deviated from the question (hardware vs software)
            "Cat"  # Grounded answer
        ]
        data = list(zip(queries, responses, references))
        
        validated_input = HallucinationData(instances=data)
        result = validated_input.model_dump_json()
        
        expected = {
            "instances": [
                {
                    "query": "Why did the chicken cross the road?",
                    "response": "To get to the other side",
                    "reference": "The chicken crossed the road to get to the other side"
                },
                {
                    "query": "What computer has the best screen?",
                    "response": "Apple mac has the best screen",
                    "reference": "The apple mac has the best hardware"
                },
                {
                    "query": "What pet does the context reference?",
                    "response": "Cat",
                    "reference": "The cat is hungry"
                }
            ]
        }
        
        assert json.loads(result) == expected
    
    def test_hallucination_validator_empty_data(self):
        """Test hallucination validator with empty data - should raise validation error."""
        data = []
        with pytest.raises(ValueError, match="At least one instance is required"):
            HallucinationData(instances=data)
    
    def test_hallucination_validator_mixed_reference_data(self):
        """Test hallucination validator with mixed reference and no-reference data - should fail due to length mismatch."""
        data = [
            ("Query 1", "Response 1"),  # No reference
            ("Query 2", "Response 2", "Reference 2"),  # With reference
        ]
        
        # This should fail because all instances must have the same length
        with pytest.raises(ValueError, match="All instances must have the same number of elements"):
            HallucinationData(instances=data)
    
    def test_hallucination_validator_invalid_tuple_length(self):
        """Test that hallucination validator raises error with wrong tuple length."""
        with pytest.raises(ValueError):
            HallucinationData(instances=[("only_one_item",)])  # Single item tuple
    
    def test_hallucination_validator_invalid_data_type(self):
        """Test that hallucination validator raises error with invalid data type."""
        with pytest.raises(ValueError):
            HallucinationData(instances=["not_a_tuple"])  # String instead of tuple


class TestValidatorEdgeCases:
    """Test edge cases for all validators."""
    
    def test_all_validators_with_none_data(self):
        """Test that all validators handle None data appropriately."""
        with pytest.raises((ValueError, TypeError)):
            ToxicityData(instances=None)
        
        with pytest.raises((ValueError, TypeError)):
            RagData(instances=None)
        
        with pytest.raises((ValueError, TypeError)):
            HallucinationData(instances=None)
    
    def test_validators_with_large_text(self):
        """Test validators with large text inputs."""
        large_text = "A" * 10000  # 10k character string
        
        # Toxicity validator
        toxicity_data = [large_text]
        validated = ToxicityData(instances=toxicity_data)
        assert len(validated.instances) == 1
        assert validated.instances[0].text == large_text
        
        # RAG validator - (query, context) format
        rag_data = [("Short query", large_text)]
        validated = RagData(instances=rag_data)
        assert len(validated.instances) == 1
        assert validated.instances[0].query == "Short query"
        assert validated.instances[0].context == large_text
        
        # Hallucination validator
        hall_data = [("Query", large_text)]
        validated = HallucinationData(instances=hall_data)
        assert len(validated.instances) == 1
        assert validated.instances[0].response == large_text
    
    def test_validators_with_empty_strings(self):
        """Test validators with empty string inputs."""
        # Toxicity validator
        toxicity_data = ["", "non-empty"]
        validated = ToxicityData(instances=toxicity_data)
        assert validated.instances[0].text == ""
        assert validated.instances[1].text == "non-empty"
        
        # RAG validator - (query, context) format
        rag_data = [("", ""), ("query", "context")]
        validated = RagData(instances=rag_data)
        assert validated.instances[0].query == ""
        assert validated.instances[0].context == ""
        
        # Hallucination validator
        hall_data = [("", ""), ("query", "response")]
        validated = HallucinationData(instances=hall_data)
        assert validated.instances[0].query == ""
        assert validated.instances[0].response == ""