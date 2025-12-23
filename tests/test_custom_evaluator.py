"""
Test the custom evaluator functionality without heavy dependencies.
"""
import pytest
from grounded_ai.evaluators.custom_evaluator import CustomEvaluator


def mock_model_runner(prompt: str) -> str:
    """Mock model that returns predictable responses for testing."""
    if "positive" in prompt.lower():
        return "POSITIVE"
    elif "negative" in prompt.lower():
        return "NEGATIVE"
    else:
        return "NEUTRAL"


def test_custom_evaluator_basic():
    """Test basic CustomEvaluator functionality."""
    
    # Create a simple evaluation function
    def simple_evaluate_func(data, responses):
        positive_count = sum(1 for r in responses if "POSITIVE" in r)
        return {"positive": positive_count, "total": len(responses)}
    
    # Create evaluator
    evaluator = CustomEvaluator(
        model_runner=mock_model_runner,
        evaluate_func=simple_evaluate_func,
        prompt_template="Classify sentiment: {{ text }}"
    )
    
    # Test data
    data = [
        {"text": "I love this positive thing"},
        {"text": "This is terrible and negative"},
        {"text": "Just neutral content"}
    ]
    
    # Run evaluation
    results = evaluator.evaluate(data)
    
    # Verify results
    assert results["positive"] == 1
    assert results["total"] == 3
    assert "metadata" in results


def test_custom_evaluator_with_format_functions():
    """Test CustomEvaluator with custom format functions."""
    
    def format_input_func(instance):
        # Custom input formatting
        return f"CUSTOM: {instance['text'].upper()}"
    
    def format_output_func(response):
        # Custom output formatting
        return response.lower()
    
    def evaluate_func(data, responses):
        return {"lowercase_responses": responses}
    
    evaluator = CustomEvaluator(
        model_runner=mock_model_runner,
        evaluate_func=evaluate_func,
        format_input_func=format_input_func,
        format_output_func=format_output_func
    )
    
    data = [{"text": "positive message"}]
    results = evaluator.evaluate(data)
    
    # Response should be lowercase due to format_output_func
    assert results["lowercase_responses"][0] == "positive"


def test_custom_evaluator_template_only():
    """Test CustomEvaluator with just template (no format functions)."""
    
    def evaluate_func(data, responses):
        return {"responses": responses}
    
    evaluator = CustomEvaluator(
        model_runner=mock_model_runner,
        evaluate_func=evaluate_func,
        prompt_template="Process: {{ text }}"
    )
    
    data = [{"text": "positive content"}]
    results = evaluator.evaluate(data)
    
    assert len(results["responses"]) == 1
    assert results["responses"][0] == "POSITIVE"


def test_custom_evaluator_validation():
    """Test that CustomEvaluator validates required parameters."""
    
    def dummy_evaluate(data, responses):
        return {}
    
    # Should raise error without model_runner
    with pytest.raises(ValueError, match="model_runner is required"):
        CustomEvaluator(
            evaluate_func=dummy_evaluate,
            prompt_template="test"
        )
    
    # Should raise error without evaluate_func
    with pytest.raises(ValueError, match="evaluate_func is required"):
        CustomEvaluator(
            model_runner=mock_model_runner,
            prompt_template="test"
        )
    
    # Should raise error without any formatting method
    with pytest.raises(ValueError, match="Either format_input_func or prompt_template is required"):
        CustomEvaluator(
            model_runner=mock_model_runner,
            evaluate_func=dummy_evaluate
        )


def test_template_rendering():
    """Test Jinja2 template rendering with various inputs."""
    
    def evaluate_func(data, responses):
        return {"count": len(responses)}
    
    evaluator = CustomEvaluator(
        model_runner=mock_model_runner,
        evaluate_func=evaluate_func,
        prompt_template="Task: {{ task }}\nInput: {{ text }}\nContext: {{ context }}"
    )
    
    data = [{
        "task": "sentiment analysis",
        "text": "positive message",
        "context": "review data"
    }]
    
    # This should not raise an error
    results = evaluator.evaluate(data)
    assert results["count"] == 1


def test_format_input_func_priority():
    """Test that format_input_func takes priority over prompt_template."""
    
    def format_input_func(instance):
        return f"CUSTOM FORMAT: {instance['text']}"
    
    def evaluate_func(data, responses):
        return {"first_response": responses[0] if responses else ""}
    
    # Mock model that echoes the prompt
    def echo_model_runner(prompt):
        return prompt[:20]  # Return first 20 chars
    
    evaluator = CustomEvaluator(
        model_runner=echo_model_runner,
        evaluate_func=evaluate_func,
        format_input_func=format_input_func,
        prompt_template="TEMPLATE: {{ text }}"  # Should be ignored
    )
    
    data = [{"text": "hello"}]
    results = evaluator.evaluate(data)
    
    # Should use format_input_func, not template
    assert "CUSTOM FORMAT" in results["first_response"]
    assert "TEMPLATE" not in results["first_response"]


def test_debug_output(capsys):
    """Test debug output functionality."""
    
    def evaluate_func(data, responses):
        return {"total": len(responses)}
    
    evaluator = CustomEvaluator(
        model_runner=mock_model_runner,
        evaluate_func=evaluate_func,
        prompt_template="Test: {{ text }}",
        debug=True
    )
    
    data = [{"text": "test"}]
    evaluator.evaluate(data)
    
    captured = capsys.readouterr()
    assert "CustomEvaluator initialized" in captured.out
    assert "Evaluating 1 instances" in captured.out


def test_metadata_inclusion():
    """Test that metadata is properly included in results."""
    
    def evaluate_func(data, responses):
        return {"custom_result": "test"}
    
    evaluator = CustomEvaluator(
        model_runner=mock_model_runner,
        evaluate_func=evaluate_func,
        prompt_template="{{ text }}",
        metadata={"version": "1.0", "model": "test_model"}
    )
    
    data = [{"text": "test"}]
    results = evaluator.evaluate(data)
    
    assert "metadata" in results
    assert results["metadata"]["total_instances"] == 1
    assert results["metadata"]["evaluator_config"]["version"] == "1.0"
    assert results["metadata"]["evaluator_config"]["model"] == "test_model"


def test_evaluate_func_non_dict_return():
    """Test handling when evaluate_func doesn't return a dict."""
    
    def evaluate_func(data, responses):
        # Return non-dict value
        return 42
    
    evaluator = CustomEvaluator(
        model_runner=mock_model_runner,
        evaluate_func=evaluate_func,
        prompt_template="{{ text }}"
    )
    
    data = [{"text": "test"}]
    results = evaluator.evaluate(data)
    
    # Should wrap non-dict return in dict
    assert results["evaluation_result"] == 42
    assert "metadata" in results