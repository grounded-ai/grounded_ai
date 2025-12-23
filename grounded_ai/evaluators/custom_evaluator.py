"""
Custom evaluator implementation for maximum flexibility.

This module provides a CustomEvaluator class that allows users to bring their own:
- Models (any model type - local, API, custom inference)
- Prompts (completely custom prompt templates)
- Evaluation logic (custom scoring/evaluation functions)
- Data formats (flexible input/output handling)
"""
from abc import abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Union
from jinja2 import Template


@dataclass
class CustomEvaluator:
    """
    A flexible evaluator framework that allows users to bring their own models and evaluation logic.
    
    The user provides:
    1. model_runner: Function that takes a prompt and returns a response
    2. evaluate_func: Function that processes responses and returns evaluation results
    3. Optional format functions for input/output processing
    
    Example Usage:
    
    # Basic example with OpenAI
    ```python
    import openai
    
    def my_model_runner(prompt: str) -> str:
        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1
        )
        return response.choices[0].message.content
    
    def my_evaluate_func(data: List[dict], responses: List[str]) -> dict:
        # Your custom evaluation logic
        positive_count = sum(1 for r in responses if "positive" in r.lower())
        return {
            "positive": positive_count,
            "total": len(responses),
            "positive_rate": positive_count / len(responses) if responses else 0
        }
    
    evaluator = CustomEvaluator(
        model_runner=my_model_runner,
        evaluate_func=my_evaluate_func,
        prompt_template="Classify sentiment: {{ text }}\nSentiment:"
    )
    
    data = [{"text": "I love this!"}, {"text": "This is terrible"}]
    results = evaluator.evaluate(data)
    # Output: {"positive": 1, "total": 2, "positive_rate": 0.5}
    ```
    
    # Advanced example with custom local model
    ```python
    class MyModel:
        def __init__(self):
            # Load your custom model
            pass
        
        def predict(self, text: str) -> str:
            # Your inference logic
            return "model_output"
    
    my_model = MyModel()
    
    def format_input_func(instance: dict) -> str:
        # Custom input formatting beyond just template rendering
        formatted = f"Context: {instance.get('context', '')}\nQuery: {instance['text']}"
        return formatted
    
    def format_output_func(response: str) -> str:
        # Custom output processing
        return response.strip().upper()
    
    def evaluate_func(data: List[dict], responses: List[str]) -> dict:
        # Custom evaluation logic
        scores = []
        for response in responses:
            if "HIGH" in response:
                scores.append(3)
            elif "MEDIUM" in response:
                scores.append(2)
            else:
                scores.append(1)
        
        return {
            "average_score": sum(scores) / len(scores) if scores else 0,
            "scores": scores,
            "high_count": sum(1 for s in scores if s == 3)
        }
    
    evaluator = CustomEvaluator(
        model_runner=my_model.predict,
        evaluate_func=evaluate_func,
        format_input_func=format_input_func,
        format_output_func=format_output_func,
        prompt_template="Rate quality: {{ text }}"  # Optional if format_input_func handles everything
    )
    ```
    """
    
    # Required components
    model_runner: Callable[[str], str] = None
    """Function that takes a prompt string and returns model response"""
    
    evaluate_func: Callable[[List[dict], List[str]], dict] = None
    """Function that takes (original_data, model_responses) and returns evaluation results"""
    
    # Optional prompt template (Jinja2)
    prompt_template: Optional[str] = None
    """Optional Jinja2 template string for formatting prompts. Not needed if format_input_func is provided."""
    
    # Optional formatting functions
    format_input_func: Optional[Callable[[dict], str]] = None
    """Optional function to format input data into prompts. If not provided, uses prompt_template."""
    
    format_output_func: Optional[Callable[[str], str]] = None
    """Optional function to postprocess model responses before evaluation."""
    
    # Configuration
    debug: bool = False
    """Enable debug logging"""
    
    metadata: Dict[str, Any] = field(default_factory=dict)
    """Additional metadata for tracking evaluation runs"""
    
    def __post_init__(self):
        """Validate required components are provided."""
        if not self.model_runner:
            raise ValueError("model_runner is required")
        if not self.evaluate_func:
            raise ValueError("evaluate_func is required")
        if not self.format_input_func and not self.prompt_template:
            raise ValueError("Either format_input_func or prompt_template is required")
        
        # Compile the Jinja2 template if provided
        if self.prompt_template:
            self._template = Template(self.prompt_template)
        
        if self.debug:
            print(f"CustomEvaluator initialized")
            print(f"  - Has template: {bool(self.prompt_template)}")
            print(f"  - Has format_input_func: {bool(self.format_input_func)}")
            print(f"  - Has format_output_func: {bool(self.format_output_func)}")
    
    def format_input(self, instance: Dict[str, Any]) -> str:
        """Format a single instance into a prompt string."""
        if self.format_input_func:
            # Use custom formatting function
            return self.format_input_func(instance)
        elif self.prompt_template:
            # Use Jinja2 template
            return self._template.render(**instance)
        else:
            raise ValueError("No input formatting method available")
    
    def format_output(self, response: str) -> str:
        """Format/clean a model response."""
        if self.format_output_func:
            return self.format_output_func(response)
        else:
            # Default: just strip whitespace
            return response.strip()
    
    def run_single(self, instance: Dict[str, Any]) -> str:
        """Process a single instance through the entire pipeline."""
        # Format the prompt
        prompt = self.format_input(instance)
        
        if self.debug:
            print(f"Formatted prompt: {prompt[:200]}...")
        
        # Run through model
        response = self.model_runner(prompt)
        
        if self.debug:
            print(f"Raw model response: {response}")
        
        # Format the output
        formatted_response = self.format_output(response)
        
        if self.debug:
            print(f"Formatted response: {formatted_response}")
        
        return formatted_response
    
    def evaluate(self, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Evaluate a list of instances.
        
        Args:
            data: List of dictionaries containing instance data
            
        Returns:
            Dictionary containing evaluation results from the evaluate_func
        """
        if self.debug:
            print(f"Evaluating {len(data)} instances")
        
        responses = []
        
        # Process all instances
        for i, instance in enumerate(data):
            if self.debug:
                print(f"Processing instance {i+1}/{len(data)}")
            
            response = self.run_single(instance)
            responses.append(response)
        
        # Run custom evaluation function
        results = self.evaluate_func(data, responses)
        
        # Add metadata
        if not isinstance(results, dict):
            # If evaluate_func doesn't return a dict, wrap it
            results = {"evaluation_result": results}
        
        results["metadata"] = {
            "total_instances": len(data),
            "evaluator_config": {
                "has_template": bool(self.prompt_template),
                "has_format_input": bool(self.format_input_func),
                "has_format_output": bool(self.format_output_func),
                **self.metadata
            }
        }
        
        if self.debug:
            print(f"Final evaluation results: {results}")
        
        return results

