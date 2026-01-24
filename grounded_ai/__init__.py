from typing import Optional, Type, Union, Any, Dict

from .base import BaseEvaluator
from .schemas import EvaluationInput, EvaluationOutput, EvaluationError
# Import backends lazily or generally
# For now, we import locally within the factory to avoid heavy dependency loading if unused

class Evaluator:
    """
    Main entry point for Grounded AI evaluation.
    Auto-detects the appropriate backend based on the model string.
    """

    def __init__(self, model: str, **kwargs):
        self.backend = self._load_backend(model, **kwargs)

    def _load_backend(self, model: str, **kwargs) -> BaseEvaluator:
        """
        Routes the model string to the correct backend implementation.
        """
        if model.startswith("grounded-ai/"):
            from .backends.grounded_ai_slm.backend import GroundedAISLMBackend
            return GroundedAISLMBackend(model_id=model, **kwargs)
        
        elif model.startswith("openai/") or model.startswith("gpt-"):
            from .backends.openai import OpenAIBackend
            # Strip prefix if needed, or pass full string depending on implementation
            return OpenAIBackend(model_name=model.replace("openai/", ""), **kwargs)
        
        elif model.startswith("anthropic/") or "claude" in model:
            from .backends.anthropic import AnthropicBackend
            return AnthropicBackend(model_name=model.replace("anthropic/", ""), **kwargs)
            
        elif model.startswith("hf/") or model.startswith("huggingface/"):
            from .backends.huggingface import HuggingFaceBackend
            return HuggingFaceBackend(model_id=model, **kwargs)
            
        else:
            # customizable fallback logic or raise error
            raise ValueError(f"Unknown model provider for '{model}'. Supported: 'grounded-ai/', 'openai/', 'anthropic/', 'hf/'.")

    def evaluate(self, 
                 text: str, 
                 context: Optional[str] = None, 
                 query: Optional[str] = None, 
                 reference: Optional[str] = None,
                 **kwargs) -> Union[EvaluationOutput, EvaluationError]:
        """
        Main evaluation wrapper.
        Constructs the EvaluationInput and passes it to the backend.
        
        Args:
            text: The main text content to evaluate (response, doc, etc.)
            context: Additional context
            query: The user query
            reference: Ground truth or knowledge base reference
        """
        input_data = EvaluationInput(
            text=text,
            context=context,
            query=query,
            reference=reference
        )
        return self.backend.evaluate(input_data)

__all__ = ["Evaluator", "EvaluationInput", "EvaluationOutput", "EvaluationError"]
