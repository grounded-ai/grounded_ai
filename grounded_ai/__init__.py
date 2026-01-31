from typing import Any, Dict, Type, Union

from pydantic import BaseModel

from .base import BaseEvaluator
from .otel import (
    GenAIConversation,
    GenAIMessage,
    GenAISpan,
    MessagePart,
    TokenUsage,
    TraceConverter,
    convert_traces,
)
from .schemas import EvaluationError, EvaluationInput, EvaluationOutput

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

            return AnthropicBackend(
                model_name=model.replace("anthropic/", ""), **kwargs
            )

        elif model.startswith("hf/") or model.startswith("huggingface/"):
            from .backends.huggingface import HuggingFaceBackend

            return HuggingFaceBackend(model_id=model, **kwargs)

        else:
            # customizable fallback logic or raise error
            raise ValueError(
                f"Unknown model provider for '{model}'. Supported: 'grounded-ai/', 'openai/', 'anthropic/', 'hf/'."
            )

    def evaluate(
        self,
        input_data: Union[BaseModel, Dict[str, Any], str] = None,
        output_schema: Type[BaseModel] = None,
        **kwargs,
    ) -> Union[BaseModel, EvaluationError]:
        """
        Main evaluation wrapper.

        Args:
            input_data: Pydantic model, Dict, or string (interpreted as 'response' field).
            output_schema: Optional override for the output structure.
            **kwargs: If input_data is not provided, kwargs are used to construct EvaluationInput.
        """
        # Handle string passed as positional argument or direct string
        if isinstance(input_data, str):
            input_data = EvaluationInput(response=input_data, **kwargs)

        # Handle kwargs-based construction if input_data is None
        elif input_data is None:
            input_data = EvaluationInput(**kwargs)

        return self.backend.evaluate(input_data, output_schema=output_schema)


__all__ = [
    "Evaluator",
    "EvaluationInput",
    "EvaluationOutput",
    "EvaluationError",
    # OTel types
    "GenAIConversation",
    "GenAISpan",
    "GenAIMessage",
    "MessagePart",
    "TokenUsage",
    "TraceConverter",
    "convert_traces",
]
