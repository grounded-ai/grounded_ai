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

# Fields that belong to EvaluationInput construction vs. backend runtime kwargs
_INPUT_FIELDS = {"response", "query", "context", "base_template"}


def _prepare_input(
    backend: BaseEvaluator, input_data: Union[BaseModel, Dict[str, Any], str, None], kwargs: Dict[str, Any]
):
    """
    Normalize input_data into the backend's input model.
    Splits kwargs into input-construction fields and backend runtime args.
    Returns (input_data, backend_kwargs).
    """
    # A backend with its own input class (the Decider backend's DeciderInput) names the field
    # a bare string goes into; every other backend takes the stock EvaluationInput.
    primary = getattr(backend, "primary_input_field", None)
    if primary:
        schema, fields = backend.input_schema, set(backend.input_schema.model_fields)
    else:
        schema, fields, primary = EvaluationInput, _INPUT_FIELDS, "response"
    input_kwargs = {k: v for k, v in kwargs.items() if k in fields}
    backend_kwargs = {k: v for k, v in kwargs.items() if k not in fields}

    if isinstance(input_data, GenAIConversation):
        other = {k: v for k, v in input_kwargs.items() if k not in _INPUT_FIELDS and k != primary}
        input_data = schema(**{primary: input_data.to_evaluation_string()}, **other)
    elif isinstance(input_data, str):
        input_data = schema(**{primary: input_data}, **input_kwargs)
    elif input_data is None:
        input_data = schema(**input_kwargs)

    return input_data, backend_kwargs


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

            return OpenAIBackend(model_name=model.replace("openai/", ""), **kwargs)

        elif model.startswith("anthropic/") or (
            "claude" in model and not model.startswith("bedrock/")
        ):
            from .backends.anthropic import AnthropicBackend

            return AnthropicBackend(
                model_name=model.replace("anthropic/", ""), **kwargs
            )

        elif model.startswith("hf/") or model.startswith("huggingface/"):
            from .backends.huggingface import HuggingFaceBackend

            return HuggingFaceBackend(model_id=model, **kwargs)

        elif model.startswith("bedrock/"):
            from .backends.bedrock import BedrockBackend

            return BedrockBackend(model_id=model.replace("bedrock/", ""), **kwargs)

        elif model.startswith("decider/"):
            from .backends.decider import DeciderBackend

            return DeciderBackend(model_name=model[len("decider/"):], **kwargs)

        else:
            raise ValueError(
                f"Unknown model provider for '{model}'. Supported: 'grounded-ai/', 'openai/', 'anthropic/', 'hf/', 'bedrock/', 'decider/'."
            )

    def _prepare_input(
        self, input_data: Union[BaseModel, Dict[str, Any], str, None], kwargs: Dict[str, Any]
    ):
        """Normalize input_data for this evaluator's backend. Returns (input_data, backend_kwargs)."""
        return _prepare_input(self.backend, input_data, kwargs)

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
            **kwargs: Input fields (response, query, context, base_template) or backend
                      runtime args (temperature, max_tokens, etc.) forwarded to the backend.
        """
        input_data, backend_kwargs = self._prepare_input(input_data, kwargs)
        return self.backend.evaluate(input_data, output_schema=output_schema, **backend_kwargs)

class AsyncEvaluator(Evaluator):
    """
    Async entry point for Grounded AI evaluation. Same interface as Evaluator
    but evaluate() is a coroutine — use with `await`.

    OpenAI and Anthropic backends use native async clients; all others
    (Bedrock, HuggingFace, SLM) run in a thread pool via asyncio.to_thread.

    Example::

        evaluator = AsyncEvaluator("openai/gpt-4o")
        result = await evaluator.evaluate(response="London is in France.", context="London is in England.")

        # Batch with rate limiting
        import asyncio
        sem = asyncio.Semaphore(5)
        async def bounded(item):
            async with sem:
                return await evaluator.evaluate(response=item)
        results = await asyncio.gather(*[bounded(x) for x in dataset])
    """

    async def evaluate(
        self,
        input_data: Union[BaseModel, Dict[str, Any], str] = None,
        output_schema: Type[BaseModel] = None,
        **kwargs,
    ) -> Union[BaseModel, EvaluationError]:
        input_data, backend_kwargs = self._prepare_input(input_data, kwargs)
        return await self.backend.evaluate_async(input_data, output_schema=output_schema, **backend_kwargs)


def __getattr__(name: str):
    # CascadeEvaluator is imported on first use, so `import grounded_ai` does not load the Decider
    # backend (see "Lazy backend imports" in CLAUDE.md).
    if name == "CascadeEvaluator":
        from .cascade import CascadeEvaluator

        return CascadeEvaluator
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "Evaluator",
    "AsyncEvaluator",
    "CascadeEvaluator",
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
