from typing import Type, Union
from pydantic import BaseModel

from ..base import BaseEvaluator
from ..schemas import EvaluationInput, EvaluationOutput, EvaluationError

try:
    from transformers import pipeline
except ImportError:
    pipeline = None


class HuggingFaceBackend(BaseEvaluator):
    """
    Generic backend for running standard HuggingFace models locally using `transformers`.
    Supports both 'text-generation' (LLMs) and 'text-classification' (BERT-style) pipelines.
    """

    def __init__(
        self,
        model_id: str,
        device: str = None,
        task: str = None,
        input_schema: Type[BaseModel] = EvaluationInput,
        output_schema: Type[BaseModel] = EvaluationOutput,
        **kwargs,
    ):
        super().__init__(
            input_schema=input_schema,
            output_schema=output_schema,
            system_prompt=kwargs.pop("system_prompt", None),
        )
        if pipeline is None:
            raise ImportError(
                "transformers package is not installed. Please install it via `pip install transformers`."
            )

        self.model_id = model_id.replace("hf/", "").replace("huggingface/", "")

        # Simple task heuristic if not provided
        if not task:
            task = "text-generation"

        self.task = task
        self.pipeline = pipeline(task, model=self.model_id, device=device, **kwargs)

    def _call_backend(
        self, input_data: BaseModel, output_schema: Type[BaseModel]
    ) -> Union[BaseModel, EvaluationError]:
        if self.task == "text-classification":
            return self._evaluate_classification(input_data, output_schema)
        else:
            return self._evaluate_generation(input_data, output_schema)

    def _evaluate_classification(
        self, input_data: EvaluationInput, output_schema: Type[BaseModel] = None
    ) -> BaseModel:
        """Handle BERT-style classification models."""
        eval_text = input_data.response
        if input_data.context:
            eval_text = f"Context: {input_data.context}\nText: {eval_text}"

        results = self.pipeline(eval_text, top_k=1)
        top_result = results[0] if isinstance(results, list) else results

        label = top_result.get("label", "unknown")
        score = top_result.get("score", 0.0)

        return output_schema(
            score=score,
            label=label,
            confidence=score,
            reasoning=f"Classified as {label} with score {score:.4f}",
        )

    def _evaluate_generation(
        self, input_data: EvaluationInput, output_schema: Type[BaseModel] = None
    ) -> BaseModel:
        """Handle Generative LLMs using chat-style messaging."""

        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})

        # User Content
        if hasattr(input_data, "formatted_prompt"):
            user_content = input_data.formatted_prompt
        elif hasattr(input_data, "response"):
            user_content = f"Evaluate: {input_data.response}"
        else:
            # Fallback for custom models
            user_content = str(input_data.model_dump())

        messages.append({"role": "user", "content": user_content})

        # Generate
        # When passing messages to text-generation pipeline, it returns a list of dicts (the conversation)
        output = self.pipeline(messages, max_new_tokens=100, return_full_text=False)

        # Extract the assistant's response.
        # Output format is typically: [{'generated_text': [{'role': '...', 'content': '...'}, ...]}]
        # BUT if return_full_text=False, it might return just the string for some models/versions.
        generated_content = output[0]["generated_text"]

        if isinstance(generated_content, list):
            # It's a list of messages
            generated_text = generated_content[-1]["content"].strip()
        else:
            # It's a raw string
            generated_text = str(generated_content).strip()

        return output_schema(
            score=0.0, label="generated_text", confidence=0.0, reasoning=generated_text
        )
