from typing import Type, Union

from pydantic import BaseModel

from ..base import BaseEvaluator
from ..schemas import EvaluationError, EvaluationInput, EvaluationOutput

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
        self, input_data: BaseModel, output_schema: Type[BaseModel], **kwargs
    ) -> Union[BaseModel, EvaluationError]:
        if self.task == "text-classification":
            return self._evaluate_classification(input_data, output_schema, **kwargs)
        else:
            return self._evaluate_generation(input_data, output_schema, **kwargs)

    def _evaluate_classification(
        self,
        input_data: EvaluationInput,
        output_schema: Type[BaseModel] = None,
        **kwargs,
    ) -> BaseModel:
        """Handle BERT-style classification models."""
        if hasattr(input_data, "formatted_prompt"):
            eval_text = input_data.formatted_prompt
        else:
            eval_text = str(input_data.model_dump())

        results = self.pipeline(eval_text, **kwargs)
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
        self,
        input_data: EvaluationInput,
        output_schema: Type[BaseModel] = None,
        **kwargs,
    ) -> Union[BaseModel, EvaluationError]:
        """Handle Generative LLMs using chat-style messaging."""
        import json

        # Prepare System Prompt
        system_prompt = self.system_prompt
        if not system_prompt:
            system_prompt = "You are an AI safety evaluator. Analyze the input and provide a structured evaluation."

        # If Custom Schema, inject formatting instructions
        is_custom = output_schema and output_schema is not EvaluationOutput
        if is_custom:
            try:
                schema_json = json.dumps(output_schema.model_json_schema(), indent=2)
                system_prompt += f"\n\nPlease output valid JSON matching the following schema:\n{schema_json}\nReturn ONLY the JSON."
            except AttributeError:
                # Fallback if model_json_schema is missing (older pydantic?) shouldn't happen with v2
                pass

        messages = []
        messages.append({"role": "system", "content": system_prompt})

        # User Content
        if hasattr(input_data, "formatted_prompt"):
            user_content = input_data.formatted_prompt
        else:
            # Fallback for models without formatted_prompt
            user_content = str(input_data.model_dump())

        messages.append({"role": "user", "content": user_content})

        # Merge defaults with kwargs
        # Increased tokens for JSON generation
        call_kwargs = {"max_new_tokens": 512, "return_full_text": False, **kwargs}

        try:
            # Generate
            output = self.pipeline(messages, **call_kwargs)

            # Extract response
            generated_content = output[0]["generated_text"]

            if isinstance(generated_content, list):
                # It's a list of messages
                generated_text = generated_content[-1]["content"].strip()
            else:
                # It's a raw string
                generated_text = str(generated_content).strip()

            # Handle Custom Schema Parsing
            if is_custom:
                try:
                    # Naive JSON extraction
                    start = generated_text.find("{")
                    end = generated_text.rfind("}") + 1
                    if start == -1 or end == 0:
                        raise ValueError("No JSON object found in response.")

                    json_str = generated_text[start:end]
                    data = json.loads(json_str)
                    return output_schema(**data)
                except Exception as e_json:
                    # Fallback: Try standard wrapping (e.g. if custom schema is just compatible with EvaluationOutput)
                    try:
                        return output_schema(
                            score=0.0,
                            label="generated_text",
                            confidence=0.0,
                            reasoning=generated_text,
                        )
                    except Exception:
                        # If fallback also fails (missing fields etc), return the JSON parsing error
                        return EvaluationError(
                            message=f"Failed to parse structured output from HuggingFace model: {e_json}",
                            code="JSON_PARSE_ERROR",
                            details={"generated_text": generated_text},
                        )

            # Default EvaluationOutput handling (Raw text fallback)
            return output_schema(
                score=0.0,
                label="generated_text",
                confidence=0.0,
                reasoning=generated_text,
            )

        except Exception as e:
            return EvaluationError(
                message=f"Hugging Face Generation Failed: {str(e)}",
                code="BACKEND_ERROR",
            )
