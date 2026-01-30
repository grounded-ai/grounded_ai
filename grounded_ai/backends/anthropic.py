import json
import os
from typing import Any, Optional, Type, Union

from pydantic import BaseModel

from ..base import BaseEvaluator
from ..schemas import EvaluationError, EvaluationInput, EvaluationOutput

try:
    from anthropic import Anthropic
except ImportError:
    Anthropic = None


class AnthropicBackend(BaseEvaluator):
    """
    Anthropic backend using the 'structured-outputs' beta feature.
    """

    def __init__(
        self,
        model_name: str,
        api_key: str = None,
        client: Optional[Any] = None,
        input_schema: Type[BaseModel] = EvaluationInput,
        output_schema: Type[BaseModel] = EvaluationOutput,
        **kwargs,
    ):
        super().__init__(
            input_schema=input_schema,
            output_schema=output_schema,
            system_prompt=kwargs.pop("system_prompt", None),
        )
        if Anthropic is None:
            raise ImportError(
                "anthropic package is not installed. Please install it via `pip install anthropic`."
            )

        self.model_name = model_name
        self.client = client or Anthropic(
            api_key=api_key or os.getenv("ANTHROPIC_API_KEY")
        )
        self.kwargs = kwargs

    def _call_backend(
        self, input_data: BaseModel, output_schema: Type[BaseModel], **kwargs
    ) -> Union[BaseModel, EvaluationError]:
        # Construct Prompt
        system_prompt = (
            self.system_prompt
            or "You are an AI safety evaluator. Analyze the input and provide a structured evaluation."
        )

        # Use the input model's own formatting logic (or default template)
        if hasattr(input_data, "formatted_prompt"):
            user_content = input_data.formatted_prompt
        else:
            # Fallback for models without formatted_prompt
            user_content = str(input_data.model_dump())

        # Pydantic's model_json_schema() generates a compliant JSON schema but Anthropic needs additionalProperties=False
        json_schema = output_schema.model_json_schema()

        # Strip title to reduce tokens and noise if desired, but mainly enforce strictness
        if "title" in json_schema:
            del json_schema["title"]

        def _enforce_strict_schema(schema):
            if isinstance(schema, dict):
                # Remove unsuported validation keywords
                for key in [
                    "minimum",
                    "maximum",
                    "exclusiveMinimum",
                    "exclusiveMaximum",
                    "title",
                ]:
                    if key in schema:
                        del schema[key]

                if schema.get("type") == "object":
                    schema["additionalProperties"] = False
                    if "properties" in schema:
                        for prop in schema["properties"].values():
                            _enforce_strict_schema(prop)
            return schema

        json_schema = _enforce_strict_schema(json_schema)

        # Merge init kwargs with runtime kwargs (runtime overrides init)
        request_kwargs = {**self.kwargs, **kwargs}

        try:
            # Use Beta Structured Outputs
            response = self.client.beta.messages.create(
                model=self.model_name,
                system=system_prompt,
                messages=[{"role": "user", "content": user_content}],
                betas=["structured-outputs-2025-11-13"],
                output_format={"type": "json_schema", "schema": json_schema},
                **request_kwargs,
            )

            # Response is raw string in content[0].text which we parse
            raw_json = response.content[0].text
            data = json.loads(raw_json)
            return output_schema(**data)

        except Exception as e:
            # Catch API errors
            error_code = str(getattr(e, "status_code", "UNKNOWN_ERROR"))
            error_msg = str(e)

            if hasattr(e, "body") and isinstance(e.body, dict):
                if "error" in e.body and "message" in e.body["error"]:
                    error_msg = e.body["error"]["message"]
                elif "message" in e.body:
                    error_msg = e.body["message"]

            return EvaluationError(
                error_code=error_code,
                message=error_msg,
                details={"exception_type": type(e).__name__},
            )
