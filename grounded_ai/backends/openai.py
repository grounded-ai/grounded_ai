import os
from typing import Any, Optional, Type, Union

from pydantic import BaseModel

from ..base import BaseEvaluator
from ..schemas import EvaluationError, EvaluationInput, EvaluationOutput

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None


class OpenAIBackend(BaseEvaluator):
    """
    OpenAI backend using the Structured Outputs (beta.chat.completions.parse) implementation.
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
        if OpenAI is None:
            raise ImportError(
                "openai package is not installed. Please install it via `pip install openai`."
            )

        self.model_name = model_name
        self.client = client or OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
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

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]

        # Merge init kwargs with runtime kwargs (runtime overrides init)
        request_kwargs = {**self.kwargs, **kwargs}

        try:
            # Call OpenAI with Structured Outputs
            completion = self.client.beta.chat.completions.parse(
                model=self.model_name,
                messages=messages,
                response_format=output_schema,
                **request_kwargs,
            )

            message = completion.choices[0].message

            # Handle Refusals
            if message.refusal:
                return EvaluationError(
                    error_code="MODEL_REFUSAL", message=message.refusal
                )

            return message.parsed

        except Exception as e:
            error_code = str(getattr(e, "status_code", "UNKNOWN_ERROR"))
            error_msg = str(e)

            # Try to extract cleaner message
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
