from typing import Any, Dict, Type, Union, Optional
import os
import json
from pydantic import BaseModel

from ..base import BaseEvaluator
from ..schemas import EvaluationInput, EvaluationOutput, EvaluationError

try:
    from anthropic import Anthropic
except ImportError:
    Anthropic = None

class AnthropicBackend(BaseEvaluator):
    """
    Anthropic backend using the 'structured-outputs' beta feature.
    """

    def __init__(self, model_name: str, api_key: str = None, client: Optional[Any] = None, **kwargs):
        if Anthropic is None:
            raise ImportError("anthropic package is not installed. Please install it via `pip install anthropic`.")
        
        self.model_name = model_name
        self.client = client or Anthropic(api_key=api_key or os.getenv("ANTHROPIC_API_KEY"))
        self.kwargs = kwargs

    @property
    def input_schema(self) -> Type[BaseModel]:
        return EvaluationInput

    @property
    def output_schema(self) -> Type[BaseModel]:
        return EvaluationOutput

    def evaluate(self, input_data: Union[EvaluationInput, Dict[str, Any]]) -> Union[EvaluationOutput, EvaluationError]:
        # Standardize Input
        if isinstance(input_data, dict):
            input_data = EvaluationInput(**input_data)

        # Construct Prompt
        system_prompt = "You are an AI safety evaluator. Analyze the input and provide a structured evaluation."
        
        user_content = f"Task: Evaluate the following content.\n"
        if input_data.query:
            user_content += f"Query: {input_data.query}\n"
        if input_data.context:
            user_content += f"Context: {input_data.context}\n"
        if input_data.reference:
            user_content += f"Reference: {input_data.reference}\n"
        
        user_content += f"Content to Evaluate: {input_data.text}"

        json_schema = EvaluationOutput.model_json_schema()
        
        try:
            response = self.client.messages.create(
                model=self.model_name,
                max_tokens=1024,
                system=system_prompt,
                messages=[
                    {"role": "user", "content": user_content}
                ],
                tools=[{
                    "name": "evaluate_content",
                    "description": "Output the evaluation result in a structured format.",
                    "input_schema": json_schema
                }],
                tool_choice={"type": "tool", "name": "evaluate_content"},
                **self.kwargs
            )

            # Parse the tool use output
            for content_block in response.content:
                if content_block.type == "tool_use" and content_block.name == "evaluate_content":
                    data = content_block.input
                    return EvaluationOutput(**data)

            return EvaluationError(
                error_code="OUTPUT_FORMAT_ERROR",
                message="Anthropic model did not output a structured evaluation using the tool."
            )

        except Exception as e:
            # Catch API errors
            error_code = str(getattr(e, "status_code", "UNKNOWN_ERROR"))
            error_msg = str(e)
            
            # Try to extract cleaner message from Anthropic/OpenAI style errors
            if hasattr(e, "body") and isinstance(e.body, dict):
                 if "error" in e.body and "message" in e.body["error"]:
                     error_msg = e.body["error"]["message"]
                 elif "message" in e.body:
                     error_msg = e.body["message"]
            
            return EvaluationError(
                error_code=error_code, 
                message=error_msg,
                details={"exception_type": type(e).__name__}
            )
