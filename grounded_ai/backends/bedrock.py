import json
from typing import Optional, Type, Union

from pydantic import BaseModel

from ..base import BaseEvaluator
from ..schemas import EvaluationError, EvaluationInput, EvaluationOutput

try:
    import boto3
except ImportError:
    boto3 = None


class BedrockBackend(BaseEvaluator):
    """
    AWS Bedrock backend using the 'structured-outputs' feature for Anthropic models.
    """

    def __init__(
        self,
        model_id: str,
        region_name: Optional[str] = None,
        aws_access_key_id: Optional[str] = None,
        aws_secret_access_key: Optional[str] = None,
        aws_session_token: Optional[str] = None,
        input_schema: Type[BaseModel] = EvaluationInput,
        output_schema: Type[BaseModel] = EvaluationOutput,
        **kwargs,
    ):
        super().__init__(
            input_schema=input_schema,
            output_schema=output_schema,
            system_prompt=kwargs.pop("system_prompt", None),
        )
        if boto3 is None:
            raise ImportError(
                "boto3 package is not installed. Please install it via `pip install boto3`."
            )

        self.model_id = model_id

        # Boto3 client initialization
        self.session_kwargs = {}
        if region_name:
            self.session_kwargs["region_name"] = region_name
        if aws_access_key_id:
            self.session_kwargs["aws_access_key_id"] = aws_access_key_id
        if aws_secret_access_key:
            self.session_kwargs["aws_secret_access_key"] = aws_secret_access_key
        if aws_session_token:
            self.session_kwargs["aws_session_token"] = aws_session_token

        self.client = boto3.client("bedrock-runtime", **self.session_kwargs)
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

        # Pydantic's model_json_schema() generates a compliant JSON schema but we need to enforce strictness
        # to ensure high quality structured outputs.
        json_schema = output_schema.model_json_schema()

        # Strip title to reduce tokens and noise if desired
        # Strip title to reduce tokens and noise if desired
        if "title" in json_schema:
            del json_schema["title"]

        def _enforce_strict_schema(schema):
            if isinstance(schema, dict):
                # Remove validation keywords that might confuse the model or aren't supported strictly
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
                    if "additionalProperties" not in schema:
                        schema["additionalProperties"] = False
                    if "properties" in schema:
                        for prop in schema["properties"].values():
                            _enforce_strict_schema(prop)

                # Handle array items
                if schema.get("type") == "array":
                    if "items" in schema:
                        _enforce_strict_schema(schema["items"])

                # Recursively process definitions ($defs or definitions)
                for def_key in ["$defs", "definitions"]:
                    if def_key in schema:
                        for def_item in schema[def_key].values():
                            _enforce_strict_schema(def_item)

            return schema

        json_schema = _enforce_strict_schema(json_schema)

        # Merge init kwargs with runtime kwargs (runtime overrides init)
        request_kwargs = {**self.kwargs, **kwargs}

        # Converse API parameters
        # inferenceConfig contains maxTokens, temperature, etc.
        inference_config = {}
        if "max_tokens" in request_kwargs:
            inference_config["maxTokens"] = request_kwargs.pop("max_tokens")
        else:
            inference_config["maxTokens"] = 1024

        if "temperature" in request_kwargs:
            inference_config["temperature"] = request_kwargs.pop("temperature")
        try:
            # Prepare Tool/Output Config
            # Note: The schema must be passed as a JSON string inside the structure
            tool_config = {
                "jsonSchema": {
                    "schema": json.dumps(json_schema),
                    "name": output_schema.__name__.lower() or "evaluation_output",
                    "description": "Structured evaluation output",
                }
            }

            output_config = {
                "textFormat": {"type": "json_schema", "structure": tool_config}
            }

            # Prepare Messages
            messages = [{"role": "user", "content": [{"text": user_content}]}]

            system_message = [{"text": system_prompt}]

            # Prepare arguments for converse
            converse_kwargs = {
                "modelId": self.model_id,
                "messages": messages,
                "system": system_message,
                "inferenceConfig": inference_config,
            }

            # Add outputConfig if present (Structured Outputs)
            if output_config:
                converse_kwargs["outputConfig"] = output_config

            # Add any other additional fields (like headers, or specific future API params)
            if request_kwargs:
                converse_kwargs["additionalModelRequestFields"] = request_kwargs

            # Call Converse API
            response = self.client.converse(**converse_kwargs)

            # Parse Response
            if "output" in response and "message" in response["output"]:
                content = response["output"]["message"]["content"]
                if content:
                    # Iterate through content blocks to find text or toolUse
                    # Some models return 'reasoningContent' first, so we can't just check content[0]
                    for block in content:
                        if "text" in block:
                            response_text = block["text"]
                            try:
                                data = json.loads(response_text)
                                return output_schema(**data)
                            except json.JSONDecodeError:
                                # Continue looking if this text block isn't valid JSON
                                # (e.g. might be a generic preamble)
                                continue

                        elif "toolUse" in block:
                            tool_use = block["toolUse"]
                            # We accept the first valid tool use
                            return output_schema(**tool_use["input"])

            return EvaluationError(
                error_code="EMPTY_RESPONSE",
                message="Bedrock Converse response contained no valid content.",
                details={"full_response": str(response)},
            )

        except Exception as e:
            return EvaluationError(
                error_code="BEDROCK_ERROR",
                message=str(e),
                details={"exception_type": type(e).__name__},
            )
