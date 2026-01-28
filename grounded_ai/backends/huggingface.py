import json
import logging
from typing import Type, Union, Set, List
from pydantic import BaseModel

try:
    import torch
    from transformers import (
        pipeline,
        AutoModelForCausalLM,
        AutoTokenizer,
        LogitsProcessor,
    )
except ImportError:
    pipeline = None
    torch = None
    AutoModelForCausalLM = None
    AutoTokenizer = None
    LogitsProcessor = object  # Mock for type checking if missing

from ..base import BaseEvaluator
from ..schemas import EvaluationInput, EvaluationOutput, EvaluationError


class PydanticSchemaBoostProcessor(LogitsProcessor):
    """Boost tokens that are part of a Pydantic model schema"""

    def __init__(
        self,
        tokenizer,
        pydantic_model: Type[BaseModel],
        boost_factor: float = 3.0,
        boost_structural: bool = True,
    ):
        self.tokenizer = tokenizer
        self.boost_factor = boost_factor
        self.schema = pydantic_model.model_json_schema()

        # Extract all relevant tokens
        self.schema_tokens = self._extract_all_schema_tokens()

        # Add structural JSON tokens
        if boost_structural:
            structural = ["{", "}", "[", "]", ":", ",", '"', "true", "false", "null"]
            self.schema_tokens.update(self._tokenize_variants(structural))

    def _extract_all_schema_tokens(self) -> Set[int]:
        """Extract all tokens from Pydantic schema"""
        tokens = set()

        # Extract from properties (field names)
        if "properties" in self.schema:
            for field_name, field_info in self.schema["properties"].items():
                tokens.update(self._tokenize_variants([field_name]))

                # Handle enum values (Literal types)
                if "enum" in field_info:
                    tokens.update(self._tokenize_variants(field_info["enum"]))

                # Handle nested schemas
                if "properties" in field_info:
                    tokens.update(self._extract_from_nested(field_info))

                # Handle anyOf (Optional types)
                if "anyOf" in field_info:
                    for option in field_info["anyOf"]:
                        if "properties" in option:
                            tokens.update(self._extract_from_nested(option))

        # Extract from definitions/defs (nested models)
        for key in ["definitions", "$defs"]:
            if key in self.schema:
                for def_name, def_schema in self.schema[key].items():
                    if "properties" in def_schema:
                        for field_name in def_schema["properties"].keys():
                            tokens.update(self._tokenize_variants([field_name]))

        return tokens

    def _extract_from_nested(self, schema_obj: dict) -> Set[int]:
        """Extract tokens from nested schema objects"""
        tokens = set()
        if "properties" in schema_obj:
            for field_name, field_info in schema_obj["properties"].items():
                tokens.update(self._tokenize_variants([field_name]))
                if "enum" in field_info:
                    tokens.update(self._tokenize_variants(field_info["enum"]))
        return tokens

    def _tokenize_variants(self, words: List[str]) -> Set[int]:
        """Tokenize words with common JSON formatting variants"""
        token_ids = set()

        for word in words:
            word_str = str(word)  # Handle non-string enums

            # Common variants in JSON
            variants = [
                word_str,  # raw
                f" {word_str}",  # with leading space
                f'"{word_str}"',  # quoted
                f' "{word_str}"',  # quoted with space
                f': "{word_str}"',  # after colon
                f': {word_str}',  # after colon unquoted
                f', "{word_str}"',  # after comma
                f'{{\n  "{word_str}"',  # in new object
            ]

            for variant in variants:
                try:
                    encoded = self.tokenizer.encode(variant, add_special_tokens=False)
                    token_ids.update(encoded)
                except Exception:
                    pass  # Skip if encoding fails

        return token_ids

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor
    ) -> torch.FloatTensor:
        """Apply boost to schema tokens"""
        for token_id in self.schema_tokens:
            if token_id < scores.shape[-1]:  # Ensure token exists in vocab
                scores[:, token_id] += self.boost_factor

        return scores


class HuggingFaceBackend(BaseEvaluator):
    """
    Generic backend for running standard HuggingFace models locally using `transformers`.
    Supports 'text-classification' (using pipeline) and 'text-generation' (using AutoModel + LogitsProcessor).
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
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Simple task heuristic if not provided
        if not task:
            task = "text-generation"

        self.task = task
        self.model = None
        self.tokenizer = None
        self.pipeline = None

        if self.task == "text-classification":
            # Use pipeline for classification tasks
            self.pipeline = pipeline(
                task, model=self.model_id, device=self.device, **kwargs
            )
        else:
            # Use AutoModel for generation tasks to support custom logits processing
            logging.info(f"Loading {self.model_id} for generation...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                device_map="auto" if self.device == "cuda" else None,
                torch_dtype="auto",
                **kwargs,
            )
            if self.device != "cuda" and self.device is not None:
                self.model.to(self.device)

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
    ) -> BaseModel:
        """Handle Generative LLMs using schema-boosted generation."""

        # 1. Prepare Schema
        target_schema = output_schema or self.output_schema
        schema_json = json.dumps(target_schema.model_json_schema(), indent=2)

        # 2. Construct Prompt embedding the schema
        # We try to use a standardized prompt format asking for JSON
        system_prompt = (
            self.system_prompt
            or "You are an AI assistant that outputs strictly valid JSON."
        )

        user_content = ""
        if hasattr(input_data, "formatted_prompt"):
            user_content = input_data.formatted_prompt
        else:
            # Smart default construction
            parts = []
            if getattr(input_data, "response", None):
                parts.append(f"Input Text: {input_data.response}")
            if getattr(input_data, "query", None):
                parts.append(f"Query: {input_data.query}")
            if getattr(input_data, "context", None):
                parts.append(f"Context: {input_data.context}")
            user_content = "\n".join(parts)

        final_prompt = (
            f"{system_prompt}\n\n"
            f"Generate a JSON object matching this schema:\n{schema_json}\n\n"
            f"Task Input:\n{user_content}\n\n"
            f"Output only valid JSON:"
        )

        # 3. Setup Logits Processor
        boost_factor = kwargs.pop("boost_factor", 3.0)
        processor = PydanticSchemaBoostProcessor(
            self.tokenizer, target_schema, boost_factor=boost_factor
        )

        # 4. Generate
        inputs = self.tokenizer(final_prompt, return_tensors="pt")
        if self.device == "cuda":
            inputs = inputs.to("cuda")
        elif self.device == "mps" and torch.backends.mps.is_available():
             inputs = inputs.to("mps")

        generation_kwargs = {
            "max_new_tokens": 512,
            "temperature": 0.7,
            "do_sample": True,
            "pad_token_id": self.tokenizer.eos_token_id,
        }
        generation_kwargs.update(kwargs)

        outputs = self.model.generate(
            inputs.input_ids,
            logits_processor=[processor],
            **generation_kwargs,
        )

        # 5. Decode and Parse
        generated_text = self.tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1] :], skip_special_tokens=True
        )

        try:
            # Attempt to find JSON block
            json_start = generated_text.find("{")
            json_end = generated_text.rfind("}") + 1
            if json_start != -1 and json_end > json_start:
                json_str = generated_text[json_start:json_end]
                parsed_data = json.loads(json_str)
                return target_schema(**parsed_data)
            else:
                # Fallback if no specific JSON block found, try parsing whole text
                parsed_data = json.loads(generated_text)
                return target_schema(**parsed_data)

        except Exception as e:
            # Fallback to a generic error/unparsed response
            logging.error(f"Failed to parse JSON generation: {e}")
            return EvaluationError(
                message=f"Failed to parse JSON generation. Error: {str(e)}",
                details={"raw_output": generated_text}
            )

