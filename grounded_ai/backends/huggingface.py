from typing import Any, Dict, Type, Union, Optional
from pydantic import BaseModel

from ..base import BaseEvaluator
from ..schemas import EvaluationInput, EvaluationOutput

try:
    from transformers import pipeline
except ImportError:
    pipeline = None

class HuggingFaceBackend(BaseEvaluator):
    """
    Generic backend for running standard HuggingFace models locally using `transformers`.
    Does NOT enforce Grounded AI specific strict validation or prompts (unlike `grounded_ai_slm`).
    """

    def __init__(self, model_id: str, device: str = None, task: str = "text-generation", **kwargs):
        if pipeline is None:
            raise ImportError("transformers package is not installed. Please install it via `pip install transformers`.")
        
        self.model_id = model_id.replace("hf/", "").replace("huggingface/", "")
        self.pipeline = pipeline(task, model=self.model_id, device=device, **kwargs)

    @property
    def input_schema(self) -> Type[BaseModel]:
        return EvaluationInput

    @property
    def output_schema(self) -> Type[BaseModel]:
        return EvaluationOutput

    def evaluate(self, input_data: Union[EvaluationInput, Dict[str, Any]]) -> EvaluationOutput:
        # Standardize Input
        if isinstance(input_data, dict):
            input_data = EvaluationInput(**input_data)

        # Simple Prompting (Generic)
        user_content = f"Task: Evaluate the following content.\n"
        if input_data.query:
            user_content += f"Query: {input_data.query}\n"
        if input_data.context:
            user_content += f"Context: {input_data.context}\n"
        if input_data.reference:
            user_content += f"Reference: {input_data.reference}\n"
        
        user_content += f"Content to Evaluate: {input_data.text}\n"
        user_content += "Provide a JSON response with 'score', 'label', 'confidence', 'reasoning'."

        # Run Generation
        output = self.pipeline(user_content, max_new_tokens=200)
        generated_text = output[0]["generated_text"]
        
        # Naive parsing for generic backend - real usage would likely require output parsers or specific prompting
        # This is a stub implementation as agreed in design doc
        
        # Try to find JSON-like structure or just return dummy for now if unstructured
        return EvaluationOutput(
            score=0.0, 
            label="unknown", 
            confidence=0.0, 
            reasoning=f"Generic HF output: {generated_text[:50]}..."
        )
