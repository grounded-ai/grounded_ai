from typing import Any, Dict, Type, Union, Optional
import json
import re
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

    def __init__(self, model_id: str, device: str = None, task: str = None, **kwargs):
        if pipeline is None:
            raise ImportError("transformers package is not installed. Please install it via `pip install transformers`.")
        
        self.model_id = model_id.replace("hf/", "").replace("huggingface/", "")
        
        # Simple task heuristic if not provided
        if not task:
            task = "text-generation"
            
        self.task = task
        self.pipeline = pipeline(task, model=self.model_id, device=device, **kwargs)

    @property
    def input_schema(self) -> Type[BaseModel]:
        return EvaluationInput

    @property
    def output_schema(self) -> Type[BaseModel]:
        return EvaluationOutput

    def evaluate(self, input_data: Union[EvaluationInput, Dict[str, Any]]) -> Union[EvaluationOutput, EvaluationError]:
        if isinstance(input_data, dict):
            input_data = EvaluationInput(**input_data)

        if self.task == "text-classification":
            return self._evaluate_classification(input_data)
        else:
            return self._evaluate_generation(input_data)

    def _evaluate_classification(self, input_data: EvaluationInput) -> EvaluationOutput:
        """Handle BERT-style classification models."""
        eval_text = input_data.text
        if input_data.context:
             eval_text = f"Context: {input_data.context}\nText: {input_data.text}"

        results = self.pipeline(eval_text, top_k=1) 
        top_result = results[0] if isinstance(results, list) else results

        label = top_result.get("label", "unknown")
        score = top_result.get("score", 0.0)
        
        return EvaluationOutput(
            score=score,
            label=label,
            confidence=score,
            reasoning=f"Classified as {label} with score {score:.4f}"
        )

    def _evaluate_generation(self, input_data: EvaluationInput) -> EvaluationOutput:
        """Handle Generative LLMs with simple text generation."""
        
        # Simple Prompt
        prompt = f"""Task: Evaluate the following content.
Query: {input_data.query or "N/A"}
Context: {input_data.context or "N/A"}
Reference: {input_data.reference or "N/A"}

Content to Evaluate:
{input_data.text}

Evaluation:"""

        # Generate
        output = self.pipeline(prompt, max_new_tokens=100, return_full_text=False)
        generated_text = output[0]["generated_text"].strip()
        
        return EvaluationOutput(
            score=0.0,
            label="generated_text",
            confidence=0.0,
            reasoning=generated_text
        )
