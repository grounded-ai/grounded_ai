from dataclasses import dataclass
from typing import List, Tuple, Optional

import torch
# from grounded_ai.validators.rag_data import RagData
from transformers import pipeline

from .base import BaseEvaluator
from .prompt_hub import RAG_RELEVANCE_EVAL_BASE, HALLUCINATION_EVAL_BASE, TOXICITY_EVAL_BASE, ANY_EVAL_BASE
from .utils import (
    evaluate_hallucination, 
    evaluate_rag, 
    evaluate_toxicity,
    format_hallucination,
    format_rag,
    format_toxicity,
)

PROMPT_MAP = {
    "TOXICITY": TOXICITY_EVAL_BASE,
    "RAG_RELEVANCE": RAG_RELEVANCE_EVAL_BASE,
    "HALLUCINATION": HALLUCINATION_EVAL_BASE,
    "ANY": ANY_EVAL_BASE,
}

EVAL_MAP = {
    "TOXICITY": evaluate_toxicity,
    "RAG_RELEVANCE": evaluate_rag,
    "HALLUCINATION": evaluate_hallucination,
    "ANY": None,
}

FORMAT_MAP = {
    "TOXICITY": format_toxicity,
    "RAG_RELEVANCE": format_rag,
    "HALLUCINATION": format_hallucination,
}

@dataclass
class GroundedAIEvaluator(BaseEvaluator):

    """
    GroundedAIEvaluator is a flexible evaluation class for various NLP tasks such as RAG relevance,
    hallucination detection, and toxicity assessment. It leverages prompt templates and evaluation functions
    to assess model outputs according to the selected evaluation mode.

    Example Usage:
        evaluator = GroundedAIEvaluator(eval_mode=EvalMode.RAG_RELEVANCE)
        evaluator.warmup()
        data = [
            ("What is the capital of France?", "Paris is the capital of France."),
            ("What is the largest planet in our solar system?", "Jupiter is the largest planet in our solar system.")
        ]
        response = evaluator.evaluate(data)
        # Output: {'relevant': 2, 'unrelated': 0, 'percentage_relevant': 100.0}
    """

    groundedai_eval_id = "grounded-ai/phi4-mini-judge"
    quantization: bool = False
    add_reasoning: bool = False
    custom_prompt: Optional[str] = None
    generation_args: Optional[dict] = None

    @property
    def base_prompt(self) -> str:
        if self.custom_prompt:
            return self.custom_prompt
        return PROMPT_MAP.get(self.eval_mode.value, ANY_EVAL_BASE)

    def format_input(self, **kwargs):
        format_func = self.format_func_map.get(self.eval_mode.value)
        if not format_func:
            raise ValueError(f"No formatter for eval_mode: {self.eval_mode.value}")
        return format_func(**kwargs)

    def run_model(self, **kwargs):
        if not self.generation_args:
            self.generation_args = {
                "max_new_tokens": 10,
                "temperature": 0.1,
                "do_sample": True,
                "pad_token_id": self.tokenizer.eos_token_id
            }
        
        input_prompt = self.format_input(**kwargs)
        
        # TODO add check for reasoning being true, if so just change the system prompt
        messages = [{"role": "user", "content": input_prompt}]
        if self._pipeline is None:
            self._pipeline = pipeline("text-generation", model=self.merged_model, tokenizer=self.tokenizer)
        output = self._pipeline(messages, **self.generation_args)
        torch.cuda.empty_cache()
        return output[0]["generated_text"].strip().lower()

    def evaluate(self, data: List[Tuple[str, str]]) -> dict:
        eval_func = EVAL_MAP.get(self.eval_mode.value)
        return eval_func(self, data) if eval_func else {}
