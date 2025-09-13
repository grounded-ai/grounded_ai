from dataclasses import dataclass
from typing import List, Tuple, Optional

import torch
# from grounded_ai.validators.rag_data import RagData
from transformers import pipeline

from .base import BaseEvaluator
from .prompt_hub import RAG_RELEVANCE_EVAL_BASE, HALLUCINATION_EVAL_BASE, TOXICITY_EVAL_BASE, ANY_EVAL_BASE
from .utils import *

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

@dataclass
class GroundedAIEvaluator(BaseEvaluator):
    """
    The RAG (Retrieval-Augmented Generation) Evaluator class is used to evaluate the relevance
    of a given text with respect to a query.

    Example Usage:
    ```python
    evaluator = RagRelevanceEvaluator()
    evaluator.warmup()
    data = [
        ("What is the capital of France?", "Paris is the capital of France."),
        ("What is the largest planet in our solar system?", "Jupiter is the largest planet in our solar system.")
    ]
    response = evaluator.evaluate(data)
    # Output
    # {'relevant': 2, 'unrelated': 0, 'percentage_relevant': 100.0}
    ```
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

    def __post_init__(self):
        self.format_func_map = {
            "TOXICITY": self.format_toxicity,
            "RAG_RELEVANCE": self.format_rag,
            "HALLUCINATION": self.format_hallucination,
        }

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
        
        # TODO add check for reasoning beaing true, if so just change the system prompt
        messages = [{"role": "user", "content": input_prompt}]
        pipe = pipeline("text-generation", model=self.merged_model, tokenizer=self.tokenizer)
        output = pipe(messages, **self.generation_args)
        torch.cuda.empty_cache()
        return output[0]["generated_text"].strip().lower()

    def evaluate(self, data: List[Tuple[str, str]]) -> dict:
        eval_func = EVAL_MAP.get(self.eval_mode.value)
        return eval_func(self, data) if eval_func else {}
