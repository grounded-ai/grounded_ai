from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
from transformers import pipeline

from .base import BaseEvaluator, EvalMode
from .prompt_hub import (
    HALLUCINATION_EVAL_BASE,
    RAG_RELEVANCE_EVAL_BASE,
    SYSTEM_PROMPT_BASE,
    TOXICITY_EVAL_BASE,
)
from .utils import (
    evaluate_hallucination,
    evaluate_rag,
    evaluate_toxicity,
    format_hallucination,
    format_rag,
    format_toxicity,
    format_system
)

PROMPT_MAP = {
    EvalMode.TOXICITY: TOXICITY_EVAL_BASE,
    EvalMode.RAG_RELEVANCE: RAG_RELEVANCE_EVAL_BASE,
    EvalMode.HALLUCINATION: HALLUCINATION_EVAL_BASE,
}

EVAL_MAP = {
    EvalMode.TOXICITY: evaluate_toxicity,
    EvalMode.RAG_RELEVANCE: evaluate_rag,
    EvalMode.HALLUCINATION: evaluate_hallucination,
}

FORMAT_MAP = {
    EvalMode.TOXICITY: format_toxicity,
    EvalMode.RAG_RELEVANCE: format_rag,
    EvalMode.HALLUCINATION: format_hallucination,
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

    grounded_ai_eval_id: str = "grounded-ai/phi4-mini-judge"
    quantization: bool = False
    generation_args: Optional[dict] = None
    pipeline: object = None
    custom_prompt: Optional[str] = None

    @property
    def system_prompt(self) -> str:
        return format_system(SYSTEM_PROMPT_BASE)

    @property
    def base_prompt(self) -> str:
        if self.custom_prompt:
            return self.custom_prompt
        return PROMPT_MAP.get(self.eval_mode, "")

    def format_input(self, instance: dict) -> str:
        format_func = FORMAT_MAP.get(self.eval_mode)
        if not format_func:
            raise ValueError(f"No formatter for eval_mode: {self.eval_mode}")
        return format_func(self, instance)

    def run_model(self, instance: dict) -> str:
        if not self.generation_args:
            self.generation_args = {
                "max_new_tokens": 10,
                "temperature": 0.1,
                "do_sample": True,
                "pad_token_id": self.tokenizer.eos_token_id,
            }

        input_prompt = self.format_input(instance)
        
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": input_prompt},
        ]
        if self.pipeline is None:
            self.pipeline = pipeline(
                "text-generation", model=self.merged_model, tokenizer=self.tokenizer
            )
        output = self.pipeline(messages, **self.generation_args)
        torch.cuda.empty_cache()
        return output[0]["generated_text"][2]["content"].strip().lower()

    def evaluate(self, data: List[Tuple[str, str]]) -> dict:
        eval_func = EVAL_MAP.get(self.eval_mode)
        return eval_func(self, data) if eval_func else {}
