from dataclasses import dataclass

import torch
from jinja2 import Template
from .prompt_hub import RAG_RELEVANCE_EVAL_BASE
from transformers import pipeline

from .base import BaseEvaluator


@dataclass
class RagRelevanceEvaluator(BaseEvaluator):
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

    groundedai_eval_id = "grounded-ai/phi3-rag-relevance-judge"
    quantization: bool = False
    base_prompt = RAG_RELEVANCE_EVAL_BASE

    def format_input(self, text, query):
        template = Template(self.base_prompt)
        rendered_prompt = template.render(text=text, query=query)
        return rendered_prompt

    def run_model(self, text, query):
        input_prompt = self.format_input(text, query)
        messages = [{"role": "user", "content": input_prompt}]

        pipe = pipeline(
            "text-generation", model=self.merged_model, tokenizer=self.tokenizer
        )

        generation_args = {
            "max_new_tokens": 5,
            "return_full_text": False,
            "temperature": 0.01,
            "do_sample": True,
        }

        output = pipe(messages, **generation_args)
        torch.cuda.empty_cache()
        return output[0]["generated_text"].strip().lower()

    def evaluate(self, data):
        relevant = 0
        unrelated = 0
        for query, text in data:
            output = self.run_model(text, query)
            if output == "relevant":
                relevant += 1
            elif output == "unrelated":
                unrelated += 1
        percentage_relevant = (relevant / len(data)) * 100 if data else 0
        return {
            "relevant": relevant,
            "unrelated": unrelated,
            "percentage_relevant": percentage_relevant,
        }
