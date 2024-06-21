from dataclasses import dataclass
from typing import List

import torch
from jinja2 import Template
from transformers import pipeline

from grounded_ai.schema.base_data import EvaluationData

from .base import BaseEvaluator
from .prompt_hub import TOXICITY_EVAL_BASE


@dataclass
class ToxicityEvaluator(BaseEvaluator):
    """
    The Toxicity Evaluation class is used to evaluate the toxicity of a given text.

    Example Usage:
    ```python
    toxicity_evaluator = ToxicityEvaluator()
    toxicity_evaluator.warmup()
    data = [
        "That guy is so stupid and ugly",
        "Bunnies are the cutest animals in the world"
    ]
    response = toxicity_evaluator.evaluate(data)
    # Output
    # {'toxic': 1, 'non-toxic': 1, 'percentage_toxic': 50.0}
    ```
    """

    add_reason: bool = False
    groundedai_eval_id = "grounded-ai/phi3-toxicity-judge"
    quantization: bool = False
    base_prompt: str = TOXICITY_EVAL_BASE

    def format_input(self, text):
        """This function formats the input text for the model"""
        template = Template(self.base_prompt)
        rendered_prompt = template.render(text=text, add_reason=self.add_reason)
        return rendered_prompt

    def run_model(self, query: str) -> str:
        """This function runs the model on the given query to make its toxicity prediction"""
        input = self.format_input(query)
        messages = [{"role": "user", "content": input}]

        pipe = pipeline(
            "text-generation",
            model=self.merged_model,
            tokenizer=self.tokenizer,
        )

        max_tokens = 56 if self.add_reason else 4
        generation_args = {
            "max_new_tokens": max_tokens,
            "return_full_text": False,
            "temperature": 0.01,
            "do_sample": True,
        }

        output = pipe(messages, **generation_args)
        torch.cuda.empty_cache()
        return output[0]["generated_text"].strip().lower()

    def evaluate(self, data: List[str]) -> dict:
        try:
            evaluation_data = EvaluationData(instances=data)
        except ValueError as e:
            print(f"Error validating input data: {e}")
            return {}

        toxic = 0
        non_toxic = 0
        reasons = []
        for instance in evaluation_data.instances:
            output = self.run_model(instance)
            if "non-toxic" in output:
                non_toxic += 1
            elif "toxic" in output:
                toxic += 1
            if self.add_reason:
                reasons.append((instance, output))

        percentage_toxic = (
            (toxic / len(evaluation_data.instances)) * 100
            if evaluation_data.instances
            else 0
        )
        return {
            "toxic": toxic,
            "non-toxic": non_toxic,
            "percentage_toxic": percentage_toxic,
            "reasons": reasons,
        }
