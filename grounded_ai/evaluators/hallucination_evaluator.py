from dataclasses import dataclass

import torch
from jinja2 import Template
from transformers import pipeline

from grounded_ai.schema.hallucination_data import EvaluationData

from .base import BaseEvaluator
from .prompt_hub import HALLUCINATION_EVAL_BASE


@dataclass
class HallucinationEvaluator(BaseEvaluator):
    """
    HallucinationEvaluator is a class that evaluates whether a machine learning model has hallucinated or not.

    Example Usage:
    ```python
    evaluator = HallucinationEvaluator(quantization=True)
    evaluator.warmup()
    data = [
        ['Based on the following <context>Walrus are the largest mammal</context> answer the question <query> What is the best PC?</query>', 'The best PC is the mac'],
        ['What is the color of an apple', "Apples are usually red or green"],
    ]
    response = evaluator.evaluate(data)
    # Output
    # {'hallucinated': 1, 'percentage_hallucinated': 50.0, 'truthful': 1}
    ```

    Example Usage with References:
    ```python
    references = [
        "The chicken crossed the road to get to the other side",
        "The apple mac has the best hardware",
        "The cat is hungry"
    ]
    queries = [
        "Why did the chicken cross the road?",
        "What computer has the best software?",
        "What pet does the context reference?"
    ]
    responses = [
        "To get to the other side", # Grounded answer
        "Apple mac",                # Deviated from the question (hardware vs software)
        "Cat"                       # Grounded answer
    ]
    data = list(zip(queries, responses, references))
    response = evaluator.evaluate(data)
    # Output
    # {'hallucinated': 1, 'truthful': 2, 'percentage_hallucinated': 33.33333333333333}
    ```
    """

    groundedai_eval_id = "grounded-ai/phi3-hallucination-judge"
    quantization: bool = False
    base_prompt: str = HALLUCINATION_EVAL_BASE

    def format_input(self, query: str, response: str, reference: str = None) -> str:
        template = Template(self.base_prompt)
        rendered_prompt = template.render(
            reference=reference, query=query, response=response
        )
        return rendered_prompt

    def run_model(self, query: str, response: str, reference: str = "") -> str:
        input = self.format_input(query, response, reference)
        messages = [{"role": "user", "content": input}]

        pipe = pipeline(
            "text-generation",
            model=self.merged_model,
            tokenizer=self.tokenizer,
        )

        generation_args = {
            "max_new_tokens": 2,
            "return_full_text": False,
            "temperature": 0.01,
            "do_sample": True,
        }

        output = pipe(messages, **generation_args)
        torch.cuda.empty_cache()
        return output[0]["generated_text"].strip().lower()

    def evaluate(self, data: list) -> dict:
        try:
            evaluation_data = EvaluationData(instances=data)
        except ValueError as e:
            print(f"Error validating input data: {e}")
            return {}

        hallucinated: int = 0
        truthful: int = 0
        for instance in evaluation_data.instances:
            output = self.run_model(
                instance.query, instance.response, instance.reference
            )
            if output == "yes":
                hallucinated += 1
            elif output == "no":
                truthful += 1

        percentage_hallucinated: float = (
            hallucinated / len(evaluation_data.instances)
        ) * 100
        return {
            "hallucinated": hallucinated,
            "truthful": truthful,
            "percentage_hallucinated": percentage_hallucinated,
        }
