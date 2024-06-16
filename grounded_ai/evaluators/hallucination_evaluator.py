from transformers import pipeline
import torch
from dataclasses import dataclass
from .base import BaseEvaluator

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

    def format_input(self, query: str, response: str, reference: str = None) -> str:
        knowledge_line = f"[Knowledge]: {reference}\n" if reference is not None else ""
        prompt = f"""Your job is to evaluate whether a machine learning model has hallucinated or not.
    A hallucination occurs when the response is coherent but factually incorrect or nonsensical
    outputs that are not grounded in the provided context.
    You are given the following information:
        ####INFO####
        {knowledge_line}[User Input]: {query}
        [Model Response]: {response}
        ####END INFO####
    Based on the information provided is the model output a hallucination? Respond with only "yes" or "no"
    """
        return prompt

    def run_model(self, query: str, response: str, reference: str = None) -> str:
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
        hallucinated: int = 0
        truthful: int = 0
        for item in data:
            if len(item) == 2:
                query, response = item
                output = self.run_model(query, response)
            elif len(item) == 3:
                query, response, reference = item
                output = self.run_model(query, response, reference)
            if output == "yes":
                hallucinated += 1
            elif output == "no":
                truthful += 1
        percentage_hallucinated: float = (hallucinated / len(data)) * 100
        return {
            "hallucinated": hallucinated,
            "truthful": truthful,
            "percentage_hallucinated": percentage_hallucinated,
        }
