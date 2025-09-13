from typing import Optional, List, Tuple, Union
import re
from jinja2 import Template
from ..validators.hallucination_data import HallucinationData
from ..validators.rag_data import RagData
from ..validators.toxic_data import ToxicityData

def extract_rating(response: str) -> Optional[int]:
    """
    Extracts a rating from the model's response.

    Args:
        response (str): The model's response containing the rating.

    Returns:
        Optional[int]: The extracted rating as an integer, or None if not found.
    """
    match = re.search(r"<rating>(\d+)</rating>", response)
    if match:
        return int(match.group(1))
    return "No rating found in response."

def extract_reasoning(response: str) -> Optional[str]:
    """
    Extracts reasoning from the model's response.

    Args:
        response (str): The model's response containing the reasoning.

    Returns:
        Optional[str]: The extracted reasoning as a string, or None if not found.
    """
    match = re.search(r"<reasoning>(.*?)</reasoning>", response, re.DOTALL)
    if match:
        return match.group(1).strip()
    return "No reasoning found in response."

def evaluate_rag(evaluator, data: List[Tuple[str, str]]) -> dict:
        try:
            evaluation_data = RagData(instances=data)
        except ValueError as e:
            print(f"Error validating input data: {e}")
            return {}

        relevant = 0
        unrelated = 0
        for instance in evaluation_data.instances:
            output = evaluator.run_model(instance.context, instance.query)
            if output == "relevant":
                relevant += 1
            elif output == "unrelated":
                unrelated += 1

        percentage_relevant = (
            (relevant / len(evaluation_data.instances)) * 100
            if evaluation_data.instances
            else 0
        )
        return {
            "relevant": relevant,
            "unrelated": unrelated,
            "percentage_relevant": percentage_relevant,
        }

def evaluate_toxicity(evaluator, data: List[str]) -> dict:
        try:
            evaluation_data = ToxicityData(instances=data)
        except ValueError as e:
            print(f"Error validating input data: {e}")
            return {}

        toxic = 0
        non_toxic = 0
        reasons = []
        for instance in evaluation_data.instances:
            output = evaluator.run_model(instance.text)
            if "non-toxic" in output:
                non_toxic += 1
            elif "toxic" in output:
                toxic += 1
            if evaluator.add_reason:
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

def evaluate_hallucination(
        evaluator, data: List[Union[Tuple[str, str], Tuple[str, str, Optional[str]]]]
    ) -> dict:
        try:
            evaluation_data = HallucinationData(instances=data)
            hallucinated: int = 0
            truthful: int = 0
            for instance in evaluation_data.instances:
                output = evaluator.run_model(instance.query, instance.response)
                if output == "yes":
                    hallucinated += 1
                elif output == "no":
                    truthful += 1

            percentage_hallucinated: float = (
                (hallucinated / len(evaluation_data.instances)) * 100
                if evaluation_data.instances
                else 0
            )
            return {
                "hallucinated": hallucinated,
                "truthful": truthful,
                "percentage_hallucinated": percentage_hallucinated,
            }
        except ValueError as e:
            print(f"Error validating input data: {e}")
            return {"hallucinated": 0, "truthful": 0, "percentage_hallucinated": 0.0}

def evaluate_hallucination_with_references(
    evaluator, data: List[Union[Tuple[str, str], Tuple[str, str, Optional[str]]]]
) -> dict:
    try:
        evaluation_data = HallucinationData(instances=data)
        hallucinated: int = 0
        truthful: int = 0
        for instance in evaluation_data.instances:
            output = evaluator.run_model(
                instance.query, instance.response, instance.reference
            )
            if output == "yes":
                hallucinated += 1
            elif output == "no":
                truthful += 1

        percentage_hallucinated: float = (
            (hallucinated / len(evaluation_data.instances)) * 100
            if evaluation_data.instances
            else 0
        )
        return {
            "hallucinated": hallucinated,
            "truthful": truthful,
            "percentage_hallucinated": percentage_hallucinated,
        }
    except ValueError as e:
        print(f"Error validating input data: {e}")
        return {"hallucinated": 0, "truthful": 0, "percentage_hallucinated": 0.0}
    
def format_toxicity(self, **kwargs):
        text = kwargs.get("text", "")
        template = Template(self.base_prompt)
        rendered_prompt = template.render(text=text)
        return rendered_prompt

def format_rag(self, **kwargs):
    text = kwargs.get("text", "")
    query = kwargs.get("query", "")
    template = Template(self.base_prompt)
    rendered_prompt = template.render(text=text, query=query)
    return rendered_prompt

def format_hallucination(self, **kwargs):
    query = kwargs.get("query", "")
    response = kwargs.get("response", "")
    reference = kwargs.get("reference", "")
    template = Template(self.base_prompt)
    rendered_prompt = template.render(
        reference=reference, query=query, response=response
    )
    return rendered_prompt