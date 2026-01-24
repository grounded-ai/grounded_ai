import logging
import os
import re
from functools import wraps
from typing import List, Optional, Tuple, Union

from jinja2 import Template
from tqdm import tqdm

from .validators.hallucination_data import HallucinationData
from .validators.query_response_data import HallucinationData as QueryResponseData # Optional if needed
from .validators.rag_data import RagData
from .validators.toxic_data import ToxicityData
from .validators.output_data import OutputInstance


# TODO document GROUNDED_AI_DEBUG=false/true can be set to enable/disable debug logging
def debug_eval_logging(func):
    """
    Decorator to add optional debug logging to evaluation functions.

    Logs input data and final results when GROUNDED_AI_DEBUG environment variable is set to 'true'.
    Individual instance logging should be added directly in the evaluation functions.
    """

    @wraps(func)
    def wrapper(evaluator, data, *args, **kwargs):
        debug_enabled = os.getenv("GROUNDED_AI_DEBUG", "false").lower() == "true"

        if not debug_enabled:
            return func(evaluator, data, *args, **kwargs)

        logger = logging.getLogger(f"grounded_ai.{func.__name__}")
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.DEBUG)

        logger.info(f"Starting {func.__name__} with {len(data)} instances")
        logger.debug(
            f"Input data sample: {str(data[:2]) if len(data) > 2 else str(data)}"
        )

        result = func(evaluator, data, *args, **kwargs)
        logger.info(f"{func.__name__} completed: {result}")
        return result

    return wrapper


def log_instance(
    instance_num: int, instance_data: dict, output: str, classification: str
):
    """
    Helper function to log individual instance processing when debug mode is enabled.

    Args:
        instance_num: The instance number being processed
        instance_data: The input data for this instance
        output: Raw model output
        classification: Extracted classification
    """
    debug_enabled = os.getenv("GROUNDED_AI_DEBUG", "false").lower() == "true"
    if not debug_enabled:
        return

    logger = logging.getLogger("grounded_ai.instance_logging")
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)

    # Log input based on data type
    if "context" in instance_data and "query" in instance_data:  # RAG
        logger.debug(
            f"Instance {instance_num}: Context='{str(instance_data.get('context', ''))[:100]}...', Query='{instance_data.get('query', '')}'"
        )
    elif "text" in instance_data:  # Toxicity
        logger.debug(
            f"Instance {instance_num}: Text='{str(instance_data.get('text', ''))[:100]}...'"
        )
    elif "query" in instance_data and "response" in instance_data:  # Hallucination
        logger.debug(
            f"Instance {instance_num}: Query='{str(instance_data.get('query', ''))[:50]}...', Response='{str(instance_data.get('response', ''))[:50]}...'"
        )

    logger.debug(
        f"Instance {instance_num} - Raw output: '{output}', Classification: '{classification}'"
    )


@debug_eval_logging
def evaluate_rag(evaluator, data: List[Tuple[str, str]]) -> dict:
    try:
        evaluation_data = RagData(instances=data)
    except ValueError as e:
        print(f"Error validating input data: {e}")
        return {}

    relevant = 0
    unrelated = 0
    for i, instance in enumerate(
        tqdm(evaluation_data.instances, desc="Evaluating RAG Relevance")
    ):
        instance = instance.model_dump()
        output = evaluator.run_model(instance)
        output_instance = OutputInstance(raw_response=output)
        classification = output_instance.rating

        log_instance(i + 1, instance, output, classification)

        if classification == "relevant":
            relevant += 1
        elif classification == "unrelated":
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


@debug_eval_logging
def evaluate_toxicity(evaluator, data: List[str]) -> dict:
    try:
        evaluation_data = ToxicityData(instances=data)
    except ValueError as e:
        print(f"Error validating input data: {e}")
        return {}

    toxic = 0
    non_toxic = 0
    reasons = []

    for i, instance in enumerate(
        tqdm(evaluation_data.instances, desc="Evaluating Toxicity")
    ):
        instance = instance.model_dump()
        output = evaluator.run_model(instance)
        output_instance = OutputInstance(raw_response=output)
        classification = output_instance.rating

        log_instance(i + 1, instance, output, classification)

        if "non-toxic" in classification:
            non_toxic += 1
        elif "toxic" in classification:
            toxic += 1
        # if evaluator.add_reasoning:
        #     reasons.append((output_instance.reasoning))

    percentage_toxic = (
        (toxic / len(evaluation_data.instances)) * 100
        if evaluation_data.instances
        else 0
    )
    return {
        "toxic": toxic,
        "non-toxic": non_toxic,
        "percentage_toxic": percentage_toxic,
        # "reasons": reasons ,
    }


@debug_eval_logging
def evaluate_hallucination(
    evaluator, data: List[Union[Tuple[str, str], Tuple[str, str, Optional[str]]]]
) -> dict:
    try:
        evaluation_data = HallucinationData(instances=data)
        hallucinated: int = 0
        truthful: int = 0
        for i, instance in enumerate(
            tqdm(evaluation_data.instances, desc="Evaluating Hallucination")
        ):
            instance = instance.model_dump()
            output = evaluator.run_model(instance)
            output_instance = OutputInstance(raw_response=output)
            classification = output_instance.rating

            # Log individual instance if debug mode is enabled
            log_instance(i + 1, instance, output, classification)

            if classification == "yes":
                hallucinated += 1
            elif classification == "no":
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


@debug_eval_logging
def evaluate_hallucination_with_references(
    evaluator, data: List[Union[Tuple[str, str], Tuple[str, str, Optional[str]]]]
) -> dict:
    try:
        evaluation_data = HallucinationData(instances=data)
        hallucinated: int = 0
        truthful: int = 0
        for i, instance in enumerate(
            tqdm(evaluation_data.instances, desc="Evaluating Hallucination")
        ):
            instance = instance.model_dump()
            output = evaluator.run_model(instance)
            output_instance = OutputInstance(raw_response=output)
            classification = output_instance.rating

            log_instance(i + 1, instance, output, classification)

            if classification == "yes":
                hallucinated += 1
            elif classification == "no":
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


def format_toxicity(evaluator, instance):
    text = instance.get("text", "")
    template = Template(evaluator.base_prompt)
    rendered_prompt = template.render(text=text)
    return rendered_prompt


def format_rag(evaluator, instance):
    context = instance.get("context", "")
    query = instance.get("query", "")
    template = Template(evaluator.base_prompt)
    rendered_prompt = template.render(text=context, query=query)
    return rendered_prompt


def format_hallucination(evaluator, instance):
    query = instance.get("query", "")
    response = instance.get("response", "")
    reference = instance.get("reference", "")
    template = Template(evaluator.base_prompt)
    rendered_prompt = template.render(
        reference=reference, query=query, response=response
    )
    return rendered_prompt

def format_system(system):
    # template = Template(system)
    # rendered_system = template.render()
    return system