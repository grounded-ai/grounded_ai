from peft import PeftModel, PeftConfig
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from transformers import pipeline
import torch


class HallucinationEvaluator:
    """
    HallucinationEvaluator is a class that evaluates whether a machine learning model has hallucinated or not.

    Example Usage:
    ```python
    base_model_id = "microsoft/Phi-3-mini-4k-instruct"
    groundedai_eval_id = "grounded-ai/phi3-hallucination-judge"
    evaluator = HallucinationEvaluator(base_model_id, groundedai_eval_id, quantization=True)
    evaluator.load_model(base_model_id, groundedai_eval_id)
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

    def __init__(
        self, base_model_id: str, groundedai_eval_id: str, quantization: bool = False
    ):
        self.base_model_id: str = base_model_id
        self.groundedai_eval_id: str = groundedai_eval_id
        self.model = None
        self.tokenizer = None
        self.quantization: bool = quantization

    def load_model(self, base_model_id: str, groundedai_eval_id: str):
        if torch.cuda.is_bf16_supported():
            compute_dtype = torch.bfloat16
            attn_implementation = "flash_attention_2"
            # If bfloat16 is not supported, 'compute_dtype' is set to 'torch.float16' and 'attn_implementation' is set to 'sdpa'.
        else:
            compute_dtype = torch.float16
            attn_implementation = "sdpa"

        config = PeftConfig.from_pretrained(groundedai_eval_id)
        tokenizer = AutoTokenizer.from_pretrained(base_model_id)
        if self.quantization:
            bnb_config = BitsAndBytesConfig(
                load_in_8bit=True,
            )
            base_model = AutoModelForCausalLM.from_pretrained(
                base_model_id,
                attn_implementation=attn_implementation,
                torch_dtype=compute_dtype,
                quantization_config=bnb_config,
            )
            model_peft = PeftModel.from_pretrained(
                base_model, groundedai_eval_id, config=config
            )
            merged_model = model_peft.merge_and_unload()
        else:
            base_model = AutoModelForCausalLM.from_pretrained(
                base_model_id,
                attn_implementation=attn_implementation,
                torch_dtype=compute_dtype,
            )
            model_peft = PeftModel.from_pretrained(
                base_model, groundedai_eval_id, config=config
            )

            merged_model = model_peft.merge_and_unload()
            merged_model.to("cuda")

        self.model = merged_model
        self.tokenizer = tokenizer

    def format_func(self, query: str, response: str, reference: str = None) -> str:
        # TODO implement promt hub and optionally pass in user defined prompt
        if reference is None:
            prompt = f"""Your job is to evaluate whether a machine learning model has hallucinated or not.
            A hallucination occurs when the response is coherent but factually incorrect or nonsensical
            outputs that are not grounded in the provided context.
            You are given the following information:
                ####INFO####
                [User Input]: {query}
                [Model Response]: {response}
                ####END INFO####
                Based on the information provided is the model output a hallucination? Respond with only "yes" or "no"
                """
        else:
            prompt = f"""Your job is to evaluate whether a machine learning model has hallucinated or not.
              A hallucination occurs when the response is coherent but factually incorrect or nonsensical
              outputs that are not grounded in the provided context.
              You are given the following information:
                  ####INFO####
                  [Knowledge]: {reference}
                  [User Input]: {query}
                  [Model Response]: {response}
                  ####END INFO####
                  Based on the information provided is the model output a hallucination? Respond with only "yes" or "no"
                """
        return prompt

    def run_model(self, query: str, response: str, reference: str = None) -> str:
        input = self.format_func(query, response, reference)
        messages = [{"role": "user", "content": input}]

        pipe = pipeline(
            "text-generation",
            model=self.model,
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
