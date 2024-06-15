from peft import PeftModel, PeftConfig
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from transformers import pipeline
import torch


class RagEvaluator:
    """
    The RAG (Retrieval-Augmented Generation) Evaluator class is used to evaluate the relevance
    of a given text with respect to a query.

    Example Usage:
    ```python
    base_model_id = "microsoft/Phi-3-mini-4k-instruct"
    groundedai_eval_id = "grounded-ai/phi3-rag-relevance-judge"
    evaluator = RagEvaluator(base_model_id, groundedai_eval_id, quantization=True)
    evaluator.load_model(base_model_id, groundedai_eval_id)
    data = [
        ("What is the capital of France?", "Paris is the capital of France."),
        ("What is the largest planet in our solar system?", "Jupiter is the largest planet in our solar system.")
    ]
    response = evaluator.evaluate(data)
    # Output
    # {'relevant': 2, 'unrelated': 0, 'percentage_relevant': 100.0}
    ```
    """

    def __init__(
        self, base_model_id: str, groundedai_eval_id: str, quantization: bool = False
    ):
        self.base_model_id = base_model_id
        self.groundedai_eval_id = groundedai_eval_id
        self.model = None
        self.tokenizer = None
        self.quantization = quantization

    def load_model(self, base_model_id: str, groundedai_eval_id: str):
        if torch.cuda.is_bf16_supported():
            compute_dtype = torch.bfloat16
            attn_implementation = "flash_attention_2"
        else:
            compute_dtype = torch.float16
            attn_implementation = "sdpa"

        config = PeftConfig.from_pretrained(groundedai_eval_id)
        tokenizer = AutoTokenizer.from_pretrained(base_model_id)
        if self.quantization:
            bnb_config = BitsAndBytesConfig(load_in_8bit=True)
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

    def format_input(self, text, query):
        input_prompt = f"""
        You are comparing a reference text to a question and trying to determine if the reference text
        contains information relevant to answering the question. Here is the data:
        [BEGIN DATA]
        ************
        [Question]: {query}
        ************
        [Reference text]: {text}
        ************
        [END DATA]
        Compare the Question above to the Reference text. You must determine whether the Reference text
        contains information that can answer the Question. Please focus on whether the very specific
        question can be answered by the information in the Reference text.
        Your response must be single word, either "relevant" or "unrelated",
        and should not contain any text or characters aside from that word.
        "unrelated" means that the reference text does not contain an answer to the Question.
        "relevant" means the reference text contains an answer to the Question."""
        return input_prompt

    def run_model(self, text, query):
        input_prompt = self.format_input(text, query)
        messages = [{"role": "user", "content": input_prompt}]

        pipe = pipeline("text-generation", model=self.model, tokenizer=self.tokenizer)

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
