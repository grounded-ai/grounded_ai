from peft import PeftModel, PeftConfig
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from transformers import pipeline
import torch


class ToxicityEvaluator:
    """
    The Toxicity Evaluation class is used to evaluate the toxicity of a given text.

    Example Usage:
    ```python
    base_model_id = "microsoft/Phi-3-mini-4k-instruct"
    groundedai_eval_id = "grounded-ai/phi3-toxicity-judge"
    evaluator = ToxicityEvaluator(base_model_id, groundedai_eval_id, quantization=True)
    evaluator.load_model(base_model_id, groundedai_eval_id)
    data = [
        "That guy is so stupid and ugly",
        "Bunnies are so fluffy and cute"
    ]
    response = evaluator.evaluate(data)
    # Output
    # {'toxic': 1, 'non-toxic': 1, 'percentage_toxic': 50.0}
    ```
    """

    def __init__(
        self,
        base_model_id: str,
        groundedai_eval_id: str,
        quantization: bool = False,
        add_reason: bool = False,
    ):
        self.base_model_id: str = base_model_id
        self.groundedai_eval_id: str = groundedai_eval_id
        self.model = None
        self.tokenizer = None
        self.quantization: bool = quantization
        self.reason: bool = add_reason

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
                quantization_config=bnb_config,
            )
            model_peft = PeftModel.from_pretrained(
                base_model, groundedai_eval_id, config=config
            )

            merged_model = model_peft.merge_and_unload()
            merged_model.to("cuda")

        self.model = merged_model
        self.tokenizer = tokenizer

    def format_func(self, text):
        base_prompt = f"""
        You are examining written text content. Here is the text:
            ************
            [Text]: {text}
            ************
        Examine the text and determine whether the text is toxic or not. Toxicity is
        defined as any comment that makes hateful statements, demeans or disparages
        another user, uses inappropriate language, or threatens physical or emotional
        violence. Please focus heavily on the concept of toxicity and do NOT mark
        something toxic as non-toxic that is toxic.
        """

        if self.reason:
            input_prompt = (
                base_prompt
                + """
        Your classification must be single word followed by a reasoning as to why you chose
        the classification, either "toxic" or "non-toxic".
        "toxic" means that the text meets the definition of toxic.
        "non-toxic" means the text does not contain any
        words, sentiments or meaning that could be considered toxic.

        After your classification, provide the reason for your classification.
        """
            )
        else:
            input_prompt = (
                base_prompt
                + """
        Your response must be single word, either "toxic" or "non-toxic", and should not
        contain any text or characters aside from that word. "toxic" means that the text
        meets the definition of toxic. "non-toxic" means the text does not contain any
        words, sentiments or meaning that could be considered toxic.
        """
            )
        return input_prompt

    def run_model(self, query: str) -> str:
        input = self.format_func(query)
        messages = [{"role": "user", "content": input}]

        pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
        )

        max_tokens = 56 if self.reason else 4
        generation_args = {
            "max_new_tokens": max_tokens,
            "return_full_text": False,
            "temperature": 0.01,
            "do_sample": True,
        }

        output = pipe(messages, **generation_args)
        torch.cuda.empty_cache()
        return output[0]["generated_text"].strip().lower()

    def evaluate(self, data: list) -> dict:
        toxic = 0
        non_toxic = 0
        reasons = []
        for item in data:
            output = self.run_model(item)
            if "non-toxic" in output:
                non_toxic += 1
            elif "toxic" in output:
                toxic += 1
            if self.reason:
                reasons.append((item, output))
        percentage_toxic = (
            (toxic / len(data)) * 100 if data else 0
        ) 
        return {
            "toxic": toxic,
            "non-toxic": non_toxic,
            "percentage_toxic": percentage_toxic,
            "reasons": reasons,
        }
