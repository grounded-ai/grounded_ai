import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional
from enum import Enum

import torch
from peft import PeftConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

BASE_MODEL_ID = os.getenv("BASE_MODEL_ID", "microsoft/Phi-4-mini-instruct")

class EvalMode(str, Enum):
    ANY = "ANY"
    TOXICITY = "TOXICITY"
    RAG_RELEVANCE = "RAG_RELEVANCE"
    HALLUCINATION = "HALLUCINATION"

@dataclass
class BaseEvaluator(ABC):
    base_model: AutoModelForCausalLM = None
    tokenizer: AutoTokenizer = None
    merged_model: Optional[PeftModel] = None
    use_peft: Optional[bool] = True
    quantization: Optional[bool] = False
    eval_mode: EvalMode = EvalMode.ANY

    @property
    @abstractmethod
    def groundedai_eval_id(self) -> str:
        ...

    @property
    @abstractmethod
    def base_prompt(self) -> str:
        ...

    def warmup(self):
        """Warmup the model by loading it and merging the adapter if necessary."""
        self.load_model()
        if self.use_peft:
            self.merge_adapter(self.groundedai_eval_id)

    def load_model(self):
        """Loads the base model with or without quantization."""
        compute_dtype = (
            torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        )
        attn_implementation = (
            "flash_attention_2" if torch.cuda.is_bf16_supported() else "sdpa"
        )

        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
        model_kwargs = {
            "attn_implementation": attn_implementation,
            "torch_dtype": compute_dtype,
        }
        if self.quantization:
            model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
        base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_ID, **model_kwargs)

        self.base_model = base_model
        self.tokenizer = tokenizer

    def merge_adapter(self, groundedai_eval_id: str):
        """Merges the PEFT adapter into the base model."""
        config = PeftConfig.from_pretrained(groundedai_eval_id)
        model_peft = PeftModel.from_pretrained(
            self.base_model, groundedai_eval_id, config=config
        )
        self.merged_model = model_peft.merge_and_unload()
        if not self.quantization:
            self.merged_model.to("cuda")

    @abstractmethod
    def format_input(self, input_text: str) -> str:
        pass

    @abstractmethod
    def run_model(self, input_text: str) -> str:
        pass

    @abstractmethod
    def evaluate(self, data: list) -> dict:
        pass
