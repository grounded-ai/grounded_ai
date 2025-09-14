import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Optional

import torch
from peft import PeftConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

BASE_MODEL_ID = os.getenv("BASE_MODEL_ID", "microsoft/Phi-4-mini-instruct")


class EvalMode(str, Enum):
    TOXICITY = "TOXICITY"
    RAG_RELEVANCE = "RAG_RELEVANCE"
    HALLUCINATION = "HALLUCINATION"


@dataclass
class BaseEvaluator(ABC):
    base_model: str = BASE_MODEL_ID
    merged_model: Optional[PeftModel] = None
    use_peft: bool = True
    quantization: bool = False
    eval_mode: EvalMode = EvalMode.ANY
    grounded_ai_eval_id: Optional[str] = None

    def warmup(self):
        """Warmup the model by loading it and merging the adapter if necessary."""
        self.load_model()
        if self.use_peft:
            self.merge_adapter(self.grounded_ai_eval_id)

    def load_model(self):
        """Loads the base model with or without quantization."""
        compute_dtype = (
            torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        )

        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
        model_kwargs = {
            "dtype": compute_dtype,
        }
        if self.quantization:
            model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
        self.base_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_ID, **model_kwargs
        )
        self.tokenizer = tokenizer

    def merge_adapter(self, grounded_ai_eval_id: str):
        """Merges the PEFT adapter into the base model."""
        config = PeftConfig.from_pretrained(grounded_ai_eval_id)
        model_peft = PeftModel.from_pretrained(
            self.base_model, grounded_ai_eval_id, config=config
        )
        self.merged_model = model_peft.merge_and_unload()
        if not self.quantization:
            self.merged_model.to("cuda")

    @abstractmethod
    def base_prompt(self) -> str:
        pass

    @abstractmethod
    def format_input(self, instance: dict) -> str:
        pass

    @abstractmethod
    def run_model(self, instance: dict) -> str:
        pass

    @abstractmethod
    def evaluate(self, data: list) -> dict:
        pass
