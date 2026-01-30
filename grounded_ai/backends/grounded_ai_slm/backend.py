import logging
import os
import re
from enum import Enum
from typing import Optional, Type, Union

import torch
from jinja2 import Template
from peft import PeftConfig, PeftModel
from pydantic import BaseModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    pipeline,
)

from ...base import BaseEvaluator
from ...schemas import EvaluationInput, EvaluationOutput
from .prompts import (
    HALLUCINATION_EVAL_BASE,
    RAG_RELEVANCE_EVAL_BASE,
    SYSTEM_PROMPT_BASE,
    TOXICITY_EVAL_BASE,
)

# Constants
BASE_MODEL_ID = os.getenv("BASE_MODEL_ID", "microsoft/Phi-4-mini-instruct")


class EvalMode(str, Enum):
    TOXICITY = "TOXICITY"
    RAG_RELEVANCE = "RAG_RELEVANCE"
    HALLUCINATION = "HALLUCINATION"


class GroundedAISLMBackend(BaseEvaluator):
    """
    Specialized backend for Grounded AI's fine-tuned SLMs.
    Enforces strict internal validation using custom Pydantic models.
    """

    def __init__(
        self,
        model_id: str,
        device: str = None,
        quantization: bool = False,
        eval_mode: Optional[Union[EvalMode, str]] = None,
        input_schema: Type[BaseModel] = EvaluationInput,
        output_schema: Type[BaseModel] = EvaluationOutput,
        system_prompt: str = None,
    ):
        super().__init__(
            input_schema=input_schema,
            output_schema=output_schema,
            system_prompt=system_prompt,
        )
        self.model_id = model_id
        self.quantization = quantization
        if torch is not None:
            self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = "cpu"

        self.task = None
        if eval_mode:
            self.set_eval_mode(eval_mode)

        self.tokenizer = None
        self.base_model = None
        self.merged_model = None
        self.pipeline = None
        self._load_model()

    def set_eval_mode(self, mode: Union[EvalMode, str]):
        """Sets the evaluation mode/task for the backend."""
        # Robust enum conversion: handle string (case-insensitive) or Enum input
        if isinstance(mode, str):
            try:
                self.task = EvalMode(mode.upper())
            except ValueError:
                raise ValueError(
                    f"Invalid eval_mode '{mode}'. Options: {[e.value for e in EvalMode]}"
                )
        else:
            self.task = EvalMode(mode)
        self.prompt_template = self._get_template(self.task)

    def _get_template(self, task: EvalMode) -> str:
        prompt_map = {
            EvalMode.HALLUCINATION: HALLUCINATION_EVAL_BASE,
            EvalMode.RAG_RELEVANCE: RAG_RELEVANCE_EVAL_BASE,
            EvalMode.TOXICITY: TOXICITY_EVAL_BASE,
        }
        return prompt_map.get(task, "")

    def _load_model(self):
        """Loads and merges the model."""
        logging.info(f"Loading Grounded AI SLM: {self.model_id} on {self.device}")
        compute_dtype = (
            torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        )

        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
        model_kwargs = {"dtype": compute_dtype}

        if self.quantization:
            model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)

        base_model_obj = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_ID, **model_kwargs
        )

        try:
            config = PeftConfig.from_pretrained(self.model_id)
            model_peft = PeftModel.from_pretrained(
                base_model_obj, self.model_id, config=config
            )
            self.merged_model = model_peft.merge_and_unload()
        except Exception as e:
            logging.warning(
                f"Failed to load PEFT adapter for {self.model_id}: {e}. Using base model."
            )
            self.merged_model = base_model_obj

        if not self.quantization and self.device == "cuda":
            self.merged_model.to("cuda")

        self.tokenizer = tokenizer

        self.pipeline = pipeline(
            "text-generation", model=self.merged_model, tokenizer=self.tokenizer
        )

    def _call_backend(
        self, input_data: BaseModel, output_schema: Type[BaseModel], **kwargs
    ) -> BaseModel:
        if not self.task:
            raise ValueError(
                "Eval mode not set. Please provide 'eval_mode' in init or call set_eval_mode()."
            )

        # 3. Format Prompt
        prompt = self._format_prompt(input_data)

        # 4. Generate
        messages = [
            {"role": "system", "content": self.system_prompt or SYSTEM_PROMPT_BASE},
            {"role": "user", "content": prompt},
        ]

        # Defaults
        generation_args = {
            "max_new_tokens": 100,
            "temperature": 0.1,
            "do_sample": True,
            "pad_token_id": self.tokenizer.eos_token_id,
        }

        # Explicitly allow-list generation arguments to prevent pollution from wrapper kwargs
        allowed_gen_args = {
            "temperature",
            "max_new_tokens",
            "do_sample",
            "top_p",
            "top_k",
            "repetition_penalty",
        }
        for k, v in kwargs.items():
            if k in allowed_gen_args:
                generation_args[k] = v

        outputs = self.pipeline(messages, **generation_args)
        raw_output = outputs[0]["generated_text"][-1]["content"].strip()

        # 5. Parse Output
        return self._parse_output(raw_output, output_schema)

    def _format_prompt(self, input_model: EvaluationInput) -> str:
        template = Template(self.prompt_template)

        if self.task == EvalMode.TOXICITY:
            # Toxicity Base expects 'text'
            return template.render(text=input_model.response)

        elif self.task == EvalMode.RAG_RELEVANCE:
            # RAG Base expects 'text' (the doc) and 'query'
            # Fallback: if user passed 'context' but no 'response', treat context as the doc
            doc_text = input_model.response or input_model.context
            return template.render(text=doc_text, query=input_model.query)

        elif self.task == EvalMode.HALLUCINATION:
            # Hallucination expects 'response', 'query', 'reference'
            return template.render(
                response=input_model.response,
                query=input_model.query,
                reference=input_model.context or "",
            )

        return input_model.response

    def _parse_output(
        self, raw_response: str, output_schema: Type[BaseModel] = None
    ) -> BaseModel:
        rating_match = re.search(r"<rating>(.*?)</rating>", raw_response)
        reasoning_match = re.search(
            r"<reasoning>(.*?)</reasoning>", raw_response, re.DOTALL
        )

        rating_str = (
            rating_match.group(1).strip().lower() if rating_match else "unknown"
        )
        reasoning = reasoning_match.group(1).strip() if reasoning_match else None

        score = 0.0
        label = rating_str

        if self.task == EvalMode.TOXICITY:
            if "non-toxic" in rating_str:
                score = 0.0
                label = "non-toxic"
            else:
                score = 1.0
                label = "toxic"
        elif self.task == EvalMode.RAG_RELEVANCE:
            if "relevant" in rating_str:
                score = 1.0
                label = "relevant"
            else:
                score = 0.0
                label = "unrelated"
        elif self.task == EvalMode.HALLUCINATION:
            if "accurate" in rating_str or "no" in rating_str:
                score = 0.0
                label = "faithful"
            else:
                score = 1.0
                label = "hallucination"

        return output_schema(
            score=score, label=label, confidence=1.0, reasoning=reasoning
        )
