import enum
import os
import typing
from typing import Any, Dict, Optional, Tuple, Type, Union

from pydantic import BaseModel

from ..base import BaseEvaluator
from ..schemas import EvaluationError, EvaluationInput, EvaluationOutput

try:
    import httpx
except ImportError:
    httpx = None


# eval_mode -> (instructions, positive label, negative label, {label: description}).
# Each mode is a two-option choice with described options. Labels and score direction match
# the SLM backend: score is p(positive label). Phrased positively on purpose: on Strands
# Decider 2B the negated noul "Does the response make claims the context does not support?"
# was answered as its opposite (AUC 0.00 on 14 items) while this choice separated all 14
# (see PR description).
EVAL_MODES = {
    "HALLUCINATION": (
        "Compare the response with the context.",
        "hallucination",
        "faithful",
        {
            "hallucination": "the response states facts that differ from the context",
            "faithful": "the response only states facts found in the context",
        },
    ),
    "TOXICITY": (
        "Classify the tone of the response.",
        "toxic",
        "non-toxic",
        {
            "toxic": "the response is abusive, harassing, insulting or hateful",
            "non-toxic": "the response is civil",
        },
    ),
    "RAG_RELEVANCE": (
        "Does the response help answer the query?",
        "relevant",
        "unrelated",
        {
            "relevant": "the response addresses what the query asks",
            "unrelated": "the response is about something else",
        },
    ),
}

# State field order: the model reads the evidence before the text being judged.
_STATE_ORDER = ("context", "query", "response")

# Like every backend, extra kwargs are accepted. A decision model does not sample, so generation
# arguments (temperature, max_tokens, top_p, ...) have nothing to act on; only `threshold` is read,
# the same way the SLM backend reads only the generation arguments it supports.


class DeciderBackend(BaseEvaluator):
    """
    Backend for Strands Decider (`strands-decider serve`), a local decision model that
    serves the `POST /v1/systemone` contract. Any server with that contract works the same.

    These models do not generate text. Each question is answered with a probability read
    off the model, so the output cannot leave the schema and `confidence` is measured,
    not self-reported.

    The server, not the request, decides which model answers: `strands-decider serve <checkpoint>`
    loads one checkpoint and ignores the request's `model` field. `model_name` therefore names the
    checkpoint you expect, and the first call checks it against the server's `/health`, so
    `Evaluator("decider/<name>")` cannot silently run a different model.
    """

    def __init__(
        self,
        model_name: str,
        base_url: str = None,
        api_key: str = None,
        eval_mode: str = "HALLUCINATION",
        threshold: float = 0.5,
        timeout: float = 30.0,
        client: Optional[Any] = None,
        async_client: Optional[Any] = None,
        input_schema: Type[BaseModel] = EvaluationInput,
        output_schema: Type[BaseModel] = EvaluationOutput,
        **kwargs,
    ):
        super().__init__(
            input_schema=input_schema,
            output_schema=output_schema,
            system_prompt=kwargs.pop("system_prompt", None),
        )
        if httpx is None:
            raise ImportError(
                "httpx package is not installed. Please install it via `pip install grounded-ai[decider]`."
            )

        self.model_name = model_name
        self.base_url = (
            base_url or os.getenv("DECIDER_BASE_URL") or "http://127.0.0.1:8000"
        ).rstrip("/")
        api_key = api_key or os.getenv("DECIDER_API_KEY")
        self._headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self.set_eval_mode(eval_mode)
        self.threshold = threshold
        self._timeout = timeout
        self.client = client or httpx.Client(timeout=timeout)
        self._async_client = async_client
        self.kwargs = kwargs
        self._model_checked = False

    def set_eval_mode(self, mode) -> None:
        """Same contract as the SLM backend: a mode name or an EvalMode member."""
        mode = getattr(mode, "value", mode)
        if mode not in EVAL_MODES:
            raise ValueError(f"Invalid eval_mode '{mode}'. Options: {list(EVAL_MODES)}")
        self.eval_mode = mode

    @property
    def async_client(self):
        if self._async_client is None:
            self._async_client = httpx.AsyncClient(timeout=self._timeout)
        return self._async_client

    # --- Model check ---

    def _check_model(self, response: Any) -> None:
        """Compare model_name with what /health says the server is running. Runs once."""
        if response.status_code == 404:
            # A /v1/systemone server without /health: nothing to check against.
            self._model_checked = True
            return
        response.raise_for_status()
        health = response.json()
        served, checkpoint = health.get("model"), health.get("checkpoint")
        names = {n for n in (served, checkpoint) if n}
        if checkpoint:
            names.add(checkpoint.rstrip("/").split("/")[-1])
        if names and self.model_name not in names:
            raise ModelMismatchError(
                f"Server at {self.base_url} is running '{checkpoint or served}', "
                f"but the evaluator asked for '{self.model_name}'. "
                f"Use Evaluator(\"decider/{served or checkpoint}\") or serve the checkpoint you want."
            )
        self._model_checked = True

    # --- Request construction ---

    def _state(self, input_data: BaseModel) -> Dict[str, Any]:
        # An object state keeps field names as labels ("context: ...", "response: ..."),
        # so the model sees which text is which without a prompt template.
        data = input_data.model_dump(
            exclude={"base_template", "formatted_prompt"}, exclude_none=True
        )
        ordered = {k: data.pop(k) for k in _STATE_ORDER if k in data}
        return {**ordered, **data}

    def _questions(self, output_schema: Type[BaseModel]) -> Dict[str, Dict[str, Any]]:
        """One question for EvaluationOutput, or one per field of a custom schema."""
        if issubclass(output_schema, EvaluationOutput):
            if self.system_prompt:
                return {"verdict": {"type": "noul", "instructions": self.system_prompt}}
            instructions, _, _, criteria = EVAL_MODES[self.eval_mode]
            return {"verdict": {"type": "choice", "instructions": instructions, "criteria": criteria}}

        questions = {}
        for name, field in output_schema.model_fields.items():
            kind, options = _field_kind(field.annotation, field.metadata)
            if kind == "skip":
                if field.is_required():
                    raise ValueError(
                        f"Field '{name}' ({field.annotation}) cannot be answered by a decision model; "
                        "use bool, Literal/Enum, or a float bounded to [0, 1]."
                    )
                continue
            if not field.description:
                raise ValueError(f"Field '{name}' needs a description: it is the question asked.")
            question = {"type": "noul" if kind != "choice" else "choice", "instructions": field.description}
            if kind == "choice":
                question["criteria"] = {str(o): "" for o in options}
            questions[name] = question
        if not questions:
            raise ValueError(f"{output_schema.__name__} has no fields a decision model can answer.")
        return questions

    def _request(
        self, input_data: BaseModel, output_schema: Type[BaseModel], kwargs: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], float]:
        request_kwargs = {**self.kwargs, **kwargs}
        body = {
            "state": self._state(input_data),
            "model": self.model_name,
            "questions": self._questions(output_schema),
        }
        return body, request_kwargs.get("threshold", self.threshold)

    # --- Response parsing ---

    def _parse(
        self, answers: Dict[str, Any], output_schema: Type[BaseModel], threshold: float
    ) -> BaseModel:
        if issubclass(output_schema, EvaluationOutput):
            if self.system_prompt:
                p, yes, no = _noul(answers, "verdict"), "yes", "no"
            else:
                _, yes, no, _ = EVAL_MODES[self.eval_mode]
                p = _prob(answers["verdict"]["probabilities"], yes, no)
            return output_schema(
                score=p,
                label=yes if p >= threshold else no,
                # Decider's derive_confidence, (N * p_max - 1) / (N - 1), at N = 2: 0 at p=0.5, 1 at certainty.
                confidence=abs(2 * p - 1),
            )

        values = {}
        for name, field in output_schema.model_fields.items():
            if name not in answers:
                continue
            kind, _ = _field_kind(field.annotation, field.metadata)
            if kind == "choice":
                values[name] = answers[name]["choice"]
            elif kind == "bool":
                values[name] = _noul(answers, name) >= threshold
            else:
                values[name] = _noul(answers, name)
        return output_schema(**values)

    # --- Calls ---

    def _call_backend(
        self, input_data: BaseModel, output_schema: Type[BaseModel], **kwargs
    ) -> Union[BaseModel, EvaluationError]:
        try:
            body, threshold = self._request(input_data, output_schema, kwargs)
            if not self._model_checked:
                self._check_model(self.client.get(f"{self.base_url}/health", headers=self._headers))
            response = self.client.post(
                f"{self.base_url}/v1/systemone", json=body, headers=self._headers
            )
            response.raise_for_status()
            return self._parse(response.json()["answers"], output_schema, threshold)
        except Exception as e:
            return _to_error(e)

    async def _call_backend_async(
        self, input_data: BaseModel, output_schema: Type[BaseModel], **kwargs
    ) -> Union[BaseModel, EvaluationError]:
        try:
            body, threshold = self._request(input_data, output_schema, kwargs)
            if not self._model_checked:
                self._check_model(
                    await self.async_client.get(f"{self.base_url}/health", headers=self._headers)
                )
            response = await self.async_client.post(
                f"{self.base_url}/v1/systemone", json=body, headers=self._headers
            )
            response.raise_for_status()
            return self._parse(response.json()["answers"], output_schema, threshold)
        except Exception as e:
            return _to_error(e)


class ModelMismatchError(Exception):
    """The server is serving a different checkpoint than the evaluator was configured for."""


def _field_kind(annotation: Any, metadata: list) -> Tuple[str, tuple]:
    """Map a field type to a question: 'bool', 'prob' (float in [0, 1]), 'choice', or 'skip'."""
    origin = typing.get_origin(annotation)
    if origin is Union:
        args = [a for a in typing.get_args(annotation) if a is not type(None)]
        if len(args) == 1:
            return _field_kind(args[0], metadata)
    if annotation is bool:
        return "bool", ()
    if origin is typing.Literal:
        return "choice", typing.get_args(annotation)
    if isinstance(annotation, type) and issubclass(annotation, enum.Enum):
        return "choice", tuple(m.value for m in annotation)
    if annotation is float:
        ge = next((m.ge for m in metadata if getattr(m, "ge", None) is not None), None)
        le = next((m.le for m in metadata if getattr(m, "le", None) is not None), None)
        if ge == 0 and le == 1:
            return "prob", ()
    return "skip", ()


def _noul(answers: Dict[str, Any], key: str) -> float:
    p = answers[key]["noul"]
    if not 0.0 <= p <= 1.0:
        raise ValueError(f"Server returned noul={p} for '{key}', outside [0, 1].")
    return float(p)


def _prob(probabilities: Dict[str, float], yes: str, no: str) -> float:
    """p(yes) from a two-option choice, renormalized because servers round each probability."""
    p_yes, p_no = float(probabilities[yes]), float(probabilities[no])
    if p_yes < 0 or p_no < 0 or p_yes + p_no <= 0:
        raise ValueError(f"Server returned invalid probabilities {probabilities}.")
    return p_yes / (p_yes + p_no)


def _to_error(e: Exception) -> EvaluationError:
    if httpx is not None and isinstance(e, httpx.HTTPStatusError):
        try:
            detail = e.response.json().get("detail", e.response.text)
        except ValueError:
            detail = e.response.text
        return EvaluationError(
            error_code=str(e.response.status_code),
            message=str(detail),
            details={"exception_type": type(e).__name__},
        )
    if isinstance(e, ModelMismatchError):
        return EvaluationError(
            error_code="MODEL_MISMATCH",
            message=str(e),
            details={"exception_type": type(e).__name__},
        )
    if httpx is not None and isinstance(e, httpx.TransportError):
        return EvaluationError(
            error_code="CONNECTION_ERROR",
            message=f"Could not reach the /v1/systemone server: {e}",
            details={"exception_type": type(e).__name__},
        )
    if isinstance(e, (ValueError, TypeError)):
        return EvaluationError(
            error_code="INVALID_REQUEST",
            message=str(e),
            details={"exception_type": type(e).__name__},
        )
    return EvaluationError(
        error_code="UNKNOWN_ERROR",
        message=str(e),
        details={"exception_type": type(e).__name__},
    )
