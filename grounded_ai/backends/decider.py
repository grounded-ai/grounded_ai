import enum
import json
import os
import types
import typing
import warnings
from typing import Any, Dict, Generic, Optional, Tuple, Type, TypeVar, Union

from pydantic import BaseModel, Field

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
# arguments have nothing to act on and are dropped. `threshold` and `eval_mode` are read. Anything
# else is probably a typo and gets a warning instead of silently doing nothing.
_IGNORED_KWARGS = {
    "temperature", "max_tokens", "max_new_tokens", "max_completion_tokens", "top_p", "top_k",
    "do_sample", "repetition_penalty", "frequency_penalty", "presence_penalty", "seed", "stop",
}
_READ_KWARGS = {"threshold", "eval_mode"}

# Fields of EvaluationOutput that the verdict question fills (reasoning stays None: nothing is generated).
_VERDICT_FIELDS = {"score", "label", "confidence", "reasoning"}
_VERDICT_KEY = "verdict"
# The server's `score` question takes 2 to 10 ordered levels.
_MIN_LEVELS, _MAX_LEVELS = 2, 10
_UNION_TYPES = (Union,) + ((types.UnionType,) if hasattr(types, "UnionType") else ())


T = TypeVar("T")


class Noul(BaseModel):
    """The whole answer to a yes/no question, as `/v1/systemone` returns it."""

    noul: float = Field(ge=0.0, le=1.0, description="Probability the statement is true")


class Choice(BaseModel, Generic[T]):
    """The whole answer to a pick-one question. Give the options: `Choice[Literal["a", "b", "c"]]`."""

    choice: T = Field(description="The option the model picked")
    probabilities: Dict[str, float] = Field(description="Probability of each option")
    confidence: float = Field(description="How concentrated the distribution is: 0 uniform, 1 certain")


class Score(BaseModel, Generic[T]):
    """The whole answer to a rating. Give the levels, lowest first: `Score[Literal["poor", "ok", "great"]]`."""

    score: float = Field(description="Expected level index, counting from 0 at the lowest level")
    legend: Dict[str, str] = Field(default_factory=dict, description="Level index -> the level it stands for")
    probabilities: Dict[str, float] = Field(description="Probability of each level index")
    confidence: float = Field(description="How tightly the mass clusters on the scale")


class DeciderBackend(BaseEvaluator):
    """
    Backend for Strands Decider (`strands-decider serve`), a local decision model that
    serves the `POST /v1/systemone` contract. Any server with that contract works the same.

    These models do not generate text. Each question is answered with a probability read
    off the model, so the output cannot leave the schema and `confidence` is measured,
    not self-reported.

    The output schema maps one-to-one onto what the model returns. `/v1/systemone` has three
    question types, and every field of an output schema is exactly one of them:

        noul    float in [0, 1]                 -> the probability the statement is true
                bool                            -> that probability >= threshold
        choice  Literal / Enum                  -> the option the model picked
        score   int or float with ge/le bounds  -> a rating over 2-10 ordered levels

    Those fields hold the value only. To get the whole answer, with its probabilities and
    confidence, type the field as the answer itself:

        Noul                              -> noul
        Choice[Literal["a", "b", "c"]]    -> choice, probabilities, confidence
        Score[Literal["poor", "great"]]   -> score, legend, probabilities, confidence

    The field's `description` is the question (the field name is used when there is none).
    Nothing else can be answered: a required field of any other type is refused with
    INVALID_REQUEST, and one with a default is left at its default. The stock EvaluationOutput
    is one choice question, set by `eval_mode`: score = p(positive label), label, confidence.

    The input side is free: the request's `state` is a string or a JSON object, so any input
    model works. Its fields are sent as an object, or its own `formatted_prompt` / `base_template`
    is rendered and sent as text.

    There is no system message in the `/v1/systemone` contract, so `system_prompt` is not sent.
    What the model is asked comes from `eval_mode` and the field descriptions.

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
        eval_mode: Any = None,
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
        if self.system_prompt:
            warnings.warn(
                "system_prompt is not sent to a decision model: /v1/systemone has no system message. "
                "Put the question in eval_mode or in the output schema's field descriptions.",
                stacklevel=3,
            )
        self.set_eval_mode(eval_mode if eval_mode is not None else "HALLUCINATION")
        self.threshold = _check_threshold(threshold)
        self._timeout = timeout
        self.client = client or httpx.Client(timeout=timeout)
        self._async_client = async_client
        self.kwargs = kwargs
        self._model_checked = False
        self._max_length = None  # the served model's token window, from /health

    def set_eval_mode(self, mode) -> None:
        """A built-in mode name, or an EvalMode member as on the SLM backend: HALLUCINATION,
        TOXICITY or RAG_RELEVANCE. `eval_mode=` can also be passed to a single evaluate() call.

        For an evaluation of your own, pass an output_schema: an EvaluationOutput subclass whose
        `label` is a Literal or Enum of your options, the same way as on the other backends.
        """
        self._mode, self.eval_mode = _resolve_mode(mode)

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
        try:
            health = response.json()
        except ValueError:
            health = None
        if not isinstance(health, dict):
            # /health exists but is not the strands-decider JSON: nothing to check against.
            self._model_checked = True
            return
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
        if isinstance(health.get("max_length"), int):
            self._max_length = health["max_length"]
        self._model_checked = True

    def _warn_if_long(self, state: Union[str, Dict[str, Any]]) -> None:
        """The server cuts a state that overflows the model's window from the end, without an
        error, so the last field (the response being judged) is what goes missing. There is no
        tokenizer here, so this is a rough guard: over ~4 characters per token of window."""
        if not self._max_length:
            return
        chars = len(state) if isinstance(state, str) else len(json.dumps(state, ensure_ascii=False))
        if chars > 4 * self._max_length:
            warnings.warn(
                f"Decider input is about {chars} characters but the served model reads at most "
                f"{self._max_length} tokens; the server truncates the end of the input, so the text "
                "being judged may be cut off. Shorten the context.",
                stacklevel=2,
            )

    # --- Request construction ---

    def _state(self, input_data: BaseModel) -> Union[str, Dict[str, Any]]:
        # Like every other backend, the input model's own prompt formatting is honored: a custom
        # base_template (per instance or as a subclass default), or a model that defines its own
        # formatted_prompt. Its rendered text is what the model reads.
        if hasattr(input_data, "formatted_prompt"):
            stock_prompt = getattr(type(input_data), "formatted_prompt", None) is getattr(
                EvaluationInput, "formatted_prompt", None
            )
            stock_template = (
                getattr(input_data, "base_template", None)
                == EvaluationInput.model_fields["base_template"].default
            )
            if not (stock_prompt and stock_template):
                state = input_data.formatted_prompt
                if not isinstance(state, str) or not state.strip():
                    raise ValueError("The input's formatted_prompt is empty: there is nothing to evaluate.")
                return state
        # Stock template: an object state keeps the field names as keys, so the model sees which
        # text is which without a prompt template.
        data = input_data.model_dump(
            mode="json", exclude={"base_template", "formatted_prompt"}, exclude_none=True
        )
        if not data:
            raise ValueError("The input has no fields set: there is nothing to evaluate.")
        ordered = {k: data.pop(k) for k in _STATE_ORDER if k in data}
        return {**ordered, **data}

    def _verdict(self, output_schema: Type[BaseModel], mode: tuple) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """The one choice question behind score/label/confidence of an EvaluationOutput.

        Returns (question, labels); `labels` maps each option, positive first, to the value the
        schema's `label` field takes. The options are the eval mode's two labels, unless the
        schema narrows `label` to a Literal or Enum of its own: then those are the options and
        the field's description is the question.
        """
        instructions, yes, no, criteria = mode
        label_field = output_schema.model_fields["label"]
        kind, options = _field_spec("label", label_field, strict=False)
        if kind != "choice":
            return {"type": "choice", "instructions": instructions, "criteria": criteria}, {yes: yes, no: no}
        if set(options) == {yes, no}:
            # Same labels as the mode: keep the mode's question and label descriptions.
            return (
                {"type": "choice", "instructions": instructions, "criteria": criteria},
                {yes: options[yes], no: options[no]},
            )
        stock = EvaluationOutput.model_fields["label"].description
        question = label_field.description if label_field.description not in (None, stock) else "Classify the response."
        return {"type": "choice", "instructions": question, "criteria": {o: "" for o in options}}, options

    def _field_questions(self, schema: Type[BaseModel], skip: set, out: Dict[str, Dict[str, Any]]) -> None:
        """One question per answerable field of `schema`, of the type the field maps to."""
        for name, field in schema.model_fields.items():
            if name in skip:
                continue
            kind, info = _field_spec(name, field)
            if kind == "skip":
                continue
            text = field.description or name.replace("_", " ")
            if kind in ("choice", "choice_answer"):
                out[name] = {"type": "choice", "instructions": text, "criteria": {o: "" for o in info}}
            elif kind == "score_answer":
                out[name] = {"type": "score", "instructions": text, "criteria": list(info)}
            elif kind == "level":
                low, high, _ = info
                out[name] = {"type": "score", "instructions": text, "criteria": [str(v) for v in range(low, high + 1)]}
            else:  # bool, prob, noul_answer
                out[name] = {"type": "noul", "instructions": text}

    def _request(
        self, input_data: BaseModel, output_schema: Type[BaseModel], kwargs: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """The request body, and what _parse needs to read the answer back."""
        request_kwargs = {**self.kwargs, **kwargs}
        unknown = set(request_kwargs) - _IGNORED_KWARGS - _READ_KWARGS
        if unknown:
            warnings.warn(
                f"DeciderBackend ignores unknown argument(s) {sorted(unknown)}; "
                f"it reads {sorted(_READ_KWARGS)}.",
                stacklevel=2,
            )
        plan = {"threshold": _check_threshold(request_kwargs.get("threshold", self.threshold)), "labels": None}
        questions: Dict[str, Dict[str, Any]] = {}
        skip = set()
        if issubclass(output_schema, EvaluationOutput):
            mode = self._mode
            if request_kwargs.get("eval_mode") is not None:
                mode = _resolve_mode(request_kwargs["eval_mode"])[0]
            questions[_VERDICT_KEY], plan["labels"] = self._verdict(output_schema, mode)
            skip = _VERDICT_FIELDS
            if _VERDICT_KEY in output_schema.model_fields:
                raise ValueError(f"'{_VERDICT_KEY}' is reserved on an EvaluationOutput schema; rename the field.")
        elif request_kwargs.get("eval_mode") is not None:
            raise ValueError(
                "eval_mode fills EvaluationOutput's score/label/confidence; "
                f"{output_schema.__name__} is asked field by field instead."
            )
        # Fields a subclass adds to EvaluationOutput are asked too, in the same request.
        self._field_questions(output_schema, skip, questions)
        if not questions:
            raise ValueError(f"{output_schema.__name__} has no fields a decision model can answer.")
        body = {"state": self._state(input_data), "model": self.model_name, "questions": questions}
        return body, plan

    # --- Response parsing ---

    def _field_values(
        self, schema: Type[BaseModel], skip: set, answers: Dict[str, Any], threshold: float
    ) -> Dict[str, Any]:
        values = {}
        for name, field in schema.model_fields.items():
            if name in skip:
                continue
            kind, info = _field_spec(name, field)
            if kind == "choice":
                values[name] = info[answers[name]["choice"]]
            elif kind == "choice_answer":
                values[name] = {**answers[name], "choice": info[answers[name]["choice"]]}
            elif kind == "score_answer":
                values[name] = dict(answers[name])
            elif kind == "noul_answer":
                values[name] = {"noul": _noul(answers, name)}
            elif kind == "level":
                low, _, is_int = info
                values[name] = low + (_top_level(answers[name]) if is_int else float(answers[name]["score"]))
            elif kind == "bool":
                values[name] = _noul(answers, name) >= threshold
            elif kind == "prob":
                values[name] = _noul(answers, name)
        return values

    def _parse(self, answers: Dict[str, Any], output_schema: Type[BaseModel], plan: Dict[str, Any]) -> BaseModel:
        threshold = plan["threshold"]
        if not issubclass(output_schema, EvaluationOutput):
            return output_schema(**self._field_values(output_schema, set(), answers, threshold))

        labels, verdict = plan["labels"], answers[_VERDICT_KEY]
        if len(labels) == 2:
            yes, no = list(labels)
            p = _prob(verdict["probabilities"], yes, no)
            label = labels[yes] if p >= threshold else labels[no]
            # Decider's derive_confidence, (N * p_max - 1) / (N - 1), at N = 2: 0 at p=0.5, 1 at certainty.
            confidence = abs(2 * p - 1)
        else:
            # More than two labels: the model's pick, with score = p(first label).
            probabilities = {o: float(verdict["probabilities"][o]) for o in labels}
            total = sum(probabilities.values())
            if total <= 0 or min(probabilities.values()) < 0:
                raise ValueError(f"Server returned invalid probabilities {verdict['probabilities']}.")
            p = probabilities[next(iter(labels))] / total
            label = labels[verdict["choice"]]
            n = len(labels)
            confidence = max(0.0, (n * max(probabilities.values()) / total - 1) / (n - 1))
        extra = self._field_values(output_schema, _VERDICT_FIELDS, answers, threshold)
        return output_schema(score=p, label=label, confidence=confidence, **extra)

    def _read(self, response: Any, output_schema: Type[BaseModel], plan: Dict[str, Any]) -> BaseModel:
        response.raise_for_status()
        try:
            return self._parse(response.json()["answers"], output_schema, plan)
        except (KeyError, TypeError, ValueError) as e:
            raise ResponseError(f"Could not read the server's answer into {output_schema.__name__}: {e!r}") from e

    # --- Calls ---

    def _call_backend(
        self, input_data: BaseModel, output_schema: Type[BaseModel], **kwargs
    ) -> Union[BaseModel, EvaluationError]:
        try:
            body, plan = self._request(input_data, output_schema, kwargs)
            if not self._model_checked:
                self._check_model(self.client.get(f"{self.base_url}/health", headers=self._headers))
            self._warn_if_long(body["state"])
            response = self.client.post(
                f"{self.base_url}/v1/systemone", json=body, headers=self._headers
            )
            return self._read(response, output_schema, plan)
        except Exception as e:
            return _to_error(e)

    async def _call_backend_async(
        self, input_data: BaseModel, output_schema: Type[BaseModel], **kwargs
    ) -> Union[BaseModel, EvaluationError]:
        try:
            body, plan = self._request(input_data, output_schema, kwargs)
            if not self._model_checked:
                self._check_model(
                    await self.async_client.get(f"{self.base_url}/health", headers=self._headers)
                )
            self._warn_if_long(body["state"])
            response = await self.async_client.post(
                f"{self.base_url}/v1/systemone", json=body, headers=self._headers
            )
            return self._read(response, output_schema, plan)
        except Exception as e:
            return _to_error(e)


def _resolve_mode(mode: Any) -> Tuple[tuple, Any]:
    """(the (instructions, positive, negative, criteria) tuple, the mode's name)."""
    mode = getattr(mode, "value", mode)
    name = mode.upper() if isinstance(mode, str) else mode  # case-insensitive, as on the SLM backend
    if not isinstance(name, str) or name not in EVAL_MODES:
        raise ValueError(
            f"Invalid eval_mode {mode!r}. Options: {list(EVAL_MODES)}. For an evaluation of your own, pass an "
            "output_schema: an EvaluationOutput subclass whose `label` is a Literal or Enum of your options."
        )
    return EVAL_MODES[name], name


def _check_threshold(threshold: Any) -> float:
    if isinstance(threshold, bool) or not isinstance(threshold, (int, float)) or not 0 <= threshold <= 1:
        raise ValueError(f"threshold must be a number in [0, 1], got {threshold!r}.")
    return float(threshold)


class ModelMismatchError(Exception):
    """The server is serving a different checkpoint than the evaluator was configured for."""


class ResponseError(Exception):
    """The server answered, but the answer could not be read into the output schema."""


def _options(annotation: Any) -> Optional[Dict[str, Any]]:
    """{option text sent to the model: value handed to the schema} for a Literal or Enum."""
    if typing.get_origin(annotation) is typing.Literal:
        values = [v.value if isinstance(v, enum.Enum) else v for v in typing.get_args(annotation)]
    elif isinstance(annotation, type) and issubclass(annotation, enum.Enum):
        values = [m.value for m in annotation]
    else:
        return None
    options = {str(v): v for v in values}
    if len(options) != len(values):
        raise ValueError(f"Options {values} are not distinct once written as text.")
    return options


def _bound(metadata: list, inclusive: str, exclusive: str, step: int) -> Optional[float]:
    """A field's lower or upper bound from its ge/gt or le/lt constraint."""
    for m in metadata:
        if getattr(m, inclusive, None) is not None:
            return getattr(m, inclusive)
    for m in metadata:
        if getattr(m, exclusive, None) is not None:
            return getattr(m, exclusive) + step
    return None


def _field_spec(name: str, field: Any, strict: bool = True) -> Tuple[str, Any]:
    """Map a field to the question type that answers it: (kind, info).

        noul    'bool'    yes/no                     'prob'  float in [0, 1]
        choice  'choice'  {option text: value}
        score   'level'   (low, high, is_int) rating
        'noul_answer', 'choice_answer', 'score_answer'  the whole answer (Noul, Choice[...], Score[...])
        'skip'  not answerable, left at its default

    A required field that is not answerable raises, unless `strict` is False.
    """
    annotation, metadata = field.annotation, field.metadata
    if typing.get_origin(annotation) in _UNION_TYPES:  # Optional[X] and X | None are asked as X
        args = [a for a in typing.get_args(annotation) if a is not type(None)]
        if len(args) == 1:
            annotation = args[0]

    if annotation is bool:
        return "bool", None
    if annotation is Noul:
        return "noul_answer", None
    generic = getattr(annotation, "__pydantic_generic_metadata__", None) or {}
    answer = Choice if annotation is Choice or generic.get("origin") is Choice else None
    answer = Score if annotation is Score or generic.get("origin") is Score else answer
    if answer is not None:
        options = _options(generic["args"][0]) if generic.get("args") else None
        if options is None or len(options) < 2:
            raise ValueError(
                f"Field '{name}' needs its options, two or more: "
                f'{answer.__name__}[Literal["a", "b"]]' + (", lowest level first." if answer is Score else ".")
            )
        if answer is Score and len(options) > _MAX_LEVELS:
            raise ValueError(f"Field '{name}' has {len(options)} levels; a score takes at most {_MAX_LEVELS}.")
        return ("choice_answer" if answer is Choice else "score_answer"), options
    options = _options(annotation)
    if options is not None:
        if len(options) < 2:
            raise ValueError(f"Field '{name}' needs at least two options: a choice question picks one of several.")
        return "choice", options
    if annotation in (int, float):
        low, high = _bound(metadata, "ge", "gt", 1), _bound(metadata, "le", "lt", -1)
        if annotation is float and low == 0 and high == 1 and not any(
            getattr(m, "gt", None) is not None or getattr(m, "lt", None) is not None for m in metadata
        ):
            return "prob", None
        if low is not None and high is not None and float(low).is_integer() and float(high).is_integer():
            if _MIN_LEVELS <= int(high) - int(low) + 1 <= _MAX_LEVELS:
                return "level", (int(low), int(high), annotation is int)
    if strict and field.is_required():
        raise ValueError(
            f"Field '{name}' ({field.annotation}) does not map to a decision model answer. Use bool or a "
            "float bounded to [0, 1] (noul), Literal/Enum (choice), or an int or float bounded to a range "
            f"of {_MIN_LEVELS}-{_MAX_LEVELS} values (score); or give the field a default to leave it unanswered."
        )
    return "skip", None


def _top_level(answer: Dict[str, Any]) -> int:
    """The most probable level index of a `score` answer."""
    probabilities = answer.get("probabilities")
    if probabilities:
        return int(max(probabilities, key=lambda level: float(probabilities[level])))
    return int(round(float(answer["score"])))


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
    if isinstance(e, ResponseError):
        return EvaluationError(
            error_code="INVALID_RESPONSE",
            message=str(e),
            details={"exception_type": type(e.__cause__ or e).__name__},
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
