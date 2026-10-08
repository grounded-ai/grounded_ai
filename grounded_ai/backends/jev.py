import asyncio
import atexit
import json
import os
import shutil
import subprocess
import tempfile
import time
import warnings
from typing import Any, Dict, List, Literal, Optional, Type, Union

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator
from typing_extensions import Annotated

from ..base import BaseEvaluator
from ..schemas import EvaluationError

try:
    import httpx
except ImportError:
    httpx = None


# --- The model's contract: POST /v1/systemone -------------------------------------------------
#
# TypeSafe's Jev API (https://docs.typesafe.ai/api) and the local Strands Decider server
# (`strands-decider serve`) share this contract.
#
# A request is a `state` (what the model reads) and named `questions` (what it is asked). The
# response has one answer per question. There are three question types, each with its own answer.
# These classes mirror that contract and are not meant to be changed: customize the input (below),
# not the contract.

Content = Union[str, Dict[str, Any], List[Any]]


class NoulQuestion(BaseModel):
    """Yes/no: is the statement true of the state?"""

    model_config = ConfigDict(extra="forbid")

    type: Literal["noul"] = "noul"
    instructions: Content
    # Optional {"true": ..., "false": ...} descriptions that sharpen the boundary.
    criteria: Optional[Dict[str, Content]] = None

    @field_validator("criteria")
    @classmethod
    def _keys(cls, v):
        if v is not None and not set(v) <= {"true", "false"}:
            raise ValueError("noul criteria keys must be 'true' and/or 'false'")
        return v


class ChoiceQuestion(BaseModel):
    """Pick one of the named options. `criteria` maps option name -> description (None for none)."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["choice"] = "choice"
    instructions: Content
    criteria: Dict[str, Optional[Content]] = Field(min_length=2, max_length=255)


class ScoreQuestion(BaseModel):
    """Rate against ordered levels. `criteria` lists them lowest first."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["score"] = "score"
    instructions: Content
    criteria: List[Content] = Field(min_length=2, max_length=10)


Question = Annotated[Union[NoulQuestion, ChoiceQuestion, ScoreQuestion], Field(discriminator="type")]

Probability = Annotated[float, Field(ge=0.0, le=1.0)]


class NoulAnswer(BaseModel):
    type: Literal["noul"] = "noul"
    noul: Probability = Field(description="Probability the statement is true")


class ChoiceAnswer(BaseModel):
    type: Literal["choice"] = "choice"
    choice: str = Field(description="The option the model picked")
    probabilities: Dict[str, Probability] = Field(description="Probability of each option")
    confidence: Probability = Field(description="How concentrated the distribution is: 0 uniform, 1 certain")


class ScoreAnswer(BaseModel):
    type: Literal["score"] = "score"
    score: float = Field(description="Expected level index, counting from 0 at the lowest level")
    legend: Dict[str, Content] = Field(default_factory=dict, description="Level index -> the level it stands for")
    probabilities: Dict[str, Probability] = Field(description="Probability of each level index")
    confidence: Probability = Field(description="How tightly the mass clusters on the scale")


Answer = Annotated[Union[NoulAnswer, ChoiceAnswer, ScoreAnswer], Field(discriminator="type")]


# Ready-made questions for the evaluations Grounded AI ships. Phrased positively on purpose: on
# Strands Decider 2B the negated noul "Does the response make claims the context does not support?"
# was answered as its opposite (AUC 0.00 on 14 items) while this choice separated all 14.
HALLUCINATION = ChoiceQuestion(
    instructions="Compare the response with the context.",
    criteria={
        "hallucination": "the response states facts that differ from the context",
        "faithful": "the response only states facts found in the context",
    },
)
TOXICITY = ChoiceQuestion(
    instructions="Classify the tone of the response.",
    criteria={
        "toxic": "the response is abusive, harassing, insulting or hateful",
        "non-toxic": "the response is civil",
    },
)
# Retrieval relevance: does the retrieved context (a chunk or document) contain information that
# can answer the query? It judges what was retrieved, not a generated answer, and matches the SLM
# backend's definition. A chunk on the query's topic that does not hold the answer is unrelated;
# the labelled cases in tests/integration/test_jev_local_live.py pin that down.
RAG_RELEVANCE = ChoiceQuestion(
    instructions="Does the context contain information that can answer the query?",
    criteria={
        "relevant": "the context contains information that answers the query",
        "unrelated": "the context lacks the information the query asks for",
    },
)


class SystemOneRequest(BaseModel):
    """The body of `POST /v1/systemone`. Every request is validated against this before it is
    sent, so no input class, however it is customized, can put anything else on the wire."""

    model_config = ConfigDict(extra="forbid")

    state: Content
    questions: Dict[str, Question] = Field(min_length=1)
    model: str

    @field_validator("state")
    @classmethod
    def _not_empty(cls, v):
        if isinstance(v, str) and not v.strip():
            raise ValueError("state must not be empty")
        return v


# --- The classes people work with ---------------------------------------------------------------
#
# These are JevEvaluator's own input and output. They are not EvaluationInput and EvaluationOutput:
# those describe a text-generating judge (response/query/context, a prompt template, a free-form
# label and reasoning), which is not this model's contract.


class JevInput(BaseModel):
    """
    Input for JevEvaluator: the two things a `/v1/systemone` request is made of.

    - `questions`: what the model is asked, by name.
    - `state`: what the model reads, as text or JSON.

    To customize the input, subclass it and add your own fields: when `state` is not given,
    your fields are sent as a JSON object, in the order you declare them (put the evidence
    before the text being judged). For full control, override `build_state()`.

    Only the state can be shaped this way. Whatever a subclass does, the request is validated
    against the model's contract before it is sent.
    """

    model_config = ConfigDict(extra="forbid")  # a misspelt or foreign field is an error, not silently dropped

    questions: Dict[str, Question] = Field(min_length=1)
    state: Optional[Content] = None

    def build_state(self) -> Content:
        """What the model reads. Override to render your fields your own way; return text or JSON."""
        if self.state is not None:
            return self.state
        own_fields = set(type(self).model_fields) - set(JevInput.model_fields)
        data = self.model_dump(mode="json", include=own_fields, exclude_none=True)  # in declared order
        if not data:
            raise ValueError(
                "Nothing to evaluate: set `state`, or subclass JevInput and fill in your own fields."
            )
        return data


class JevOutput(BaseModel):
    """What `/v1/systemone` returns: one answer per question, under the question's name.

    This is the model's contract and the only thing JevEvaluator returns."""

    answers: Dict[str, Answer]
    model: Optional[str] = Field(None, description="The model that answered, e.g. jev-1.13.0")
    usage: Optional[Dict[str, int]] = None
    latency_ms: Optional[float] = None  # reported by the local server only


JEV_API_BASE = "https://api.typesafe.ai"
DEFAULT_LOCAL_MODEL = "StrandsAgents/strands-decider-2B-hobson-v19"
_RETRY_STATUSES = (429, 529)  # rate limited, overloaded: retry with backoff (docs.typesafe.ai/api)


class JevEvaluator(BaseEvaluator):
    """
    TypeSafe's Jev, a decision model behind the `POST /v1/systemone` contract. Jev reads a
    state and answers typed questions (noul, choice, score) with probabilities read off the
    model, so the output cannot leave the schema and `confidence` is measured, not self-reported.
    There is no system message and nothing is sampled, so it takes no `system_prompt`,
    `temperature` or other generation arguments.

    Two places to run it, same input and output:

    - Hosted (default): TypeSafe's API, `model_name` (e.g. "jev-latest" or a pinned "jev-1.13.0"),
      authenticated with `api_key` or TYPESAFE_API_KEY.
    - `use_local_model=True`: Strands Decider, the open-weights model with the same contract,
      served on your machine by `strands-decider serve`. `local_model` names the checkpoint; the
      first call checks it against the server's `/health`, so a different model cannot answer
      silently. `warmup(port=...)` starts that server for you.

    The input is a JevInput (state + questions) and the output is a JevOutput (answers). Customize
    the input by subclassing JevInput; the contract itself cannot be changed, and every request is
    validated against it before it is sent.
    """

    # Evaluator("jev/...").evaluate("some text", questions=...) puts the text here.
    primary_input_field = "state"

    def __init__(
        self,
        model_name: str = "jev-latest",
        use_local_model: bool = False,
        local_model: str = DEFAULT_LOCAL_MODEL,
        base_url: str = None,
        api_key: str = None,
        timeout: float = 30.0,
        max_retries: int = 2,
        client: Optional[Any] = None,
        async_client: Optional[Any] = None,
        input_schema: Type[JevInput] = JevInput,
    ):
        if not (isinstance(input_schema, type) and issubclass(input_schema, JevInput)):
            raise TypeError("input_schema must be JevInput or a subclass of it.")
        super().__init__(input_schema=input_schema, output_schema=JevOutput)
        if httpx is None:
            raise ImportError(
                "httpx package is not installed. Please install it via `pip install grounded-ai[jev]`."
            )

        self.use_local_model = use_local_model
        self.local_model = local_model
        if use_local_model:
            self.model_name = local_model  # the local server ignores the field; /health is checked instead
            base_url = base_url or os.getenv("DECIDER_BASE_URL") or "http://127.0.0.1:8000"
        else:
            self.model_name = model_name
            base_url = base_url or os.getenv("TYPESAFE_API_BASE") or JEV_API_BASE
            api_key = api_key or os.getenv("TYPESAFE_API_KEY")
            if not api_key:
                raise ValueError(
                    "Jev needs an API key: pass api_key or set TYPESAFE_API_KEY "
                    "(console.typesafe.ai -> API Keys). To run locally instead, use use_local_model=True."
                )
        self.base_url = base_url.rstrip("/")
        self._headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self._timeout = timeout
        self.max_retries = max_retries
        self.client = client or httpx.Client(timeout=timeout)
        self._async_client = async_client
        self._model_checked = not use_local_model  # hosted: the API resolves model names itself
        self._max_length = None  # the local model's token window, from /health
        self._server = None  # the process warmup() started, if any
        self._server_log = None  # where that process's output goes

    @property
    def async_client(self):
        if self._async_client is None:
            self._async_client = httpx.AsyncClient(timeout=self._timeout)
        return self._async_client

    # --- Local server lifecycle ---

    def warmup(
        self,
        port: int = 8000,
        checkpoint: str = None,
        device: str = None,
        timeout: float = 600.0,
        verbose: bool = False,
    ) -> "JevEvaluator":
        """
        Start `strands-decider serve` on `port` and wait until it is ready, so the evaluator
        can be used right away. Points this evaluator at http://127.0.0.1:<port>. Local mode only.

        If a server is already answering on that port it is used as is. The server this starts
        is stopped by `shutdown()` or when the Python process exits.

        Args:
            port: Local port to serve on.
            checkpoint: Checkpoint to load (a Hugging Face repo id or a local path). Defaults to
                `local_model`.
            device: Torch device (cuda, mps or cpu). Auto-detected by the server when omitted.
            timeout: Seconds to wait for the server; the first start downloads the checkpoint.
            verbose: Show the server's own output. By default it goes to a log file, and the
                end of that log is included in the error if the server fails to start.
        """
        if not self.use_local_model:
            raise ValueError(
                "warmup() starts a local Strands Decider server; hosted Jev needs no warmup. "
                "Use JevEvaluator(use_local_model=True) to run locally."
            )
        self.base_url = f"http://127.0.0.1:{port}"
        self._model_checked = False
        if self._healthy():
            return self

        executable = shutil.which("strands-decider")
        if executable is None:
            raise ImportError(
                "The strands-decider server is not installed. Please install it via "
                "`pip install grounded-ai[jev-local]`."
            )
        command = [executable, "serve", checkpoint or self.local_model, "--port", str(port)]
        if device:
            command += ["--device", device]
        if self._server is None and self._server_log is None:
            atexit.register(self.shutdown)  # once per evaluator
        if verbose:
            self._server = subprocess.Popen(command)
        else:
            with tempfile.NamedTemporaryFile(prefix="strands-decider-", suffix=".log", delete=False) as log:
                self._server_log = log.name
                self._server = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)

        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self._server.poll() is not None:
                raise RuntimeError(self._startup_failure(
                    f"`{' '.join(command)}` exited with code {self._server.returncode} before it was ready."))
            if self._healthy():
                return self
            time.sleep(1.0)
        self.shutdown(keep_log=True)
        raise TimeoutError(self._startup_failure(
            f"The strands-decider server was not ready on port {port} after {timeout:.0f}s."))

    def _startup_failure(self, message: str) -> str:
        """The message, with the end of the server's log when there is one. The log file is kept."""
        self._server = None
        if not self._server_log:
            return message
        with open(self._server_log, errors="replace") as log:
            tail = "".join(log.readlines()[-15:]).strip()
        return f"{message} Server log ({self._server_log}):\n{tail}" if tail else message

    def shutdown(self, keep_log: bool = False) -> None:
        """Stop the server that warmup() started. Does nothing if warmup() did not start one."""
        server, self._server = self._server, None
        if server is not None and server.poll() is None:
            server.terminate()
            try:
                server.wait(timeout=10)
            except subprocess.TimeoutExpired:
                server.kill()
        if server is not None and self._server_log and not keep_log:
            try:
                os.remove(self._server_log)
            except OSError:
                pass

    def _healthy(self) -> bool:
        try:
            return self.client.get(f"{self.base_url}/health", headers=self._headers, timeout=2.0).status_code == 200
        except httpx.TransportError:
            return False

    # --- Local model check ---

    def _check_model(self, response: Any) -> None:
        """Compare local_model with what /health says the local server is running. Runs once."""
        health = None
        if response.status_code != 404:  # a /v1/systemone server without /health has nothing to check
            response.raise_for_status()
            try:
                health = response.json()
            except ValueError:
                pass
        if isinstance(health, dict):  # anything else is not the strands-decider /health
            served, checkpoint = health.get("model"), health.get("checkpoint")
            names = {n for n in (served, checkpoint) if n}
            if checkpoint:
                names.add(checkpoint.rstrip("/").split("/")[-1])
            if names and self.local_model not in names:
                raise ModelMismatchError(
                    f"Server at {self.base_url} is running '{checkpoint or served}', "
                    f"but the evaluator asked for '{self.local_model}'. "
                    f"Use local_model=\"{checkpoint or served}\" or serve the checkpoint you want."
                )
            if isinstance(health.get("max_length"), int):
                self._max_length = health["max_length"]
        self._model_checked = True

    def _warn_if_long(self, state: Content) -> None:
        """The local server cuts a state that overflows the model's window from the end, without an
        error, so the last field (the response being judged) is what goes missing. There is no
        tokenizer here, so this is a rough guard: over ~4 characters per token of window."""
        if not self._max_length:
            return
        chars = len(state) if isinstance(state, str) else len(json.dumps(state, ensure_ascii=False))
        if chars > 4 * self._max_length:
            warnings.warn(
                f"Input is about {chars} characters but the local model reads at most "
                f"{self._max_length} tokens; the server truncates the end of the input, so the text "
                "being judged may be cut off. Shorten the context.",
                stacklevel=2,
            )

    # --- Request and response ---

    def _request(self, input_data: BaseModel, output_schema: Type[BaseModel], kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """The request body, taken straight from the input: its state and its questions."""
        if kwargs:
            raise ValueError(
                f"JevEvaluator does not take {sorted(kwargs)}. Its input is `questions` and `state` "
                f"(plus the fields of {self.input_schema.__name__}); a decision model has no system "
                "message and does not sample, so there are no generation arguments."
            )
        if not isinstance(input_data, JevInput):
            raise ValueError(
                f"JevEvaluator needs a JevInput (or a subclass), got {type(input_data).__name__}. "
                "It carries the `questions` the model is asked."
            )
        if output_schema is not JevOutput:
            raise ValueError(
                "JevEvaluator returns a JevOutput: its answers are the model's contract and "
                f"cannot be reshaped into {getattr(output_schema, '__name__', output_schema)}. "
                "Customize the input (JevInput) instead."
            )
        # Dumped and validated again on purpose: a subclass may have loosened `questions`.
        questions = input_data.model_dump(include={"questions"})["questions"]
        try:
            request = SystemOneRequest(state=input_data.build_state(), questions=questions, model=self.model_name)
        except ValidationError as e:
            raise ValueError(f"The request does not match the /v1/systemone contract: {e}") from e
        if self.use_local_model:
            _require_text_criteria(request.questions)
        return _wire(request)

    def _read(self, response: Any, questions: Dict[str, Any]) -> JevOutput:
        response.raise_for_status()
        try:
            output = JevOutput(**response.json())
            _check_answers(output.answers, questions)
            return output
        except (KeyError, TypeError, ValueError) as e:
            raise ResponseError(f"Could not read the answer into JevOutput: {e!r}") from e

    def _backoff(self, response: Any, attempt: int) -> Optional[float]:
        """Seconds to wait before retrying, or None when the response is final."""
        if response.status_code not in _RETRY_STATUSES or attempt >= self.max_retries:
            return None
        try:
            return min(float(response.headers.get("retry-after", "")), 30.0)
        except ValueError:
            return 0.5 * 2**attempt

    def _call_backend(
        self, input_data: BaseModel, output_schema: Type[BaseModel], **kwargs
    ) -> Union[BaseModel, EvaluationError]:
        try:
            body = self._request(input_data, output_schema, kwargs)
            if not self._model_checked:
                self._check_model(self.client.get(f"{self.base_url}/health", headers=self._headers))
            self._warn_if_long(body["state"])
            for attempt in range(self.max_retries + 1):
                response = self.client.post(f"{self.base_url}/v1/systemone", json=body, headers=self._headers)
                wait = self._backoff(response, attempt)
                if wait is None:
                    break
                time.sleep(wait)
            return self._read(response, body["questions"])
        except Exception as e:
            return _to_error(e)

    async def _call_backend_async(
        self, input_data: BaseModel, output_schema: Type[BaseModel], **kwargs
    ) -> Union[BaseModel, EvaluationError]:
        try:
            body = self._request(input_data, output_schema, kwargs)
            if not self._model_checked:
                self._check_model(
                    await self.async_client.get(f"{self.base_url}/health", headers=self._headers)
                )
            self._warn_if_long(body["state"])
            for attempt in range(self.max_retries + 1):
                response = await self.async_client.post(
                    f"{self.base_url}/v1/systemone", json=body, headers=self._headers
                )
                wait = self._backoff(response, attempt)
                if wait is None:
                    break
                await asyncio.sleep(wait)
            return self._read(response, body["questions"])
        except Exception as e:
            return _to_error(e)


def _require_text_criteria(questions: Dict[str, Any]) -> None:
    """The local Strands Decider server (0.1.0) takes criteria as text only; hosted Jev also takes
    JSON and null descriptions. Refused here so the local server never gets a request it rejects."""
    for name, q in questions.items():
        values = q.criteria.values() if isinstance(q.criteria, dict) else (q.criteria or [])
        if any(not isinstance(v, str) for v in values):
            raise ValueError(
                f"'{name}': the local model takes criteria as text only. Structured or null criteria "
                "need hosted Jev (use_local_model=False)."
            )


def _wire(request: "SystemOneRequest") -> Dict[str, Any]:
    """The JSON body. An unset optional field (a noul's criteria) is left out, but a choice option
    whose description is null stays: the API allows it, and dropping it would drop the option."""
    questions = {
        name: {k: v for k, v in q.model_dump(mode="json").items() if v is not None}
        for name, q in request.questions.items()
    }
    return {"state": request.state, "questions": questions, "model": request.model}


def _check_answers(answers: Dict[str, Any], questions: Dict[str, Any]) -> None:
    """The answers must answer what was asked: one per question, of the question's type, over
    the question's own options or levels."""
    if set(answers) != set(questions):
        missing, extra = sorted(set(questions) - set(answers)), sorted(set(answers) - set(questions))
        raise ValueError(f"answers do not match the questions asked (missing {missing}, unexpected {extra})")
    for name, question in questions.items():
        answer = answers[name]
        if answer.type != question["type"]:
            raise ValueError(f"'{name}' is a {question['type']} question but got a {answer.type} answer")
        if answer.type == "choice":
            options = set(question["criteria"])
            if set(answer.probabilities) != options or answer.choice not in options:
                raise ValueError(f"'{name}' was answered over {sorted(answer.probabilities)}, not its options {sorted(options)}")
        elif answer.type == "score":
            levels = [str(i) for i in range(len(question["criteria"]))]
            if set(answer.probabilities) != set(levels) or not 0 <= answer.score <= len(levels) - 1:
                raise ValueError(f"'{name}' was answered outside its {len(levels)} levels")


class ModelMismatchError(Exception):
    """The local server is serving a different checkpoint than the evaluator was configured for."""


class ResponseError(Exception):
    """The server answered, but the answer could not be read into the output schema."""


def _to_error(e: Exception) -> EvaluationError:
    """Every failure leaves the backend as an EvaluationError, like on the other backends."""
    cause, message = e, str(e)
    if isinstance(e, httpx.HTTPStatusError):
        code = str(e.response.status_code)
        try:
            message = str(e.response.json().get("detail", e.response.text))
        except (ValueError, AttributeError):
            message = e.response.text
    elif isinstance(e, ResponseError):
        code, cause = "INVALID_RESPONSE", e.__cause__ or e
    elif isinstance(e, ModelMismatchError):
        code = "MODEL_MISMATCH"
    elif isinstance(e, httpx.TransportError):
        code, message = "CONNECTION_ERROR", f"Could not reach the /v1/systemone server: {e}"
    elif isinstance(e, (ValueError, TypeError)):
        code = "INVALID_REQUEST"
    else:
        code = "UNKNOWN_ERROR"
    return EvaluationError(error_code=code, message=message, details={"exception_type": type(cause).__name__})
