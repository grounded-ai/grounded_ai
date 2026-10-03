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
    # Optional {"true": "...", "false": "..."} descriptions that sharpen the boundary.
    criteria: Optional[Dict[str, str]] = None

    @field_validator("criteria")
    @classmethod
    def _keys(cls, v):
        if v is not None and not set(v) <= {"true", "false"}:
            raise ValueError("noul criteria keys must be 'true' and/or 'false'")
        return v


class ChoiceQuestion(BaseModel):
    """Pick one of the named options. `criteria` maps option name -> description."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["choice"] = "choice"
    instructions: Content
    criteria: Dict[str, str] = Field(min_length=2, max_length=255)


class ScoreQuestion(BaseModel):
    """Rate against ordered levels. `criteria` lists them lowest first."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["score"] = "score"
    instructions: Content
    criteria: List[str] = Field(min_length=2, max_length=10)


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
    legend: Dict[str, str] = Field(default_factory=dict, description="Level index -> the level it stands for")
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
RAG_RELEVANCE = ChoiceQuestion(
    instructions="Does the response help answer the query?",
    criteria={
        "relevant": "the response addresses what the query asks",
        "unrelated": "the response is about something else",
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
# These are the Decider backend's own input and output. They are not EvaluationInput and
# EvaluationOutput: those describe a text-generating judge (response/query/context, a prompt
# template, a free-form label and reasoning), which is not this model's contract.


class DeciderInput(BaseModel):
    """
    Input for the Decider backend: the two things a `/v1/systemone` request is made of.

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
        own_fields = set(type(self).model_fields) - set(DeciderInput.model_fields)
        data = self.model_dump(mode="json", include=own_fields, exclude_none=True)  # in declared order
        if not data:
            raise ValueError(
                "Nothing to evaluate: set `state`, or subclass DeciderInput and fill in your own fields."
            )
        return data


class DeciderOutput(BaseModel):
    """What `/v1/systemone` returns: one answer per question, under the question's name.

    This is the model's contract and the only thing the Decider backend returns."""

    answers: Dict[str, Answer]
    model: Optional[str] = Field(None, description="The model name the server reports")
    usage: Optional[Dict[str, int]] = None
    latency_ms: Optional[float] = None


class DeciderBackend(BaseEvaluator):
    """
    Backend for Strands Decider (`strands-decider serve`), a local decision model that
    serves the `POST /v1/systemone` contract. Any server with that contract works the same.

    These models do not generate text. Each question is answered with probabilities read off
    the model, so the output cannot leave the schema and `confidence` is measured, not
    self-reported. There is no system message and nothing is sampled, so this backend takes
    no `system_prompt`, `temperature` or other generation arguments.

    The input is a DeciderInput (state + questions) and the output is a DeciderOutput (answers).
    They are this backend's own classes, not EvaluationInput/EvaluationOutput, and they mirror
    the model's contract. Customize the input by subclassing DeciderInput; the contract itself
    cannot be changed, and every request is validated against it before it is sent.

    The server, not the request, decides which model answers: `strands-decider serve <checkpoint>`
    loads one checkpoint and ignores the request's `model` field. `model_name` therefore names the
    checkpoint you expect, and the first call checks it against the server's `/health`, so
    `Evaluator("decider/<name>")` cannot silently run a different model.

    `warmup(port=...)` starts that server for you.
    """

    # Evaluator("decider/...").evaluate("some text", questions=...) puts the text here.
    primary_input_field = "state"

    def __init__(
        self,
        model_name: str,
        base_url: str = None,
        api_key: str = None,
        timeout: float = 30.0,
        client: Optional[Any] = None,
        async_client: Optional[Any] = None,
        input_schema: Type[DeciderInput] = DeciderInput,
    ):
        if not (isinstance(input_schema, type) and issubclass(input_schema, DeciderInput)):
            raise TypeError("input_schema must be DeciderInput or a subclass of it.")
        super().__init__(input_schema=input_schema, output_schema=DeciderOutput)
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
        self._timeout = timeout
        self.client = client or httpx.Client(timeout=timeout)
        self._async_client = async_client
        self._model_checked = False
        self._max_length = None  # the served model's token window, from /health
        self._server = None  # the process warmup() started, if any
        self._server_log = None  # where that process's output goes

    @property
    def async_client(self):
        if self._async_client is None:
            self._async_client = httpx.AsyncClient(timeout=self._timeout)
        return self._async_client

    # --- Server lifecycle ---

    def warmup(
        self,
        port: int = 8000,
        checkpoint: str = None,
        device: str = None,
        timeout: float = 600.0,
        verbose: bool = False,
    ) -> "DeciderBackend":
        """
        Start `strands-decider serve` on `port` and wait until it is ready, so the evaluator
        can be used right away. Points this backend at http://127.0.0.1:<port>.

        If a server is already answering on that port it is used as is. The server this starts
        is stopped by `shutdown()` or when the Python process exits.

        Args:
            port: Local port to serve on.
            checkpoint: Checkpoint to load (a Hugging Face repo id or a local path). Defaults to
                `model_name`, so use the full repo id there: "decider/StrandsAgents/<name>".
            device: Torch device (cuda, mps or cpu). Auto-detected by the server when omitted.
            timeout: Seconds to wait for the server; the first start downloads the checkpoint.
            verbose: Show the server's own output. By default it goes to a log file, and the
                end of that log is included in the error if the server fails to start.
        """
        self.base_url = f"http://127.0.0.1:{port}"
        self._model_checked = False
        if self._healthy():
            return self

        executable = shutil.which("strands-decider")
        if executable is None:
            raise ImportError(
                "The strands-decider server is not installed. Please install it via "
                "`pip install grounded-ai[decider]` (Python >= 3.10)."
            )
        command = [executable, "serve", checkpoint or self.model_name, "--port", str(port)]
        if device:
            command += ["--device", device]
        if self._server is None and self._server_log is None:
            atexit.register(self.shutdown)  # once per backend
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

    # --- Model check ---

    def _check_model(self, response: Any) -> None:
        """Compare model_name with what /health says the server is running. Runs once."""
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
            if names and self.model_name not in names:
                raise ModelMismatchError(
                    f"Server at {self.base_url} is running '{checkpoint or served}', "
                    f"but the evaluator asked for '{self.model_name}'. "
                    f"Use Evaluator(\"decider/{served or checkpoint}\") or serve the checkpoint you want."
                )
            if isinstance(health.get("max_length"), int):
                self._max_length = health["max_length"]
        self._model_checked = True

    def _warn_if_long(self, state: Content) -> None:
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

    # --- Request and response ---

    def _request(self, input_data: BaseModel, output_schema: Type[BaseModel], kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """The request body, taken straight from the input: its state and its questions."""
        if kwargs:
            raise ValueError(
                f"DeciderBackend does not take {sorted(kwargs)}. Its input is `questions` and `state` "
                f"(plus the fields of {self.input_schema.__name__}); a decision model has no system "
                "message and does not sample, so there are no generation arguments."
            )
        if not isinstance(input_data, DeciderInput):
            raise ValueError(
                f"The Decider backend needs a DeciderInput (or a subclass), got {type(input_data).__name__}. "
                "It carries the `questions` the model is asked."
            )
        if output_schema is not DeciderOutput:
            raise ValueError(
                "The Decider backend returns a DeciderOutput: its answers are the model's contract and "
                f"cannot be reshaped into {getattr(output_schema, '__name__', output_schema)}. "
                "Customize the input (DeciderInput) instead."
            )
        # Dumped and validated again on purpose: a subclass may have loosened `questions`.
        questions = input_data.model_dump(include={"questions"}, exclude_none=True)["questions"]
        try:
            request = SystemOneRequest(state=input_data.build_state(), questions=questions, model=self.model_name)
        except ValidationError as e:
            raise ValueError(f"The request does not match the /v1/systemone contract: {e}") from e
        return request.model_dump(exclude_none=True)

    def _read(self, response: Any, questions: Dict[str, Any]) -> DeciderOutput:
        response.raise_for_status()
        try:
            output = DeciderOutput(**response.json())
            _check_answers(output.answers, questions)
            return output
        except (KeyError, TypeError, ValueError) as e:
            raise ResponseError(f"Could not read the server's answer into DeciderOutput: {e!r}") from e

    def _call_backend(
        self, input_data: BaseModel, output_schema: Type[BaseModel], **kwargs
    ) -> Union[BaseModel, EvaluationError]:
        try:
            body = self._request(input_data, output_schema, kwargs)
            if not self._model_checked:
                self._check_model(self.client.get(f"{self.base_url}/health", headers=self._headers))
            self._warn_if_long(body["state"])
            response = self.client.post(
                f"{self.base_url}/v1/systemone", json=body, headers=self._headers
            )
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
            response = await self.async_client.post(
                f"{self.base_url}/v1/systemone", json=body, headers=self._headers
            )
            return self._read(response, body["questions"])
        except Exception as e:
            return _to_error(e)


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
    """The server is serving a different checkpoint than the evaluator was configured for."""


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
