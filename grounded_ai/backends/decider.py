import atexit
import json
import os
import shutil
import subprocess
import time
import warnings
from typing import Any, Dict, List, Literal, Optional, Type, Union

from pydantic import BaseModel, Field, field_validator
from typing_extensions import Annotated

from ..base import BaseEvaluator
from ..schemas import EvaluationError, EvaluationInput

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

    type: Literal["choice"] = "choice"
    instructions: Content
    criteria: Dict[str, str] = Field(min_length=2, max_length=255)


class ScoreQuestion(BaseModel):
    """Rate against ordered levels. `criteria` lists them lowest first."""

    type: Literal["score"] = "score"
    instructions: Content
    criteria: List[str] = Field(min_length=2, max_length=10)


Question = Annotated[Union[NoulQuestion, ChoiceQuestion, ScoreQuestion], Field(discriminator="type")]


class NoulAnswer(BaseModel):
    type: Literal["noul"] = "noul"
    noul: float = Field(description="Probability the statement is true")


class ChoiceAnswer(BaseModel):
    type: Literal["choice"] = "choice"
    choice: str = Field(description="The option the model picked")
    probabilities: Dict[str, float] = Field(description="Probability of each option")
    confidence: float = Field(description="How concentrated the distribution is: 0 uniform, 1 certain")


class ScoreAnswer(BaseModel):
    type: Literal["score"] = "score"
    score: float = Field(description="Expected level index, counting from 0 at the lowest level")
    legend: Dict[str, str] = Field(default_factory=dict, description="Level index -> the level it stands for")
    probabilities: Dict[str, float] = Field(description="Probability of each level index")
    confidence: float = Field(description="How tightly the mass clusters on the scale")


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

# State field order: the model reads the evidence before the text being judged.
_STATE_ORDER = ("context", "query", "response")
_NOT_STATE = {"questions", "state", "base_template", "formatted_prompt"}


class DeciderInput(EvaluationInput):
    """
    Input for the Decider backend: the two things a `/v1/systemone` request is made of.

    - `questions`: what the model is asked, by name. Sent as is.
    - `state`: what the model reads. Sent as is when set. When left unset it is built from the
      other fields: `response`, `query`, `context` and any fields a subclass adds are sent as a
      JSON object, or, with a custom `base_template`, the rendered template is sent as text.

    Subclass it to add your own fields or a default template. The questions and the answers
    they produce are the model's contract and stay as they are.
    """

    questions: Dict[str, Question] = Field(min_length=1)
    state: Optional[Content] = None

    def request_state(self) -> Content:
        """The `state` of the request."""
        if self.state is not None:
            if isinstance(self.state, str) and not self.state.strip():
                raise ValueError("state is empty: there is nothing to evaluate.")
            return self.state
        stock_prompt = type(self).formatted_prompt is EvaluationInput.formatted_prompt
        stock_template = self.base_template == EvaluationInput.model_fields["base_template"].default
        if not (stock_prompt and stock_template):
            text = self.formatted_prompt
            if not isinstance(text, str) or not text.strip():
                raise ValueError("The rendered template is empty: there is nothing to evaluate.")
            return text
        data = self.model_dump(mode="json", exclude=_NOT_STATE, exclude_none=True)
        if not data:
            raise ValueError("The input has no fields set: there is nothing to evaluate.")
        ordered = {k: data.pop(k) for k in _STATE_ORDER if k in data}
        return {**ordered, **data}


class DeciderOutput(BaseModel):
    """What `/v1/systemone` returns: one answer per question, under the question's name."""

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
    Both mirror the model's contract. Customize the input by subclassing DeciderInput.

    The server, not the request, decides which model answers: `strands-decider serve <checkpoint>`
    loads one checkpoint and ignores the request's `model` field. `model_name` therefore names the
    checkpoint you expect, and the first call checks it against the server's `/health`, so
    `Evaluator("decider/<name>")` cannot silently run a different model.

    `warmup(port=...)` starts that server for you.
    """

    def __init__(
        self,
        model_name: str,
        base_url: str = None,
        api_key: str = None,
        timeout: float = 30.0,
        client: Optional[Any] = None,
        async_client: Optional[Any] = None,
        input_schema: Type[BaseModel] = DeciderInput,
        output_schema: Type[BaseModel] = DeciderOutput,
    ):
        super().__init__(input_schema=input_schema, output_schema=output_schema)
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
        self._server = subprocess.Popen(command)
        atexit.register(self.shutdown)

        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self._server.poll() is not None:
                code, self._server = self._server.returncode, None
                raise RuntimeError(f"`{' '.join(command)}` exited with code {code} before it was ready.")
            if self._healthy():
                return self
            time.sleep(1.0)
        self.shutdown()
        raise TimeoutError(f"The strands-decider server was not ready on port {port} after {timeout:.0f}s.")

    def shutdown(self) -> None:
        """Stop the server that warmup() started. Does nothing if warmup() did not start one."""
        server, self._server = self._server, None
        if server is not None and server.poll() is None:
            server.terminate()
            try:
                server.wait(timeout=10)
            except subprocess.TimeoutExpired:
                server.kill()

    def _healthy(self) -> bool:
        try:
            return self.client.get(f"{self.base_url}/health", headers=self._headers, timeout=2.0).status_code == 200
        except httpx.TransportError:
            return False

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
                f"DeciderBackend takes no runtime arguments, got {sorted(kwargs)}. A decision model "
                "has no system message and does not sample; what it is asked comes from `questions`."
            )
        if not isinstance(input_data, DeciderInput):
            raise ValueError(
                f"The Decider backend needs a DeciderInput (or a subclass), got {type(input_data).__name__}. "
                "It carries the `questions` the model is asked."
            )
        if not (isinstance(output_schema, type) and issubclass(output_schema, DeciderOutput)):
            raise ValueError(
                "The Decider backend returns a DeciderOutput: its answers are the model's contract and "
                f"cannot be reshaped into {getattr(output_schema, '__name__', output_schema)}. "
                "Customize the input (DeciderInput) instead."
            )
        return {
            "state": input_data.request_state(),
            "model": self.model_name,
            "questions": {name: q.model_dump(exclude_none=True) for name, q in input_data.questions.items()},
        }

    def _read(self, response: Any, output_schema: Type[BaseModel]) -> BaseModel:
        response.raise_for_status()
        try:
            return output_schema(**response.json())
        except (KeyError, TypeError, ValueError) as e:
            raise ResponseError(f"Could not read the server's answer into {output_schema.__name__}: {e!r}") from e

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
            return self._read(response, output_schema)
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
            return self._read(response, output_schema)
        except Exception as e:
            return _to_error(e)


class ModelMismatchError(Exception):
    """The server is serving a different checkpoint than the evaluator was configured for."""


class ResponseError(Exception):
    """The server answered, but the answer could not be read into the output schema."""


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
