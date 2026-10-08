import json

import httpx
import pytest
from pydantic import ValidationError

import grounded_ai.backends.jev as jev
from grounded_ai import AsyncEvaluator, Evaluator
from grounded_ai.backends.jev import (
    HALLUCINATION,
    RAG_RELEVANCE,
    TOXICITY,
    JevEvaluator,
    JevInput,
    JevOutput,
)
from grounded_ai.schemas import EvaluationError, EvaluationInput, EvaluationOutput

MODEL = "strands-decider-2B-hobson-v19"

# /health as `strands-decider serve StrandsAgents/strands-decider-2B-hobson-v19` reports it (trimmed).
HEALTH = {
    "status": "ok",
    "model": MODEL,
    "checkpoint": f"StrandsAgents/{MODEL}",
    "max_length": 100,
    "device": "mps",
}


def answer(question, p=0.9, pick=0):
    """What the server returns for one question: option number `pick` gets probability p."""
    if question["type"] == "noul":
        return {"type": "noul", "noul": p}
    choice = question["type"] == "choice"
    names = list(question["criteria"]) if choice else [str(i) for i in range(len(question["criteria"]))]
    rest = round((1 - p) / (len(names) - 1), 4)
    probabilities = {n: (p if i == pick else rest) for i, n in enumerate(names)}
    if choice:
        return {"type": "choice", "choice": names[pick], "probabilities": probabilities, "confidence": 0.8}
    legend = {str(i): level for i, level in enumerate(question["criteria"])}
    return {"type": "score", "score": float(pick), "legend": legend, "probabilities": probabilities, "confidence": 0.7}


def make_backend(seen=None, p=0.9, pick=0, health=HEALTH, health_calls=None, respond=None,
                 model_name=MODEL, async_=False, **kwargs):
    """Backend whose HTTP client plays the server: /health returns `health`, /v1/systemone answers
    every question asked. `seen` records request bodies; `respond` replaces the /v1/systemone reply."""

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            if health_calls is not None:
                health_calls.append(1)
            return httpx.Response(404) if health is None else httpx.Response(200, json=health)
        body = json.loads(request.content)
        if seen is not None:
            seen.append({"url": str(request.url), "headers": request.headers, "body": body})
        if respond is not None:
            return respond(body)
        answers = {name: answer(q, p, pick) for name, q in body["questions"].items()}
        return httpx.Response(200, json={
            "model": MODEL, "answers": answers, "usage": {"input_tokens": 42, "output_tokens": len(answers)},
            "latency_ms": 12.5,
        })

    transport = httpx.MockTransport(handler)
    client_kwargs = {"async_client": httpx.AsyncClient(transport=transport)} if async_ else {}
    return JevEvaluator(use_local_model=True, local_model=model_name, client=httpx.Client(transport=transport), **client_kwargs, **kwargs)


def ask(state="x", **fields):
    return JevInput(questions={"verdict": HALLUCINATION}, state=state, **fields)


class RagInput(JevInput):
    """A customized input: its own fields are what the model reads, in this order."""

    context: str
    response: str


class TestContract:
    """The request is the input's state and questions; the output is the server's answers."""

    def test_factory_routing(self):
        evaluator = Evaluator("jev", use_local_model=True, local_model=MODEL, base_url="http://127.0.0.1:8010/")
        assert isinstance(evaluator.backend, JevEvaluator)
        assert evaluator.backend.model_name == MODEL
        assert evaluator.backend.base_url == "http://127.0.0.1:8010"

    def test_request_is_the_input(self):
        seen = []
        backend = make_backend(seen, api_key="k")
        backend.evaluate(RagInput(
            context="London is in England.", response="London is in France.", questions={"verdict": HALLUCINATION}))

        (call,) = seen
        assert call["url"] == "http://127.0.0.1:8000/v1/systemone"
        assert call["headers"]["authorization"] == "Bearer k"
        assert call["body"] == {
            "state": {"context": "London is in England.", "response": "London is in France."},
            "model": MODEL,
            "questions": {"verdict": HALLUCINATION.model_dump()},
        }

    @pytest.mark.parametrize(
        "question,labels",
        [
            (HALLUCINATION, {"hallucination", "faithful"}),
            (TOXICITY, {"toxic", "non-toxic"}),
            (RAG_RELEVANCE, {"relevant", "unrelated"}),
        ],
    )
    def test_shipped_questions(self, question, labels):
        result = make_backend().evaluate(JevInput(state="x", questions={"verdict": question}))
        assert set(result.answers["verdict"].probabilities) == labels

    def test_rag_relevance_judges_a_retrieved_chunk_against_the_query(self):
        """RAG relevance is about retrieval: is the retrieved context relevant to the query?
        It is not about whether a generated response answers the query."""
        text = " ".join([str(RAG_RELEVANCE.instructions), *RAG_RELEVANCE.criteria.values()]).lower()
        assert "context" in text and "query" in text
        assert "response" not in text

    def test_output_contract_cannot_be_replaced(self):
        class Reshaped(JevOutput):
            answers: dict

        for schema in (EvaluationOutput, Reshaped):
            result = make_backend().evaluate(ask(), output_schema=schema)
            assert isinstance(result, EvaluationError)
            assert result.error_code == "INVALID_REQUEST"
            assert "JevOutput" in result.message
        with pytest.raises(TypeError):
            make_backend(output_schema=EvaluationOutput)

    def test_jev_classes_are_not_the_stock_ones(self):
        assert not issubclass(JevInput, EvaluationInput)
        assert not issubclass(JevOutput, EvaluationOutput)
        assert set(JevInput.model_fields) == {"questions", "state"}
        with pytest.raises(TypeError):
            make_backend(input_schema=EvaluationInput)

    def test_plain_evaluation_input_refused(self):
        result = make_backend().evaluate(EvaluationInput(response="x"))
        assert result.error_code == "INVALID_REQUEST"
        assert "questions" in result.message

    def test_no_system_prompt_or_generation_arguments(self):
        """A decision model has no system message and does not sample."""
        with pytest.raises(TypeError):
            make_backend(system_prompt="You are a strict auditor.")
        with pytest.raises(TypeError):
            make_backend(temperature=0.1)
        with pytest.raises(TypeError):
            make_backend(eval_mode="TOXICITY")
        seen = []
        result = make_backend(seen).evaluate(ask(), temperature=0.7)
        assert result.error_code == "INVALID_REQUEST"
        assert "temperature" in result.message
        assert seen == []


class TestEvaluatorKeywords:
    """Evaluator.evaluate() builds the JevInput from keywords. (How inputs map onto the request
    and how answers parse is covered in test_jev_contract.py.)"""

    def test_evaluator_takes_jev_fields_as_keywords(self):
        seen = []
        evaluator = Evaluator.__new__(Evaluator)
        evaluator.backend = make_backend(seen)
        result = evaluator.evaluate(
            state={"context": "London is in England.", "response": "London is in France."},
            questions={"verdict": HALLUCINATION})
        assert result.answers["verdict"].choice == "hallucination"

        evaluator.evaluate("Just the text.", questions={"tone": TOXICITY})  # a bare string is the state
        assert seen[1]["body"]["state"] == "Just the text."
        assert list(seen[1]["body"]["questions"]) == ["tone"]

        # The stock input fields are not Jev fields.
        result = evaluator.evaluate(response="x", questions={"tone": TOXICITY})
        assert result.error_code == "INVALID_REQUEST"
        assert "response" in result.message
        with pytest.raises(ValidationError):
            JevInput(response="x", questions={"tone": TOXICITY})

    def test_evaluator_with_a_custom_input_class(self):
        seen = []
        evaluator = Evaluator.__new__(Evaluator)
        evaluator.backend = make_backend(seen, input_schema=RagInput)
        evaluator.evaluate(context="c", response="r", questions={"verdict": HALLUCINATION})
        assert seen[0]["body"]["state"] == {"context": "c", "response": "r"}


class TestServer:
    def test_http_error(self):
        backend = make_backend(respond=lambda body: httpx.Response(422, json={"detail": "questions: field required"}))
        result = backend.evaluate(ask())
        assert isinstance(result, EvaluationError)
        assert result.error_code == "422"
        assert "field required" in result.message

    def test_model_mismatch_refused(self):
        """The server ignores the request's model field, so a wrong name must fail loudly."""
        seen = []
        backend = make_backend(seen, model_name="strands-decider-latest")
        result = backend.evaluate(ask())
        assert result.error_code == "MODEL_MISMATCH"
        assert f"StrandsAgents/{MODEL}" in result.message
        assert seen == []  # never reached /v1/systemone

    @pytest.mark.parametrize("name", [MODEL, f"StrandsAgents/{MODEL}"])
    def test_served_name_or_checkpoint_accepted(self, name):
        assert isinstance(make_backend(model_name=name).evaluate(ask()), JevOutput)

    def test_health_checked_once(self):
        calls = []
        backend = make_backend(health_calls=calls)
        backend.evaluate(ask())
        backend.evaluate(ask("y"))
        assert len(calls) == 1

    def test_server_without_health_is_not_blocked(self):
        backend = make_backend(health=None, model_name="anything")
        assert isinstance(backend.evaluate(ask()), JevOutput)

    def test_long_input_warns(self):
        with pytest.warns(UserWarning, match="truncates"):  # /health says max_length=100
            make_backend().evaluate(ask("word " * 200))

    def test_connection_error(self):
        backend = JevEvaluator(use_local_model=True, local_model=MODEL, base_url="http://127.0.0.1:9")
        result = backend.evaluate(ask())
        assert result.error_code == "CONNECTION_ERROR"

    @pytest.mark.asyncio
    async def test_async(self):
        evaluator = AsyncEvaluator.__new__(AsyncEvaluator)
        evaluator.backend = make_backend(async_=True)
        result = await evaluator.evaluate(
            state={"context": "London is in England.", "response": "London is in France."},
            questions={"verdict": HALLUCINATION})
        assert result.answers["verdict"].choice == "hallucination"

    @pytest.mark.asyncio
    async def test_async_model_mismatch(self):
        backend = make_backend(async_=True, model_name="strands-decider-latest")
        assert (await backend.evaluate_async(ask())).error_code == "MODEL_MISMATCH"


class FakeServer:
    """Stands in for the `strands-decider serve` process."""

    def __init__(self, command, exits_with=None):
        self.command, self.returncode, self.terminated = command, exits_with, False

    def poll(self):
        return self.returncode

    def terminate(self):
        self.terminated, self.returncode = True, 0

    def wait(self, timeout=None):
        return self.returncode


class TestWarmup:
    def _backend(self, up, model_name=f"StrandsAgents/{MODEL}"):
        """`up` holds the ports with a server answering /health."""

        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.port not in up:
                raise httpx.ConnectError("refused", request=request)
            if request.url.path == "/health":
                return httpx.Response(200, json=HEALTH)
            body = json.loads(request.content)
            return httpx.Response(200, json={"answers": {n: answer(q) for n, q in body["questions"].items()}})

        return JevEvaluator(use_local_model=True, local_model=model_name, client=httpx.Client(transport=httpx.MockTransport(handler)))

    def test_starts_the_server_on_the_port_and_waits(self, monkeypatch):
        up, started = set(), []

        def popen(command, **kwargs):
            started.append(FakeServer(command))
            up.add(8123)  # ready by the next health check
            return started[-1]

        monkeypatch.setattr(jev.shutil, "which", lambda name: "/usr/bin/strands-decider")
        monkeypatch.setattr(jev.subprocess, "Popen", popen)
        monkeypatch.setattr(jev.time, "sleep", lambda s: None)

        backend = self._backend(up)
        assert backend.warmup(port=8123, device="cpu") is backend
        assert started[0].command == [
            "/usr/bin/strands-decider", "serve", f"StrandsAgents/{MODEL}", "--port", "8123", "--device", "cpu"]
        assert backend.base_url == "http://127.0.0.1:8123"
        assert isinstance(backend.evaluate(ask()), JevOutput)

        backend.shutdown()
        assert started[0].terminated

    def test_reuses_a_server_already_on_the_port(self, monkeypatch):
        monkeypatch.setattr(jev.subprocess, "Popen", lambda command, **kw: pytest.fail("should not start a server"))
        backend = self._backend({8000})
        backend.warmup(port=8000)
        assert backend.base_url == "http://127.0.0.1:8000"

    def test_checkpoint_argument_overrides_the_model_name(self, monkeypatch):
        up, started = set(), []
        monkeypatch.setattr(jev.shutil, "which", lambda name: "strands-decider")
        monkeypatch.setattr(jev.subprocess, "Popen", lambda c, **kw: (started.append(c), up.add(8000), FakeServer(c))[2])
        monkeypatch.setattr(jev.time, "sleep", lambda s: None)
        self._backend(up, model_name=MODEL).warmup(checkpoint="/models/hobson")
        assert started[0][:3] == ["strands-decider", "serve", "/models/hobson"]

    def test_server_that_exits_is_reported_with_its_log(self, monkeypatch):
        def popen(command, stdout=None, **kwargs):
            stdout.write(b"OSError: checkpoint not found\n")  # what the server printed before dying
            return FakeServer(command, exits_with=1)

        monkeypatch.setattr(jev.shutil, "which", lambda name: "strands-decider")
        monkeypatch.setattr(jev.subprocess, "Popen", popen)
        with pytest.raises(RuntimeError, match="exited with code 1(.|\n)*checkpoint not found"):
            self._backend(set()).warmup(port=8000)

    def test_server_output_is_kept_out_of_the_way_unless_verbose(self, monkeypatch):
        up, seen = set(), []

        def popen(command, **kwargs):
            seen.append(kwargs)
            up.add(8000)
            return FakeServer(command)

        monkeypatch.setattr(jev.shutil, "which", lambda name: "strands-decider")
        monkeypatch.setattr(jev.subprocess, "Popen", popen)
        monkeypatch.setattr(jev.time, "sleep", lambda s: None)
        self._backend(up).warmup(port=8000)
        up.clear()
        self._backend(up).warmup(port=8000, verbose=True)
        assert seen[0]["stdout"] is not None and seen[0]["stderr"] is jev.subprocess.STDOUT
        assert seen[1] == {}  # verbose: the server writes straight to the terminal

    def test_missing_server_package(self, monkeypatch):
        monkeypatch.setattr(jev.shutil, "which", lambda name: None)
        with pytest.raises(ImportError, match="grounded-ai\\[jev-local\\]"):
            self._backend(set()).warmup(port=8000)

    def test_times_out(self, monkeypatch):
        started = []
        monkeypatch.setattr(jev.shutil, "which", lambda name: "strands-decider")
        monkeypatch.setattr(jev.subprocess, "Popen", lambda c, **kw: (started.append(FakeServer(c)), started[-1])[1])
        monkeypatch.setattr(jev.time, "sleep", lambda s: None)
        with pytest.raises(TimeoutError):
            self._backend(set()).warmup(port=8000, timeout=0.05)
        assert started[0].terminated



# --- Hosted Jev (the default) ----------------------------------------------------------------------


def hosted(responses, seen=None, async_=False, **kwargs):
    """A hosted JevEvaluator whose HTTP client replays `responses` (status, json) in order."""
    queue = list(responses)

    def handler(request: httpx.Request) -> httpx.Response:
        if seen is not None:
            seen.append(request)
        status, body = queue.pop(0)
        headers = body.pop("_headers", {}) if isinstance(body, dict) else {}
        return httpx.Response(status, json=body, headers=headers)

    transport = httpx.MockTransport(handler)
    clients = {"client": httpx.Client(transport=transport)}
    if async_:
        clients["async_client"] = httpx.AsyncClient(transport=transport)
    return JevEvaluator(api_key=kwargs.pop("api_key", "ts-key"), **clients, **kwargs)


URGENT = {"is_urgent": {"type": "noul", "instructions": "Does this convey urgency?"}}
OK = (200, {"model": "jev-1.13.0", "answers": {"is_urgent": {"type": "noul", "noul": 0.95}},
            "usage": {"input_tokens": 296, "output_tokens": 20}})


class TestHosted:
    def test_calls_typesafe_with_the_model_and_a_bearer_key(self):
        seen = []
        result = hosted([OK], seen).evaluate(JevInput(state="Help! Payouts failing.", questions=URGENT))
        assert isinstance(result, JevOutput) and result.answers["is_urgent"].noul == 0.95
        assert result.model == "jev-1.13.0"
        (request,) = seen  # no /health call: the hosted API resolves model names itself
        assert str(request.url) == "https://api.typesafe.ai/v1/systemone"
        assert request.headers["authorization"] == "Bearer ts-key"
        assert json.loads(request.content) == {
            "state": "Help! Payouts failing.", "model": "jev-latest", "questions": URGENT}

    def test_factory_routes_jev_to_hosted(self, monkeypatch):
        monkeypatch.setenv("TYPESAFE_API_KEY", "env-key")
        assert Evaluator("jev").backend.model_name == "jev-latest"
        backend = Evaluator("jev/jev-1.13.0").backend
        assert (backend.model_name, backend.use_local_model) == ("jev-1.13.0", False)
        assert backend._headers["Authorization"] == "Bearer env-key"

    def test_base_url_from_the_environment(self, monkeypatch):
        monkeypatch.setenv("TYPESAFE_API_BASE", "http://proxy.local/typesafe/")
        assert JevEvaluator(api_key="k").base_url == "http://proxy.local/typesafe"

    def test_a_missing_key_is_an_error_up_front(self, monkeypatch):
        monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
        with pytest.raises(ValueError, match="TYPESAFE_API_KEY.*use_local_model=True"):
            JevEvaluator()

    def test_local_mode_needs_no_key(self, monkeypatch):
        monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
        local = Evaluator("jev", use_local_model=True).backend
        assert (local.model_name, local.base_url) == (jev.DEFAULT_LOCAL_MODEL, "http://127.0.0.1:8000")

    def test_warmup_is_for_local_mode_only(self):
        with pytest.raises(ValueError, match="use_local_model=True"):
            hosted([]).warmup(port=8000)

    @pytest.mark.parametrize("status", [429, 529])
    def test_rate_limits_and_overload_are_retried(self, monkeypatch, status):
        waits = []
        monkeypatch.setattr(jev.time, "sleep", waits.append)
        busy = (status, {"detail": "slow down", "_headers": {"retry-after": "2"}})
        result = hosted([busy, (status, {"detail": "slow down"}), OK]).evaluate(
            JevInput(state="x", questions=URGENT))
        assert isinstance(result, JevOutput)
        assert waits == [2.0, 1.0]  # retry-after when given, else backoff

    def test_gives_up_after_max_retries(self, monkeypatch):
        monkeypatch.setattr(jev.time, "sleep", lambda s: None)
        busy = (429, {"detail": "Rate limit exceeded"})
        seen = []
        result = hosted([busy] * 3, seen, max_retries=2).evaluate(JevInput(state="x", questions=URGENT))
        assert (result.error_code, result.message, len(seen)) == ("429", "Rate limit exceeded", 3)

    def test_other_errors_are_not_retried(self):
        seen = []
        result = hosted([(401, {"detail": "Invalid API key"})], seen).evaluate(JevInput(state="x", questions=URGENT))
        assert (result.error_code, len(seen)) == ("401", 1)

    @pytest.mark.asyncio
    async def test_async_retries_too(self, monkeypatch):
        waits = []

        async def sleep(s):
            waits.append(s)

        monkeypatch.setattr(jev.asyncio, "sleep", sleep)
        result = await hosted([(529, {"detail": "overloaded"}), OK], async_=True).evaluate_async(
            JevInput(state="x", questions=URGENT))
        assert isinstance(result, JevOutput) and waits == [0.5]

