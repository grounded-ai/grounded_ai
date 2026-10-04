import json

import httpx
import pytest
from pydantic import ValidationError

import grounded_ai.backends.decider as decider
from grounded_ai import AsyncEvaluator, Evaluator
from grounded_ai.backends.decider import (
    HALLUCINATION,
    RAG_RELEVANCE,
    TOXICITY,
    DeciderBackend,
    DeciderInput,
    DeciderOutput,
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
    return DeciderBackend(model_name=model_name, client=httpx.Client(transport=transport), **client_kwargs, **kwargs)


def ask(state="x", **fields):
    return DeciderInput(questions={"verdict": HALLUCINATION}, state=state, **fields)


class RagInput(DeciderInput):
    """A customized input: its own fields are what the model reads, in this order."""

    context: str
    response: str


class TestContract:
    """The request is the input's state and questions; the output is the server's answers."""

    def test_factory_routing(self):
        evaluator = Evaluator(f"decider/{MODEL}", base_url="http://127.0.0.1:8010/")
        assert isinstance(evaluator.backend, DeciderBackend)
        assert evaluator.backend.model_name == MODEL
        assert evaluator.backend.base_url == "http://127.0.0.1:8010"
        assert Evaluator("decider/my-org/decider/ckpt").backend.model_name == "my-org/decider/ckpt"

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
        result = make_backend().evaluate(DeciderInput(state="x", questions={"verdict": question}))
        assert set(result.answers["verdict"].probabilities) == labels

    def test_rag_relevance_judges_a_retrieved_chunk_against_the_query(self):
        """RAG relevance is about retrieval: is the retrieved context relevant to the query?
        It is not about whether a generated response answers the query."""
        text = " ".join([str(RAG_RELEVANCE.instructions), *RAG_RELEVANCE.criteria.values()]).lower()
        assert "context" in text and "query" in text
        assert "response" not in text

    def test_output_contract_cannot_be_replaced(self):
        class Reshaped(DeciderOutput):
            answers: dict

        for schema in (EvaluationOutput, Reshaped):
            result = make_backend().evaluate(ask(), output_schema=schema)
            assert isinstance(result, EvaluationError)
            assert result.error_code == "INVALID_REQUEST"
            assert "DeciderOutput" in result.message
        with pytest.raises(TypeError):
            make_backend(output_schema=EvaluationOutput)

    def test_decider_classes_are_not_the_stock_ones(self):
        assert not issubclass(DeciderInput, EvaluationInput)
        assert not issubclass(DeciderOutput, EvaluationOutput)
        assert set(DeciderInput.model_fields) == {"questions", "state"}
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
    """Evaluator.evaluate() builds the DeciderInput from keywords. (How inputs map onto the request
    and how answers parse is covered in test_decider_contract.py.)"""

    def test_evaluator_takes_decider_fields_as_keywords(self):
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

        # The stock input fields are not Decider fields.
        result = evaluator.evaluate(response="x", questions={"tone": TOXICITY})
        assert result.error_code == "INVALID_REQUEST"
        assert "response" in result.message
        with pytest.raises(ValidationError):
            DeciderInput(response="x", questions={"tone": TOXICITY})

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
        assert isinstance(make_backend(model_name=name).evaluate(ask()), DeciderOutput)

    def test_health_checked_once(self):
        calls = []
        backend = make_backend(health_calls=calls)
        backend.evaluate(ask())
        backend.evaluate(ask("y"))
        assert len(calls) == 1

    def test_server_without_health_is_not_blocked(self):
        backend = make_backend(health=None, model_name="anything")
        assert isinstance(backend.evaluate(ask()), DeciderOutput)

    def test_long_input_warns(self):
        with pytest.warns(UserWarning, match="truncates"):  # /health says max_length=100
            make_backend().evaluate(ask("word " * 200))

    def test_connection_error(self):
        backend = DeciderBackend(model_name=MODEL, base_url="http://127.0.0.1:9")
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

        return DeciderBackend(model_name=model_name, client=httpx.Client(transport=httpx.MockTransport(handler)))

    def test_starts_the_server_on_the_port_and_waits(self, monkeypatch):
        up, started = set(), []

        def popen(command, **kwargs):
            started.append(FakeServer(command))
            up.add(8123)  # ready by the next health check
            return started[-1]

        monkeypatch.setattr(decider.shutil, "which", lambda name: "/usr/bin/strands-decider")
        monkeypatch.setattr(decider.subprocess, "Popen", popen)
        monkeypatch.setattr(decider.time, "sleep", lambda s: None)

        backend = self._backend(up)
        assert backend.warmup(port=8123, device="cpu") is backend
        assert started[0].command == [
            "/usr/bin/strands-decider", "serve", f"StrandsAgents/{MODEL}", "--port", "8123", "--device", "cpu"]
        assert backend.base_url == "http://127.0.0.1:8123"
        assert isinstance(backend.evaluate(ask()), DeciderOutput)

        backend.shutdown()
        assert started[0].terminated

    def test_reuses_a_server_already_on_the_port(self, monkeypatch):
        monkeypatch.setattr(decider.subprocess, "Popen", lambda command, **kw: pytest.fail("should not start a server"))
        backend = self._backend({8000})
        backend.warmup(port=8000)
        assert backend.base_url == "http://127.0.0.1:8000"

    def test_checkpoint_argument_overrides_the_model_name(self, monkeypatch):
        up, started = set(), []
        monkeypatch.setattr(decider.shutil, "which", lambda name: "strands-decider")
        monkeypatch.setattr(decider.subprocess, "Popen", lambda c, **kw: (started.append(c), up.add(8000), FakeServer(c))[2])
        monkeypatch.setattr(decider.time, "sleep", lambda s: None)
        self._backend(up, model_name=MODEL).warmup(checkpoint="/models/hobson")
        assert started[0][:3] == ["strands-decider", "serve", "/models/hobson"]

    def test_server_that_exits_is_reported_with_its_log(self, monkeypatch):
        def popen(command, stdout=None, **kwargs):
            stdout.write(b"OSError: checkpoint not found\n")  # what the server printed before dying
            return FakeServer(command, exits_with=1)

        monkeypatch.setattr(decider.shutil, "which", lambda name: "strands-decider")
        monkeypatch.setattr(decider.subprocess, "Popen", popen)
        with pytest.raises(RuntimeError, match="exited with code 1(.|\n)*checkpoint not found"):
            self._backend(set()).warmup(port=8000)

    def test_server_output_is_kept_out_of_the_way_unless_verbose(self, monkeypatch):
        up, seen = set(), []

        def popen(command, **kwargs):
            seen.append(kwargs)
            up.add(8000)
            return FakeServer(command)

        monkeypatch.setattr(decider.shutil, "which", lambda name: "strands-decider")
        monkeypatch.setattr(decider.subprocess, "Popen", popen)
        monkeypatch.setattr(decider.time, "sleep", lambda s: None)
        self._backend(up).warmup(port=8000)
        up.clear()
        self._backend(up).warmup(port=8000, verbose=True)
        assert seen[0]["stdout"] is not None and seen[0]["stderr"] is decider.subprocess.STDOUT
        assert seen[1] == {}  # verbose: the server writes straight to the terminal

    def test_missing_server_package(self, monkeypatch):
        monkeypatch.setattr(decider.shutil, "which", lambda name: None)
        with pytest.raises(ImportError, match="grounded-ai\\[decider\\]"):
            self._backend(set()).warmup(port=8000)

    def test_times_out(self, monkeypatch):
        started = []
        monkeypatch.setattr(decider.shutil, "which", lambda name: "strands-decider")
        monkeypatch.setattr(decider.subprocess, "Popen", lambda c, **kw: (started.append(FakeServer(c)), started[-1])[1])
        monkeypatch.setattr(decider.time, "sleep", lambda s: None)
        with pytest.raises(TimeoutError):
            self._backend(set()).warmup(port=8000, timeout=0.05)
        assert started[0].terminated

