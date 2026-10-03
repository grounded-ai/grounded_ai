import json

import httpx
import pytest
from pydantic import ValidationError

from grounded_ai import AsyncEvaluator, Evaluator
from grounded_ai.backends.decider import (
    HALLUCINATION,
    RAG_RELEVANCE,
    TOXICITY,
    ChoiceAnswer,
    ChoiceQuestion,
    DeciderBackend,
    DeciderInput,
    DeciderOutput,
    NoulAnswer,
    NoulQuestion,
    ScoreAnswer,
    ScoreQuestion,
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


def ask(**fields):
    return DeciderInput(questions={"verdict": HALLUCINATION}, **fields)


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
        backend.evaluate(ask(response="London is in France.", context="London is in England."))

        (call,) = seen
        assert call["url"] == "http://127.0.0.1:8000/v1/systemone"
        assert call["headers"]["authorization"] == "Bearer k"
        assert call["body"] == {
            # Context first: the model reads the evidence before the text it judges.
            "state": {"context": "London is in England.", "response": "London is in France."},
            "model": MODEL,
            "questions": {"verdict": HALLUCINATION.model_dump()},
        }

    def test_all_three_question_types_round_trip(self):
        questions = {
            "risky": NoulQuestion(instructions="Is this a security risk?"),
            "severity": ChoiceQuestion(
                instructions="How severe is it?", criteria={"low": "cosmetic", "medium": "", "high": "data loss"}
            ),
            "complexity": ScoreQuestion(instructions="Rate the complexity.", criteria=["trivial", "simple", "hard"]),
        }
        seen = []
        result = make_backend(seen, p=0.7, pick=2).evaluate(DeciderInput(response="x", questions=questions))

        sent = seen[0]["body"]["questions"]
        assert sent["risky"] == {"type": "noul", "instructions": "Is this a security risk?"}
        assert sent["severity"]["criteria"] == {"low": "cosmetic", "medium": "", "high": "data loss"}
        assert sent["complexity"] == {
            "type": "score", "instructions": "Rate the complexity.", "criteria": ["trivial", "simple", "hard"]}

        assert isinstance(result, DeciderOutput)
        assert result.answers["risky"] == NoulAnswer(noul=0.7)
        assert result.answers["severity"] == ChoiceAnswer(
            choice="high", probabilities={"low": 0.15, "medium": 0.15, "high": 0.7}, confidence=0.8)
        assert result.answers["complexity"] == ScoreAnswer(
            score=2.0, legend={"0": "trivial", "1": "simple", "2": "hard"},
            probabilities={"0": 0.15, "1": 0.15, "2": 0.7}, confidence=0.7)
        assert (result.model, result.usage, result.latency_ms) == (
            MODEL, {"input_tokens": 42, "output_tokens": 3}, 12.5)

    @pytest.mark.parametrize(
        "question,labels",
        [
            (HALLUCINATION, {"hallucination", "faithful"}),
            (TOXICITY, {"toxic", "non-toxic"}),
            (RAG_RELEVANCE, {"relevant", "unrelated"}),
        ],
    )
    def test_shipped_questions(self, question, labels):
        result = make_backend().evaluate(DeciderInput(response="x", questions={"verdict": question}))
        assert set(result.answers["verdict"].probabilities) == labels

    def test_questions_are_validated_like_the_server_does(self):
        with pytest.raises(ValidationError):
            DeciderInput(response="x", questions={})  # at least one question
        with pytest.raises(ValidationError):
            ChoiceQuestion(instructions="Q?", criteria={"only": ""})  # a choice needs two options
        with pytest.raises(ValidationError):
            ScoreQuestion(instructions="Q?", criteria=[str(i) for i in range(11)])  # at most 10 levels
        with pytest.raises(ValidationError):
            NoulQuestion(instructions="Q?", criteria={"maybe": ""})

    def test_questions_from_plain_dicts(self):
        seen = []
        make_backend(seen).evaluate({
            "response": "x",
            "questions": {"safe": {"type": "noul", "instructions": "Is it safe?"}},
        })
        assert seen[0]["body"]["questions"]["safe"] == {"type": "noul", "instructions": "Is it safe?"}

    def test_output_contract_cannot_be_replaced(self):
        result = make_backend().evaluate(ask(response="x"), output_schema=EvaluationOutput)
        assert isinstance(result, EvaluationError)
        assert result.error_code == "INVALID_REQUEST"
        assert "DeciderOutput" in result.message

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
        result = make_backend(seen).evaluate(ask(response="x"), temperature=0.7)
        assert result.error_code == "INVALID_REQUEST"
        assert "temperature" in result.message
        assert seen == []


class TestInputCustomization:
    """The input is the part people shape: its fields, a template, or the state itself."""

    def test_subclass_fields_become_the_state(self):
        class SupportTurn(DeciderInput):
            customer_message: str
            agent_reply: str
            tags: list = []

        seen = []
        make_backend(seen).evaluate(SupportTurn(
            customer_message="Why was I charged twice?", agent_reply="Refunded.", tags=["billing"],
            questions={"apologizes": NoulQuestion(instructions="Does the agent apologize?")},
        ))
        assert seen[0]["body"]["state"] == {
            "customer_message": "Why was I charged twice?", "agent_reply": "Refunded.", "tags": ["billing"]}

    def test_custom_base_template_is_the_state(self):
        seen = []
        make_backend(seen).evaluate(ask(response="Port 8080.", base_template="Rule: ports must be 443. Text: {{ response }}"))
        assert seen[0]["body"]["state"] == "Rule: ports must be 443. Text: Port 8080."

    def test_subclass_default_template_is_the_state(self):
        class PortPolicy(DeciderInput):
            base_template: str = "Rule: ports must be 443. Text: {{ response }}"

        seen = []
        make_backend(seen).evaluate(PortPolicy(response="Port 8080.", questions={"verdict": HALLUCINATION}))
        assert seen[0]["body"]["state"] == "Rule: ports must be 443. Text: Port 8080."

    def test_explicit_state_is_sent_as_is(self):
        seen = []
        backend = make_backend(seen)
        backend.evaluate(ask(state={"ticket": {"id": 7, "body": "Charged twice."}}, response="ignored"))
        backend.evaluate(ask(state="Charged twice."))
        assert seen[0]["body"]["state"] == {"ticket": {"id": 7, "body": "Charged twice."}}
        assert seen[1]["body"]["state"] == "Charged twice."

    def test_empty_input_refused(self):
        seen = []
        backend = make_backend(seen)
        assert backend.evaluate(ask()).error_code == "INVALID_REQUEST"
        assert backend.evaluate(ask(state="  ")).error_code == "INVALID_REQUEST"
        assert seen == []

    def test_evaluator_takes_decider_fields_as_keywords(self):
        seen = []
        evaluator = Evaluator.__new__(Evaluator)
        evaluator.backend = make_backend(seen)
        result = evaluator.evaluate(
            response="London is in France.", context="London is in England.", questions={"verdict": HALLUCINATION})
        assert result.answers["verdict"].choice == "hallucination"
        assert seen[0]["body"]["state"] == {"context": "London is in England.", "response": "London is in France."}

        evaluator.evaluate("Just the response.", questions={"tone": TOXICITY})
        assert seen[1]["body"]["state"] == {"response": "Just the response."}
        assert list(seen[1]["body"]["questions"]) == ["tone"]


class TestServer:
    def test_http_error(self):
        backend = make_backend(respond=lambda body: httpx.Response(422, json={"detail": "questions: field required"}))
        result = backend.evaluate(ask(response="x"))
        assert isinstance(result, EvaluationError)
        assert result.error_code == "422"
        assert "field required" in result.message

    def test_unreadable_answer_is_invalid_response(self):
        backend = make_backend(respond=lambda body: httpx.Response(200, json={"answers": {"verdict": {"type": "choice"}}}))
        assert backend.evaluate(ask(response="x")).error_code == "INVALID_RESPONSE"

    def test_model_mismatch_refused(self):
        """The server ignores the request's model field, so a wrong name must fail loudly."""
        seen = []
        backend = make_backend(seen, model_name="strands-decider-latest")
        result = backend.evaluate(ask(response="x"))
        assert result.error_code == "MODEL_MISMATCH"
        assert f"StrandsAgents/{MODEL}" in result.message
        assert seen == []  # never reached /v1/systemone

    @pytest.mark.parametrize("name", [MODEL, f"StrandsAgents/{MODEL}"])
    def test_served_name_or_checkpoint_accepted(self, name):
        assert isinstance(make_backend(model_name=name).evaluate(ask(response="x")), DeciderOutput)

    def test_health_checked_once(self):
        calls = []
        backend = make_backend(health_calls=calls)
        backend.evaluate(ask(response="x"))
        backend.evaluate(ask(response="y"))
        assert len(calls) == 1

    def test_server_without_health_is_not_blocked(self):
        backend = make_backend(health=None, model_name="anything")
        assert isinstance(backend.evaluate(ask(response="x")), DeciderOutput)

    def test_long_input_warns(self):
        with pytest.warns(UserWarning, match="truncates"):  # /health says max_length=100
            make_backend().evaluate(ask(response="r", context="word " * 200))

    def test_connection_error(self):
        backend = DeciderBackend(model_name=MODEL, base_url="http://127.0.0.1:9")
        result = backend.evaluate(ask(response="x"))
        assert result.error_code == "CONNECTION_ERROR"

    @pytest.mark.asyncio
    async def test_async(self):
        evaluator = AsyncEvaluator.__new__(AsyncEvaluator)
        evaluator.backend = make_backend(async_=True)
        result = await evaluator.evaluate(
            response="London is in France.", context="London is in England.", questions={"verdict": HALLUCINATION})
        assert result.answers["verdict"].choice == "hallucination"

    @pytest.mark.asyncio
    async def test_async_model_mismatch(self):
        backend = make_backend(async_=True, model_name="strands-decider-latest")
        assert (await backend.evaluate_async(ask(response="x"))).error_code == "MODEL_MISMATCH"


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
        import grounded_ai.backends.decider as decider

        up, started = set(), []

        def popen(command):
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
        assert isinstance(backend.evaluate(ask(response="x")), DeciderOutput)

        backend.shutdown()
        assert started[0].terminated

    def test_reuses_a_server_already_on_the_port(self, monkeypatch):
        import grounded_ai.backends.decider as decider

        monkeypatch.setattr(decider.subprocess, "Popen", lambda command: pytest.fail("should not start a server"))
        backend = self._backend({8000})
        backend.warmup(port=8000)
        assert backend.base_url == "http://127.0.0.1:8000"

    def test_checkpoint_argument_overrides_the_model_name(self, monkeypatch):
        import grounded_ai.backends.decider as decider

        up, started = set(), []
        monkeypatch.setattr(decider.shutil, "which", lambda name: "strands-decider")
        monkeypatch.setattr(decider.subprocess, "Popen", lambda c: (started.append(c), up.add(8000), FakeServer(c))[2])
        monkeypatch.setattr(decider.time, "sleep", lambda s: None)
        self._backend(up, model_name=MODEL).warmup(checkpoint="/models/hobson")
        assert started[0][:3] == ["strands-decider", "serve", "/models/hobson"]

    def test_server_that_exits_is_reported(self, monkeypatch):
        import grounded_ai.backends.decider as decider

        monkeypatch.setattr(decider.shutil, "which", lambda name: "strands-decider")
        monkeypatch.setattr(decider.subprocess, "Popen", lambda command: FakeServer(command, exits_with=1))
        with pytest.raises(RuntimeError, match="exited with code 1"):
            self._backend(set()).warmup(port=8000)

    def test_missing_server_package(self, monkeypatch):
        import grounded_ai.backends.decider as decider

        monkeypatch.setattr(decider.shutil, "which", lambda name: None)
        with pytest.raises(ImportError, match="grounded-ai\\[decider\\]"):
            self._backend(set()).warmup(port=8000)

    def test_times_out(self, monkeypatch):
        import grounded_ai.backends.decider as decider

        started = []
        monkeypatch.setattr(decider.shutil, "which", lambda name: "strands-decider")
        monkeypatch.setattr(decider.subprocess, "Popen", lambda c: (started.append(FakeServer(c)), started[-1])[1])
        monkeypatch.setattr(decider.time, "sleep", lambda s: None)
        with pytest.raises(TimeoutError):
            self._backend(set()).warmup(port=8000, timeout=0.05)
        assert started[0].terminated

