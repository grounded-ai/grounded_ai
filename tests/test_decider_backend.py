import json
from typing import Literal, Optional

import httpx
import pytest
from pydantic import BaseModel, Field

from grounded_ai import AsyncEvaluator, Evaluator
from grounded_ai.backends.decider import DeciderBackend
from grounded_ai.schemas import EvaluationError, EvaluationInput, EvaluationOutput


def verdict(p_yes, yes="hallucination", no="faithful"):
    """A two-option choice answer for the eval-mode question."""
    return {"verdict": {"type": "choice", "choice": yes if p_yes >= 0.5 else no, "confidence": abs(2 * p_yes - 1),
                        "probabilities": {yes: p_yes, no: round(1 - p_yes, 4)}}}


# /health as `strands-decider serve StrandsAgents/strands-decider-2B-hobson-v19` reports it (trimmed).
HEALTH = {
    "status": "ok",
    "model": "strands-decider-2B-hobson-v19",
    "checkpoint": "StrandsAgents/strands-decider-2B-hobson-v19",
    "device": "mps",
}


def make_backend(answers, status=200, seen=None, async_=False, health=HEALTH, health_calls=None,
                 model_name="strands-decider-2B-hobson-v19", **kwargs):
    """Backend whose HTTP client answers /health with `health` and /v1/systemone with `answers`.
    `seen` records /v1/systemone requests; `health_calls` counts /health requests."""

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            if health_calls is not None:
                health_calls.append(1)
            return httpx.Response(404) if health is None else httpx.Response(200, json=health)
        if seen is not None:
            seen.append({"url": str(request.url), "headers": request.headers, "body": json.loads(request.content)})
        if status != 200:
            return httpx.Response(status, json={"detail": answers})
        return httpx.Response(200, json={"answers": answers})

    transport = httpx.MockTransport(handler)
    client_kwargs = {"async_client": httpx.AsyncClient(transport=transport)} if async_ else {}
    return DeciderBackend(
        model_name=model_name,
        client=httpx.Client(transport=transport),
        **client_kwargs,
        **kwargs,
    )


class TestDeciderBackend:
    def test_factory_routing(self):
        evaluator = Evaluator("decider/strands-decider-2B-hobson-v19", base_url="http://127.0.0.1:8010/")
        assert isinstance(evaluator.backend, DeciderBackend)
        assert evaluator.backend.model_name == "strands-decider-2B-hobson-v19"
        assert evaluator.backend.base_url == "http://127.0.0.1:8010"

    def test_request_shape(self):
        """Inputs become an object state; the eval mode becomes one noul question."""
        seen = []
        backend = make_backend(verdict(0.9), seen=seen, api_key="k")
        backend.evaluate(EvaluationInput(response="London is in France.", context="London is in England."))

        (call,) = seen
        assert call["url"] == "http://127.0.0.1:8000/v1/systemone"
        assert call["headers"]["authorization"] == "Bearer k"
        assert call["body"]["model"] == "strands-decider-2B-hobson-v19"
        # Context first: the model reads the evidence before the text it judges.
        assert list(call["body"]["state"]) == ["context", "response"]
        question = call["body"]["questions"]["verdict"]
        assert question["type"] == "choice"
        assert set(question["criteria"]) == {"hallucination", "faithful"}

    @pytest.mark.parametrize(
        "mode,yes,no,p,label,score",
        [
            ("HALLUCINATION", "hallucination", "faithful", 0.9, "hallucination", 0.9),
            ("HALLUCINATION", "hallucination", "faithful", 0.2, "faithful", 0.2),
            ("TOXICITY", "toxic", "non-toxic", 0.7, "toxic", 0.7),
            ("RAG_RELEVANCE", "relevant", "unrelated", 0.1, "unrelated", 0.1),
        ],
    )
    def test_eval_modes(self, mode, yes, no, p, label, score):
        backend = make_backend(verdict(p, yes, no), eval_mode=mode)
        result = backend.evaluate(EvaluationInput(response="x"))
        assert isinstance(result, EvaluationOutput)
        assert result.label == label
        assert result.score == pytest.approx(score)
        assert result.confidence == pytest.approx(abs(2 * p - 1))

    def test_system_prompt_is_the_question(self):
        seen = []
        backend = make_backend({"verdict": {"type": "noul", "noul": 0.6}}, seen=seen, system_prompt="Is this safe?")
        result = backend.evaluate(EvaluationInput(response="x"))
        assert seen[0]["body"]["questions"]["verdict"]["instructions"] == "Is this safe?"
        assert result.label == "yes"

    def test_threshold_runtime_override(self):
        backend = make_backend(verdict(0.6))
        assert backend.evaluate(EvaluationInput(response="x")).label == "hallucination"
        assert backend.evaluate(EvaluationInput(response="x"), threshold=0.9).label == "faithful"

    def test_custom_schema_one_question_per_field(self):
        class Ticket(BaseModel):
            urgent: bool = Field(description="Does this need a reply within the hour?")
            area: Literal["billing", "bug", "account"] = Field(description="Which team owns it?")
            p_harmful: float = Field(ge=0, le=1, description="Is this request harmful?")
            reasoning: Optional[str] = None

        seen = []
        backend = make_backend(
            {
                "urgent": {"type": "noul", "noul": 0.8},
                "area": {"type": "choice", "choice": "billing", "confidence": 0.9, "probabilities": {}},
                "p_harmful": {"type": "noul", "noul": 0.05},
            },
            seen=seen,
        )
        result = backend.evaluate(EvaluationInput(response="Charged twice."), output_schema=Ticket)

        assert result == Ticket(urgent=True, area="billing", p_harmful=0.05)
        questions = seen[0]["body"]["questions"]
        assert set(questions) == {"urgent", "area", "p_harmful"}  # all in one request
        assert questions["area"]["criteria"] == {"billing": "", "bug": "", "account": ""}  # Decider requires str descriptions

    def test_required_str_field_refused(self):
        class Bad(BaseModel):
            summary: str = Field(description="Summarize it.")

        backend = make_backend({})
        result = backend.evaluate(EvaluationInput(response="x"), output_schema=Bad)
        assert isinstance(result, EvaluationError)
        assert result.error_code == "INVALID_REQUEST"

    def test_generation_kwargs_accepted_like_other_backends(self):
        """temperature & co. are accepted at init and per call, as on every backend; a decision
        model does not sample, so they do not change the request."""
        seen = []
        backend = make_backend(verdict(0.9), seen=seen, temperature=0.1, max_tokens=50)
        result = backend.evaluate(EvaluationInput(response="x"), temperature=0.7, top_p=0.9)
        assert result.label == "hallucination"
        assert set(seen[0]["body"]) == {"state", "model", "questions"}

    def test_set_eval_mode_like_slm_backend(self):
        from enum import Enum

        class EvalMode(str, Enum):  # same shape as the SLM backend's enum, which imports torch
            TOXICITY = "TOXICITY"

        backend = make_backend(verdict(0.9, "toxic", "non-toxic"))
        backend.set_eval_mode(EvalMode.TOXICITY)
        assert backend.evaluate(EvaluationInput(response="x")).label == "toxic"
        with pytest.raises(ValueError):
            backend.set_eval_mode("NOPE")

    def test_http_error(self):
        backend = make_backend("questions: field required", status=422)
        result = backend.evaluate(EvaluationInput(response="x"))
        assert isinstance(result, EvaluationError)
        assert result.error_code == "422"
        assert "field required" in result.message

    def test_out_of_range_probability(self):
        backend = make_backend({"verdict": {"type": "choice", "probabilities": {"hallucination": -0.2, "faithful": 0.1}}})
        result = backend.evaluate(EvaluationInput(response="x"))
        assert isinstance(result, EvaluationError)

    def test_model_mismatch_refused(self):
        """The server ignores the request's model field, so a wrong name must fail loudly."""
        seen = []
        backend = make_backend(verdict(0.9), seen=seen, model_name="strands-decider-latest")
        result = backend.evaluate(EvaluationInput(response="x"))
        assert isinstance(result, EvaluationError)
        assert result.error_code == "MODEL_MISMATCH"
        assert "StrandsAgents/strands-decider-2B-hobson-v19" in result.message
        assert seen == []  # never reached /v1/systemone

    @pytest.mark.parametrize(
        "name", ["strands-decider-2B-hobson-v19", "StrandsAgents/strands-decider-2B-hobson-v19"]
    )
    def test_served_name_or_checkpoint_accepted(self, name):
        backend = make_backend(verdict(0.9), model_name=name)
        assert backend.evaluate(EvaluationInput(response="x")).label == "hallucination"

    def test_health_checked_once(self):
        calls = []
        backend = make_backend(verdict(0.9), health_calls=calls)
        backend.evaluate(EvaluationInput(response="x"))
        backend.evaluate(EvaluationInput(response="y"))
        assert len(calls) == 1

    def test_server_without_health_is_not_blocked(self):
        backend = make_backend(verdict(0.9), health=None, model_name="anything")
        assert backend.evaluate(EvaluationInput(response="x")).label == "hallucination"

    @pytest.mark.asyncio
    async def test_async_model_mismatch(self):
        backend = make_backend(verdict(0.9), async_=True, model_name="strands-decider-latest")
        result = await backend.evaluate_async(EvaluationInput(response="x"))
        assert result.error_code == "MODEL_MISMATCH"

    def test_rounded_probabilities_renormalized(self):
        """Servers round each probability to 4 decimals, so the pair may not sum to exactly 1."""
        backend = make_backend({"verdict": {"type": "choice", "probabilities": {"hallucination": 0.6, "faithful": 0.3}}})
        assert backend.evaluate(EvaluationInput(response="x")).score == pytest.approx(2 / 3)

    def test_connection_error(self):
        backend = DeciderBackend(model_name="strands-decider-latest", base_url="http://127.0.0.1:9")
        result = backend.evaluate(EvaluationInput(response="x"))
        assert isinstance(result, EvaluationError)
        assert result.error_code == "CONNECTION_ERROR"

    @pytest.mark.asyncio
    async def test_async(self):
        backend = make_backend(verdict(0.95), async_=True)
        evaluator = AsyncEvaluator.__new__(AsyncEvaluator)
        evaluator.backend = backend
        result = await evaluator.evaluate(response="London is in France.", context="London is in England.")
        assert result.label == "hallucination"
