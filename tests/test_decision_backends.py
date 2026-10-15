"""
One backend for every /v1/systemone decision model (DecisionEvaluator: Jev is a host preset) and OpenAI's Decisions API
(OpenAIDecisionsEvaluator), with one definition of confidence and requests checked against what each
model accepts.
"""

import json
from types import SimpleNamespace

import httpx
import pytest

from grounded_ai import AsyncEvaluator, CascadeEvaluator, Evaluator
from grounded_ai.backends.jev import (
    Capabilities,
    ChoiceAnswer,
    ChoiceQuestion,
    DecisionEvaluator,
    JevEvaluator,
    JevInput,
    JevOutput,
    NoulAnswer,
    NoulQuestion,
    ScoreAnswer,
    ScoreQuestion,
    confidence,
)
from grounded_ai.backends.openai_decisions import OpenAIDecisionsEvaluator, to_decisions_request
from grounded_ai.schemas import EvaluationError
from tests.test_cascade import FakeJudge

CALL = ChoiceQuestion(instructions="What should the assistant do next?",
                      criteria={"call_tool": "call one of the tools", "respond": "reply without a tool"})
URGENT = NoulQuestion(instructions="This needs a reply within the hour.", criteria={"true": "a deadline today"})
CLARITY = ScoreQuestion(instructions="How clear is it?", criteria=["unclear", "partly clear", "clear"])

# What d1-3B (agent-decision-3b) served locally returned today for a 53/47 answer.
D1_ANSWER = {"type": "choice", "choice": "call_tool", "confidence": 0.5302,
             "probabilities": {"call_tool": 0.5302, "respond": 0.4698}}


def server(answers, seen=None):
    def handler(request):
        if seen is not None:
            seen.append(request)
        body = json.loads(request.content)
        return httpx.Response(200, json={"model": body["model"], "answers": {k: answers[k] for k in body["questions"]}})

    transport = httpx.MockTransport(handler)
    return dict(client=httpx.Client(transport=transport), async_client=httpx.AsyncClient(transport=transport))


class TestConfidence:
    """TypeSafe's definitions and its own worked examples (docs.typesafe.ai/confidence)."""

    def choice(self, *p):
        names = [f"o{i}" for i in range(len(p))]
        return ChoiceAnswer(choice=names[p.index(max(p))], probabilities=dict(zip(names, p)), confidence=0)

    def score(self, *p):
        return ScoreAnswer(score=sum(i * v for i, v in enumerate(p)), probabilities={str(i): v for i, v in enumerate(p)},
                           confidence=0)

    def test_choice_is_normalised_for_the_number_of_options(self):
        assert confidence(self.choice(0.6, 0.3, 0.1)) == pytest.approx(0.4)
        assert confidence(self.choice(0.6, 0.2, 0.2)) == pytest.approx(0.4)
        assert confidence(self.choice(0.5, 0.5)) == pytest.approx(0.0)
        assert confidence(self.choice(0.5302, 0.4698)) == pytest.approx(0.0604)

    def test_score_is_spread_around_the_most_likely_level(self):
        assert confidence(self.score(0, 0.5, 0.5)) == pytest.approx(0.25)
        assert confidence(self.score(0.5, 0, 0.5)) == pytest.approx(0.0)
        assert confidence(self.score(0, 0.57, 0.43)) == pytest.approx(1 - 0.43 / (2 / 3))
        assert confidence(self.score(0, 0, 1)) == pytest.approx(1.0)

    def test_noul_is_distance_from_a_coin_flip(self):
        assert confidence(NoulAnswer(noul=0.9)) == pytest.approx(0.8)
        assert confidence(NoulAnswer(noul=0.1)) == pytest.approx(0.8)


class TestSystemOne:
    def test_routing_and_where_requests_go(self):
        seen = []
        evaluator = Evaluator("systemone/agent-decision-3b", base_url="http://gpu:8765/", **server({"next": D1_ANSWER}, seen))
        assert isinstance(evaluator.backend, DecisionEvaluator)
        result = evaluator.evaluate(state="tools and chat", questions={"next": CALL})
        assert isinstance(result, JevOutput)
        assert str(seen[0].url) == "http://gpu:8765/v1/systemone"
        assert json.loads(seen[0].content)["model"] == "agent-decision-3b"
        assert "authorization" not in seen[0].headers  # no key unless one is given

    def test_needs_an_address(self, monkeypatch):
        monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
        with pytest.raises(ValueError, match="base_url=.*host="):
            Evaluator("systemone/laya")
        with pytest.raises(ValueError, match="Unknown host 'lmstudio'"):
            Evaluator("systemone/laya", host="lmstudio")

    def test_host_presets(self, monkeypatch):
        monkeypatch.setenv("TYPESAFE_API_KEY", "ts-key")
        assert Evaluator("systemone/laya", host="llamacpp").backend.base_url == "http://127.0.0.1:8080"
        assert Evaluator("systemone/laya", host="ollama").backend.base_url == "http://127.0.0.1:11434"
        typesafe = Evaluator("systemone/jev-latest", host="typesafe").backend
        assert typesafe.base_url == "https://api.typesafe.ai"
        assert typesafe._headers == {"Authorization": "Bearer ts-key"}
        assert Evaluator("systemone/x", host="ollama", base_url="http://box:11434").backend.base_url == "http://box:11434"

    def test_confidence_is_recomputed_from_the_probabilities(self):
        evaluator = Evaluator("systemone/agent-decision-3b", base_url="http://gpu:8765", **server({"next": D1_ANSWER}))
        answer = evaluator.evaluate(state="s", questions={"next": CALL}).answers["next"]
        assert answer.confidence == pytest.approx(0.0604)  # d1 itself said 0.53
        assert answer.probabilities == D1_ANSWER["probabilities"]  # what the model said is untouched

    def test_jev_keeps_its_own_confidence(self):
        jev = DecisionEvaluator(api_key="k", **server({"next": D1_ANSWER}))  # host="typesafe"
        assert jev.evaluate(JevInput(state="s", questions={"next": CALL})).answers["next"].confidence == 0.5302

    @pytest.mark.asyncio
    async def test_async(self):
        evaluator = AsyncEvaluator("systemone/agent-decision-3b", base_url="http://gpu:8765", **server({"next": D1_ANSWER}))
        answer = (await evaluator.evaluate(state="s", questions={"next": CALL})).answers["next"]
        assert answer.confidence == pytest.approx(0.0604)

    def test_a_coin_flip_is_escalated_by_the_cascade(self):
        """At 0.5 the d1-reported 0.53 would have been kept; recomputed, it goes to the judge."""
        first = DecisionEvaluator("agent-decision-3b", base_url="http://gpu:8765", **server({"next": D1_ANSWER}))
        judge = FakeJudge({"next": "respond"})
        result = CascadeEvaluator(jev=first, judge=judge, min_confidence=0.5).evaluate(state="s", questions={"next": CALL})
        assert result.escalated == ["next"] and result.answers["next"].answer == "respond"


class TestCapabilities:
    def refused(self, evaluator, **questions):
        result = evaluator.evaluate(JevInput(state="s", questions=questions))
        assert isinstance(result, EvaluationError) and result.error_code == "INVALID_REQUEST"
        return result.message

    def test_question_types(self):
        seen = []
        choice_only = DecisionEvaluator("tev1", base_url="http://x", capabilities=Capabilities(question_types=("choice",)),
                                         **server({}, seen))
        assert "answers choice questions, not score" in self.refused(choice_only, clarity=CLARITY)
        assert seen == []  # never sent

    def test_option_and_level_limits(self):
        small = DecisionEvaluator("m", base_url="http://x", capabilities=Capabilities(max_options=2, max_levels=2), **server({}))
        three = ChoiceQuestion(instructions="?", criteria={"a": "a", "b": "b", "c": "c"})
        assert "at most 2 options" in self.refused(small, q=three)
        assert "at most 2 levels" in self.refused(small, q=CLARITY)

    def test_strands_takes_text_criteria_only(self):
        strands = Evaluator("systemone/strands-decider", host="strands", **server({})).backend
        structured = ChoiceQuestion(instructions="?", criteria={"a": {"rule": "x"}, "b": None})
        assert "criteria as text only" in self.refused(strands, q=structured)


class FakeDecisions:
    """Stands in for openai.OpenAI(): records the request, returns a canned /v1/decisions response."""

    def __init__(self, response=None, error=None):
        self.requests, self.response, self.error = [], response, error
        self.decisions = SimpleNamespace(create=self.create)

    def create(self, **body):
        self.requests.append(body)
        if self.error:
            raise self.error
        return self.response


class FakeAsyncDecisions(FakeDecisions):
    async def acreate(self, **body):
        return self.create(**body)

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.decisions = SimpleNamespace(create=self.acreate)


LUNA = {
    "model": "gpt-6-luna",
    "answers": [
        {"type": "choice", "name": "next", "choice": "call_tool", "confidence": 0.6,
         "probabilities": [{"value": "call_tool", "probability": 0.8}, {"value": "respond", "probability": 0.2}]},
        {"type": "predicate", "name": "urgent", "probability": 0.9},
        {"type": "score", "name": "clarity", "score": 1.6, "confidence": 0.4,
         "probabilities": [{"value": 0, "label": "unclear", "probability": 0.1},
                           {"value": 1, "label": "partly clear", "probability": 0.2},
                           {"value": 2, "label": "clear", "probability": 0.7}]},
    ],
    "usage": {"input_tokens": 120, "output_tokens": 0, "total_tokens": 120,
              "input_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 0},
              "output_tokens_details": {"reasoning_tokens": 0}},
}


class TestOpenAIDecisions:
    def test_routing(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        backend = Evaluator("openai-decisions/gpt-6-luna").backend
        assert isinstance(backend, OpenAIDecisionsEvaluator) and backend.model_name == "gpt-6-luna"

    def test_request_is_translated(self):
        client = FakeDecisions(LUNA)
        OpenAIDecisionsEvaluator(client=client).evaluate(
            JevInput(state={"tools": ["search"], "user": "hi"}, questions={"next": CALL, "urgent": URGENT, "clarity": CLARITY}))
        body = client.requests[0]
        assert body["model"] == "gpt-6-luna"
        assert json.loads(body["input"]) == {"tools": ["search"], "user": "hi"}  # JSON state as JSON text
        next_, urgent, clarity = body["questions"]
        assert next_ == {"type": "choice", "name": "next", "instructions": "What should the assistant do next?",
                         "choices": [{"value": "call_tool", "description": "call one of the tools"},
                                     {"value": "respond", "description": "reply without a tool"}]}
        assert urgent == {"type": "predicate", "name": "urgent",
                          "instructions": "This needs a reply within the hour.\nTrue means: a deadline today"}
        assert clarity["levels"] == [{"label": "unclear"}, {"label": "partly clear"}, {"label": "clear"}]

    def test_option_without_description(self):
        body = to_decisions_request({"model": "m", "state": "s", "questions": {
            "q": {"type": "choice", "instructions": "?", "criteria": {"a": None, "b": "bee"}}}})
        assert body["questions"][0]["choices"] == [{"value": "a"}, {"value": "b", "description": "bee"}]

    def test_response_is_translated(self):
        result = OpenAIDecisionsEvaluator(client=FakeDecisions(LUNA)).evaluate(
            JevInput(state="s", questions={"next": CALL, "urgent": URGENT, "clarity": CLARITY}))
        assert result.answers["next"] == ChoiceAnswer(choice="call_tool", probabilities={"call_tool": 0.8, "respond": 0.2},
                                                      confidence=0.6)
        assert result.answers["urgent"] == NoulAnswer(noul=0.9)
        assert result.answers["clarity"] == ScoreAnswer(
            score=1.6, legend={"0": "unclear", "1": "partly clear", "2": "clear"},
            probabilities={"0": 0.1, "1": 0.2, "2": 0.7}, confidence=0.4)
        assert result.model == "gpt-6-luna" and result.usage == {"input_tokens": 120, "output_tokens": 0}

    def test_refusal_fails_the_call_naming_the_question(self):
        refused = {"model": "gpt-6-luna", "answers": [{"type": "refusal", "name": "next"}]}
        result = OpenAIDecisionsEvaluator(client=FakeDecisions(refused)).evaluate(JevInput(state="s", questions={"next": CALL}))
        assert isinstance(result, EvaluationError) and result.error_code == "REFUSED" and "next" in result.message

    def test_an_answer_over_other_options_is_an_invalid_response(self):
        wrong = {"model": "gpt-6-luna", "answers": [{**LUNA["answers"][0], "choice": "maybe",
                                                     "probabilities": [{"value": "maybe", "probability": 1.0}]}]}
        result = OpenAIDecisionsEvaluator(client=FakeDecisions(wrong)).evaluate(JevInput(state="s", questions={"next": CALL}))
        assert result.error_code == "INVALID_RESPONSE"

    def test_api_errors(self):
        # Shaped like the SDK's own (other tests replace the `openai` module, so it is not imported here).
        class APIStatusError(Exception):
            def __init__(self, message, status_code):
                super().__init__(message)
                self.message, self.status_code = message, status_code

        class RateLimitError(APIStatusError):
            pass

        error = RateLimitError("Rate limit reached", 429)
        result = OpenAIDecisionsEvaluator(client=FakeDecisions(error=error)).evaluate(JevInput(state="s", questions={"next": CALL}))
        assert (result.error_code, result.message) == ("429", "Rate limit reached")

    def test_generation_arguments_are_refused(self):
        client = FakeDecisions(LUNA)
        result = OpenAIDecisionsEvaluator(client=client).evaluate(JevInput(state="s", questions={"next": CALL}), temperature=0)
        assert result.error_code == "INVALID_REQUEST" and client.requests == []

    @pytest.mark.asyncio
    async def test_async(self):
        evaluator = OpenAIDecisionsEvaluator(client=FakeDecisions(LUNA), async_client=FakeAsyncDecisions(LUNA))
        result = await evaluator.evaluate_async(JevInput(state="s", questions={"next": CALL}))
        assert result.answers["next"].choice == "call_tool"

    def test_as_the_cascades_first_stage(self):
        unsure = {"model": "gpt-6-luna", "answers": [{**LUNA["answers"][0], "confidence": 0.3}]}
        judge = FakeJudge({"next": "respond"})
        result = CascadeEvaluator(jev=OpenAIDecisionsEvaluator(client=FakeDecisions(unsure)), judge=judge,
                                  min_confidence=0.5).evaluate(state="s", questions={"next": CALL})
        assert result.escalated == ["next"] and result.answers["next"].answer == "respond"


class TestOneBackend:
    """Jev is a host of DecisionEvaluator; the Jev-only class remains as a deprecated alias."""

    def test_jev_strings_use_the_one_backend(self, monkeypatch):
        monkeypatch.setenv("TYPESAFE_API_KEY", "k")
        hosted, local = Evaluator("jev/jev-latest").backend, Evaluator("jev", use_local_model=True).backend
        assert type(hosted) is DecisionEvaluator and hosted.host == "typesafe"
        assert type(local) is DecisionEvaluator and local.host == "strands" and local.use_local_model

    def test_jev_evaluator_still_works_and_says_it_is_deprecated(self, monkeypatch):
        monkeypatch.setenv("TYPESAFE_API_KEY", "k")
        with pytest.warns(DeprecationWarning, match="DecisionEvaluator"):
            old = JevEvaluator(**server({"next": D1_ANSWER}))
        assert isinstance(old, DecisionEvaluator) and old.host == "typesafe"
        assert old.evaluate(JevInput(state="s", questions={"next": CALL})).answers["next"].confidence == 0.5302
        with pytest.warns(DeprecationWarning):
            assert JevEvaluator(use_local_model=True, local_model="m").host == "strands"
