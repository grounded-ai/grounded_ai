"""
CascadeEvaluator: Jev answers every question; the ones it is not confident about are
forwarded, as a custom evaluation input, to an LLM judge.
"""

import json

import httpx
import pytest
from pydantic import BaseModel

from grounded_ai import CascadeEvaluator, Evaluator
from grounded_ai.backends.jev import (
    ChoiceAnswer,
    ChoiceQuestion,
    JevEvaluator,
    JevInput,
    JevOutput,
    NoulAnswer,
    NoulQuestion,
    ScoreAnswer,
    ScoreQuestion,
)
from grounded_ai.base import BaseEvaluator
from grounded_ai.cascade import CascadeOutput, JevLeftover, JudgedAnswer
from grounded_ai.schemas import EvaluationError, EvaluationOutput

MODEL = "strands-decider-2B-hobson-v19"

AREA = ChoiceQuestion(instructions="Which team owns it?", criteria={"billing": "charges", "bug": "defects", "account": "login"})
URGENT = NoulQuestion(instructions="This needs a reply within the hour.")
CLARITY = ScoreQuestion(instructions="How clearly is the problem described?", criteria=["unclear", "partly clear", "clear"])

CONFIDENT = {
    "area": {"type": "choice", "choice": "billing", "probabilities": {"billing": 0.97, "bug": 0.02, "account": 0.01}, "confidence": 0.955},
    "urgent": {"type": "noul", "noul": 0.98},  # |2p - 1| = 0.96
    "clarity": {"type": "score", "score": 1.9, "legend": {"0": "unclear", "1": "partly clear", "2": "clear"},
                "probabilities": {"0": 0.0, "1": 0.1, "2": 0.9}, "confidence": 0.93},
}
UNSURE = {
    "area": {"type": "choice", "choice": "bug", "probabilities": {"billing": 0.4, "bug": 0.45, "account": 0.15}, "confidence": 0.175},
    "urgent": {"type": "noul", "noul": 0.6},  # |2p - 1| = 0.2
    "clarity": {"type": "score", "score": 1.0, "legend": {"0": "unclear", "1": "partly clear", "2": "clear"},
                "probabilities": {"0": 0.3, "1": 0.4, "2": 0.3}, "confidence": 0.3},
}


def fake_jev(answers, seen=None, status=200):
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            return httpx.Response(404)
        body = json.loads(request.content)
        if seen is not None:
            seen.append(body)
        if status != 200:
            return httpx.Response(status, json={"detail": "boom"})
        return httpx.Response(200, json={"model": MODEL, "answers": {k: answers[k] for k in body["questions"]}})

    return JevEvaluator(use_local_model=True, local_model=MODEL, client=httpx.Client(transport=httpx.MockTransport(handler)),
                          async_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)))


class FakeJudge(BaseEvaluator):
    """An LLM backend: records what it was asked and answers with fixed picks."""

    def __init__(self, picks=None, error=None):
        super().__init__()
        self.picks, self.error, self.calls = picks or {}, error, []

    def _call_backend(self, input_data, output_schema, **kwargs):
        self.calls.append((input_data, output_schema))
        if self.error:
            return self.error
        fields = output_schema.model_fields
        values = {}
        for field, info in fields.items():
            if field.startswith("answer_"):
                name = info.description
                values[field] = self.picks[name]
                values["reasoning_" + field[len("answer_"):]] = f"because {name}"
        return output_schema(**values)


def cascade(answers, judge=None, **kwargs):
    return CascadeEvaluator(jev=fake_jev(answers), judge=judge or FakeJudge(), **kwargs)


def ask(**questions):
    return JevInput(state="You charged me twice and my account is overdrawn.", questions=questions)


class TestRouting:
    def test_confident_answers_never_reach_the_judge(self):
        judge = FakeJudge()
        result = cascade(CONFIDENT, judge).evaluate(ask(area=AREA, urgent=URGENT, clarity=CLARITY))
        assert isinstance(result, CascadeOutput)
        assert judge.calls == []
        assert result.escalated == []
        assert result.answers["area"] == ChoiceAnswer(**CONFIDENT["area"])
        assert result.answers["urgent"] == NoulAnswer(**CONFIDENT["urgent"])
        assert result.answers["clarity"] == ScoreAnswer(**CONFIDENT["clarity"])

    def test_only_unsure_questions_are_escalated_in_one_judge_call(self):
        answers = {"area": UNSURE["area"], "urgent": CONFIDENT["urgent"], "clarity": UNSURE["clarity"]}
        judge = FakeJudge({"area": "billing", "clarity": "clear"})
        result = cascade(answers, judge).evaluate(ask(area=AREA, urgent=URGENT, clarity=CLARITY))

        assert len(judge.calls) == 1
        assert result.escalated == ["area", "clarity"]
        assert result.answers["urgent"] == NoulAnswer(**CONFIDENT["urgent"])
        area = result.answers["area"]
        assert isinstance(area, JudgedAnswer)
        assert (area.question_type, area.answer, area.reasoning) == ("choice", "billing", "because area")
        assert result.answers["clarity"].answer == "clear"

    def test_noul_confidence_is_derived_from_the_probability(self):
        """A yes/no answer has no confidence field: the probability is the uncertainty, |2p - 1|."""
        for p, escalated in [(0.98, False), (0.02, False), (0.95, False), (0.6, True), (0.5, True), (0.1, True)]:
            judge = FakeJudge({"urgent": True})
            result = cascade({"urgent": {"type": "noul", "noul": p}}, judge, min_confidence=0.88).evaluate(ask(urgent=URGENT))
            assert (result.escalated == ["urgent"]) is escalated, p

    def test_threshold_is_inclusive_and_configurable(self):
        at_threshold = {"area": {**UNSURE["area"], "confidence": 0.9}}
        assert cascade(at_threshold).evaluate(ask(area=AREA)).escalated == []  # default 0.9
        judge = FakeJudge({"area": "billing"})
        assert cascade(at_threshold, judge, min_confidence=0.95).evaluate(ask(area=AREA)).escalated == ["area"]

    def test_jevs_own_answers_are_kept_for_escalated_questions(self):
        judge = FakeJudge({"area": "billing"})
        result = cascade({"area": UNSURE["area"]}, judge).evaluate(ask(area=AREA))
        assert isinstance(result.jev, JevOutput)
        assert result.jev.answers["area"] == ChoiceAnswer(**UNSURE["area"])

    def test_threshold_must_be_a_confidence(self):
        for bad in (-0.1, 1.5):
            with pytest.raises(ValueError):
                cascade(CONFIDENT, min_confidence=bad)


class TestWhatTheJudgeReceives:
    def test_leftover_is_a_custom_evaluation_input(self):
        """The judge gets the original state and only the leftover questions, as an input model
        it renders like any other custom evaluation input."""
        judge = FakeJudge({"area": "billing"})
        cascade({"area": UNSURE["area"], "urgent": CONFIDENT["urgent"]}, judge).evaluate(ask(area=AREA, urgent=URGENT))

        (input_data, _), = judge.calls
        assert isinstance(input_data, JevLeftover)
        assert input_data.state == "You charged me twice and my account is overdrawn."
        assert set(input_data.questions) == {"area"}
        prompt = input_data.formatted_prompt
        assert "You charged me twice" in prompt
        assert "Which team owns it?" in prompt and "billing" in prompt and "defects" in prompt
        assert "This needs a reply" not in prompt  # confident questions are not re-asked

    def test_output_schema_restricts_answers_to_each_questions_options(self):
        judge = FakeJudge({"area": "billing", "urgent": True, "clarity": "partly clear"})
        cascade(UNSURE, judge).evaluate(ask(area=AREA, urgent=URGENT, clarity=CLARITY))
        (_, schema), = judge.calls
        answers = {info.description: info.annotation for f, info in schema.model_fields.items() if f.startswith("answer_")}
        assert set(answers) == {"area", "urgent", "clarity"}
        assert set(answers["area"].__args__) == {"billing", "bug", "account"}
        assert answers["urgent"] is bool
        assert set(answers["clarity"].__args__) == {"unclear", "partly clear", "clear"}
        # flat (no nested models), reasoning before each answer, so every backend's structured output accepts it
        names = list(schema.model_fields)
        assert names == ["reasoning_0", "answer_0", "reasoning_1", "answer_1", "reasoning_2", "answer_2"]
        assert "$defs" not in schema.model_json_schema()
        with pytest.raises(Exception):
            schema(**{f: ("x" if f.startswith("reasoning") else "refund") for f in names})

    def test_state_keeps_its_declared_order_in_the_prompt(self):
        """The evidence is declared before the text being judged; the judge must read it that way."""
        leftover = JevLeftover(state={"policy": "Refunds within 30 days.", "answer": "You have 90 days."},  # reverse alphabetical
                                   questions={"verdict": AREA})
        prompt = leftover.formatted_prompt
        assert prompt.index("Refunds within 30 days.") < prompt.index("You have 90 days.")

    def test_question_names_need_not_be_identifiers(self):
        judge = FakeJudge({"which team?": "billing"})
        result = cascade({"which team?": UNSURE["area"]}, judge).evaluate(
            JevInput(state="x", questions={"which team?": AREA}))
        assert result.answers["which team?"].answer == "billing"

    def test_custom_input_classes_keep_their_state(self):
        class Ticket(JevInput):
            customer: str
            body: str

        judge = FakeJudge({"area": "billing"})
        cascade({"area": UNSURE["area"]}, judge).evaluate(Ticket(customer="Ada", body="Charged twice.", questions={"area": AREA}))
        (input_data, _), = judge.calls
        assert input_data.state == {"customer": "Ada", "body": "Charged twice."}


class TestReviewFixes:
    def test_importing_the_package_does_not_load_the_jev_backend(self):
        import subprocess
        import sys

        code = "import sys, grounded_ai; print('grounded_ai.backends.jev' in sys.modules)"
        out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout.strip()
        assert out == "False"
        from grounded_ai import CascadeEvaluator as again  # still importable from the package
        assert again is CascadeEvaluator

    def test_escalated_follows_the_order_the_questions_were_asked(self):
        def handler(request):
            if request.url.path == "/health":
                return httpx.Response(404)
            body = json.loads(request.content)
            reply = {k: UNSURE[k] for k in reversed(list(body["questions"]))}  # server answers in another order
            return httpx.Response(200, json={"answers": reply})

        backend = JevEvaluator(use_local_model=True, local_model=MODEL, client=httpx.Client(transport=httpx.MockTransport(handler)))
        judge = FakeJudge({"area": "billing", "urgent": True, "clarity": "clear"})
        result = CascadeEvaluator(jev=backend, judge=judge).evaluate(ask(area=AREA, urgent=URGENT, clarity=CLARITY))
        assert result.escalated == ["area", "urgent", "clarity"]

    def test_judged_lists_what_the_judge_actually_answered(self):
        answers = {"area": UNSURE["area"], "urgent": CONFIDENT["urgent"]}
        ok = cascade(answers, FakeJudge({"area": "billing"})).evaluate(ask(area=AREA, urgent=URGENT))
        assert (ok.escalated, ok.judged) == (["area"], ["area"])
        failed = cascade(answers, FakeJudge(error=EvaluationError(error_code="500", message="x"))).evaluate(
            ask(area=AREA, urgent=URGENT))
        assert (failed.escalated, failed.judged) == (["area"], [])

    def test_a_judge_that_raises_keeps_the_jev_answers(self):
        class Raising(FakeJudge):
            def _call_backend(self, input_data, output_schema, **kwargs):
                raise RuntimeError("CUDA out of memory")

        result = cascade({"area": UNSURE["area"]}, Raising()).evaluate(ask(area=AREA))
        assert isinstance(result, CascadeOutput)
        assert result.answers["area"] == ChoiceAnswer(**UNSURE["area"])
        assert result.judge_error.error_code == "JUDGE_ERROR"
        assert "CUDA out of memory" in result.judge_error.message

    def test_a_judge_that_ignores_the_schema_is_a_judge_error(self):
        class WrongShape(FakeJudge):
            def _call_backend(self, input_data, output_schema, **kwargs):
                return EvaluationOutput(score=1.0, label="x", confidence=1.0)

        result = cascade({"area": UNSURE["area"]}, WrongShape()).evaluate(ask(area=AREA))
        assert result.judge_error.error_code == "JUDGE_ERROR"
        assert result.judged == []

    def test_judges_that_cannot_honour_an_output_schema_are_refused(self):
        with pytest.raises(TypeError, match="judge"):
            CascadeEvaluator(jev=fake_jev(CONFIDENT), judge=fake_jev(CONFIDENT))

        class GroundedAISLMBackend(FakeJudge):  # stands in for the SLM backend without loading torch
            pass

        with pytest.raises(TypeError, match="judge"):
            CascadeEvaluator(jev=fake_jev(CONFIDENT), judge=GroundedAISLMBackend())

        class HuggingFaceBackend(FakeJudge):
            task = "text-classification"

        with pytest.raises(TypeError, match="judge"):
            CascadeEvaluator(jev=fake_jev(CONFIDENT), judge=HuggingFaceBackend())

    def test_object_instructions_keep_their_key_order_in_the_prompt(self):
        question = NoulQuestion(instructions={"rule": "ports must be 443", "claim": "it listens on 8080"})
        prompt = JevLeftover(state="x", questions={"q": question}).formatted_prompt
        assert prompt.index("ports must be 443") < prompt.index("it listens on 8080")

    def test_leftover_has_only_what_the_judge_needs(self):
        assert set(JevLeftover.model_fields) == {"state", "questions"}


class TestFailures:
    def test_jev_failure_is_returned(self):
        result = CascadeEvaluator(jev=fake_jev(CONFIDENT, status=500), judge=FakeJudge()).evaluate(ask(area=AREA))
        assert isinstance(result, EvaluationError)
        assert result.error_code == "500"

    def test_judge_failure_keeps_the_jev_answers(self):
        boom = EvaluationError(error_code="429", message="rate limited")
        result = cascade({"area": UNSURE["area"], "urgent": CONFIDENT["urgent"]}, FakeJudge(error=boom)).evaluate(
            ask(area=AREA, urgent=URGENT))
        assert isinstance(result, CascadeOutput)
        assert result.judge_error == boom
        assert result.escalated == ["area"]
        assert result.answers["area"] == ChoiceAnswer(**UNSURE["area"])  # nothing better to return
        assert result.answers["urgent"] == NoulAnswer(**CONFIDENT["urgent"])


class TestConstruction:
    def test_from_model_strings(self):
        evaluator = CascadeEvaluator(jev="jev", jev_kwargs={"use_local_model": True, "local_model": MODEL}, judge="openai/gpt-4o-mini", judge_kwargs={"api_key": "k"})
        assert isinstance(evaluator.jev, JevEvaluator)
        assert evaluator.jev.model_name == MODEL
        assert evaluator.judge.model_name == "gpt-4o-mini"

    def test_from_evaluators(self):
        evaluator = CascadeEvaluator(jev=Evaluator("jev", use_local_model=True, local_model=MODEL), judge=FakeJudge())
        assert isinstance(evaluator.jev, JevEvaluator)

    def test_first_stage_must_be_jev(self):
        with pytest.raises(TypeError):
            CascadeEvaluator(jev=FakeJudge(), judge=FakeJudge())

    def test_keyword_inputs_like_evaluator(self):
        judge = FakeJudge()
        result = cascade(CONFIDENT, judge).evaluate(state="Charged twice.", questions={"area": AREA})
        assert result.answers["area"].choice == "billing"


@pytest.mark.asyncio
async def test_async():
    judge = FakeJudge({"area": "billing"})
    result = await cascade({"area": UNSURE["area"], "urgent": CONFIDENT["urgent"]}, judge).evaluate_async(
        ask(area=AREA, urgent=URGENT))
    assert result.escalated == ["area"]
    assert result.answers["area"].answer == "billing"


def test_judged_answer_is_not_a_measured_answer():
    """A judged answer carries the pick and the judge's reasoning, never probabilities."""
    fields = set(JudgedAnswer.model_fields)
    assert {"answer", "reasoning", "question_type", "judge"} <= fields
    assert not fields & {"probabilities", "confidence", "noul", "score"}
    assert issubclass(JudgedAnswer, BaseModel)


class TestStructuredCriteriaForTheJudge:
    def test_json_criteria_reach_the_judge_as_json(self):
        questions = {
            "team": ChoiceQuestion(instructions="Which team?", criteria={"billing": {"covers": ["refunds"]}, "bug": None}),
            "urgent": NoulQuestion(instructions="Urgent?", criteria={"true": {"signal": "deadline"}}),
            "size": ScoreQuestion(instructions="How big?", criteria=[{"level": "small"}, {"level": "large"}]),
        }
        prompt = JevLeftover(state="x", questions=questions).formatted_prompt
        assert "{'" not in prompt  # no Python repr
        assert '"covers"' in prompt and '"signal"' in prompt and '"level": "small"' in prompt

    def test_json_score_levels_make_a_usable_judge_schema(self):
        from grounded_ai.cascade import _judge_schema

        q = {"size": ScoreQuestion(instructions="How big?", criteria=[{"level": "small"}, {"level": "large"}])}
        schema = _judge_schema(q)
        answer = schema.model_json_schema()["properties"]["answer_0"]
        assert all(isinstance(v, str) for v in answer["enum"])
        schema(reasoning_0="r", answer_0=answer["enum"][1])  # a level the judge can actually return

    def test_missing_questions_is_an_evaluation_error(self):
        result = CascadeEvaluator(jev=fake_jev(CONFIDENT), judge=FakeJudge()).evaluate(state="s")
        assert isinstance(result, EvaluationError) and result.error_code == "INVALID_REQUEST"
