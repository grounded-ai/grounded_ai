"""
The Decider backend's contract, both directions:

- Input: a variety of DeciderInput shapes must map onto the `/v1/systemone` request exactly.
- Output: a variety of server responses must parse into the right answer fields, and anything
  that does not answer what was asked must be refused.

When `strands-decider` is installed (it is in the live CI job), every request and response here
is also checked against that package's own wire types, so the two cannot drift apart.
"""

import datetime
import enum
import json
from typing import List, Optional

import httpx
import pytest
from pydantic import BaseModel, ValidationError

from grounded_ai.backends.decider import (
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
from grounded_ai.schemas import EvaluationError

try:
    from strands_decider import schema as server_schema
except ImportError:  # the unit-test CI job does not install the server
    server_schema = None

MODEL = "strands-decider-2B-hobson-v19"


def default_answer(question):
    if question["type"] == "noul":
        return {"type": "noul", "noul": 0.5}
    if question["type"] == "choice":
        names = list(question["criteria"])
        return {
            "type": "choice", "choice": names[0], "confidence": 0.0,
            "probabilities": {n: round(1 / len(names), 4) for n in names},
        }
    levels = len(question["criteria"])
    return {
        "type": "score", "score": 0.0, "confidence": 0.0,
        "legend": {str(i): text for i, text in enumerate(question["criteria"])},
        "probabilities": {str(i): round(1 / levels, 4) for i in range(levels)},
    }


def backend(seen=None, answers=None, body=None):
    """A backend whose server replies with `answers` (default: a valid answer per question),
    or with the raw `body` when given. `seen` collects the request bodies."""

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            return httpx.Response(404)
        sent = json.loads(request.content)
        if seen is not None:
            seen.append(sent)
        if body is not None:
            return body if isinstance(body, httpx.Response) else httpx.Response(200, json=body)
        reply = answers if answers is not None else {n: default_answer(q) for n, q in sent["questions"].items()}
        return httpx.Response(200, json={"model": MODEL, "answers": reply, "usage": {"input_tokens": 9, "output_tokens": len(reply)}})

    return DeciderBackend(model_name=MODEL, client=httpx.Client(transport=httpx.MockTransport(handler)))


def sent_body(input_data):
    seen = []
    result = backend(seen).evaluate(input_data)
    assert isinstance(result, DeciderOutput), result
    (body,) = seen
    if server_schema is not None:  # the real server must accept exactly what we send
        server_schema.SystemOneRequest(**body)
    return body


# --- Input: questions -----------------------------------------------------------------------

QUESTIONS = [
    pytest.param(
        NoulQuestion(instructions="The response is polite."),
        {"type": "noul", "instructions": "The response is polite."},
        id="noul",
    ),
    pytest.param(
        NoulQuestion(instructions="It is urgent.", criteria={"true": "needs a reply today", "false": "can wait"}),
        {"type": "noul", "instructions": "It is urgent.", "criteria": {"true": "needs a reply today", "false": "can wait"}},
        id="noul-with-both-criteria",
    ),
    pytest.param(
        NoulQuestion(instructions="It is urgent.", criteria={"true": "needs a reply today"}),
        {"type": "noul", "instructions": "It is urgent.", "criteria": {"true": "needs a reply today"}},
        id="noul-with-one-criterion",
    ),
    pytest.param(
        NoulQuestion(instructions={"claim": "The refund window is 30 days.", "scope": "policy"}),
        {"type": "noul", "instructions": {"claim": "The refund window is 30 days.", "scope": "policy"}},
        id="noul-instructions-as-object",
    ),
    pytest.param(
        NoulQuestion(instructions=["Check tone.", "Check facts."]),
        {"type": "noul", "instructions": ["Check tone.", "Check facts."]},
        id="noul-instructions-as-list",
    ),
    pytest.param(
        ChoiceQuestion(instructions="Which team?", criteria={"billing": "charges", "bug": "defects"}),
        {"type": "choice", "instructions": "Which team?", "criteria": {"billing": "charges", "bug": "defects"}},
        id="choice-two-options",
    ),
    pytest.param(
        ChoiceQuestion(instructions="Which team?", criteria={"billing": "", "bug": "", "account": ""}),
        {"type": "choice", "instructions": "Which team?", "criteria": {"billing": "", "bug": "", "account": ""}},
        id="choice-empty-descriptions",
    ),
    pytest.param(
        ChoiceQuestion(instructions="Quelle langue ?", criteria={"français": "écrit en français", "日本語": "日本語で書かれている"}),
        {"type": "choice", "instructions": "Quelle langue ?", "criteria": {"français": "écrit en français", "日本語": "日本語で書かれている"}},
        id="choice-unicode",
    ),
    pytest.param(
        ChoiceQuestion(instructions="Pick one.", criteria={f"option-{i}": f"description {i}" for i in range(255)}),
        {"type": "choice", "instructions": "Pick one.", "criteria": {f"option-{i}": f"description {i}" for i in range(255)}},
        id="choice-255-options",
    ),
    pytest.param(
        ScoreQuestion(instructions="How clear?", criteria=["unclear", "clear"]),
        {"type": "score", "instructions": "How clear?", "criteria": ["unclear", "clear"]},
        id="score-two-levels",
    ),
    pytest.param(
        ScoreQuestion(instructions="Rate 1-10.", criteria=[str(i) for i in range(1, 11)]),
        {"type": "score", "instructions": "Rate 1-10.", "criteria": ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10"]},
        id="score-ten-levels",
    ),
]


@pytest.mark.parametrize("question,wire", QUESTIONS)
def test_question_maps_to_the_wire(question, wire):
    body = sent_body(DeciderInput(state="x", questions={"q": question}))
    assert body == {"state": "x", "model": MODEL, "questions": {"q": wire}}


@pytest.mark.parametrize("question,wire", QUESTIONS)
def test_question_given_as_a_dict_maps_the_same(question, wire):
    body = sent_body(DeciderInput(state="x", questions={"q": wire}))
    assert body["questions"] == {"q": wire}


def test_many_questions_keep_their_names_and_order():
    questions = {
        "z_last_alphabetically_first_here": NoulQuestion(instructions="A."),
        "area": ChoiceQuestion(instructions="B?", criteria={"x": "", "y": ""}),
        "a score": ScoreQuestion(instructions="C?", criteria=["low", "high"]),
        "mixed": {"type": "noul", "instructions": "D."},
    }
    body = sent_body(DeciderInput(state="x", questions=questions))
    assert list(body["questions"]) == list(questions)
    assert [q["type"] for q in body["questions"].values()] == ["noul", "choice", "score", "noul"]


BAD_QUESTIONS = [
    pytest.param({}, id="no-questions"),
    pytest.param({"q": {"type": "essay", "instructions": "Write."}}, id="unknown-type"),
    pytest.param({"q": {"instructions": "No type."}}, id="missing-type"),
    pytest.param({"q": {"type": "noul"}}, id="noul-without-instructions"),
    pytest.param({"q": {"type": "noul", "instructions": "Q.", "criteria": {"maybe": ""}}}, id="noul-criteria-wrong-key"),
    pytest.param({"q": {"type": "noul", "instructions": "Q.", "temperature": 0.2}}, id="stray-key"),
    pytest.param({"q": {"type": "choice", "instructions": "Q?"}}, id="choice-without-criteria"),
    pytest.param({"q": {"type": "choice", "instructions": "Q?", "criteria": {"only": ""}}}, id="choice-one-option"),
    pytest.param({"q": {"type": "choice", "instructions": "Q?", "criteria": ["a", "b"]}}, id="choice-criteria-as-list"),
    pytest.param({"q": {"type": "choice", "instructions": "Q?", "criteria": {str(i): "" for i in range(256)}}}, id="choice-256-options"),
    pytest.param({"q": {"type": "choice", "instructions": "Q?", "criteria": {"a": None, "b": ""}}}, id="choice-null-description"),
    pytest.param({"q": {"type": "score", "instructions": "Q?", "criteria": ["only"]}}, id="score-one-level"),
    pytest.param({"q": {"type": "score", "instructions": "Q?", "criteria": [str(i) for i in range(11)]}}, id="score-eleven-levels"),
    pytest.param({"q": {"type": "score", "instructions": "Q?", "criteria": {"low": "", "high": ""}}}, id="score-criteria-as-dict"),
]


@pytest.mark.parametrize("questions", BAD_QUESTIONS)
def test_bad_questions_are_refused_by_the_input_class(questions):
    with pytest.raises(ValidationError):
        DeciderInput(state="x", questions=questions)


@pytest.mark.parametrize("questions", BAD_QUESTIONS)
def test_bad_questions_never_reach_the_server_even_from_a_loose_subclass(questions):
    class Loose(DeciderInput):
        questions: dict

    seen = []
    result = backend(seen).evaluate(Loose(state="x", questions=questions))
    assert isinstance(result, EvaluationError)
    assert result.error_code == "INVALID_REQUEST"
    assert seen == []


# --- Input: state ---------------------------------------------------------------------------

STATES = [
    pytest.param("Charged twice.", id="text"),
    pytest.param("  leading and trailing space is kept  ", id="text-with-whitespace"),
    pytest.param("Zażółć gęślą jaźń 🙂 日本語", id="unicode"),
    pytest.param({"context": "c", "response": "r"}, id="object"),
    pytest.param({"ticket": {"id": 7, "tags": ["billing", "urgent"], "paid": True, "notes": None, "amount": 12.5}}, id="nested-object"),
    pytest.param(["turn 1", "turn 2"], id="list-of-text"),
    pytest.param([{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}], id="list-of-objects"),
    pytest.param({}, id="empty-object"),
]


@pytest.mark.parametrize("state", STATES)
def test_state_is_sent_as_given(state):
    body = sent_body(DeciderInput(state=state, questions={"q": NoulQuestion(instructions="Q.")}))
    assert body["state"] == state


class Tier(str, enum.Enum):
    FREE = "free"
    PRO = "pro"


class Customer(BaseModel):
    name: str
    tier: Tier


class Ticket(DeciderInput):
    """A customized input with one field of each kind people are likely to add."""

    customer: Customer
    body: str
    attempts: int
    refunded: bool
    amount: float
    tags: List[str] = []
    opened: datetime.date
    note: Optional[str] = None


def test_subclass_fields_map_to_a_json_object_state():
    ticket = Ticket(
        customer=Customer(name="Ada", tier=Tier.PRO), body="Charged twice.", attempts=2, refunded=False,
        amount=19.99, tags=["billing"], opened=datetime.date(2026, 10, 3),
        questions={"q": NoulQuestion(instructions="Q.")},
    )
    body = sent_body(ticket)
    assert body["state"] == {
        "customer": {"name": "Ada", "tier": "pro"},
        "body": "Charged twice.",
        "attempts": 2,
        "refunded": False,
        "amount": 19.99,
        "tags": ["billing"],
        "opened": "2026-10-03",
    }
    assert list(body["state"]) == ["customer", "body", "attempts", "refunded", "amount", "tags", "opened"]  # declared order
    assert "note" not in body["state"]  # unset optional fields are left out
    assert "questions" not in body["state"]  # the request's own fields never leak into the state


@pytest.mark.parametrize(
    "rendered",
    [
        pytest.param("Rule: ports must be 443.\nText: Port 8080.", id="text"),
        pytest.param({"rule": "ports must be 443", "text": "Port 8080."}, id="object"),
        pytest.param(["Rule: ports must be 443.", "Port 8080."], id="list"),
    ],
)
def test_build_state_override_is_what_is_sent(rendered):
    class Custom(DeciderInput):
        text: str

        def build_state(self):
            return rendered

    body = sent_body(Custom(text="ignored by the override", questions={"q": NoulQuestion(instructions="Q.")}))
    assert body["state"] == rendered


def test_explicit_state_wins_over_subclass_fields():
    class Turn(DeciderInput):
        message: str

    body = sent_body(Turn(message="from the field", state="from state", questions={"q": NoulQuestion(instructions="Q.")}))
    assert body["state"] == "from state"


BAD_STATES = [
    pytest.param(None, id="nothing-set"),
    pytest.param("", id="empty-text"),
    pytest.param("   \n", id="blank-text"),
]


@pytest.mark.parametrize("state", BAD_STATES)
def test_empty_state_is_refused(state):
    seen = []
    result = backend(seen).evaluate(DeciderInput(state=state, questions={"q": NoulQuestion(instructions="Q.")}))
    assert result.error_code == "INVALID_REQUEST"
    assert seen == []


@pytest.mark.parametrize("value", [42, 3.14, True, object()], ids=["int", "float", "bool", "object"])
def test_build_state_returning_something_else_is_refused(value):
    class Broken(DeciderInput):
        def build_state(self):
            return value

    seen = []
    result = backend(seen).evaluate(Broken(questions={"q": NoulQuestion(instructions="Q.")}))
    assert result.error_code == "INVALID_REQUEST"
    assert seen == []


def test_the_request_has_exactly_the_contract_keys():
    body = sent_body(DeciderInput(state="x", questions={"q": NoulQuestion(instructions="Q.")}))
    assert set(body) == {"state", "questions", "model"}


# --- Output: parsing ------------------------------------------------------------------------

NOUL = {"q": NoulQuestion(instructions="Q.")}
CHOICE = {"q": ChoiceQuestion(instructions="Q?", criteria={"billing": "", "bug": "", "account": ""})}
SCORE = {"q": ScoreQuestion(instructions="Q?", criteria=["unclear", "partly clear", "clear"])}


def parse(questions, answers=None, body=None):
    return backend(answers=answers, body=body).evaluate(DeciderInput(state="x", questions=questions))


@pytest.mark.parametrize("value", [0.0, 0.0731, 0.5, 1.0, 1, 0], ids=str)
def test_noul_answer_fields(value):
    result = parse(NOUL, {"q": {"type": "noul", "noul": value}})
    answer = result.answers["q"]
    assert isinstance(answer, NoulAnswer)
    assert answer.noul == float(value) and isinstance(answer.noul, float)
    assert answer.model_dump() == {"type": "noul", "noul": float(value)}


def test_choice_answer_fields():
    raw = {"type": "choice", "choice": "billing", "probabilities": {"billing": 0.937, "bug": 0.0351, "account": 0.028}, "confidence": 0.9054}
    answer = parse(CHOICE, {"q": raw}).answers["q"]
    assert isinstance(answer, ChoiceAnswer)
    assert answer.choice == "billing"
    assert answer.probabilities == {"billing": 0.937, "bug": 0.0351, "account": 0.028}
    assert list(answer.probabilities) == ["billing", "bug", "account"]  # the order the options were given in
    assert answer.confidence == 0.9054
    assert answer.model_dump() == raw


def test_score_answer_fields():
    raw = {
        "type": "score", "score": 1.3147, "legend": {"0": "unclear", "1": "partly clear", "2": "clear"},
        "probabilities": {"0": 0.1876, "1": 0.31, "2": 0.5024}, "confidence": 0.3382,
    }
    answer = parse(SCORE, {"q": raw}).answers["q"]
    assert isinstance(answer, ScoreAnswer)
    assert answer.score == 1.3147
    assert answer.legend == {"0": "unclear", "1": "partly clear", "2": "clear"}
    assert answer.probabilities == {"0": 0.1876, "1": 0.31, "2": 0.5024}
    assert answer.confidence == 0.3382
    assert answer.model_dump() == raw


@pytest.mark.parametrize("score", [0.0, 2.0, 0, 2], ids=str)
def test_score_at_the_ends_of_the_scale(score):
    raw = {"type": "score", "score": score, "probabilities": {"0": 0.5, "1": 0.0, "2": 0.5}, "confidence": 0.0}
    answer = parse(SCORE, {"q": raw}).answers["q"]
    assert answer.score == float(score)
    assert answer.legend == {}  # optional on the wire


def test_mixed_answers_keep_their_names_and_types():
    questions = {"a": NOUL["q"], "b": CHOICE["q"], "c": SCORE["q"]}
    result = parse(questions)
    assert list(result.answers) == ["a", "b", "c"]
    assert [type(a) for a in result.answers.values()] == [NoulAnswer, ChoiceAnswer, ScoreAnswer]


def test_envelope_fields():
    body = {"model": MODEL, "answers": {"q": {"type": "noul", "noul": 0.5}}, "usage": {"input_tokens": 146, "output_tokens": 1}, "latency_ms": 3091.74}
    result = parse(NOUL, body=body)
    assert (result.model, result.usage, result.latency_ms) == (MODEL, {"input_tokens": 146, "output_tokens": 1}, 3091.74)


def test_envelope_fields_are_optional_and_unknown_ones_are_ignored():
    result = parse(NOUL, body={"answers": {"q": {"type": "noul", "noul": 0.5, "future_field": 1}}, "request_id": "abc"})
    assert (result.model, result.usage, result.latency_ms) == (None, None, None)
    assert result.answers["q"] == NoulAnswer(noul=0.5)


BAD_ANSWERS = [
    # malformed answers
    pytest.param(NOUL, {"q": {"type": "noul"}}, id="noul-missing-value"),
    pytest.param(NOUL, {"q": {"type": "noul", "noul": "high"}}, id="noul-not-a-number"),
    pytest.param(NOUL, {"q": {"type": "noul", "noul": None}}, id="noul-null"),
    pytest.param(NOUL, {"q": {"type": "noul", "noul": 1.2}}, id="noul-above-one"),
    pytest.param(NOUL, {"q": {"type": "noul", "noul": -0.2}}, id="noul-below-zero"),
    pytest.param(NOUL, {"q": {"noul": 0.5}}, id="answer-missing-type"),
    pytest.param(NOUL, {"q": {"type": "essay", "text": "..."}}, id="unknown-answer-type"),
    pytest.param(NOUL, {"q": 0.5}, id="answer-not-an-object"),
    pytest.param(CHOICE, {"q": {"type": "choice", "choice": "billing", "confidence": 0.9}}, id="choice-missing-probabilities"),
    pytest.param(CHOICE, {"q": {"type": "choice", "probabilities": {"billing": 1.0, "bug": 0.0, "account": 0.0}, "confidence": 1.0}}, id="choice-missing-choice"),
    pytest.param(CHOICE, {"q": {"type": "choice", "choice": "billing", "probabilities": {"billing": 1.0, "bug": 0.0, "account": 0.0}}}, id="choice-missing-confidence"),
    pytest.param(CHOICE, {"q": {"type": "choice", "choice": "billing", "probabilities": {"billing": 1.4, "bug": 0.0, "account": 0.0}, "confidence": 1.0}}, id="choice-probability-above-one"),
    pytest.param(CHOICE, {"q": {"type": "choice", "choice": "billing", "probabilities": {"billing": 1.0, "bug": 0.0, "account": 0.0}, "confidence": 1.7}}, id="choice-confidence-above-one"),
    pytest.param(SCORE, {"q": {"type": "score", "probabilities": {"0": 1.0, "1": 0.0, "2": 0.0}, "confidence": 1.0}}, id="score-missing-score"),
    pytest.param(SCORE, {"q": {"type": "score", "score": 1.0, "confidence": 1.0}}, id="score-missing-probabilities"),
    # well-formed answers that do not answer what was asked
    pytest.param(NOUL, {}, id="no-answer-for-the-question"),
    pytest.param(NOUL, {"other": {"type": "noul", "noul": 0.5}}, id="answer-under-another-name"),
    pytest.param(NOUL, {"q": {"type": "noul", "noul": 0.5}, "extra": {"type": "noul", "noul": 0.5}}, id="answer-nobody-asked-for"),
    pytest.param(NOUL, {"q": {"type": "choice", "choice": "a", "probabilities": {"a": 1.0, "b": 0.0}, "confidence": 1.0}}, id="choice-answer-to-a-noul-question"),
    pytest.param(CHOICE, {"q": {"type": "noul", "noul": 0.5}}, id="noul-answer-to-a-choice-question"),
    pytest.param(CHOICE, {"q": {"type": "choice", "choice": "refund", "probabilities": {"billing": 0.5, "bug": 0.3, "account": 0.2}, "confidence": 0.3}}, id="choice-not-among-the-options"),
    pytest.param(CHOICE, {"q": {"type": "choice", "choice": "billing", "probabilities": {"billing": 0.6, "bug": 0.4}, "confidence": 0.3}}, id="choice-probabilities-missing-an-option"),
    pytest.param(CHOICE, {"q": {"type": "choice", "choice": "billing", "probabilities": {"billing": 0.5, "bug": 0.3, "account": 0.1, "other": 0.1}, "confidence": 0.3}}, id="choice-probabilities-for-an-unknown-option"),
    pytest.param(SCORE, {"q": {"type": "score", "score": 2.6, "probabilities": {"0": 0.1, "1": 0.2, "2": 0.7}, "confidence": 0.5}}, id="score-beyond-the-top-level"),
    pytest.param(SCORE, {"q": {"type": "score", "score": -0.1, "probabilities": {"0": 0.1, "1": 0.2, "2": 0.7}, "confidence": 0.5}}, id="score-below-zero"),
    pytest.param(SCORE, {"q": {"type": "score", "score": 1.0, "probabilities": {"0": 0.5, "1": 0.5}, "confidence": 0.5}}, id="score-probabilities-missing-a-level"),
    pytest.param(SCORE, {"q": {"type": "score", "score": 1.0, "probabilities": {"low": 0.2, "mid": 0.3, "high": 0.5}, "confidence": 0.5}}, id="score-probabilities-not-keyed-by-level-index"),
]


@pytest.mark.parametrize("questions,answers", BAD_ANSWERS)
def test_bad_answers_are_invalid_response(questions, answers):
    result = parse(questions, answers)
    assert isinstance(result, EvaluationError), result
    assert result.error_code == "INVALID_RESPONSE"


BAD_BODIES = [
    pytest.param({}, id="no-answers-key"),
    pytest.param({"answers": None}, id="answers-null"),
    pytest.param({"answers": []}, id="answers-as-list"),
    pytest.param([], id="body-is-a-list"),
    pytest.param(httpx.Response(200, text="<html>502 Bad Gateway</html>"), id="body-is-not-json"),
    pytest.param(httpx.Response(200, text=""), id="empty-body"),
]


@pytest.mark.parametrize("body", BAD_BODIES)
def test_bad_bodies_are_invalid_response(body):
    result = parse(NOUL, body=body)
    assert isinstance(result, EvaluationError), result
    assert result.error_code == "INVALID_RESPONSE"


# --- Parity with the server's own wire types ------------------------------------------------

needs_server = pytest.mark.skipif(server_schema is None, reason="strands-decider is not installed")


@needs_server
@pytest.mark.parametrize(
    "ours,theirs",
    [
        (NoulQuestion, "NoulQuestion"), (ChoiceQuestion, "ChoiceQuestion"), (ScoreQuestion, "ScoreQuestion"),
        (NoulAnswer, "NoulAnswer"), (ChoiceAnswer, "ChoiceAnswer"), (ScoreAnswer, "ScoreAnswer"),
    ],
    ids=lambda v: v if isinstance(v, str) else "",
)
def test_our_classes_have_the_same_fields_as_the_servers(ours, theirs):
    assert set(ours.model_fields) == set(getattr(server_schema, theirs).model_fields)


@needs_server
@pytest.mark.parametrize("questions", BAD_QUESTIONS)
def test_the_server_refuses_the_same_bad_questions(questions):
    with pytest.raises(ValidationError):
        server_schema.SystemOneRequest(state="x", questions=questions)


@needs_server
@pytest.mark.parametrize("questions", [NOUL, CHOICE, SCORE], ids=["noul", "choice", "score"])
def test_a_response_built_by_the_servers_types_parses(questions):
    wire = {n: q.model_dump(exclude_none=True) for n, q in questions.items()}
    answers = {n: default_answer(q) for n, q in wire.items()}
    body = server_schema.SystemOneResponse(model=MODEL, answers=answers, usage=server_schema.Usage(input_tokens=5, output_tokens=1)).model_dump()
    result = parse(questions, body=body)
    assert isinstance(result, DeciderOutput), result
    assert result.answers["q"].model_dump() == body["answers"]["q"]
