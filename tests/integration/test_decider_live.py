"""
Runs the Decider backend against a real `strands-decider serve` and a real checkpoint.

Skipped unless DECIDER_LIVE=1, because it downloads the checkpoint and loads a 2B model:

    pip install -e ".[decider]" pytest
    DECIDER_LIVE=1 pytest tests/integration -s

DECIDER_CHECKPOINT, DECIDER_PORT and DECIDER_DEVICE override the defaults.
"""

import os

import pytest

from grounded_ai import Evaluator
from grounded_ai.backends.decider import (
    HALLUCINATION,
    RAG_RELEVANCE,
    TOXICITY,
    ChoiceAnswer,
    ChoiceQuestion,
    DeciderInput,
    DeciderOutput,
    NoulAnswer,
    NoulQuestion,
    ScoreAnswer,
    ScoreQuestion,
)
from grounded_ai.schemas import EvaluationError

pytestmark = pytest.mark.skipif(os.getenv("DECIDER_LIVE") != "1", reason="set DECIDER_LIVE=1 to run against a real model")

CHECKPOINT = os.getenv("DECIDER_CHECKPOINT", "StrandsAgents/strands-decider-2B-hobson-v19")
PORT = int(os.getenv("DECIDER_PORT", "8000"))

APOLLO = """
The Apollo 11 mission landed the first humans on the Moon.
Neil Armstrong and Buzz Aldrin walked on the lunar surface.
Michael Collins remained in orbit in the Command Module.
"""


@pytest.fixture(scope="module")
def evaluator():
    evaluator = Evaluator(f"decider/{CHECKPOINT}", timeout=300.0)
    evaluator.backend.warmup(port=PORT, device=os.getenv("DECIDER_DEVICE"), timeout=1800.0)
    yield evaluator
    evaluator.backend.shutdown()


def ask(evaluator, state, **questions) -> DeciderOutput:
    result = evaluator.evaluate(DeciderInput(state=state, questions=questions))
    assert isinstance(result, DeciderOutput), result
    print(f"\n{state if isinstance(state, str) else dict(state)}\n  -> {result.model_dump_json()}")
    return result


def p(answer: ChoiceAnswer, option: str) -> float:
    assert sum(answer.probabilities.values()) == pytest.approx(1.0, abs=0.01)
    return answer.probabilities[option]


def test_warmup_serves_the_checkpoint(evaluator):
    health = evaluator.backend.client.get(f"http://127.0.0.1:{PORT}/health").json()
    print("\n/health ->", health)
    assert health["checkpoint"] == CHECKPOINT


def test_hallucination_separates_an_accurate_answer_from_a_wrong_one(evaluator):
    def verdict(response):
        state = {"context": APOLLO, "query": "Who stayed in orbit?", "response": response}
        return ask(evaluator, state, verdict=HALLUCINATION).answers["verdict"]

    accurate = verdict("Michael Collins remained in orbit.")
    wrong = verdict("Buzz Aldrin stayed in the orbiter while Neil went down alone.")
    assert isinstance(accurate, ChoiceAnswer)
    assert set(accurate.probabilities) == {"hallucination", "faithful"}
    assert p(wrong, "hallucination") > p(accurate, "hallucination")
    assert (accurate.choice, wrong.choice) == ("faithful", "hallucination")


def test_toxicity(evaluator):
    def tone(text):
        return ask(evaluator, {"response": text}, tone=TOXICITY).answers["tone"]

    toxic = tone("You are a worthless idiot and everyone hates you.")
    civil = tone("Thanks for the quick reply, that fixed it.")
    assert (toxic.choice, civil.choice) == ("toxic", "non-toxic")
    assert 0.0 <= civil.confidence <= 1.0


def test_rag_relevance(evaluator):
    def relevance(response):
        state = {"query": "What are the benefits of vitamin D?", "response": response}
        return ask(evaluator, state, relevance=RAG_RELEVANCE).answers["relevance"]

    on_topic = relevance("Vitamin D helps the body use calcium, which keeps bones strong.")
    off_topic = relevance("The Eiffel Tower is located in Paris, France.")
    assert p(on_topic, "relevant") > p(off_topic, "relevant")


def test_all_three_question_types_in_one_request(evaluator):
    result = ask(
        evaluator,
        "You have charged me twice and my account is now overdrawn. I need this reversed today.",
        urgent=NoulQuestion(instructions="This needs a reply within the hour."),
        area=ChoiceQuestion(
            instructions="Which team owns it?",
            criteria={
                "billing": "charges and refunds",
                "bug": "the product misbehaves",
                "account": "login and profile",
            },
        ),
        clarity=ScoreQuestion(
            instructions="How clearly is the problem described?",
            criteria=["unclear", "partly clear", "clear"],
        ),
    )
    urgent, area, clarity = result.answers["urgent"], result.answers["area"], result.answers["clarity"]
    assert isinstance(urgent, NoulAnswer) and 0.0 <= urgent.noul <= 1.0
    assert isinstance(area, ChoiceAnswer) and area.choice == "billing"
    assert isinstance(clarity, ScoreAnswer) and 0.0 <= clarity.score <= 2.0
    assert set(clarity.probabilities) == {"0", "1", "2"}
    assert clarity.legend == {"0": "unclear", "1": "partly clear", "2": "clear"}
    assert result.usage["output_tokens"] == 3


def test_custom_input_class(evaluator):
    class CodeReview(DeciderInput):
        language: str
        code: str

    result = evaluator.evaluate(CodeReview(
        language="python",
        code="def run(user_input):\n    db.execute(f\"SELECT * FROM users WHERE name = '{user_input}'\")",
        questions={"security_risk": NoulQuestion(instructions="The code has a security vulnerability.")},
    ))
    print("\n  ->", result.model_dump_json())
    assert isinstance(result.answers["security_risk"], NoulAnswer)


def test_wrong_model_name_is_refused(evaluator):
    other = Evaluator("decider/some-other-checkpoint", base_url=f"http://127.0.0.1:{PORT}")
    result = other.evaluate(DeciderInput(state="x", questions={"tone": TOXICITY}))
    assert isinstance(result, EvaluationError)
    assert result.error_code == "MODEL_MISMATCH"
