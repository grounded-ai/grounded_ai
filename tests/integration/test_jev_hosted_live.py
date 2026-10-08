"""
JevEvaluator against TypeSafe's hosted Jev.

Skipped unless TYPESAFE_API_KEY is set. Each run costs a few thousand input tokens:

    TYPESAFE_API_KEY=... pytest tests/integration/test_jev_hosted_live.py -s
"""

import os

import pytest

from grounded_ai import Evaluator
from grounded_ai.backends.jev import (
    HALLUCINATION,
    ChoiceQuestion,
    JevInput,
    JevOutput,
    NoulQuestion,
    ScoreQuestion,
)

pytestmark = pytest.mark.skipif(
    not os.getenv("TYPESAFE_API_KEY"), reason="set TYPESAFE_API_KEY to call hosted Jev"
)

CONTEXT = "Michael Collins remained in orbit in the Command Module while Armstrong and Aldrin walked on the Moon."


@pytest.fixture(scope="module")
def evaluator():
    return Evaluator(os.getenv("JEV_MODEL", "jev/jev-latest"))


def ask(evaluator, state, questions):
    result = evaluator.evaluate(JevInput(state=state, questions=questions))
    assert isinstance(result, JevOutput), result
    print(
        f"\n{result.model}: "
        + ", ".join(
            f"{k}={v.model_dump(exclude={'legend'})}" for k, v in result.answers.items()
        )
    )
    return result


def test_every_question_type_in_one_request(evaluator):
    result = ask(
        evaluator,
        "Help! My payouts have been failing for 3 days and I need this fixed today.",
        {
            "urgent": NoulQuestion(instructions="Does this convey urgency?"),
            "department": ChoiceQuestion(
                instructions="Which team should handle this?",
                criteria={
                    "billing": "Payments, invoicing, refunds",
                    "technical": "Bugs, outages, integrations",
                    "sales": "Pricing, upgrades, new accounts",
                },
            ),
            "frustration": ScoreQuestion(
                instructions="How frustrated is the customer?",
                criteria=["Calm", "Frustrated", "Very angry"],
            ),
        },
    )
    assert result.model.startswith("jev-")  # the versioned model that answered
    assert result.answers["urgent"].noul > 0.5
    assert result.answers["department"].choice in {"billing", "technical"}
    assert result.answers["frustration"].score >= 1.0


@pytest.mark.parametrize(
    "response,expected",
    [
        ("Michael Collins stayed in orbit.", "faithful"),
        (
            "Buzz Aldrin stayed in orbit while Collins walked on the Moon.",
            "hallucination",
        ),
    ],
)
def test_hallucination(evaluator, response, expected):
    result = ask(
        evaluator,
        {"context": CONTEXT, "response": response},
        {"verdict": HALLUCINATION},
    )
    assert result.answers["verdict"].choice == expected


def test_structured_and_null_criteria_are_accepted(evaluator):
    """Hosted Jev takes JSON and null descriptions, which the local model refuses."""
    result = ask(
        evaluator,
        "Refund request for order 1182, charged twice.",
        {
            "team": ChoiceQuestion(
                instructions="Which team?",
                criteria={
                    "billing": {"covers": ["refunds", "double charges"]},
                    "shipping": None,
                },
            ),
            "severity": ScoreQuestion(
                instructions="How severe?",
                criteria=[{"level": "minor"}, {"level": "major"}],
            ),
        },
    )
    assert result.answers["team"].choice == "billing"
