"""
Runs CascadeEvaluator end to end: a real `strands-decider serve` and a real LLM judge.

Skipped unless DECIDER_LIVE=1 and a judge is configured (CASCADE_JUDGE, or ANTHROPIC_API_KEY for the default):

    DECIDER_LIVE=1 pytest tests/integration/test_cascade_live.py -s

CASCADE_JUDGE overrides the judge (any Evaluator model string); CASCADE_JUDGE_REGION sets the Bedrock region.
"""

import os

import pytest

from grounded_ai import CascadeEvaluator
from grounded_ai.backends.jev import HALLUCINATION, ChoiceQuestion, JevInput, NoulQuestion
from grounded_ai.cascade import CascadeOutput, JudgedAnswer

pytestmark = pytest.mark.skipif(
    os.getenv("DECIDER_LIVE") != "1" or not (os.getenv("CASCADE_JUDGE") or os.getenv("ANTHROPIC_API_KEY")),
    reason="set DECIDER_LIVE=1 and CASCADE_JUDGE (or ANTHROPIC_API_KEY) to run against a real model and judge",
)

CHECKPOINT = os.getenv("DECIDER_CHECKPOINT", "StrandsAgents/strands-decider-2B-hobson-v19")
JUDGE = os.getenv("CASCADE_JUDGE", "anthropic/claude-haiku-4-5-20251001")


@pytest.fixture(scope="module")
def cascade():
    judge_kwargs = {"region_name": os.environ["CASCADE_JUDGE_REGION"]} if os.getenv("CASCADE_JUDGE_REGION") else {}
    evaluator = CascadeEvaluator(
        jev="jev", judge=JUDGE, jev_kwargs={"use_local_model": True, "local_model": CHECKPOINT, "timeout": 300.0}, judge_kwargs=judge_kwargs
    )
    evaluator.jev.warmup(port=int(os.getenv("DECIDER_PORT", "8000")), device=os.getenv("DECIDER_DEVICE"), timeout=1800.0)
    yield evaluator
    evaluator.jev.shutdown()


def show(result: CascadeOutput) -> None:
    print(f"\n  escalated: {result.escalated}")
    for name, answer in result.answers.items():
        print(f"  {name}: {answer.model_dump()}")


def test_confident_questions_stay_local_and_unsure_ones_are_judged(cascade):
    result = cascade.evaluate(JevInput(
        state={
            "context": "Michael Collins remained in orbit in the Command Module while Armstrong and Aldrin walked on the Moon.",
            "response": "Buzz Aldrin stayed in the orbiter while Neil went down alone.",
        },
        questions={
            # The real model answers this correctly but at confidence 0.73: below 0.9, so it should escalate.
            "verdict": HALLUCINATION,
            # An easy one the model is sure of: it should stay local.
            "english": ChoiceQuestion(instructions="Which language is the response in?",
                                      criteria={"english": "written in English", "french": "written in French"}),
        },
    ))
    show(result)
    assert isinstance(result, CascadeOutput), result
    assert result.judge_error is None
    assert "english" not in result.escalated
    assert result.answers["english"].choice == "english"
    assert "verdict" in result.escalated
    verdict = result.answers["verdict"]
    assert isinstance(verdict, JudgedAnswer)
    assert verdict.answer == "hallucination"
    assert verdict.reasoning


def test_judge_answers_a_yes_no_question(cascade):
    result = cascade.evaluate(JevInput(
        state={"language": "python",
               "code": "def run(user_input):\n    db.execute(f\"SELECT * FROM users WHERE name = '{user_input}'\")"},
        # The real model scores this about 0.5, so |2p - 1| is near 0: it should escalate.
        questions={"security_risk": NoulQuestion(instructions="The code has a security vulnerability.")},
    ))
    show(result)
    assert result.escalated == ["security_risk"]
    assert result.answers["security_risk"].answer is True
