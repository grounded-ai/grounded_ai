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


# (query, retrieved chunk, relevant?) Hard cases on purpose: several unrelated chunks are on the
# query's topic but do not hold the answer.
RAG_CASES = [
    ("What are the benefits of vitamin D?", "Vitamin D helps the body absorb calcium, which keeps bones strong.", True),
    ("What are the benefits of vitamin D?", "The Eiffel Tower was completed in 1889 and is 330 metres tall.", False),
    ("What are the benefits of vitamin D?", "Vitamin D was first isolated in the 1920s by researchers studying rickets.", False),
    ("How long is the refund window?", "Customers may return items within 30 days of delivery for a full refund.", True),
    ("How long is the refund window?", "Our support team is available Monday to Friday, 9am to 5pm.", False),
    ("How long is the refund window?", "Refunds are issued to the original payment method once the return is received.", False),
    ("What port does the API listen on?", "By default the API server binds to 0.0.0.0 on port 8443.", True),
    ("What port does the API listen on?", "The API supports JSON and MessagePack request bodies.", False),
    ("Who wrote Pride and Prejudice?", "Pride and Prejudice is an 1813 novel by Jane Austen.", True),
    ("Who wrote Pride and Prejudice?", "The novel has been adapted for film and television many times.", False),
    ("What is the capital of Australia?", "Canberra was selected as the capital in 1908 as a compromise between Sydney and Melbourne.", True),
    ("What is the capital of Australia?", "Sydney is Australia's largest city and home to its famous opera house.", False),
    ("When does the store open on Sundays?", "On Sundays we open at 10am and close at 4pm.", True),
    ("When does the store open on Sundays?", "We are closed on public holidays, including Christmas Day.", False),
    ("When does the store open on Sundays?", "The store has been family-owned since 1972.", False),
    ("What is the maximum file size for uploads?", "Uploads are limited to 25 MB per file.", True),
    ("What is the maximum file size for uploads?", "Supported upload formats are PNG, JPEG and PDF.", False),
    ("How do I reset my password?", "Click 'Forgot password' on the sign-in page and follow the emailed link.", True),
    ("How do I reset my password?", "Passwords must be at least 12 characters long.", False),
    ("What causes tides?", "Tides are caused mainly by the gravitational pull of the Moon on Earth's oceans.", True),
    ("What causes tides?", "The highest tides in the world occur in the Bay of Fundy.", False),
    ("Which language is the backend written in?", "The backend service is implemented in Go 1.22.", True),
    ("Which language is the backend written in?", "The frontend is a React single-page app.", False),
    ("Is the warranty transferable?", "The two-year warranty stays with the product if it is sold or given away.", True),
    ("Is the warranty transferable?", "The warranty covers manufacturing defects but not accidental damage.", False),
]


def test_rag_relevance_judges_retrieved_chunks(evaluator):
    """Every labelled chunk is classified correctly, and every relevant chunk scores above every unrelated one."""
    relevant_p, unrelated_p, wrong = [], [], []
    for query, chunk, relevant in RAG_CASES:
        answer = evaluator.evaluate(
            DeciderInput(state={"query": query, "context": chunk}, questions={"relevance": RAG_RELEVANCE})
        ).answers["relevance"]
        (relevant_p if relevant else unrelated_p).append(p(answer, "relevant"))
        if (answer.choice == "relevant") != relevant:
            wrong.append(f"{answer.probabilities} for {chunk!r}")
    print(f"\nRAG relevance: {len(RAG_CASES) - len(wrong)}/{len(RAG_CASES)} correct, "
          f"lowest relevant {min(relevant_p):.3f}, highest unrelated {max(unrelated_p):.3f}")
    assert not wrong, wrong
    assert min(relevant_p) > max(unrelated_p)


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


VARIETY = [
    pytest.param("Charged twice.", NoulQuestion(instructions="This is a complaint."), NoulAnswer, id="noul"),
    pytest.param(
        "Charged twice.",
        NoulQuestion(instructions="It is urgent.", criteria={"true": "needs a reply today", "false": "can wait"}),
        NoulAnswer,
        id="noul-with-criteria",
    ),
    pytest.param(
        {"policy": "Refunds within 30 days.", "response": "You have 90 days."},
        NoulQuestion(instructions={"claim": "The response agrees with the policy.", "scope": "refund window"}),
        NoulAnswer,
        id="noul-instructions-as-object",
    ),
    pytest.param(
        [{"role": "user", "content": "Why was I charged twice?"}, {"role": "assistant", "content": "Sorry! Refunded."}],
        ChoiceQuestion(instructions="How does the assistant sound?", criteria={"apologetic": "", "defensive": "", "neutral": ""}),
        ChoiceAnswer,
        id="choice-over-a-list-state",
    ),
    pytest.param(
        "Merci beaucoup, tout fonctionne maintenant.",
        ChoiceQuestion(instructions="Quelle langue ?", criteria={"français": "écrit en français", "日本語": "日本語で書かれている"}),
        ChoiceAnswer,
        id="choice-unicode",
    ),
    pytest.param(
        "The invoice total is wrong.",
        ChoiceQuestion(instructions="Which topic?", criteria={f"topic-{i}": f"about subject number {i}" for i in range(24)}),
        ChoiceAnswer,
        id="choice-24-options",
    ),
    pytest.param(
        {"ticket": {"id": 7, "tags": ["billing", "urgent"], "paid": True, "notes": None, "amount": 12.5}},
        ScoreQuestion(instructions="How serious is it?", criteria=["minor", "serious"]),
        ScoreAnswer,
        id="score-two-levels-nested-state",
    ),
    pytest.param(
        "The explanation was clear and complete.",
        ScoreQuestion(instructions="Rate the explanation from 1 to 10.", criteria=[str(i) for i in range(1, 11)]),
        ScoreAnswer,
        id="score-ten-levels",
    ),
]


@pytest.mark.parametrize("state,question,answer_type", VARIETY)
def test_the_real_server_accepts_and_answers_every_shape(evaluator, state, question, answer_type):
    """A DeciderOutput back means the request was accepted and the answer matched the question:
    its type, and its probabilities over exactly the options or levels that were asked."""
    answer = ask(evaluator, state, q=question).answers["q"]
    assert isinstance(answer, answer_type)
    if isinstance(question, ChoiceQuestion):
        assert list(answer.probabilities) == list(question.criteria)
        assert answer.choice == max(answer.probabilities, key=answer.probabilities.get)
    if isinstance(question, ScoreQuestion):
        assert answer.legend == {str(i): level for i, level in enumerate(question.criteria)}
        expected = sum(int(level) * probability for level, probability in answer.probabilities.items())
        assert answer.score == pytest.approx(expected, abs=0.01)  # the score is the expected level


def test_the_real_server_refuses_what_the_input_class_refuses(evaluator):
    """The contract check is not stricter or looser than the server on a one-option choice."""
    response = evaluator.backend.client.post(
        f"http://127.0.0.1:{PORT}/v1/systemone",
        json={"state": "x", "model": CHECKPOINT, "questions": {"q": {"type": "choice", "instructions": "Q?", "criteria": {"only": ""}}}},
    )
    assert response.status_code == 422


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
