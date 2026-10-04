"""
CascadeEvaluator: a decision model answers first; an LLM judge answers only what it was unsure of.

The Decider answers every question in one cheap request, with measured probabilities. Each answer
whose confidence is below `min_confidence` is escalated: the original state and just those leftover
questions go to an LLM judge as a custom evaluation input (DeciderLeftover), in one call. Confident
answers are returned as the Decider gave them; escalated ones come back as JudgedAnswer, which
carries the judge's pick and reasoning and no probabilities, because an LLM does not measure any.

Confidence follows the Decider's own definitions. A choice or score answer has a `confidence`
field (normalised max-probability for a choice, normalised spread for a score). A yes/no (noul)
answer has none, because its probability is the uncertainty; it is read as |2p - 1|, which is the
choice formula at two options.

The default threshold, 0.9, is the top band of the Decider's routing convention: on held-out
short classification, v19's answers at 0.9 or above were right 0.952 of the time, against 0.655
from 0.5 to 0.9. On long documents the model is under-confident, so the same threshold escalates
more than it needs to there. Measure on your own traffic.
"""

import json
from typing import Any, Dict, List, Literal, Optional, Tuple, Type, Union

from pydantic import BaseModel, Field, TypeAdapter, computed_field, create_model
from typing_extensions import Annotated

from .base import BaseEvaluator
from .backends.decider import (
    ChoiceAnswer,
    Content,
    DeciderBackend,
    DeciderInput,
    DeciderOutput,
    NoulAnswer,
    Question,
    ScoreAnswer,
)
from .schemas import EvaluationError, EvaluationInput

_QUESTIONS = TypeAdapter(Dict[str, Question])


class JudgedAnswer(BaseModel):
    """An answer the LLM judge gave for a question the Decider was not confident about.

    It has the judge's pick and reasoning, and deliberately no probabilities or confidence.
    """

    type: Literal["judged"] = "judged"
    question_type: Literal["noul", "choice", "score"] = Field(description="The type of question it answers")
    answer: Union[bool, str] = Field(description="True/False for a noul, the option for a choice, the level for a score")
    reasoning: str = Field(description="The judge's reasoning, written before its answer")
    judge: str = Field(description="The model that answered")


CascadeAnswer = Annotated[Union[NoulAnswer, ChoiceAnswer, ScoreAnswer, JudgedAnswer], Field(discriminator="type")]


class CascadeOutput(BaseModel):
    """One answer per question: the Decider's where it was confident, the judge's where it was not."""

    answers: Dict[str, CascadeAnswer]
    escalated: List[str] = Field(description="Questions sent to the judge, in the order asked")
    decider: DeciderOutput = Field(description="The Decider's full first-stage answers, escalated ones included")
    judge_error: Optional[EvaluationError] = Field(
        None, description="Set when the judge failed; escalated questions then keep the Decider's answer"
    )


_LEFTOVER_TEMPLATE = """Read the state, then answer every question below about it.

<state>
{{ state_text }}
</state>
{% for name, q in questions.items() %}
Question {{ loop.index0 }} ("{{ name }}"):
{% if q.instructions is string %}{{ q.instructions }}{% else %}{{ q.instructions | tojson }}{% endif %}
{% if q.type == "noul" -%}
Answer true if this statement holds for the state, false if it does not.
{%- if q.criteria %}{% for value, meaning in q.criteria.items() %}
- {{ value }}: {{ meaning }}{% endfor %}{% endif %}
{%- elif q.type == "choice" -%}
Pick exactly one option:
{%- for option, description in q.criteria.items() %}
- {{ option }}{% if description %}: {{ description }}{% endif %}{% endfor %}
{%- else -%}
Pick exactly one level (they are ordered, lowest first):
{%- for level in q.criteria %}
- {{ level }}{% endfor %}
{%- endif %}
{% endfor %}"""


class DeciderLeftover(EvaluationInput):
    """The custom evaluation input the judge receives: the Decider's state and the questions it
    was not confident about, rendered as one prompt."""

    state: Content
    questions: Dict[str, Question]
    base_template: str = _LEFTOVER_TEMPLATE

    @computed_field
    @property
    def state_text(self) -> str:
        """The state as the judge reads it. JSON keeps the declared field order (Jinja's tojson
        would sort the keys), so evidence still comes before the text being judged."""
        if isinstance(self.state, str):
            return self.state
        return json.dumps(self.state, indent=2, ensure_ascii=False)


def _confidence(answer: Union[NoulAnswer, ChoiceAnswer, ScoreAnswer]) -> float:
    if isinstance(answer, NoulAnswer):
        return abs(2 * answer.noul - 1)
    return answer.confidence


def _judge_schema(questions: Dict[str, Any]) -> Type[BaseModel]:
    """A flat output schema: reasoning_i then answer_i per question, answers restricted to the
    question's own options. Flat on purpose: every backend's structured output accepts it."""
    fields: Dict[str, Tuple[Any, Any]] = {}
    for i, (name, question) in enumerate(questions.items()):
        if question.type == "noul":
            answer_type: Any = bool
        else:
            options = list(question.criteria)  # choice: option names; score: levels, lowest first
            answer_type = Literal[tuple(options)]
        fields[f"reasoning_{i}"] = (str, Field(description=f"Reasoning for question {i} ({name!r}), before its answer"))
        fields[f"answer_{i}"] = (answer_type, Field(description=name))
    return create_model("JudgedAnswers", **fields)


def _backend(target: Any, kwargs: Dict[str, Any]) -> BaseEvaluator:
    if isinstance(target, str):
        from . import Evaluator

        return Evaluator(target, **kwargs).backend
    backend = getattr(target, "backend", target)  # an Evaluator, or a backend itself
    if not isinstance(backend, BaseEvaluator):
        raise TypeError(f"Expected a model string, an Evaluator or a backend, got {type(target).__name__}.")
    return backend


def _model_name(backend: BaseEvaluator) -> str:
    return str(getattr(backend, "model_name", None) or getattr(backend, "model_id", None) or type(backend).__name__)


class CascadeEvaluator:
    """
    Decider first, LLM judge for what it is unsure of.

        cascade = CascadeEvaluator(
            decider="decider/StrandsAgents/strands-decider-2B-hobson-v19",
            judge="anthropic/claude-haiku-4-5",
            min_confidence=0.9,
        )
        result = cascade.evaluate(DeciderInput(state=..., questions={...}))
        result.answers["verdict"]   # the Decider's answer, or a JudgedAnswer
        result.escalated            # which questions went to the judge

    `decider` and `judge` take a model string, an Evaluator or a backend. `decider_kwargs` and
    `judge_kwargs` are passed to Evaluator when a model string is given (base_url, api_key, ...).
    """

    def __init__(
        self,
        decider: Any,
        judge: Any,
        min_confidence: float = 0.9,
        decider_kwargs: Optional[Dict[str, Any]] = None,
        judge_kwargs: Optional[Dict[str, Any]] = None,
    ):
        if not 0.0 <= min_confidence <= 1.0:
            raise ValueError(f"min_confidence must be in [0, 1], got {min_confidence!r}.")
        self.decider = _backend(decider, decider_kwargs or {})
        if not isinstance(self.decider, DeciderBackend):
            raise TypeError(f"The first stage must be a Decider backend, got {type(self.decider).__name__}.")
        self.judge = _backend(judge, judge_kwargs or {})
        self.min_confidence = min_confidence

    # Shared by evaluate() and evaluate_async(): everything except the two backend calls.

    def _input(self, input_data: Any, kwargs: Dict[str, Any]) -> Tuple[DeciderInput, Dict[str, Any]]:
        from . import prepare_input

        input_data, backend_kwargs = prepare_input(self.decider, input_data, kwargs)
        if isinstance(input_data, dict):
            input_data = self.decider.input_schema(**input_data)
        return input_data, backend_kwargs

    def _leftover(self, input_data: DeciderInput, first: DeciderOutput) -> Tuple[List[str], Optional[DeciderLeftover]]:
        escalated = [name for name, answer in first.answers.items() if _confidence(answer) < self.min_confidence]
        if not escalated:
            return [], None
        questions = _QUESTIONS.validate_python(
            input_data.model_dump(include={"questions"}, exclude_none=True)["questions"]
        )
        leftover = DeciderLeftover(
            state=input_data.build_state(), questions={name: questions[name] for name in escalated}
        )
        return escalated, leftover

    def _combine(self, first: DeciderOutput, escalated: List[str], leftover: Optional[DeciderLeftover], judged: Any) -> CascadeOutput:
        answers: Dict[str, Any] = dict(first.answers)
        if isinstance(judged, EvaluationError):
            return CascadeOutput(answers=answers, escalated=escalated, decider=first, judge_error=judged)
        if leftover is not None:
            judge = _model_name(self.judge)
            for i, (name, question) in enumerate(leftover.questions.items()):
                answers[name] = JudgedAnswer(
                    question_type=question.type,
                    answer=getattr(judged, f"answer_{i}"),
                    reasoning=getattr(judged, f"reasoning_{i}"),
                    judge=judge,
                )
        return CascadeOutput(answers=answers, escalated=escalated, decider=first)

    def evaluate(self, input_data: Any = None, **kwargs) -> Union[CascadeOutput, EvaluationError]:
        """Takes the same inputs as Evaluator("decider/...").evaluate(): a DeciderInput (or subclass),
        a dict, a bare string as the state, or keywords such as state= and questions=."""
        input_data, backend_kwargs = self._input(input_data, kwargs)
        first = self.decider.evaluate(input_data, **backend_kwargs)
        if isinstance(first, EvaluationError):
            return first
        escalated, leftover = self._leftover(input_data, first)
        judged = None
        if leftover is not None:
            judged = self.judge.evaluate(leftover, output_schema=_judge_schema(leftover.questions))
        return self._combine(first, escalated, leftover, judged)

    async def evaluate_async(self, input_data: Any = None, **kwargs) -> Union[CascadeOutput, EvaluationError]:
        input_data, backend_kwargs = self._input(input_data, kwargs)
        first = await self.decider.evaluate_async(input_data, **backend_kwargs)
        if isinstance(first, EvaluationError):
            return first
        escalated, leftover = self._leftover(input_data, first)
        judged = None
        if leftover is not None:
            judged = await self.judge.evaluate_async(leftover, output_schema=_judge_schema(leftover.questions))
        return self._combine(first, escalated, leftover, judged)
