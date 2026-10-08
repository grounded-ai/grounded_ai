"""
CascadeEvaluator: a decision model answers first; an LLM judge answers only what it was unsure of.

Jev answers every question in one cheap request, with measured probabilities. Each answer
whose confidence is below `min_confidence` is escalated: the original state and just those leftover
questions go to an LLM judge as a custom evaluation input (JevLeftover), in one call. Confident
answers are returned as Jev gave them; escalated ones come back as JudgedAnswer, which
carries the judge's pick and reasoning and no probabilities, because an LLM does not measure any.

Confidence follows TypeSafe's definitions (docs.typesafe.ai/confidence), which Strands Decider
shares. A choice or score answer has a `confidence` field (normalised max-probability for a choice,
normalised spread for a score). A yes/no (noul) answer has none, because its probability is the
uncertainty; it is read as |2p - 1|, which is the choice formula at two options.

The default threshold, 0.9, is where TypeSafe's own examples act without confirmation. The bands
behind it were measured on the local model, Strands Decider v19: on held-out short classification
its answers at 0.9 or above were right 0.952 of the time, against 0.655 from 0.5 to 0.9; on long
documents it is under-confident, so the same threshold escalates more than it needs to.
Measure on your own traffic.
"""

import json
from typing import Any, Dict, List, Literal, Optional, Tuple, Type, Union

from jinja2 import Template
from pydantic import BaseModel, Field, TypeAdapter, ValidationError, computed_field, create_model
from typing_extensions import Annotated

from .base import BaseEvaluator
from .backends.jev import (
    ChoiceAnswer,
    Content,
    JevEvaluator,
    JevInput,
    JevOutput,
    NoulAnswer,
    Question,
    ScoreAnswer,
)
from .schemas import EvaluationError

_QUESTIONS = TypeAdapter(Dict[str, Question])


class JudgedAnswer(BaseModel):
    """An answer the LLM judge gave for a question Jev was not confident about.

    It has the judge's pick and reasoning, and deliberately no probabilities or confidence. For a
    score question `answer` is the level as written in the question (Jev's ScoreAnswer gives
    an index instead); Jev's own answer is in CascadeOutput.jev.
    """

    type: Literal["judged"] = "judged"
    question_type: Literal["noul", "choice", "score"] = Field(description="The type of question it answers")
    answer: Union[bool, str] = Field(description="True/False for a noul, the option for a choice, the level for a score")
    reasoning: str = Field(description="The judge's reasoning, written before its answer")
    judge: str = Field(description="The model that answered")


CascadeAnswer = Annotated[Union[NoulAnswer, ChoiceAnswer, ScoreAnswer, JudgedAnswer], Field(discriminator="type")]


class CascadeOutput(BaseModel):
    """One answer per question: Jev's where it was confident, the judge's where it was not."""

    answers: Dict[str, CascadeAnswer]
    escalated: List[str] = Field(description="Questions sent to the judge, in the order they were asked")
    judged: List[str] = Field(description="Escalated questions the judge answered; empty when it failed")
    jev: JevOutput = Field(description="Jev's full first-stage answers, escalated ones included")
    judge_error: Optional[EvaluationError] = Field(
        None, description="Set when the judge failed; escalated questions then keep Jev's answer"
    )


_LEFTOVER_TEMPLATE = Template("""Read the state, then answer every question below about it.

<state>
{{ state }}
</state>
{% for name, q in questions %}
Question {{ loop.index0 }} ("{{ name }}"):
{{ q.instructions }}
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
{% endfor %}""")


def _as_text(content: Content) -> str:
    """Text as is; JSON in its declared key order (Jinja's tojson would sort the keys)."""
    return content if isinstance(content, str) else json.dumps(content, indent=2, ensure_ascii=False)


def _criteria_text(question: Question) -> Any:
    """Criteria as the judge reads them: JSON descriptions and levels as JSON text, never Python repr."""
    if question.criteria is None:
        return None
    if isinstance(question.criteria, dict):
        return {k: None if v is None else _as_text(v) for k, v in question.criteria.items()}
    return [_as_text(level) for level in question.criteria]


class JevLeftover(BaseModel):
    """The custom evaluation input the judge receives: Jev's state and the questions it
    was not confident about. Like any custom input, the LLM backends read its `formatted_prompt`."""

    state: Content
    questions: Dict[str, Question]

    @computed_field
    @property
    def formatted_prompt(self) -> str:
        questions = [
            (name, {"type": q.type, "instructions": _as_text(q.instructions), "criteria": _criteria_text(q)})
            for name, q in self.questions.items()
        ]
        return _LEFTOVER_TEMPLATE.render(state=_as_text(self.state), questions=questions)


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
            answer_type = Literal[tuple(_criteria_text(question))]  # choice: option names; score: levels as text
        fields[f"reasoning_{i}"] = (str, Field(description=f"Reasoning for question {i} ({name!r}), before its answer"))
        fields[f"answer_{i}"] = (answer_type, Field(description=name))
    return create_model("JudgedAnswers", **fields)


def _input_error(e: ValidationError) -> EvaluationError:
    from . import _input_error as report

    return report(e)


def _resolve_backend(target: Any, kwargs: Dict[str, Any]) -> BaseEvaluator:
    if isinstance(target, str):
        from . import Evaluator

        return Evaluator(target, **kwargs).backend
    backend = getattr(target, "backend", target)  # an Evaluator, or a backend itself
    if not isinstance(backend, BaseEvaluator):
        raise TypeError(f"Expected a model string, an Evaluator or a backend, got {type(target).__name__}.")
    return backend


def _refuse_unusable_judge(judge: BaseEvaluator) -> None:
    """The judge must fill an arbitrary output schema from a formatted prompt. These backends cannot:
    Jev answers only its own contract, the SLM backend renders fixed prompts into a fixed
    shape, and Hugging Face text-classification returns a label."""
    name = type(judge).__name__
    if isinstance(judge, JevEvaluator) or name == "GroundedAISLMBackend" or (
        name == "HuggingFaceBackend" and getattr(judge, "task", None) == "text-classification"
    ):
        raise TypeError(
            f"{name} cannot be the judge: it does not fill a custom output schema. Use an LLM backend "
            "such as openai/, anthropic/, bedrock/ or hf/ with task='text-generation'."
        )


def _model_name(backend: BaseEvaluator) -> str:
    return str(getattr(backend, "model_name", None) or getattr(backend, "model_id", None) or type(backend).__name__)


class CascadeEvaluator:
    """
    Jev first, LLM judge for what it is unsure of.

        cascade = CascadeEvaluator(
            jev="jev/jev-latest",  # or "jev" with jev_kwargs={"use_local_model": True}
            judge="anthropic/claude-haiku-4-5-20251001",
            min_confidence=0.9,
        )
        result = cascade.evaluate(JevInput(state=..., questions={...}))
        result.answers["verdict"]   # Jev's answer, or a JudgedAnswer
        result.escalated            # which questions went to the judge

    `jev` and `judge` take a model string, an Evaluator or a backend. `jev_kwargs` and
    `judge_kwargs` are passed to Evaluator when a model string is given (base_url, api_key, ...).
    """

    def __init__(
        self,
        jev: Any,
        judge: Any,
        min_confidence: float = 0.9,
        jev_kwargs: Optional[Dict[str, Any]] = None,
        judge_kwargs: Optional[Dict[str, Any]] = None,
    ):
        if not 0.0 <= min_confidence <= 1.0:
            raise ValueError(f"min_confidence must be in [0, 1], got {min_confidence!r}.")
        self.jev = _resolve_backend(jev, jev_kwargs or {})
        if not isinstance(self.jev, JevEvaluator):
            raise TypeError(f"The first stage must be a JevEvaluator, got {type(self.jev).__name__}.")
        self.judge = _resolve_backend(judge, judge_kwargs or {})
        _refuse_unusable_judge(self.judge)
        self.min_confidence = min_confidence

    # Shared by evaluate() and evaluate_async(): everything except the two backend calls.

    def _jev_input(self, input_data: Any, kwargs: Dict[str, Any]) -> Tuple[Any, Dict[str, Any]]:
        from . import _prepare_input

        input_data, backend_kwargs = _prepare_input(self.jev, input_data, kwargs)
        if isinstance(input_data, dict):
            input_data = self.jev.input_schema(**input_data)
        return input_data, backend_kwargs

    def _leftover(self, input_data: JevInput, jev_output: JevOutput) -> Optional[JevLeftover]:
        questions = _QUESTIONS.validate_python(
            input_data.model_dump(include={"questions"}, exclude_none=True)["questions"]
        )
        escalated = {
            name: question
            for name, question in questions.items()  # the order the questions were asked
            if _confidence(jev_output.answers[name]) < self.min_confidence
        }
        return JevLeftover(state=input_data.build_state(), questions=escalated) if escalated else None

    def _combine(self, jev_output: JevOutput, leftover: Optional[JevLeftover], judged: Any) -> CascadeOutput:
        answers: Dict[str, Any] = dict(jev_output.answers)
        escalated = list(leftover.questions) if leftover else []
        if leftover is None:
            return CascadeOutput(answers=answers, escalated=[], judged=[], jev=jev_output)
        if isinstance(judged, EvaluationError):
            return CascadeOutput(
                answers=answers, escalated=escalated, judged=[], jev=jev_output, judge_error=judged
            )
        try:
            judge = _model_name(self.judge)
            for i, (name, question) in enumerate(leftover.questions.items()):
                answers[name] = JudgedAnswer(
                    question_type=question.type,
                    answer=getattr(judged, f"answer_{i}"),
                    reasoning=getattr(judged, f"reasoning_{i}"),
                    judge=judge,
                )
        except Exception as e:  # the judge returned something other than the schema it was given
            return CascadeOutput(
                answers=dict(jev_output.answers), escalated=escalated, judged=[],
                jev=jev_output, judge_error=_judge_error(e),
            )
        return CascadeOutput(answers=answers, escalated=escalated, judged=escalated, jev=jev_output)

    def evaluate(self, input_data: Any = None, **kwargs) -> Union[CascadeOutput, EvaluationError]:
        """Takes the same inputs as Evaluator("jev/...").evaluate(): a JevInput (or subclass),
        a dict, a bare string as the state, or keywords such as state= and questions=."""
        try:
            input_data, backend_kwargs = self._jev_input(input_data, kwargs)
        except ValidationError as e:
            return _input_error(e)
        jev_output = self.jev.evaluate(input_data, **backend_kwargs)
        if isinstance(jev_output, EvaluationError):
            return jev_output
        leftover = self._leftover(input_data, jev_output)
        judged = None
        if leftover is not None:
            try:
                judged = self.judge.evaluate(leftover, output_schema=_judge_schema(leftover.questions))
            except Exception as e:  # some backends raise instead of returning EvaluationError
                judged = _judge_error(e)
        return self._combine(jev_output, leftover, judged)

    async def evaluate_async(self, input_data: Any = None, **kwargs) -> Union[CascadeOutput, EvaluationError]:
        try:
            input_data, backend_kwargs = self._jev_input(input_data, kwargs)
        except ValidationError as e:
            return _input_error(e)
        jev_output = await self.jev.evaluate_async(input_data, **backend_kwargs)
        if isinstance(jev_output, EvaluationError):
            return jev_output
        leftover = self._leftover(input_data, jev_output)
        judged = None
        if leftover is not None:
            try:
                judged = await self.judge.evaluate_async(leftover, output_schema=_judge_schema(leftover.questions))
            except Exception as e:
                judged = _judge_error(e)
        return self._combine(jev_output, leftover, judged)


def _judge_error(e: Exception) -> EvaluationError:
    return EvaluationError(
        error_code="JUDGE_ERROR",
        message=f"The judge did not answer the escalated questions: {e}",
        details={"exception_type": type(e).__name__},
    )
