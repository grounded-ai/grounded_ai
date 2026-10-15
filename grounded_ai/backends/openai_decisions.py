"""
OpenAI's Decisions API (`POST /v1/decisions`) behind the same input and output as DecisionEvaluator.

The two APIs ask the same three kinds of question and answer them the same way, with different
names on the wire, so this backend only translates:

    /v1/systemone (JevInput, JevOutput)       /v1/decisions
    state                                     input (text; JSON states are sent as JSON text)
    questions: {name: question}               questions: [{name, ...}]
    noul (+ true/false criteria)              predicate (criteria folded into the instructions)
    choice criteria {option: description}    choices [{value, description}]
    score criteria [level, ...]               levels [{label}]
    answers {name: answer}                    answers [{name, ...}]
    noul                                      probability
    probabilities {option: p}                 probabilities [{value, probability}]

OpenAI's `confidence` uses TypeSafe's definitions (its documented example, 0.95 on four options,
gives 0.93), so it is kept as reported. A question OpenAI declines to answer fails the call as
REFUSED, naming the question.
"""

import json
from typing import Any, Dict, Optional, Type, Union

from pydantic import BaseModel

from ..schemas import EvaluationError
from .jev import (
    Capabilities,
    ChoiceAnswer,
    DecisionEvaluator,
    JevOutput,
    NoulAnswer,
    ResponseError,
    ScoreAnswer,
    _check_answers,
    _to_error,
)


class RefusalError(Exception):
    """OpenAI declined to answer one or more questions."""


def _text(content: Any) -> str:
    return content if isinstance(content, str) else json.dumps(content, indent=2, ensure_ascii=False)


def to_decisions_request(body: Dict[str, Any]) -> Dict[str, Any]:
    """A validated /v1/systemone body (see DecisionEvaluator._request) as a /v1/decisions body."""
    questions = []
    for name, q in body["questions"].items():
        instructions = _text(q["instructions"])
        if q["type"] == "noul":
            criteria = q.get("criteria") or {}
            for value in ("true", "false"):
                if value in criteria:
                    instructions += f"\n{value.capitalize()} means: {_text(criteria[value])}"
            questions.append({"type": "predicate", "name": name, "instructions": instructions})
        elif q["type"] == "choice":
            choices = [{"value": option} if d is None else {"value": option, "description": _text(d)}
                       for option, d in q["criteria"].items()]
            questions.append({"type": "choice", "name": name, "instructions": instructions, "choices": choices})
        else:
            levels = [{"label": _text(level)} for level in q["criteria"]]
            questions.append({"type": "score", "name": name, "instructions": instructions, "levels": levels})
    return {"model": body["model"], "input": _text(body["state"]), "questions": questions}


def from_decisions_response(response: Dict[str, Any], body: Dict[str, Any]) -> JevOutput:
    """A /v1/decisions response as the JevOutput the same question would get from /v1/systemone."""
    by_name = {a.get("name"): a for a in response["answers"]}
    refused = [name for name, a in by_name.items() if a.get("type") == "refusal"]
    if refused:
        raise RefusalError(f"OpenAI declined to answer {sorted(refused)}.")
    answers = {}
    for name, q in body["questions"].items():
        a = by_name[name]
        if a["type"] == "predicate":
            answers[name] = NoulAnswer(noul=a["probability"])
        elif a["type"] == "choice":
            answers[name] = ChoiceAnswer(
                choice=str(a["choice"]),
                probabilities={str(p["value"]): p["probability"] for p in a["probabilities"]},
                confidence=a["confidence"],
            )
        else:
            answers[name] = ScoreAnswer(
                score=a["score"],
                legend={str(i): level for i, level in enumerate(q["criteria"])},
                probabilities={str(p["value"]): p["probability"] for p in a["probabilities"]},
                confidence=a["confidence"],
            )
    usage = response.get("usage") or {}
    return JevOutput(
        answers=answers,
        model=response.get("model"),
        usage={k: usage[k] for k in ("input_tokens", "output_tokens") if isinstance(usage.get(k), int)} or None,
    )


class OpenAIDecisionsEvaluator(DecisionEvaluator):
    """
    OpenAI's Decisions API with DecisionEvaluator's input and output, so it can stand in for Jev
    anywhere, including as the first stage of CascadeEvaluator.

        Evaluator("openai-decisions/gpt-6-luna")   # OPENAI_API_KEY

    `client` / `async_client` take an `openai.OpenAI` / `openai.AsyncOpenAI`.
    """

    def __init__(self, model_name: str = "gpt-6-luna", api_key: str = None, base_url: str = None,
                 timeout: float = 30.0, max_retries: int = 2, client: Optional[Any] = None,
                 async_client: Optional[Any] = None, **kwargs):
        import openai

        self._openai_args = dict(api_key=api_key, base_url=base_url, timeout=timeout, max_retries=max_retries)
        client = client or openai.OpenAI(**self._openai_args)
        # The openai client holds the key and the address; base_url here is only what errors report.
        super().__init__(model_name=model_name, base_url=base_url or "https://api.openai.com/v1", client=client,
                         async_client=async_client, timeout=timeout, max_retries=max_retries,
                         capabilities=Capabilities(), **kwargs)  # 2-255 choices, as the API documents
        self.recompute_confidence = False  # OpenAI reports TypeSafe's definition already

    @property
    def async_client(self):
        if self._async_client is None:
            import openai

            self._async_client = openai.AsyncOpenAI(**self._openai_args)
        return self._async_client

    def _read_decision(self, decision: Any, body: Dict[str, Any]) -> JevOutput:
        data = decision.model_dump() if hasattr(decision, "model_dump") else decision
        try:
            output = from_decisions_response(data, body)
            _check_answers(output.answers, body["questions"])
        except RefusalError:
            raise
        except (KeyError, TypeError, ValueError) as e:
            raise ResponseError(f"Could not read the decision into JevOutput: {e!r}") from e
        return output

    def _call_backend(
        self, input_data: BaseModel, output_schema: Type[BaseModel], **kwargs
    ) -> Union[BaseModel, EvaluationError]:
        try:
            body = self._request(input_data, output_schema, kwargs)
            return self._read_decision(self.client.decisions.create(**to_decisions_request(body)), body)
        except Exception as e:
            return _openai_error(e)

    async def _call_backend_async(
        self, input_data: BaseModel, output_schema: Type[BaseModel], **kwargs
    ) -> Union[BaseModel, EvaluationError]:
        try:
            body = self._request(input_data, output_schema, kwargs)
            decision = await self.async_client.decisions.create(**to_decisions_request(body))
            return self._read_decision(decision, body)
        except Exception as e:
            return _openai_error(e)


def _openai_error(e: Exception) -> EvaluationError:
    # Read off the exception rather than isinstance(openai.APIStatusError): the SDK's classes are
    # the only ones that carry these, and the check then works whatever `openai` module is loaded.
    names = {c.__name__ for c in type(e).__mro__}
    if isinstance(e, RefusalError):
        return EvaluationError(error_code="REFUSED", message=str(e), details={"exception_type": "RefusalError"})
    if "APIStatusError" in names:
        return EvaluationError(error_code=str(e.status_code), message=getattr(e, "message", str(e)),
                               details={"exception_type": type(e).__name__})
    if "APIConnectionError" in names:
        return EvaluationError(error_code="CONNECTION_ERROR", message=f"Could not reach OpenAI: {e}",
                               details={"exception_type": type(e).__name__})
    return _to_error(e)
