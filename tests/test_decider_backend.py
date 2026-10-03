import enum
import json
from typing import List, Literal, Optional

import httpx
import pytest
from pydantic import BaseModel, Field

from grounded_ai import AsyncEvaluator, Evaluator
from grounded_ai.backends.decider import DeciderBackend
from grounded_ai.schemas import EvaluationError, EvaluationInput, EvaluationOutput


def verdict(p_yes, yes="hallucination", no="faithful"):
    """A two-option choice answer for the eval-mode question."""
    return {"verdict": {"type": "choice", "choice": yes if p_yes >= 0.5 else no, "confidence": abs(2 * p_yes - 1),
                        "probabilities": {yes: p_yes, no: round(1 - p_yes, 4)}}}


# /health as `strands-decider serve StrandsAgents/strands-decider-2B-hobson-v19` reports it (trimmed).
HEALTH = {
    "status": "ok",
    "model": "strands-decider-2B-hobson-v19",
    "checkpoint": "StrandsAgents/strands-decider-2B-hobson-v19",
    "device": "mps",
}


def make_backend(answers, status=200, seen=None, async_=False, health=HEALTH, health_calls=None,
                 model_name="strands-decider-2B-hobson-v19", **kwargs):
    """Backend whose HTTP client answers /health with `health` and /v1/systemone with `answers`.
    `seen` records /v1/systemone requests; `health_calls` counts /health requests."""

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            if health_calls is not None:
                health_calls.append(1)
            return httpx.Response(404) if health is None else httpx.Response(200, json=health)
        if seen is not None:
            seen.append({"url": str(request.url), "headers": request.headers, "body": json.loads(request.content)})
        if status != 200:
            return httpx.Response(status, json={"detail": answers})
        return httpx.Response(200, json={"answers": answers})

    transport = httpx.MockTransport(handler)
    client_kwargs = {"async_client": httpx.AsyncClient(transport=transport)} if async_ else {}
    return DeciderBackend(
        model_name=model_name,
        client=httpx.Client(transport=transport),
        **client_kwargs,
        **kwargs,
    )


class TestDeciderBackend:
    def test_factory_routing(self):
        evaluator = Evaluator("decider/strands-decider-2B-hobson-v19", base_url="http://127.0.0.1:8010/")
        assert isinstance(evaluator.backend, DeciderBackend)
        assert evaluator.backend.model_name == "strands-decider-2B-hobson-v19"
        assert evaluator.backend.base_url == "http://127.0.0.1:8010"

    def test_request_shape(self):
        """Inputs become an object state; the eval mode becomes one two-option choice question."""
        seen = []
        backend = make_backend(verdict(0.9), seen=seen, api_key="k")
        backend.evaluate(EvaluationInput(response="London is in France.", context="London is in England."))

        (call,) = seen
        assert call["url"] == "http://127.0.0.1:8000/v1/systemone"
        assert call["headers"]["authorization"] == "Bearer k"
        assert call["body"]["model"] == "strands-decider-2B-hobson-v19"
        # Context first: the model reads the evidence before the text it judges.
        assert list(call["body"]["state"]) == ["context", "response"]
        question = call["body"]["questions"]["verdict"]
        assert question["type"] == "choice"
        assert set(question["criteria"]) == {"hallucination", "faithful"}

    @pytest.mark.parametrize(
        "mode,yes,no,p,label,score",
        [
            ("HALLUCINATION", "hallucination", "faithful", 0.9, "hallucination", 0.9),
            ("HALLUCINATION", "hallucination", "faithful", 0.2, "faithful", 0.2),
            ("TOXICITY", "toxic", "non-toxic", 0.7, "toxic", 0.7),
            ("RAG_RELEVANCE", "relevant", "unrelated", 0.1, "unrelated", 0.1),
        ],
    )
    def test_eval_modes(self, mode, yes, no, p, label, score):
        backend = make_backend(verdict(p, yes, no), eval_mode=mode)
        result = backend.evaluate(EvaluationInput(response="x"))
        assert isinstance(result, EvaluationOutput)
        assert result.label == label
        assert result.score == pytest.approx(score)
        assert result.confidence == pytest.approx(abs(2 * p - 1))

    def test_system_prompt_is_not_sent(self):
        """/v1/systemone has no system message: the eval mode stays the question."""
        seen = []
        with pytest.warns(UserWarning, match="system_prompt is not sent"):
            backend = make_backend(verdict(0.6), seen=seen, system_prompt="You are a strict auditor.")
        result = backend.evaluate(EvaluationInput(response="x"))
        assert "strict auditor" not in json.dumps(seen[0]["body"])
        assert result.label == "hallucination"

    def test_threshold_runtime_override(self):
        backend = make_backend(verdict(0.6))
        assert backend.evaluate(EvaluationInput(response="x")).label == "hallucination"
        assert backend.evaluate(EvaluationInput(response="x"), threshold=0.9).label == "faithful"

    def test_custom_schema_one_question_per_field(self):
        class Ticket(BaseModel):
            urgent: bool = Field(description="Does this need a reply within the hour?")
            area: Literal["billing", "bug", "account"] = Field(description="Which team owns it?")
            p_harmful: float = Field(ge=0, le=1, description="Is this request harmful?")
            reasoning: Optional[str] = None

        seen = []
        backend = make_backend(
            {
                "urgent": {"type": "noul", "noul": 0.8},
                "area": {"type": "choice", "choice": "billing", "confidence": 0.9, "probabilities": {}},
                "p_harmful": {"type": "noul", "noul": 0.05},
            },
            seen=seen,
        )
        result = backend.evaluate(EvaluationInput(response="Charged twice."), output_schema=Ticket)

        assert result == Ticket(urgent=True, area="billing", p_harmful=0.05)
        questions = seen[0]["body"]["questions"]
        assert set(questions) == {"urgent", "area", "p_harmful"}  # all in one request
        assert questions["area"]["criteria"] == {"billing": "", "bug": "", "account": ""}  # Decider requires str descriptions

    def test_required_str_field_refused(self):
        class Bad(BaseModel):
            summary: str = Field(description="Summarize it.")

        backend = make_backend({})
        result = backend.evaluate(EvaluationInput(response="x"), output_schema=Bad)
        assert isinstance(result, EvaluationError)
        assert result.error_code == "INVALID_REQUEST"

    def test_generation_kwargs_accepted_like_other_backends(self):
        """temperature & co. are accepted at init and per call, as on every backend; a decision
        model does not sample, so they do not change the request."""
        seen = []
        backend = make_backend(verdict(0.9), seen=seen, temperature=0.1, max_tokens=50)
        result = backend.evaluate(EvaluationInput(response="x"), temperature=0.7, top_p=0.9)
        assert result.label == "hallucination"
        assert set(seen[0]["body"]) == {"state", "model", "questions"}

    def test_set_eval_mode_like_slm_backend(self):
        from enum import Enum

        class EvalMode(str, Enum):  # same shape as the SLM backend's enum, which imports torch
            TOXICITY = "TOXICITY"

        backend = make_backend(verdict(0.9, "toxic", "non-toxic"))
        backend.set_eval_mode(EvalMode.TOXICITY)
        assert backend.evaluate(EvaluationInput(response="x")).label == "toxic"
        with pytest.raises(ValueError):
            backend.set_eval_mode("NOPE")

    def test_http_error(self):
        backend = make_backend("questions: field required", status=422)
        result = backend.evaluate(EvaluationInput(response="x"))
        assert isinstance(result, EvaluationError)
        assert result.error_code == "422"
        assert "field required" in result.message

    def test_out_of_range_probability(self):
        backend = make_backend({"verdict": {"type": "choice", "probabilities": {"hallucination": -0.2, "faithful": 0.1}}})
        result = backend.evaluate(EvaluationInput(response="x"))
        assert isinstance(result, EvaluationError)

    def test_model_mismatch_refused(self):
        """The server ignores the request's model field, so a wrong name must fail loudly."""
        seen = []
        backend = make_backend(verdict(0.9), seen=seen, model_name="strands-decider-latest")
        result = backend.evaluate(EvaluationInput(response="x"))
        assert isinstance(result, EvaluationError)
        assert result.error_code == "MODEL_MISMATCH"
        assert "StrandsAgents/strands-decider-2B-hobson-v19" in result.message
        assert seen == []  # never reached /v1/systemone

    @pytest.mark.parametrize(
        "name", ["strands-decider-2B-hobson-v19", "StrandsAgents/strands-decider-2B-hobson-v19"]
    )
    def test_served_name_or_checkpoint_accepted(self, name):
        backend = make_backend(verdict(0.9), model_name=name)
        assert backend.evaluate(EvaluationInput(response="x")).label == "hallucination"

    def test_health_checked_once(self):
        calls = []
        backend = make_backend(verdict(0.9), health_calls=calls)
        backend.evaluate(EvaluationInput(response="x"))
        backend.evaluate(EvaluationInput(response="y"))
        assert len(calls) == 1

    def test_server_without_health_is_not_blocked(self):
        backend = make_backend(verdict(0.9), health=None, model_name="anything")
        assert backend.evaluate(EvaluationInput(response="x")).label == "hallucination"

    @pytest.mark.asyncio
    async def test_async_model_mismatch(self):
        backend = make_backend(verdict(0.9), async_=True, model_name="strands-decider-latest")
        result = await backend.evaluate_async(EvaluationInput(response="x"))
        assert result.error_code == "MODEL_MISMATCH"

    def test_custom_base_template_is_the_state(self):
        """Like every other backend, a custom base_template's rendered text is what the model reads."""
        seen = []
        backend = make_backend(verdict(0.9), seen=seen)
        backend.evaluate(EvaluationInput(response="Port 8080.", base_template="Rule: ports must be 443. Text: {{ response }}"))
        assert seen[0]["body"]["state"] == "Rule: ports must be 443. Text: Port 8080."

    def test_default_template_keeps_labelled_fields(self):
        seen = []
        backend = make_backend(verdict(0.9), seen=seen)
        backend.evaluate(EvaluationInput(response="r", context="c"))
        assert seen[0]["body"]["state"] == {"context": "c", "response": "r"}

    def test_rounded_probabilities_renormalized(self):
        """Servers round each probability to 4 decimals, so the pair may not sum to exactly 1."""
        backend = make_backend({"verdict": {"type": "choice", "probabilities": {"hallucination": 0.6, "faithful": 0.3}}})
        assert backend.evaluate(EvaluationInput(response="x")).score == pytest.approx(2 / 3)

    def test_connection_error(self):
        backend = DeciderBackend(model_name="strands-decider-latest", base_url="http://127.0.0.1:9")
        result = backend.evaluate(EvaluationInput(response="x"))
        assert isinstance(result, EvaluationError)
        assert result.error_code == "CONNECTION_ERROR"

    @pytest.mark.asyncio
    async def test_async(self):
        backend = make_backend(verdict(0.95), async_=True)
        evaluator = AsyncEvaluator.__new__(AsyncEvaluator)
        evaluator.backend = backend
        result = await evaluator.evaluate(response="London is in France.", context="London is in England.")
        assert result.label == "hallucination"


def auto_backend(seen=None, p=0.9, pick=0, **kwargs):
    """Backend whose server answers whatever is asked, by question type: noul -> p, choice and
    score -> option number `pick` with probability p."""

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            return httpx.Response(200, json={**HEALTH, "max_length": 100})
        body = json.loads(request.content)
        if seen is not None:
            seen.append(body)
        answers = {}
        for key, q in body["questions"].items():
            if q["type"] == "noul":
                answers[key] = {"type": "noul", "noul": p}
                continue
            names = list(q["criteria"]) if q["type"] == "choice" else [str(i) for i in range(len(q["criteria"]))]
            rest = round((1 - p) / (len(names) - 1), 4)
            probabilities = {n: (p if i == pick else rest) for i, n in enumerate(names)}
            if q["type"] == "choice":
                answers[key] = {"type": "choice", "choice": names[pick], "probabilities": probabilities, "confidence": 0.8}
            else:
                answers[key] = {"type": "score", "score": float(pick), "legend": {}, "probabilities": probabilities, "confidence": 0.8}
        return httpx.Response(200, json={"answers": answers})

    return DeciderBackend(
        model_name="strands-decider-2B-hobson-v19",
        client=httpx.Client(transport=httpx.MockTransport(handler)),
        **kwargs,
    )


INPUT = EvaluationInput(response="Our product is kinda cheap.", context="c", query="q")


class TestSchemaParity:
    """Any Pydantic schema that works on the OpenAI/Anthropic backends should work here whenever
    a decision model can answer it."""

    def test_field_without_description_uses_its_name(self):
        class BrandCheck(BaseModel):
            tone_compliant: bool

        seen = []
        result = auto_backend(seen).evaluate(INPUT, output_schema=BrandCheck)
        assert result == BrandCheck(tone_compliant=True)
        assert seen[0]["questions"]["tone_compliant"] == {"type": "noul", "instructions": "tone compliant"}

    def test_free_text_field_with_default_is_left_alone(self):
        class BrandCheck(BaseModel):
            tone_compliant: bool
            forbidden_words: List[str] = []
            notes: Optional[str] = None

        seen = []
        result = auto_backend(seen).evaluate(INPUT, output_schema=BrandCheck)
        assert result == BrandCheck(tone_compliant=True)
        assert set(seen[0]["questions"]) == {"tone_compliant"}

    def test_required_free_text_field_refused(self):
        class BrandCheck(BaseModel):
            tone_compliant: bool
            forbidden_words: List[str]

        result = auto_backend().evaluate(INPUT, output_schema=BrandCheck)
        assert isinstance(result, EvaluationError)
        assert result.error_code == "INVALID_REQUEST"
        assert "forbidden_words" in result.message

    def test_int_rating_is_a_score_question(self):
        class Rating(BaseModel):
            quality: int = Field(ge=1, le=5, description="Rate the quality.")
            depth: float = Field(ge=1, le=3, description="Rate the depth.")
            strict: int = Field(gt=0, lt=4)

        seen = []
        result = auto_backend(seen, pick=2).evaluate(INPUT, output_schema=Rating)
        assert seen[0]["questions"]["quality"] == {
            "type": "score", "instructions": "Rate the quality.", "criteria": ["1", "2", "3", "4", "5"]}
        assert seen[0]["questions"]["strict"]["criteria"] == ["1", "2", "3"]
        assert result == Rating(quality=3, depth=3.0, strict=3)

    def test_unbounded_number_refused(self):
        class Bad(BaseModel):
            count: int

        result = auto_backend().evaluate(INPUT, output_schema=Bad)
        assert result.error_code == "INVALID_REQUEST"

    def test_non_string_literal_and_enum_values_round_trip(self):
        class Level(enum.IntEnum):
            LOW = 1
            HIGH = 2

        class Color(str, enum.Enum):
            RED = "red"
            BLUE = "blue"

        class Out(BaseModel):
            stars: Literal[1, 2, 3]
            level: Level
            color: Color

        seen = []
        result = auto_backend(seen, pick=1).evaluate(INPUT, output_schema=Out)
        assert seen[0]["questions"]["stars"]["criteria"] == {"1": "", "2": "", "3": ""}
        assert result == Out(stars=2, level=Level.HIGH, color=Color.BLUE)

    def test_pep604_optional_is_asked(self):
        class Out(BaseModel):
            urgent: bool | None = None
            area: Literal["billing", "bug"] | None = None

        seen = []
        result = auto_backend(seen).evaluate(INPUT, output_schema=Out)
        assert set(seen[0]["questions"]) == {"urgent", "area"}
        assert result == Out(urgent=True, area="billing")

    def test_single_option_literal_refused(self):
        class Out(BaseModel):
            kind: Literal["ticket"]

        result = auto_backend().evaluate(INPUT, output_schema=Out)  # the server refuses one-option choices
        assert result.error_code == "INVALID_REQUEST"

    def test_nested_and_list_fields_do_not_map(self):
        class Inner(BaseModel):
            ok: bool

        class Nested(BaseModel):
            inner: Inner

        class Multi(BaseModel):
            tags: List[Literal["a", "b"]]

        for schema in (Nested, Multi):
            assert auto_backend().evaluate(INPUT, output_schema=schema).error_code == "INVALID_REQUEST"

    def test_evaluation_output_subclass_extra_fields_are_asked(self):
        class Detailed(EvaluationOutput):
            severity: Literal["low", "high"] = Field(description="How severe?")

        seen = []
        result = auto_backend(seen).evaluate(INPUT, output_schema=Detailed)
        assert set(seen[0]["questions"]) == {"verdict", "severity"}  # one request
        assert (result.label, result.severity) == ("hallucination", "low")
        assert result.score == pytest.approx(0.9)


class TestCustomEvaluation:
    """A custom evaluation is an EvaluationOutput subclass with its own label options, as on the
    OpenAI/Anthropic backends. There is no custom mode object."""

    def test_label_options_are_the_choice(self):
        class BrandVoice(EvaluationOutput):
            label: Literal["on-brand", "off-brand"] = Field(description="Is the response written in our brand voice?")

        seen = []
        result = auto_backend(seen, p=0.8).evaluate(INPUT, output_schema=BrandVoice)
        assert seen[0]["questions"]["verdict"] == {
            "type": "choice",
            "instructions": "Is the response written in our brand voice?",
            "criteria": {"on-brand": "", "off-brand": ""},
        }
        assert (result.label, result.score) == ("on-brand", pytest.approx(0.8))  # score = p(first label)
        assert result.confidence == pytest.approx(0.6)
        assert auto_backend(p=0.8).evaluate(INPUT, output_schema=BrandVoice, threshold=0.9).label == "off-brand"

    def test_more_than_two_labels(self):
        class Triage(EvaluationOutput):
            label: Literal["unsafe", "borderline", "safe"] = Field(description="How risky is the response?")

        result = auto_backend(p=0.7, pick=2).evaluate(INPUT, output_schema=Triage)
        assert result.label == "safe"  # the model's pick
        assert result.score == pytest.approx(0.15)  # p(first label)
        assert result.confidence == pytest.approx((3 * 0.7 - 1) / 2)

    def test_enum_labels(self):
        class Verdict(str, enum.Enum):
            PASS = "pass"
            FAIL = "fail"

        class Gate(EvaluationOutput):
            label: Verdict = Field(description="Does the response pass review?")

        assert auto_backend(p=0.9).evaluate(INPUT, output_schema=Gate).label is Verdict.PASS

    def test_labels_matching_the_built_in_mode_use_its_question(self):
        class Tox(EvaluationOutput):
            label: Literal["toxic", "non-toxic"]

        seen = []
        auto_backend(seen, eval_mode="TOXICITY").evaluate(INPUT, output_schema=Tox)
        question = seen[0]["questions"]["verdict"]
        assert question["instructions"] == "Classify the tone of the response."
        assert question["criteria"]["toxic"].startswith("the response is abusive")

    def test_dict_eval_mode_is_gone(self):
        with pytest.raises(ValueError, match="output_schema"):
            auto_backend(eval_mode={"instructions": "Q?", "labels": {"a": "", "b": ""}})


class TestInputsAndModes:
    def test_eval_mode_is_case_insensitive_and_per_call(self):
        assert auto_backend(eval_mode="toxicity").eval_mode == "TOXICITY"
        seen = []
        backend = auto_backend(seen)
        assert backend.evaluate(INPUT, eval_mode="rag_relevance").label == "relevant"
        assert backend.eval_mode == "HALLUCINATION"  # the call did not change the backend
        assert backend.evaluate(INPUT).label == "hallucination"

    def test_eval_mode_with_plain_schema_refused(self):
        class Out(BaseModel):
            safe: bool

        result = auto_backend().evaluate(INPUT, output_schema=Out, eval_mode="TOXICITY")
        assert result.error_code == "INVALID_REQUEST"

    def test_subclass_default_template_is_the_state(self):
        class PortPolicy(EvaluationInput):
            base_template: str = "Rule: ports must be 443. Text: {{ response }}"

        seen = []
        auto_backend(seen).evaluate(PortPolicy(response="Port 8080."))
        assert seen[0]["state"] == "Rule: ports must be 443. Text: Port 8080."

    def test_input_model_with_its_own_formatted_prompt(self):
        class CodeReview(BaseModel):
            code: str

            @property
            def formatted_prompt(self) -> str:
                return f"Review this code:\n{self.code}"

        seen = []
        auto_backend(seen).evaluate(CodeReview(code="x = 1"))
        assert seen[0]["state"] == "Review this code:\nx = 1"

    def test_input_model_without_template_is_an_object_state(self):
        class Pair(BaseModel):
            question: str
            answer: str

        seen = []
        auto_backend(seen).evaluate(Pair(question="q", answer="a"))
        assert seen[0]["state"] == {"question": "q", "answer": "a"}

    def test_empty_input_refused(self):
        seen = []
        result = auto_backend(seen).evaluate(EvaluationInput())
        assert result.error_code == "INVALID_REQUEST"
        assert seen == []


class TestGuards:
    def test_unknown_kwarg_warns(self):
        with pytest.warns(UserWarning, match="treshold"):
            auto_backend().evaluate(INPUT, treshold=0.9)

    def test_bad_threshold_refused(self):
        assert auto_backend().evaluate(INPUT, threshold=1.5).error_code == "INVALID_REQUEST"
        with pytest.raises(ValueError):
            auto_backend(threshold=-1)

    def test_long_input_warns(self):
        long_input = EvaluationInput(response="r", context="word " * 200)  # /health says max_length=100
        with pytest.warns(UserWarning, match="truncates"):
            auto_backend().evaluate(long_input)

    def test_unreadable_answer_is_invalid_response(self):
        backend = make_backend({"verdict": {"type": "choice", "probabilities": {"something": 1.0}}})
        result = backend.evaluate(INPUT)
        assert result.error_code == "INVALID_RESPONSE"

    def test_model_name_keeps_inner_decider_segment(self):
        evaluator = Evaluator("decider/my-org/decider/ckpt")
        assert evaluator.backend.model_name == "my-org/decider/ckpt"

    @pytest.mark.asyncio
    async def test_async_custom_schema(self):
        class Out(BaseModel):
            quality: int = Field(ge=1, le=5)
            safe: bool

        def handler(request):
            if request.url.path == "/health":
                return httpx.Response(200, json=HEALTH)
            return httpx.Response(200, json={"answers": {
                "quality": {"type": "score", "score": 3.2, "probabilities": {"0": 0.0, "1": 0.1, "2": 0.2, "3": 0.6, "4": 0.1}},
                "safe": {"type": "noul", "noul": 0.2},
            }})

        backend = DeciderBackend(
            model_name="strands-decider-2B-hobson-v19",
            async_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        )
        assert await backend.evaluate_async(INPUT, output_schema=Out) == Out(quality=4, safe=False)
