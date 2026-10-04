"""
`grounded-ai check` (a faithfulness check from the command line) and `grounded-ai hook`
(the same check as a Stop hook for Claude Code and Codex).
"""

import io
import json

import pytest

from grounded_ai import cli
from grounded_ai.cli import Verdict

CONTEXT = "Refunds are accepted within 30 days of purchase."
GOOD = "You can get a refund within 30 days."
BAD = "You have 90 days to request a refund."


class FakeChecker:
    """Stands in for a real model: an answer containing '90 days' is unfaithful."""

    def __init__(self, error=None):
        self.calls, self.error = [], error

    def __call__(self, response, context, query):
        self.calls.append({"response": response, "context": context, "query": query})
        if self.error:
            raise self.error
        faithful = "90 days" not in response
        return Verdict(
            faithful=faithful,
            hallucination_probability=0.1 if faithful else 0.9,
            reasoning=None if faithful else "The context says 30 days, not 90.",
            model="fake/model",
        )


@pytest.fixture
def checker(monkeypatch):
    fake = FakeChecker()
    monkeypatch.setattr(cli, "make_checker", lambda model, **kwargs: fake)
    return fake


def run(argv, stdin=""):
    out, err = io.StringIO(), io.StringIO()
    code = cli.main(argv, stdin=io.StringIO(stdin), stdout=out, stderr=err)
    return code, out.getvalue(), err.getvalue()


# --- grounded-ai check ---------------------------------------------------------------------------


class TestCheck:
    def test_faithful_answer_exits_0_with_a_json_verdict(self, checker):
        code, out, _ = run(
            ["check", "--model", "fake/model", "--response", GOOD, "--context", CONTEXT]
        )
        assert code == 0
        verdict = json.loads(out)
        assert verdict["faithful"] is True
        assert verdict["model"] == "fake/model"

    def test_hallucination_exits_1(self, checker):
        code, out, _ = run(
            [
                "check",
                "--model",
                "fake/model",
                "--response",
                BAD,
                "--context",
                CONTEXT,
                "--query",
                "How long?",
            ]
        )
        assert code == 1
        assert json.loads(out)["faithful"] is False
        assert checker.calls == [
            {"response": BAD, "context": CONTEXT, "query": "How long?"}
        ]

    def test_response_from_stdin_and_context_from_a_file(self, checker, tmp_path):
        policy = tmp_path / "policy.txt"
        policy.write_text(CONTEXT)
        code, _, _ = run(
            ["check", "--model", "fake/model", "--context", f"@{policy}"], stdin=BAD
        )
        assert code == 1
        assert checker.calls[0] == {"response": BAD, "context": CONTEXT, "query": None}

    def test_missing_context_is_a_usage_error(self, checker):
        code, _, err = run(["check", "--model", "fake/model", "--response", GOOD])
        assert code == 2
        assert "context" in err

    def test_model_failure_exits_2(self, monkeypatch):
        monkeypatch.setattr(
            cli,
            "make_checker",
            lambda model, **kwargs: FakeChecker(error=RuntimeError("401 bad key")),
        )
        code, out, err = run(
            ["check", "--model", "fake/model", "--response", GOOD, "--context", CONTEXT]
        )
        assert code == 2
        assert "401 bad key" in err
        assert out == ""


# --- grounded-ai hook: transcripts ---------------------------------------------------------------


def claude_transcript(tmp_path, prompt, tool_outputs, answer, earlier_turn=True):
    """A Claude Code transcript in the shape Claude Code writes it (one JSON object per line)."""
    rows = []
    if earlier_turn:
        rows += [
            {
                "type": "user",
                "message": {"role": "user", "content": "What is the warranty?"},
            },
            {
                "type": "user",
                "message": {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "t0",
                            "content": "Warranty: two years.",
                            "is_error": False,
                        }
                    ],
                },
            },
            {
                "type": "assistant",
                "message": {
                    "role": "assistant",
                    "content": [{"type": "text", "text": "Two years."}],
                },
            },
        ]
    rows += [
        {
            "type": "user",
            "isMeta": True,
            "message": {
                "role": "user",
                "content": "<system-reminder>ignore me</system-reminder>",
            },
        },
        {"type": "user", "message": {"role": "user", "content": prompt}},
        {
            "type": "assistant",
            "message": {
                "role": "assistant",
                "content": [
                    {"type": "thinking", "thinking": "..."},
                    {"type": "tool_use", "id": "t1", "name": "Read", "input": {}},
                ],
            },
        },
    ]
    for i, output in enumerate(tool_outputs):
        content = output if isinstance(output, list) else output
        rows.append(
            {
                "type": "user",
                "message": {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": f"t{i + 1}",
                            "content": content,
                            "is_error": False,
                        }
                    ],
                },
            }
        )
    rows.append(
        {
            "type": "assistant",
            "message": {
                "role": "assistant",
                "content": [{"type": "text", "text": answer}],
            },
        }
    )
    path = tmp_path / "transcript.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return path


def codex_transcript(tmp_path, prompt, tool_outputs, answer):
    """A Codex rollout transcript in its documented shape (not verified against a real file)."""
    rows = [
        {
            "type": "response_item",
            "payload": {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "older"}],
            },
        },
        {
            "type": "response_item",
            "payload": {
                "type": "function_call_output",
                "call_id": "c0",
                "output": "old output",
            },
        },
        {
            "type": "response_item",
            "payload": {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": prompt}],
            },
        },
        {
            "type": "response_item",
            "payload": {
                "type": "function_call",
                "call_id": "c1",
                "name": "shell",
                "arguments": "{}",
            },
        },
    ]
    rows += [
        {
            "type": "response_item",
            "payload": {
                "type": "function_call_output",
                "call_id": f"c{i + 1}",
                "output": o,
            },
        }
        for i, o in enumerate(tool_outputs)
    ]
    rows.append(
        {
            "type": "response_item",
            "payload": {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": answer}],
            },
        }
    )
    path = tmp_path / "rollout.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return path


class TestTranscripts:
    def test_claude_code_turn_is_the_last_real_prompt_onward(self, tmp_path):
        path = claude_transcript(
            tmp_path, "How long do refunds take?", [CONTEXT, "Shipping is free."], BAD
        )
        turn = cli.read_turn(str(path))
        assert turn.query == "How long do refunds take?"
        assert turn.tool_outputs == [
            CONTEXT,
            "Shipping is free.",
        ]  # not the earlier turn's warranty output
        assert turn.answer == BAD

    def test_claude_code_tool_result_given_as_content_blocks(self, tmp_path):
        blocks = [
            {"type": "text", "text": "Refunds are accepted"},
            {"type": "text", "text": "within 30 days."},
        ]
        path = claude_transcript(
            tmp_path, "Refunds?", [blocks], GOOD, earlier_turn=False
        )
        assert cli.read_turn(str(path)).tool_outputs == [
            "Refunds are accepted\nwithin 30 days."
        ]

    def test_codex_turn(self, tmp_path):
        path = codex_transcript(tmp_path, "How long do refunds take?", [CONTEXT], BAD)
        turn = cli.read_turn(str(path))
        assert (turn.query, turn.tool_outputs, turn.answer) == (
            "How long do refunds take?",
            [CONTEXT],
            BAD,
        )

    def test_missing_or_unreadable_transcript_gives_an_empty_turn(self, tmp_path):
        assert cli.read_turn(None).tool_outputs == []
        assert cli.read_turn(str(tmp_path / "nope.jsonl")).tool_outputs == []
        junk = tmp_path / "junk.jsonl"
        junk.write_text("not json\n")
        assert cli.read_turn(str(junk)).tool_outputs == []


# --- grounded-ai hook: decisions ----------------------------------------------------------------


def hook_input(path, answer, **extra):
    return json.dumps(
        {
            "session_id": "s",
            "transcript_path": str(path),
            "cwd": "/x",
            "hook_event_name": "Stop",
            "last_assistant_message": answer,
            "stop_hook_active": False,
            **extra,
        }
    )


class TestHook:
    def test_unsupported_answer_blocks_with_a_reason(self, checker, tmp_path):
        path = claude_transcript(tmp_path, "How long do refunds take?", [CONTEXT], BAD)
        code, out, _ = run(
            ["hook", "--model", "fake/model"], stdin=hook_input(path, BAD)
        )
        assert code == 0
        decision = json.loads(out)
        assert decision["decision"] == "block"
        assert "The context says 30 days, not 90." in decision["reason"]
        assert checker.calls[0]["context"] == CONTEXT
        assert checker.calls[0]["query"] == "How long do refunds take?"

    def test_supported_answer_lets_the_agent_stop(self, checker, tmp_path):
        path = claude_transcript(tmp_path, "How long?", [CONTEXT], GOOD)
        code, out, _ = run(
            ["hook", "--model", "fake/model"], stdin=hook_input(path, GOOD)
        )
        assert (code, out) == (0, "")

    def test_never_blocks_twice_in_a_row(self, checker, tmp_path):
        path = claude_transcript(tmp_path, "How long?", [CONTEXT], BAD)
        code, out, _ = run(
            ["hook", "--model", "fake/model"],
            stdin=hook_input(path, BAD, stop_hook_active=True),
        )
        assert (code, out) == (0, "")
        assert checker.calls == []

    def test_turn_without_tool_output_is_not_checked(self, checker, tmp_path):
        path = claude_transcript(
            tmp_path, "Write a haiku", [], "Leaves fall quietly", earlier_turn=False
        )
        code, out, _ = run(
            ["hook", "--model", "fake/model"],
            stdin=hook_input(path, "Leaves fall quietly"),
        )
        assert (code, out) == (0, "")
        assert checker.calls == []

    def test_answer_falls_back_to_the_transcript(self, checker, tmp_path):
        path = claude_transcript(tmp_path, "How long?", [CONTEXT], BAD)
        code, out, _ = run(
            ["hook", "--model", "fake/model"], stdin=hook_input(path, None)
        )
        assert json.loads(out)["decision"] == "block"

    def test_fails_open_when_the_model_errors(self, monkeypatch, tmp_path):
        monkeypatch.setattr(
            cli,
            "make_checker",
            lambda model, **kwargs: FakeChecker(
                error=RuntimeError("connection refused")
            ),
        )
        path = claude_transcript(tmp_path, "How long?", [CONTEXT], BAD)
        code, out, err = run(
            ["hook", "--model", "fake/model"], stdin=hook_input(path, BAD)
        )
        assert (code, out) == (0, "")
        assert "connection refused" in err

    def test_fails_open_on_bad_hook_input(self, checker):
        code, out, err = run(["hook", "--model", "fake/model"], stdin="not json")
        assert (code, out) == (0, "")
        assert err

    def test_long_evidence_keeps_the_most_recent_output(self, checker, tmp_path):
        path = claude_transcript(tmp_path, "How long?", ["x" * 500, CONTEXT], BAD)
        run(
            ["hook", "--model", "fake/model", "--max-context-chars", "100"],
            stdin=hook_input(path, BAD),
        )
        context = checker.calls[0]["context"]
        assert len(context) <= 100 and context.endswith(CONTEXT)

    def test_codex_input_works_the_same(self, checker, tmp_path):
        path = codex_transcript(tmp_path, "How long?", [CONTEXT], BAD)
        code, out, _ = run(
            ["hook", "--model", "fake/model"],
            stdin=hook_input(path, BAD, turn_id="t1", model="gpt-5"),
        )
        assert json.loads(out)["decision"] == "block"


# --- the model behind the check ------------------------------------------------------------------


class TestMakeChecker:
    def test_decider_models_ask_the_hallucination_question(self, monkeypatch):
        seen = {}

        class FakeDecider:
            def evaluate(self, input_data):
                from grounded_ai.backends.decider import ChoiceAnswer, DeciderOutput

                seen["state"], seen["questions"] = (
                    input_data.state,
                    input_data.questions,
                )
                return DeciderOutput(
                    answers={
                        "verdict": ChoiceAnswer(
                            choice="hallucination",
                            probabilities={"hallucination": 0.8, "faithful": 0.2},
                            confidence=0.6,
                        )
                    }
                )

        monkeypatch.setattr(cli, "_evaluator", lambda model, **kwargs: FakeDecider())
        check = cli.make_checker("decider/StrandsAgents/strands-decider-2B-hobson-v19")
        verdict = check(BAD, CONTEXT, "How long?")
        assert seen["state"] == {
            "context": CONTEXT,
            "query": "How long?",
            "response": BAD,
        }
        assert set(seen["questions"]) == {"verdict"}
        assert (verdict.faithful, verdict.hallucination_probability) == (False, 0.8)

    def test_llm_models_fill_a_fixed_verdict_schema(self, monkeypatch):
        seen = {}

        class FakeLLM:
            def evaluate(self, input_data, output_schema=None):
                seen["prompt"], seen["schema"] = (
                    input_data.formatted_prompt,
                    output_schema,
                )
                return output_schema(
                    reasoning="It says 90, the context says 30.", verdict="unsupported"
                )

        monkeypatch.setattr(cli, "_evaluator", lambda model, **kwargs: FakeLLM())
        verdict = cli.make_checker("anthropic/claude-haiku-4-5")(BAD, CONTEXT, None)
        assert CONTEXT in seen["prompt"] and BAD in seen["prompt"]
        assert (verdict.faithful, verdict.reasoning) == (
            False,
            "It says 90, the context says 30.",
        )
        assert set(seen["schema"].model_fields) == {"reasoning", "verdict"}

    def test_an_evaluation_error_raises(self, monkeypatch):
        from grounded_ai.schemas import EvaluationError

        class Failing:
            def evaluate(self, input_data, output_schema=None):
                return EvaluationError(error_code="401", message="API key is invalid.")

        monkeypatch.setattr(cli, "_evaluator", lambda model, **kwargs: Failing())
        with pytest.raises(RuntimeError, match="API key is invalid"):
            cli.make_checker("anthropic/claude-haiku-4-5")(BAD, CONTEXT, None)

    def test_a_missing_context_file_is_a_usage_error(self, checker):
        code, out, err = run(
            [
                "check",
                "--model",
                "fake/model",
                "--response",
                GOOD,
                "--context",
                "@/no/such/file",
            ]
        )
        assert (code, out) == (2, "")
        assert "/no/such/file" in err

    def test_the_slm_is_refused_before_loading_it(self, monkeypatch):
        monkeypatch.setattr(
            cli, "_evaluator", lambda model, **kwargs: pytest.fail("loaded")
        )
        with pytest.raises(ValueError, match="faithfulness verdict"):
            cli.make_checker("grounded-ai/phi4-mini-judge")

    @pytest.mark.parametrize(
        "model, flags, message",
        [
            ("anthropic/claude-haiku-4-5", {"region": "us-east-1"}, "--region"),
            ("decider/m", {"region": "us-east-1"}, "--region"),
            ("anthropic/claude-haiku-4-5", {"base_url": "http://x"}, "--base-url"),
            ("bedrock/m", {"base_url": "http://x"}, "--base-url"),
        ],
    )
    def test_flags_for_another_backend_are_refused(
        self, monkeypatch, model, flags, message
    ):
        monkeypatch.setattr(
            cli, "_evaluator", lambda model, **kwargs: pytest.fail("built")
        )
        with pytest.raises(ValueError, match=message):
            cli.make_checker(model, **flags)

    def test_flags_reach_their_backend(self, monkeypatch):
        seen = {}
        monkeypatch.setattr(
            cli, "_evaluator", lambda model, **kwargs: seen.setdefault(model, kwargs)
        )
        cli.make_checker("bedrock/m", region="eu-west-1")
        cli.make_checker("decider/m", base_url="http://localhost:9000")
        assert seen["bedrock/m"]["region_name"] == "eu-west-1"
        assert seen["decider/m"] == {"base_url": "http://localhost:9000"}

    def test_a_model_that_ignores_the_verdict_schema_raises(self, monkeypatch):
        from grounded_ai.schemas import EvaluationOutput

        class Classifier:
            def evaluate(self, input_data, output_schema=None):
                return EvaluationOutput(score=0.9, label="INJECTION", confidence=0.9)

        monkeypatch.setattr(cli, "_evaluator", lambda model, **kwargs: Classifier())
        with pytest.raises(RuntimeError, match="faithfulness verdict"):
            cli.make_checker("hf/meta-llama/Prompt-Guard-86M")(BAD, CONTEXT, None)

    def test_the_judge_prompt_is_a_jinja_template_you_can_override(self):
        default = cli.FaithfulnessInput(context=CONTEXT, response=BAD)
        assert (
            CONTEXT in default.formatted_prompt
            and "None" not in default.formatted_prompt
        )
        custom = cli.FaithfulnessInput(
            context=CONTEXT,
            response=BAD,
            base_template="E={{ context }} A={{ response }}",
        )
        assert custom.formatted_prompt == f"E={CONTEXT} A={BAD}"


class TestTurnBoundaries:
    def _write(self, tmp_path, rows):
        path = tmp_path / "t.jsonl"
        path.write_text("\n".join(json.dumps(r) for r in rows))
        return path

    def _tool(self, text):
        return {
            "type": "user",
            "message": {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "t", "content": text}
                ],
            },
        }

    def test_a_claude_prompt_with_an_image_starts_a_turn(self, tmp_path):
        path = self._write(
            tmp_path,
            [
                {"type": "user", "message": {"role": "user", "content": "old"}},
                self._tool("old output"),
                {
                    "type": "user",
                    "message": {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "What does this screenshot say?"},
                            {
                                "type": "image",
                                "source": {"type": "base64", "data": "..."},
                            },
                        ],
                    },
                },
                self._tool("new output"),
            ],
        )
        turn = cli.read_turn(str(path))
        assert turn.query == "What does this screenshot say?"
        assert turn.tool_outputs == ["new output"]

    def test_a_compact_summary_does_not_start_a_turn(self, tmp_path):
        path = self._write(
            tmp_path,
            [
                {"type": "user", "message": {"role": "user", "content": "Fix the bug"}},
                self._tool("before compaction"),
                {
                    "type": "user",
                    "isCompactSummary": True,
                    "message": {
                        "role": "user",
                        "content": "This session is being continued...",
                    },
                },
                self._tool("after compaction"),
            ],
        )
        turn = cli.read_turn(str(path))
        assert turn.query == "Fix the bug"
        assert turn.tool_outputs == ["before compaction", "after compaction"]

    @pytest.mark.parametrize(
        "injected",
        [
            "<environment_context>\n  <cwd>/x</cwd>\n</environment_context>",
            "<user_instructions>be terse</user_instructions>",
            "# AGENTS.md instructions for /x\n\n<INSTRUCTIONS>...</INSTRUCTIONS>",
        ],
    )
    def test_codex_injected_context_does_not_start_a_turn(self, tmp_path, injected):
        def user(text):
            return {
                "type": "response_item",
                "payload": {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": text}],
                },
            }

        out = {
            "type": "response_item",
            "payload": {"type": "function_call_output", "output": "ls output"},
        }
        path = self._write(tmp_path, [user("List the files"), out, user(injected)])
        turn = cli.read_turn(str(path))
        assert (turn.query, turn.tool_outputs) == ("List the files", ["ls output"])
