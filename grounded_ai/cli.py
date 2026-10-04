"""
Command line: a faithfulness check, and the same check as a Stop hook for coding agents.

    grounded-ai check --model MODEL --context TEXT|@FILE [--query TEXT] [--response TEXT]
        Is the response supported by the context? Prints a JSON verdict.
        Exit 0: supported. 1: not supported. 2: error or bad usage.
        The response is read from stdin when --response is not given.

    grounded-ai hook --model MODEL
        A Stop hook for Claude Code and Codex. Reads the hook input on stdin, takes the agent's
        final answer and the tool output from this turn, and when the answer is not supported by
        that output, prints {"decision": "block", "reason": ...} so the agent re-checks it.
        It never blocks twice in a row, skips turns with no tool output, and fails open: any
        error lets the agent stop, with a warning on stderr.

MODEL is any Evaluator model string: "anthropic/...", "openai/...", "bedrock/...", "hf/...", or
"decider/..." (a running strands-decider server; set DECIDER_BASE_URL or --base-url).
The grounded-ai/ SLM and hf/ text-classification models answer in fixed formats and cannot be used.
"""

import argparse
import json
import sys
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, List, Literal, Optional, TextIO

from pydantic import BaseModel, Field

from .schemas import EvaluationError, EvaluationInput

SYSTEM_PROMPT = (
    "You check whether an AI assistant's answer is supported by the evidence it was given. "
    "An answer is supported only if every factual claim in it is stated in, or follows directly from, "
    "the evidence. Claims the evidence contradicts or does not mention make the answer unsupported."
)
DEFAULT_MAX_CONTEXT_CHARS = 20000


@dataclass
class Verdict:
    faithful: bool
    model: str
    hallucination_probability: Optional[float] = None  # decider/ models only
    reasoning: Optional[str] = None  # LLM judges only


@dataclass
class Turn:
    """The agent's latest turn, read from its transcript."""

    query: Optional[str] = None
    tool_outputs: List[str] = field(default_factory=list)
    answer: Optional[str] = None


class FaithfulnessInput(EvaluationInput):
    """What an LLM judge reads: the evidence, the question, and the answer to check."""

    context: str
    response: str
    base_template: str = """<evidence>
{{ context }}
</evidence>
{% if query %}
<question>
{{ query }}
</question>
{% endif %}
<answer>
{{ response }}
</answer>

Is every factual claim in the answer supported by the evidence?"""


class FaithfulnessJudgement(BaseModel):
    reasoning: str = Field(
        description="Which claims the evidence supports or not, before the verdict"
    )
    verdict: Literal["supported", "unsupported"]


# --- the model behind the check ---------------------------------------------------------------

# Backends that cannot fill a free-form verdict schema: the SLM answers in its own fixed format.
_NO_VERDICT = ("grounded-ai/",)


def _evaluator(model: str, **kwargs):
    from . import Evaluator

    return Evaluator(model, **kwargs)


def _raise_on_error(result):
    if isinstance(result, EvaluationError):
        raise RuntimeError(f"{result.error_code}: {result.message}")
    return result


def make_checker(
    model: str, base_url: Optional[str] = None, region: Optional[str] = None
) -> Callable[..., Verdict]:
    """A function (response, context, query) -> Verdict backed by `model`.
    Raises ValueError for a model or flag that cannot work, and RuntimeError when a check fails."""
    if model.startswith(_NO_VERDICT):
        raise ValueError(
            f"{model} cannot give a faithfulness verdict; use a decider/ model or an LLM judge"
        )
    if region and not model.startswith("bedrock/"):
        raise ValueError("--region only applies to bedrock/ models")
    if base_url and not model.startswith("decider/"):
        raise ValueError("--base-url only applies to decider/ models")

    if model.startswith("decider/"):
        from .backends.decider import HALLUCINATION, DeciderInput

        evaluator = _evaluator(model, **({"base_url": base_url} if base_url else {}))

        def check(response: str, context: str, query: Optional[str]) -> Verdict:
            state = {"context": context, "query": query, "response": response}
            result = _raise_on_error(
                evaluator.evaluate(
                    DeciderInput(
                        state={k: v for k, v in state.items() if v is not None},
                        questions={"verdict": HALLUCINATION},
                    )
                )
            )
            answer = result.answers["verdict"]
            return Verdict(
                faithful=answer.choice == "faithful",
                model=model,
                hallucination_probability=answer.probabilities.get("hallucination"),
            )

        return check

    kwargs: dict = {"system_prompt": SYSTEM_PROMPT}
    if region:
        kwargs["region_name"] = region
    evaluator = _evaluator(model, **kwargs)

    def check(response: str, context: str, query: Optional[str]) -> Verdict:
        result = _raise_on_error(
            evaluator.evaluate(
                FaithfulnessInput(context=context, query=query, response=response),
                output_schema=FaithfulnessJudgement,
            )
        )
        if not isinstance(result, FaithfulnessJudgement):
            raise RuntimeError(
                f"{model} returned {type(result).__name__}, not a faithfulness verdict "
                "(classifier models cannot judge faithfulness)"
            )
        return Verdict(
            faithful=result.verdict == "supported",
            model=model,
            reasoning=result.reasoning,
        )

    return check


# --- transcripts ------------------------------------------------------------------------------

# User messages Codex injects into the rollout: context for the model, not a prompt from the user.
_CODEX_INJECTED = ("<environment_context>", "<user_instructions>", "# AGENTS.md")


def _text(content: Any) -> str:
    """Tool output or message content as text: a string, or a list of content blocks."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            _text(b.get("text", b.get("content", "")))
            if isinstance(b, dict)
            else str(b)
            for b in content
        ).strip()
    return "" if content is None else str(content)


def _blocks(content: Any, kind: str) -> list:
    return [b for b in content if isinstance(b, dict) and b.get("type") == kind]


def read_turn(transcript_path: Optional[str]) -> Turn:
    """The latest turn of a Claude Code or Codex transcript: the prompt that started it, every tool
    output since, and the last assistant text. Anything unreadable gives an empty turn."""
    try:
        with open(transcript_path) as f:  # type: ignore[arg-type]
            rows = [json.loads(line) for line in f if line.strip()]
    except (TypeError, OSError, ValueError):
        return Turn()

    turn = Turn()
    for row in rows:
        # Claude Code: {"type": "user"|"assistant", "message": {"content": str | [blocks]}}
        if row.get("type") in ("user", "assistant") and isinstance(
            row.get("message"), dict
        ):
            content = row["message"].get("content")
            if row["type"] == "assistant":
                text = _text(_blocks(content or [], "text"))
                if text:
                    turn.answer = text
            elif row.get("isMeta") or row.get("isCompactSummary"):
                continue
            elif isinstance(content, str):
                turn = Turn(query=content)
            elif isinstance(content, list):
                results = _blocks(content, "tool_result")
                if results:
                    turn.tool_outputs += [_text(b.get("content")) for b in results]
                elif _blocks(content, "text"):  # a prompt with attachments
                    turn = Turn(query=_text(_blocks(content, "text")))
        # Codex: {"type": "response_item", "payload": {...}}
        elif row.get("type") == "response_item" and isinstance(
            row.get("payload"), dict
        ):
            payload = row["payload"]
            if payload.get("type") == "message" and payload.get("role") == "user":
                text = _text(payload.get("content"))
                if not text.lstrip().startswith(_CODEX_INJECTED):
                    turn = Turn(query=text)
            elif payload.get("type") == "function_call_output":
                turn.tool_outputs.append(_text(payload.get("output")))
            elif (
                payload.get("type") == "message" and payload.get("role") == "assistant"
            ):
                text = _text(payload.get("content"))
                if text:
                    turn.answer = text
    turn.tool_outputs = [t for t in turn.tool_outputs if t.strip()]
    return turn


def _evidence(tool_outputs: List[str], max_chars: int) -> str:
    """The tool outputs joined, keeping the most recent ones when they are too long."""
    joined = "\n\n".join(tool_outputs)
    return joined if len(joined) <= max_chars else joined[-max_chars:]


# --- commands ---------------------------------------------------------------------------------


def _check(args, stdin: TextIO, stdout: TextIO, stderr: TextIO) -> int:
    context = args.context
    if context and context.startswith("@"):
        try:
            with open(context[1:]) as f:
                context = f.read()
        except OSError as e:
            print(f"grounded-ai check: cannot read --context: {e}", file=stderr)
            return 2
    if not context:
        print("grounded-ai check: --context is required (text, or @file)", file=stderr)
        return 2
    response = args.response if args.response is not None else stdin.read()
    if not response.strip():
        print(
            "grounded-ai check: no response given (--response, or on stdin)",
            file=stderr,
        )
        return 2
    try:
        verdict = make_checker(args.model, base_url=args.base_url, region=args.region)(
            response, context, args.query
        )
    except Exception as e:  # noqa: BLE001 - any model failure is reported, not raised
        print(f"grounded-ai check: {e}", file=stderr)
        return 2
    print(json.dumps(asdict(verdict)), file=stdout)
    return 0 if verdict.faithful else 1


def _hook(args, stdin: TextIO, stdout: TextIO, stderr: TextIO) -> int:
    try:
        event = json.loads(stdin.read() or "{}")
        if event.get("stop_hook_active"):
            return (
                0  # this turn was already continued by a Stop hook: never block twice
            )
        turn = read_turn(event.get("transcript_path"))
        answer = event.get("last_assistant_message") or turn.answer
        if not answer or not turn.tool_outputs:
            return 0  # nothing to check, or no tool output to check it against
        verdict = make_checker(args.model, base_url=args.base_url, region=args.region)(
            answer, _evidence(turn.tool_outputs, args.max_context_chars), turn.query
        )
    except Exception as e:  # noqa: BLE001 - fail open: a broken check must never trap the agent
        print(f"grounded-ai hook: check skipped: {e}", file=stderr)
        return 0
    if verdict.faithful:
        return 0
    why = (
        f" {verdict.reasoning}"
        if verdict.reasoning
        else (
            f" (probability of a hallucination: {verdict.hallucination_probability:.2f})"
            if verdict.hallucination_probability is not None
            else ""
        )
    )
    reason = (
        f"grounded-ai ({verdict.model}): your answer may not be supported by the tool output from this turn.{why} "
        "Check each claim against the tool results, correct or remove anything they do not support, and say so if "
        "you cannot verify something."
    )
    print(json.dumps({"decision": "block", "reason": reason}), file=stdout)
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="grounded-ai", description="Faithfulness checks from the command line."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    def common(p):
        p.add_argument(
            "--model",
            required=True,
            help='Evaluator model string, e.g. "anthropic/claude-haiku-4-5"',
        )
        p.add_argument("--base-url", help="Server URL (decider/ models)")
        p.add_argument("--region", help="AWS region (bedrock/ models)")

    check = sub.add_parser(
        "check",
        help="Is a response supported by its context? Exit 0 yes, 1 no, 2 error.",
    )
    common(check)
    check.set_defaults(run=_check)
    check.add_argument("--context", help="The evidence: text, or @path to a file")
    check.add_argument("--query", help="The question the response answers (optional)")
    check.add_argument(
        "--response", help="The response to check (default: read from stdin)"
    )

    hook = sub.add_parser(
        "hook",
        help="Stop hook for Claude Code and Codex: re-check unsupported answers.",
    )
    common(hook)
    hook.set_defaults(run=_hook)
    hook.add_argument(
        "--max-context-chars",
        type=int,
        default=DEFAULT_MAX_CONTEXT_CHARS,
        help="Keep at most this much tool output, the most recent (default %(default)s)",
    )
    return parser


def main(
    argv: Optional[List[str]] = None,
    stdin: Optional[TextIO] = None,
    stdout: Optional[TextIO] = None,
    stderr: Optional[TextIO] = None,
) -> int:
    stdin, stdout, stderr = (
        stdin or sys.stdin,
        stdout or sys.stdout,
        stderr or sys.stderr,
    )
    try:
        args = _parser().parse_args(argv)
    except SystemExit as e:
        return 2 if e.code else 0
    return args.run(args, stdin, stdout, stderr)


def entrypoint() -> None:
    sys.exit(main())
