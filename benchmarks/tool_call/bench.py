"""
Does the cascade pay off? CascadeEvaluator (Jev first, Sonnet 4.6 when Jev is unsure) against
Sonnet 4.6 deciding everything, on one agent decision: should the next turn call a tool?

Both setups see the same items and are timed and priced per call:

  1. Sonnet alone:  Evaluator("bedrock/...sonnet-4-6").evaluate(...)
  2. Cascade:       CascadeEvaluator(jev="jev/jev-latest", judge=<the same Sonnet Evaluator>,
                                     min_confidence=t).evaluate(...), once per threshold t

Data: GeneralFunctionCall-Test (evalscope on ModelScope, Apache-2.0). Label = whether Kimi
K2-thinking called a tool next, so accuracy is agreement with K2-thinking.

    TYPESAFE_API_KEY=... python benchmarks/tool_call/bench.py --n 200 --thresholds 0.7 0.9
"""

import argparse
import json
import pathlib
import random
import statistics
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

from grounded_ai import CascadeEvaluator, Evaluator
from grounded_ai.backends.jev import ChoiceQuestion, JevInput
from grounded_ai.cascade import JevLeftover
from pydantic import BaseModel
from typing_extensions import Literal

JEV = "jev/jev-latest"
SONNET = "bedrock/us.anthropic.claude-sonnet-4-6"
JEV_PRICE = 0.042 / 1e6  # $ per input token; output is free
SONNET_PRICE = (3.00 / 1e6, 15.00 / 1e6)  # $ per input / output token

# The one question. Jev gets it as is; the cascade hands the same question to Sonnet when Jev
# is unsure, and Sonnet alone gets exactly what the cascade would hand it (a JevLeftover).
QUESTION = ChoiceQuestion(
    instructions="Look at the conversation and the tools the assistant has. What should the assistant do in its next turn?",
    criteria={"call_tool": "call one of the available tools", "respond": "reply directly, without calling a tool"},
)

DATA_URL = "https://www.modelscope.cn/api/v1/datasets/evalscope/GeneralFunctionCall-Test/repo?Revision=master&FilePath=test.jsonl"
DATA = pathlib.Path.home() / ".cache" / "grounded_ai_benchmarks" / "general_fc_test.jsonl"
OUT = pathlib.Path(__file__).parent / "results" / "bench.json"


class SonnetDecision(BaseModel):
    reasoning: str
    answer: Literal["call_tool", "respond"]


def load_items(n: int, seed: int) -> list:
    """n items, half where K2 called a tool and half where it replied: (state, label) pairs."""
    if not DATA.exists():
        DATA.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(DATA_URL, DATA)
    rows = [json.loads(line) for line in DATA.open() if line.strip()]
    rng = random.Random(seed)
    calls = [r for r in rows if r["should_call_tool"]]
    replies = [r for r in rows if not r["should_call_tool"]]
    picked = rng.sample(calls, n // 2) + rng.sample(replies, n - n // 2)
    return [(to_state(r), "call_tool" if r["should_call_tool"] else "respond") for r in picked]


def to_state(row: dict) -> dict:
    """What both models read: the tools (name, description, parameter names) and the conversation."""
    tools = [{"name": t["function"]["name"], "description": t["function"].get("description", ""),
              "parameters": list((t["function"].get("parameters") or {}).get("properties", {}))}
             for t in json.loads(row["tools"])]
    conversation = []
    for m in json.loads(row["messages"]):
        turn = {"role": m["role"]}
        content = m.get("content")
        if isinstance(content, list):
            content = "\n".join(part.get("text", "") for part in content if isinstance(part, dict))
        if content:
            turn["content"] = content
        if m.get("tool_calls"):
            turn["tool_calls"] = [{"name": c["function"]["name"], "arguments": c["function"].get("arguments")}
                                  for c in m["tool_calls"]]
        conversation.append(turn)
    return {"tools": tools, "conversation": conversation}


class SonnetBill:
    """Bedrock bills by tokens and the Evaluator does not return them, so read them off each
    Converse response. Observes only; the request is unchanged."""

    def __init__(self, sonnet: Evaluator):
        self._last = threading.local()
        client = sonnet.backend.client
        converse = client.converse

        def converse_and_record(**kwargs):
            response = converse(**kwargs)
            self._last.usage = response["usage"]
            return response

        client.converse = converse_and_record

    def take(self) -> float:
        """Dollars for this thread's last Sonnet call, then forget it (0 if there was none)."""
        usage, self._last.usage = getattr(self._last, "usage", None), None
        return 0.0 if usage is None else usage["inputTokens"] * SONNET_PRICE[0] + usage["outputTokens"] * SONNET_PRICE[1]


def sonnet_alone(sonnet, bill, state, label) -> dict:
    t = time.perf_counter()
    result = sonnet.evaluate(JevLeftover(state=state, questions={"next": QUESTION}), output_schema=SonnetDecision)
    seconds = time.perf_counter() - t
    return {"answer": result.answer, "right": result.answer == label, "seconds": seconds, "cost": bill.take()}


def cascade_once(cascade, bill, state, label) -> dict:
    t = time.perf_counter()
    result = cascade.evaluate(JevInput(state=state, questions={"next": QUESTION}))
    seconds = time.perf_counter() - t
    answer = result.answers["next"]
    answer = answer.answer if result.judged else answer.choice  # Sonnet's pick, or Jev's
    cost = result.jev.usage["input_tokens"] * JEV_PRICE + bill.take()
    return {"answer": answer, "right": answer == label, "seconds": seconds, "cost": cost,
            "sent_to_sonnet": bool(result.escalated)}


def run(setup, items, workers: int) -> dict:
    with ThreadPoolExecutor(workers) as pool:
        calls = list(pool.map(lambda item: setup(*item), items))
    summary = {
        "accuracy": statistics.mean(c["right"] for c in calls),
        "seconds_per_check": statistics.mean(c["seconds"] for c in calls),
        "median_seconds": statistics.median(c["seconds"] for c in calls),
        "dollars_per_million": 1e6 * statistics.mean(c["cost"] for c in calls),
    }
    if "sent_to_sonnet" in calls[0]:
        summary["sent_to_sonnet"] = statistics.mean(c["sent_to_sonnet"] for c in calls)
    return summary


def main(n: int, thresholds: list, workers: int, seed: int):
    items = load_items(n, seed)
    sonnet = Evaluator(SONNET, region_name="us-east-1")
    bill = SonnetBill(sonnet)

    results = {"n": n, "sonnet": SONNET, "jev": JEV,
               "sonnet_alone": run(lambda s, y: sonnet_alone(sonnet, bill, s, y), items, workers)}
    for t in thresholds:
        cascade = CascadeEvaluator(jev=JEV, judge=sonnet, min_confidence=t)
        results[f"cascade_{t}"] = run(lambda s, y: cascade_once(cascade, bill, s, y), items, workers)
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(results, indent=2))

    base = results["sonnet_alone"]
    print(f"{n} items. Sonnet 4.6 alone vs CascadeEvaluator (Jev first, Sonnet 4.6 below the threshold)\n")
    print(f"{'setup':16} {'to Sonnet':>9} {'s/check':>8} {'median s':>9} {'$ per 1M':>9} {'accuracy':>9} {'faster':>7} {'cheaper':>8}")
    for name, r in results.items():
        if not isinstance(r, dict):
            continue
        sent = r.get("sent_to_sonnet", 1.0)
        print(f"{name:16} {sent:>9.0%} {r['seconds_per_check']:>8.2f} {r['median_seconds']:>9.2f} "
              f"{r['dollars_per_million']:>9,.0f} {r['accuracy']:>9.3f} "
              f"{1 - r['seconds_per_check'] / base['seconds_per_check']:>7.0%} "
              f"{1 - r['dollars_per_million'] / base['dollars_per_million']:>8.0%}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=200)
    parser.add_argument("--thresholds", type=float, nargs="+", default=[0.7, 0.9])
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    main(args.n, args.thresholds, args.workers, args.seed)
