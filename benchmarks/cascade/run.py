"""
Faithfulness benchmark: LLM judge on everything vs. Decider alone vs. CascadeEvaluator at several thresholds.

Every item gets one Decider answer and one judge answer. The judge is asked exactly what the
cascade would send it (a DeciderLeftover with the item's one question, the same output schema),
so the cascade's result at any threshold is: the Decider's answer if its confidence clears the
threshold, the judge's otherwise. No extra calls per threshold.

Data: RAGTruth test split (MIT; wandb/RAGTruth-processed), downloaded at run time, nothing stored
in the repo. Real LLM responses to retrieved context, labelled by people for evident conflicts
and baseless information. A response with either is a hallucination. Two subsets, each balanced
50/50: RAG question answering (ragtruth_qa) and news summarization (ragtruth_summary). Every item
is asked the shipped HALLUCINATION question with state {context, query, response}.

Usage:
  pip install -e ".[decider]" datasets
  python benchmarks/cascade/run.py --judge bedrock/us.anthropic.claude-haiku-4-5-20251001-v1:0 \
      --judge-region us-east-1 --device mps --per-task 200
  python benchmarks/cascade/run.py --report-only        # re-print the report from results/raw.jsonl
"""

import argparse
import json
import random
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from grounded_ai import Evaluator
from grounded_ai.backends.decider import HALLUCINATION, DeciderInput, NoulAnswer
from grounded_ai.cascade import DeciderLeftover, _judge_schema
from grounded_ai.schemas import EvaluationError

HERE = Path(__file__).parent
RESULTS = HERE / "results"
THRESHOLDS = [0.0, 0.5, 0.7, 0.8, 0.9, 0.95]
TASKS = {"ragtruth_qa": "QA", "ragtruth_summary": "Summary"}


# --- Data ---------------------------------------------------------------------------------------

def ragtruth_items(task, n, rng):
    """n items from one RAGTruth subset, half hallucinated, half faithful."""
    import ast

    from datasets import load_dataset

    rows = load_dataset("wandb/RAGTruth-processed", split="test")
    rows = rows.filter(lambda r: r["task_type"] == TASKS[task])
    hallucinated, faithful = [], []
    for i, labels in enumerate(rows["hallucination_labels_processed"]):
        counts = labels if isinstance(labels, dict) else ast.literal_eval(labels)
        (hallucinated if counts["evident_conflict"] or counts["baseless_info"] else faithful).append(i)
    picks = [(i, "hallucination") for i in rng.sample(hallucinated, n // 2)]
    picks += [(i, "faithful") for i in rng.sample(faithful, n - n // 2)]
    rng.shuffle(picks)
    return [{
        "task": task, "id": f"ragtruth-test-{rows[i]['id']}", "source_model": rows[i]["model"],
        "state": {"context": rows[i]["context"], "query": rows[i]["query"], "response": rows[i]["output"]},
        "label": label,
    } for i, label in picks]


# --- Running ------------------------------------------------------------------------------------

def run(args):
    rng = random.Random(args.seed)
    items = [item for task in TASKS for item in ragtruth_items(task, args.per_task, rng)]
    print(f"{len(items)} items")

    decider = Evaluator(f"decider/{args.checkpoint}", timeout=300.0).backend
    decider.warmup(port=args.port, device=args.device, timeout=1800.0)
    judge = Evaluator(args.judge, **({"region_name": args.judge_region} if args.judge_region else {})).backend

    # Token counts from the judge's own responses (Bedrock converse reports usage per call).
    usage = {}
    if hasattr(judge, "client") and hasattr(judge.client, "converse"):
        converse = judge.client.converse

        def counting_converse(**kwargs):
            response = converse(**kwargs)
            usage[kwargs["messages"][0]["content"][0]["text"]] = response.get("usage", {})
            return response

        judge.client.converse = counting_converse

    for item in items:
        t = time.perf_counter()
        out = decider.evaluate(DeciderInput(state=item["state"], questions={"q": HALLUCINATION}))
        item["decider_ms"] = round((time.perf_counter() - t) * 1000, 1)
        if isinstance(out, EvaluationError):
            item["decider_error"] = out.message
            continue
        answer = out.answers["q"]
        item["decider_choice"] = answer.choice
        item["decider_confidence"] = abs(2 * answer.noul - 1) if isinstance(answer, NoulAnswer) else answer.confidence
        item["decider_probabilities"] = answer.probabilities
    decider.shutdown()

    def ask_judge(item):
        leftover = DeciderLeftover(state=item["state"], questions={"verdict": HALLUCINATION})
        t = time.perf_counter()
        try:
            out = judge.evaluate(leftover, output_schema=_judge_schema(leftover.questions))
        except Exception as e:  # noqa: BLE001 - recorded, not fatal
            out = EvaluationError(error_code="EXCEPTION", message=str(e))
        item["judge_ms"] = round((time.perf_counter() - t) * 1000, 1)
        if isinstance(out, EvaluationError):
            item["judge_error"] = out.message
        else:
            item["judge_choice"] = out.answer_0
        u = usage.get(leftover.formatted_prompt, {})
        item["judge_input_tokens"], item["judge_output_tokens"] = u.get("inputTokens"), u.get("outputTokens")
        return item

    with ThreadPoolExecutor(max_workers=args.judge_workers) as pool:
        list(pool.map(ask_judge, items))

    RESULTS.mkdir(exist_ok=True)
    with open(RESULTS / "raw.jsonl", "w") as f:
        for item in items:
            f.write(json.dumps({k: v for k, v in item.items() if k != "state"}) + "\n")
    meta = {"checkpoint": args.checkpoint, "judge": args.judge, "device": args.device, "per_task": args.per_task,
            "seed": args.seed, "price_in": args.price_in, "price_out": args.price_out}
    (RESULTS / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")


# --- Report -------------------------------------------------------------------------------------

def scores(picks, labels):
    """Accuracy, and F1 on the hallucination class (the usual faithfulness metric)."""
    acc = sum(p == y for p, y in zip(picks, labels)) / len(labels)
    tp = sum(p == y == "hallucination" for p, y in zip(picks, labels))
    fp = sum(p == "hallucination" != y for p, y in zip(picks, labels))
    fn = sum(y == "hallucination" != p for p, y in zip(picks, labels))
    f1 = 2 * tp / (2 * tp + fp + fn) if tp else 0.0
    return acc, f1


def report(price_in, price_out):
    items = [json.loads(line) for line in open(RESULTS / "raw.jsonl")]
    meta = json.loads((RESULTS / "meta.json").read_text())
    usable = [i for i in items if "decider_choice" in i and "judge_choice" in i and i.get("judge_input_tokens")]
    dropped = len(items) - len(usable)

    def cost(i):
        return (i["judge_input_tokens"] * price_in + i["judge_output_tokens"] * price_out) / 1e6

    lines = [
        f"Decider `{meta['checkpoint']}` on {meta['device']}; judge `{meta['judge']}`; RAGTruth test, "
        f"{len(usable)} items ({dropped} dropped for errors); judge priced at ${price_in}/${price_out} per 1M tokens.",
        "",
        "| Subset | Setup | Accuracy | Hallucination F1 | Sent to judge | Judge cost / 1,000 items |",
        "| :--- | :--- | ---: | ---: | ---: | ---: |",
    ]
    for task in ["all", *TASKS]:
        rows = [i for i in usable if task in ("all", i["task"])]
        if not rows:
            continue
        labels = [i["label"] for i in rows]
        judge_all = sum(cost(i) for i in rows) / len(rows) * 1000
        acc, f1 = scores([i["judge_choice"] for i in rows], labels)
        lines.append(f"| {task} | LLM judge on everything | {acc:.3f} | {f1:.3f} | 100% | ${judge_all:.2f} |")
        acc, f1 = scores([i["decider_choice"] for i in rows], labels)
        lines.append(f"| {task} | Decider alone | {acc:.3f} | {f1:.3f} | 0% | $0.00 |")
        for t in THRESHOLDS[1:]:
            esc = [i["decider_confidence"] < t for i in rows]
            acc, f1 = scores([i["judge_choice"] if e else i["decider_choice"] for i, e in zip(rows, esc)], labels)
            spent = sum(cost(i) for i, e in zip(rows, esc) if e) / len(rows) * 1000
            lines.append(f"| {task} | Cascade, threshold {t} | {acc:.3f} | {f1:.3f} | {sum(esc) / len(rows):.0%} | ${spent:.2f} |")
    lines += ["", "Median latency per item: Decider "
              f"{sorted(i['decider_ms'] for i in usable)[len(usable) // 2]:.0f} ms, judge "
              f"{sorted(i['judge_ms'] for i in usable)[len(usable) // 2]:.0f} ms."]
    text = "\n".join(lines) + "\n"
    (RESULTS / "report.md").write_text(text)
    print(text)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="StrandsAgents/strands-decider-2B-hobson-v19")
    p.add_argument("--judge", default="anthropic/claude-haiku-4-5")
    p.add_argument("--judge-region")
    p.add_argument("--judge-workers", type=int, default=8)
    p.add_argument("--device")
    p.add_argument("--port", type=int, default=8790)
    p.add_argument("--per-task", type=int, default=200)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--price-in", type=float, default=1.0, help="judge $ per 1M input tokens (Haiku 4.5 list price)")
    p.add_argument("--price-out", type=float, default=5.0, help="judge $ per 1M output tokens (Haiku 4.5 list price)")
    p.add_argument("--report-only", action="store_true")
    args = p.parse_args()
    if not args.report_only:
        run(args)
    report(args.price_in, args.price_out)
