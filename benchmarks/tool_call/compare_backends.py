"""
The tool-call benchmark (bench.py) across decision models, with grounded_ai.compare: which first
stage, at what threshold, and what the cascade with Sonnet 4.6 gets from each.

Same 200 items (seed 7) and the same question as bench.py. Models that need a GPU run elsewhere and
send their records here; the judge's answers can come from a saved Sonnet run (bench.py's own call)
so the comparison costs only the decision models' calls.

    # items, once:
    python benchmarks/tool_call/compare_backends.py --write-items items.jsonl
    # on a GPU box serving agent-decision-3b at /v1/systemone:
    grounded-ai compare items.jsonl --model systemone/agent-decision-3b@http://127.0.0.1:8765 --save-records gpu.jsonl
    # here, with API keys, merging the GPU records and a saved Sonnet run as the judge:
    python benchmarks/tool_call/compare_backends.py --model jev/jev-latest --model openai-decisions/gpt-6-luna \\
        --records gpu.jsonl --judge-answers sonnet.json
"""

import argparse
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import bench  # noqa: E402

from grounded_ai.comparison import collect, load_records, report, save_records  # noqa: E402

OUT = pathlib.Path(__file__).parent / "results" / "compare_backends.json"


def items(n: int = 200, seed: int = 7) -> list:
    question = bench.QUESTION.model_dump()
    return [{"state": state, "questions": {"next": question}, "labels": {"next": label}}
            for state, label in bench.load_items(n, seed)]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--write-items", help="Write the benchmark items as JSONL and stop")
    p.add_argument("--model", action="append", default=[])
    p.add_argument("--records", action="append", default=[], help="Records from models run elsewhere")
    p.add_argument("--save-records", help="Save the records of the models run here")
    p.add_argument("--judge", help="An LLM judge to ask (costs its calls)")
    p.add_argument("--judge-answers", help='A saved judge run: [{"i": item, "answer": ...}, ...] (bench.py format)')
    p.add_argument("--thresholds", type=float, nargs="+", default=[0.5, 0.6, 0.7, 0.8, 0.9])
    p.add_argument("--workers", type=int, default=6)
    args = p.parse_args()

    data = items()
    if args.write_items:
        pathlib.Path(args.write_items).write_text("".join(json.dumps(i) + "\n" for i in data))
        return
    records = []
    for path in args.records:
        records += load_records(path)
    if args.model:
        new = collect(data, args.model, workers=args.workers)
        if args.save_records:
            save_records(new, args.save_records)
        records += new
    judge_answers = None
    if args.judge_answers:
        judge_answers = {r["i"]: {"next": r["answer"]} for r in json.load(open(args.judge_answers)) if "answer" in r}
    result = report(data, records, thresholds=args.thresholds, judge=args.judge, judge_answers=judge_answers)
    print(result.table())
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(result.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
