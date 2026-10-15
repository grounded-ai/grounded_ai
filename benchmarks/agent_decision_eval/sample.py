"""
Decision models on agent-decision's own test set (grounded-ai/agent-decision-eval): 30 rows per question
family (tool calls, agent steps, call/ask/decline, call or not, grounding), seed 7, as compare() items.

    python benchmarks/agent_decision_eval/sample.py items.jsonl
    grounded-ai compare items.jsonl --model jev/jev-latest --model openai-decisions/gpt-6-luna \\
        --model systemone/agent-decision-9b@http://gpu-box:8765 --judge bedrock/us.anthropic.claude-sonnet-4-6
"""

import collections
import json
import random
import sys

from huggingface_hub import hf_hub_download


def main(out: str, per_family: int = 30, seed: int = 7):
    path = hf_hub_download("grounded-ai/agent-decision-eval", "test.jsonl", repo_type="dataset")
    groups = collections.defaultdict(list)
    for line in open(path):
        row = json.loads(line)
        families = set(row["families"].values())
        groups["tool_call" if "tool_call_ok" in families else families.pop()].append(row)
    rng = random.Random(seed)
    with open(out, "w") as f:
        for family in sorted(groups):
            for row in rng.sample(groups[family], per_family):
                f.write(json.dumps({"id": row["id"], "family": sorted(set(row["families"].values())), "state": row["state"],
                                    "questions": row["questions"], "labels": row["gold"]}) + "\n")


if __name__ == "__main__":
    main(sys.argv[1])
