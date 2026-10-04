# Cascade benchmark: faithfulness

How much of an LLM judge's bill does `CascadeEvaluator` save on faithfulness, and what does it cost in accuracy?

Every item gets one Decider answer and one judge answer. The judge is asked exactly what the cascade would send it, so the cascade's result at any threshold is the Decider's answer where its confidence clears the threshold and the judge's otherwise.

- **Data:** RAGTruth test split (MIT, `wandb/RAGTruth-processed`): real LLM responses to retrieved context, labelled by people. A response with an evident conflict or baseless information counts as a hallucination. 200 items each from RAG question answering and news summarization, balanced 50/50, seed 7. Downloaded at run time; `results/raw.jsonl` holds item ids, labels and answers, not the texts.
- **Question:** the shipped `HALLUCINATION` question, state `{context, query, response}`.
- **Models:** Decider `StrandsAgents/strands-decider-2B-hobson-v19` on Apple MPS; judge Claude Haiku 4.5 on Amazon Bedrock.
- **Cost:** the judge's reported token counts at Haiku 4.5's Anthropic list price ($1 / $5 per 1M tokens; Bedrock prices separately). The Decider runs locally and has no per-call price.

## Results (2026-10-04)

See [`results/report.md`](results/report.md) for every threshold and both subsets.

| Setup | Accuracy | Hallucination F1 | Sent to judge | Judge cost / 1,000 items |
| :--- | ---: | ---: | ---: | ---: |
| LLM judge on everything | 0.835 | 0.817 | 100% | $2.62 |
| Decider alone | 0.578 | 0.293 | 0% | $0.00 |
| Cascade, threshold 0.7 | 0.790 | 0.753 | 66% | $1.74 |
| Cascade, threshold 0.8 | 0.833 | 0.813 | 94% | $2.47 |
| Cascade, threshold 0.9 (default) | 0.835 | 0.817 | 100% | $2.62 |

Decider accuracy by its own confidence:

| Confidence | Share of items | Accuracy | Says "faithful" |
| :--- | ---: | ---: | ---: |
| 0.8 - 0.9 | 6% | 0.875 | 100% |
| 0.7 - 0.8 | 28% | 0.735 | 99% |
| 0.5 - 0.7 | 36% | 0.500 | 99% |
| below 0.5 | 29% | 0.462 | 69% |

## What it means

- **The Decider's confidence is meaningful on this data** (accuracy rises with it), **but it is confident only about "faithful"**. It labelled 39 of 400 responses as hallucinations, none confidently. In practice the cascade lets the Decider clear clearly faithful responses and sends the rest to the judge.
- **No item reached 0.9**, so at the default threshold the cascade sends everything to the judge. This matches the Decider's documented under-confidence on longer inputs.
- **The trade on faithfulness:** about a third off the judge bill for 4-5 points of accuracy (threshold 0.7), or the judge's accuracy for about 6% off (threshold 0.8).
- **Summaries are harder for the Decider than RAG answers** (alone: 0.510 vs 0.645 accuracy); see the report.
- Latency per item (median): Decider 720 ms on MPS, judge 3.3 s.

## Run it

```bash
pip install -e ".[decider]" datasets
python benchmarks/cascade/run.py --judge bedrock/us.anthropic.claude-haiku-4-5-20251001-v1:0 \
    --judge-region us-east-1 --device mps --per-task 200
python benchmarks/cascade/run.py --report-only      # re-print from results/raw.jsonl
```

The full run is 400 Decider calls and 400 judge calls (about $1 at Haiku list prices).
