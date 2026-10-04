Decider `StrandsAgents/strands-decider-2B-hobson-v19` on mps; judge `bedrock/us.anthropic.claude-haiku-4-5-20251001-v1:0`; RAGTruth test, 400 items (0 dropped for errors); judge priced at $1.0/$5.0 per 1M tokens.

| Subset | Setup | Accuracy | Hallucination F1 | Sent to judge | Judge cost / 1,000 items |
| :--- | :--- | ---: | ---: | ---: | ---: |
| all | LLM judge on everything | 0.835 | 0.817 | 100% | $2.62 |
| all | Decider alone | 0.578 | 0.293 | 0% | $0.00 |
| all | Cascade, threshold 0.5 | 0.682 | 0.557 | 29% | $0.75 |
| all | Cascade, threshold 0.7 | 0.790 | 0.753 | 66% | $1.74 |
| all | Cascade, threshold 0.8 | 0.833 | 0.813 | 94% | $2.47 |
| all | Cascade, threshold 0.9 | 0.835 | 0.817 | 100% | $2.62 |
| all | Cascade, threshold 0.95 | 0.835 | 0.817 | 100% | $2.62 |
| ragtruth_qa | LLM judge on everything | 0.820 | 0.804 | 100% | $2.38 |
| ragtruth_qa | Decider alone | 0.645 | 0.482 | 0% | $0.00 |
| ragtruth_qa | Cascade, threshold 0.5 | 0.750 | 0.691 | 42% | $1.02 |
| ragtruth_qa | Cascade, threshold 0.7 | 0.805 | 0.782 | 72% | $1.71 |
| ragtruth_qa | Cascade, threshold 0.8 | 0.820 | 0.804 | 93% | $2.21 |
| ragtruth_qa | Cascade, threshold 0.9 | 0.820 | 0.804 | 100% | $2.38 |
| ragtruth_qa | Cascade, threshold 0.95 | 0.820 | 0.804 | 100% | $2.38 |
| ragtruth_summary | LLM judge on everything | 0.850 | 0.830 | 100% | $2.86 |
| ragtruth_summary | Decider alone | 0.510 | 0.039 | 0% | $0.00 |
| ragtruth_summary | Cascade, threshold 0.5 | 0.615 | 0.384 | 16% | $0.48 |
| ragtruth_summary | Cascade, threshold 0.7 | 0.775 | 0.720 | 60% | $1.77 |
| ragtruth_summary | Cascade, threshold 0.8 | 0.845 | 0.823 | 95% | $2.74 |
| ragtruth_summary | Cascade, threshold 0.9 | 0.850 | 0.830 | 100% | $2.86 |
| ragtruth_summary | Cascade, threshold 0.95 | 0.850 | 0.830 | 100% | $2.86 |

Median latency per item: Decider 720 ms, judge 3341 ms.
