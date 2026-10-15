"""
compare(): which decision model, at what threshold, on your own labelled data.

Every decision model returns probabilities, but how often each one is right, and how often it is
sure enough to act alone, depends on the data: on the tool-call benchmark Jev settled 64% of items
at 0.7 and agent-decision-3b 25%. So the choice is measured, not read off a leaderboard:

    from grounded_ai import compare
    report = compare(items, ["jev/jev-latest", "openai-decisions/gpt-6-luna",
                             "systemone/agent-decision-9b@http://gpu-box:8765"],
                     judge="anthropic/claude-sonnet-4-6", thresholds=(0.5, 0.7, 0.9))
    print(report.table())

An item is {"state": ..., "questions": {name: question}, "labels": {name: expected}}, where expected
is the option for a choice, true/false for a noul and the level index for a score. Every question is
scored; a model answering an item fails that item's questions (counted as wrong, and reported).

With a judge, each threshold is also run as a cascade: answers below it go to the judge. The judge is
asked once per item, all questions together, and only about items some model would escalate; its
answers are reused for every model and threshold. (A live CascadeEvaluator sends only the escalated
questions, so a judge that answers differently in company could differ slightly.)

Model records can be saved and merged, so models that run in different places (a GPU box, a laptop
with API keys) end up in one report: `collect()` per place, then `report()` on all the records.
"""

import json
import statistics
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, Iterable, List, Optional, Sequence, Union

from pydantic import BaseModel

from .backends.jev import ChoiceAnswer, NoulAnswer, ScoreAnswer, confidence
from .schemas import EvaluationError

Item = Dict[str, Any]


class Record(BaseModel):
    """One model's answer to one question of one item."""

    model: str
    item: int
    question: str
    label: Any
    answer: Any = None  # the option, true/false, or the most likely level; None when the call failed
    top_probability: Optional[float] = None
    confidence: Optional[float] = None
    seconds: float = 0.0
    input_tokens: Optional[int] = None
    error: Optional[str] = None

    @property
    def right(self) -> bool:
        return self.error is None and self.answer == self.label


class ThresholdRow(BaseModel):
    threshold: float
    settled: float  # share of questions answered at or above the threshold
    settled_accuracy: Optional[float]  # how often those were right
    cascade_accuracy: Optional[float] = None  # with the judge answering the rest
    judge_share: Optional[float] = None


class ModelReport(BaseModel):
    model: str
    questions: int
    failed: int
    accuracy: float
    ece: Optional[float]  # expected calibration error of the top probability, 10 bins
    seconds_mean: float
    seconds_median: float
    input_tokens_mean: Optional[float]
    thresholds: List[ThresholdRow]
    recommended_threshold: Optional[float] = None  # lowest threshold whose settled answers reach target_accuracy


class CompareReport(BaseModel):
    models: Dict[str, ModelReport]
    judge: Optional[str] = None
    judge_accuracy: Optional[float] = None  # on the items it was asked about
    target_accuracy: Optional[float] = None

    def table(self) -> str:
        lines = [f"{'model':42} {'n':>5} {'accuracy':>8} {'ECE':>6} {'s/check':>8} {'tokens':>7}  "
                 + "  ".join(f"@{t.threshold:<4} settled/right" + ("/cascade" if self.judge else "")
                             for t in next(iter(self.models.values())).thresholds)]
        for name, m in self.models.items():
            ece = f"{m.ece:.3f}" if m.ece is not None else "-"
            tokens = f"{m.input_tokens_mean:.0f}" if m.input_tokens_mean is not None else "-"
            cells = []
            for t in m.thresholds:
                right = f"{t.settled_accuracy:.2f}" if t.settled_accuracy is not None else "-"
                cell = f"{t.settled:>5.0%}/{right}"
                if t.cascade_accuracy is not None:
                    cell += f"/{t.cascade_accuracy:.3f}"
                cells.append(f"{cell:>{22 if self.judge else 14}}")
            failed = f" ({m.failed} failed)" if m.failed else ""
            lines.append(f"{name[:42]:42} {m.questions:>5} {m.accuracy:>8.3f} {ece:>6} {m.seconds_mean:>8.3f} "
                         f"{tokens:>7}  " + "  ".join(cells) + failed)
            if m.recommended_threshold is not None:
                lines.append(f"{'':42}   act alone at >= {m.recommended_threshold} for {self.target_accuracy:.0%} right")
        if self.judge:
            lines.append(f"judge: {self.judge}" + (f", {self.judge_accuracy:.3f} right on the items it saw"
                                                   if self.judge_accuracy is not None else ""))
        return "\n".join(lines)


def _evaluator(model: Any):
    """A model string ("systemone/<model>@<base_url>" for a server address), an Evaluator or a backend."""
    from . import Evaluator

    if isinstance(model, str):
        name, _, base_url = model.partition("@")
        return Evaluator(name, **({"base_url": base_url} if base_url else {})).backend
    return getattr(model, "backend", model)


def _read(answer: Any, label: Any) -> Dict[str, Any]:
    if isinstance(answer, NoulAnswer):
        return {"answer": answer.noul >= 0.5, "top_probability": max(answer.noul, 1 - answer.noul)}
    if isinstance(answer, ChoiceAnswer):
        return {"answer": answer.choice, "top_probability": answer.probabilities[answer.choice]}
    if isinstance(answer, ScoreAnswer):
        level, p = max(((int(k), v) for k, v in answer.probabilities.items()), key=lambda kv: kv[1])
        return {"answer": level, "top_probability": p}
    raise TypeError(f"not a decision answer: {type(answer).__name__}")


def collect(items: Sequence[Item], models: Union[Sequence[Any], Dict[str, Any]], workers: int = 6) -> List[Record]:
    """Ask every model every item; one Record per model, item and question."""
    from .backends.jev import JevInput

    named = models if isinstance(models, dict) else {m if isinstance(m, str) else type(m).__name__: m for m in models}
    records: List[Record] = []
    for name, model in named.items():
        evaluator = _evaluator(model)

        def ask(i: int) -> List[Record]:
            item = items[i]
            start = time.perf_counter()
            try:
                result = evaluator.evaluate(JevInput(state=item["state"], questions=item["questions"]))
            except Exception as e:  # noqa: BLE001 - one bad item must not end the comparison
                result = EvaluationError(error_code="UNKNOWN_ERROR", message=str(e))
            seconds = time.perf_counter() - start
            out = []
            for q, label in item["labels"].items():
                base = dict(model=name, item=i, question=q, label=label, seconds=seconds)
                if isinstance(result, EvaluationError):
                    out.append(Record(**base, error=f"{result.error_code}: {result.message}"))
                    continue
                answer = result.answers[q]
                usage = result.usage or {}
                out.append(Record(**base, **_read(answer, label), confidence=confidence(answer),
                                  input_tokens=usage.get("input_tokens")))
            return out

        with ThreadPoolExecutor(workers) as pool:
            for rs in pool.map(ask, range(len(items))):
                records.extend(rs)
    return records


def _ece(records: List[Record], bins: int = 10) -> Optional[float]:
    scored = [r for r in records if r.top_probability is not None]
    if not scored:
        return None
    total = 0.0
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        inside = [r for r in scored if lo < r.top_probability <= hi or (b == 0 and r.top_probability == 0)]
        if inside:
            gap = abs(statistics.mean(r.top_probability for r in inside) - statistics.mean(r.right for r in inside))
            total += len(inside) / len(scored) * gap
    return total


def _judge_answers(items: Sequence[Item], wanted: Iterable[int], judge: Any, workers: int) -> Dict[int, Dict[str, Any]]:
    """The judge's answer to every question of each wanted item, asked once per item."""
    from .cascade import JevLeftover, _judge_schema, _QUESTIONS

    def ask(i: int):
        questions = _QUESTIONS.validate_python(
            {k: (v.model_dump() if hasattr(v, "model_dump") else v) for k, v in items[i]["questions"].items()})
        leftover = JevLeftover(state=items[i]["state"], questions=questions)
        try:
            judged = judge.evaluate(leftover, output_schema=_judge_schema(questions))
        except Exception:  # noqa: BLE001
            return i, None
        if isinstance(judged, EvaluationError):
            return i, None
        answers = {}
        for n, (name, q) in enumerate(questions.items()):
            value = getattr(judged, f"answer_{n}")
            answers[name] = list(q.criteria).index(value) if q.type == "score" else value
        return i, answers

    with ThreadPoolExecutor(workers) as pool:
        return {i: a for i, a in pool.map(ask, sorted(set(wanted))) if a is not None}


def report(
    items: Sequence[Item],
    records: Sequence[Record],
    thresholds: Sequence[float] = (0.5, 0.7, 0.8, 0.9),
    judge: Any = None,
    judge_answers: Optional[Dict[int, Dict[str, Any]]] = None,
    target_accuracy: Optional[float] = None,
    workers: int = 6,
) -> CompareReport:
    """Summarise records (from collect(), possibly from several places). `judge` is asked only about
    items some model leaves below the highest threshold; `judge_answers` supplies them instead."""
    by_model: Dict[str, List[Record]] = {}
    for r in records:
        by_model.setdefault(r.model, []).append(r)
    judged: Dict[int, Dict[str, Any]] = dict(judge_answers or {})
    judge_name = None
    if judge is not None or judge_answers:
        top = max(thresholds)
        wanted = {r.item for r in records if r.item not in judged and (r.confidence is None or r.confidence < top)}
        if judge is not None and wanted:
            backend = _evaluator(judge) if isinstance(judge, str) else getattr(judge, "backend", judge)
            judged.update(_judge_answers(items, wanted, backend, workers))
        judge_name = judge if isinstance(judge, str) else (type(judge).__name__ if judge is not None else "given answers")

    models = {}
    for name, rs in by_model.items():
        rows = []
        for t in thresholds:
            settled = [r for r in rs if r.confidence is not None and r.confidence >= t]
            row = ThresholdRow(
                threshold=t, settled=len(settled) / len(rs),
                settled_accuracy=statistics.mean(r.right for r in settled) if settled else None,
            )
            if judge_name:
                right, sent = [], 0
                for r in rs:
                    if r.confidence is not None and r.confidence >= t:
                        right.append(r.right)
                    else:
                        sent += 1
                        answer = judged.get(r.item, {}).get(r.question)
                        right.append(answer is not None and answer == r.label)
                row.cascade_accuracy, row.judge_share = statistics.mean(right), sent / len(rs)
            rows.append(row)
        recommended = None
        if target_accuracy is not None:
            recommended = next((row.threshold for row in sorted(rows, key=lambda x: x.threshold)
                                if row.settled_accuracy is not None and row.settled_accuracy >= target_accuracy), None)
        tokens = [r.input_tokens for r in rs if r.input_tokens is not None]
        per_item = {r.item: r.seconds for r in rs}  # one call per item
        models[name] = ModelReport(
            model=name, questions=len(rs), failed=sum(r.error is not None for r in rs),
            accuracy=statistics.mean(r.right for r in rs), ece=_ece(rs),
            seconds_mean=statistics.mean(per_item.values()), seconds_median=statistics.median(per_item.values()),
            input_tokens_mean=statistics.mean(tokens) if tokens else None,
            thresholds=rows, recommended_threshold=recommended,
        )
    judge_accuracy = None
    if judged:
        labels = [(items[i]["labels"][q], a) for i, answers in judged.items() for q, a in answers.items()
                  if q in items[i]["labels"]]
        judge_accuracy = statistics.mean(label == a for label, a in labels) if labels else None
    return CompareReport(models=models, judge=judge_name, judge_accuracy=judge_accuracy, target_accuracy=target_accuracy)


def compare(
    items: Sequence[Item],
    models: Union[Sequence[Any], Dict[str, Any]],
    judge: Any = None,
    thresholds: Sequence[float] = (0.5, 0.7, 0.8, 0.9),
    target_accuracy: Optional[float] = None,
    workers: int = 6,
) -> CompareReport:
    """Ask every model every item and summarise: accuracy, calibration, speed, and at each threshold
    how much it settles alone, how often those are right, and (with a judge) the cascade's accuracy."""
    return report(items, collect(items, models, workers), thresholds, judge=judge,
                  target_accuracy=target_accuracy, workers=workers)


def load_items(path: str) -> List[Item]:
    """Items from a JSONL file, one {"state", "questions", "labels"} per line."""
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def save_records(records: Sequence[Record], path: str) -> None:
    with open(path, "w") as f:
        f.writelines(r.model_dump_json() + "\n" for r in records)


def load_records(path: str) -> List[Record]:
    with open(path) as f:
        return [Record.model_validate_json(line) for line in f if line.strip()]


__all__ = ["compare", "collect", "report", "Record", "CompareReport", "ModelReport", "ThresholdRow",
           "load_items", "save_records", "load_records"]
