"""compare(): decision models measured on labelled items, alone and as a cascade's first stage."""

import io
import json

import httpx
import pytest

from grounded_ai import compare
from grounded_ai.backends.jev import DecisionEvaluator
from grounded_ai.cli import main
from grounded_ai.comparison import collect, load_records, report, save_records
from tests.test_cascade import FakeJudge

NEXT = {"type": "choice", "instructions": "Next step?", "criteria": {"call_tool": "call a tool", "respond": "reply"}}
ITEMS = [{"state": f"item {i}", "questions": {"next": NEXT}, "labels": {"next": "call_tool" if i < 2 else "respond"}}
         for i in range(4)]


def model(p_call_by_item, fail_on=()):
    """A /v1/systemone server answering item i with P(call_tool) = p_call_by_item[i]."""

    def handler(request):
        body = json.loads(request.content)
        i = int(body["state"].split()[1])
        if i in fail_on:
            return httpx.Response(500, json={"detail": "boom"})
        p = p_call_by_item[i]
        answer = {"type": "choice", "choice": "call_tool" if p >= 0.5 else "respond", "confidence": 0,
                  "probabilities": {"call_tool": p, "respond": 1 - p}}
        return httpx.Response(200, json={"answers": {"next": answer}, "usage": {"input_tokens": 10}})

    return DecisionEvaluator("m", base_url="http://x", max_retries=0,
                             client=httpx.Client(transport=httpx.MockTransport(handler)))


# Item labels: call, call, respond, respond.
SURE = model([0.99, 0.95, 0.02, 0.4])  # last one right but unsure (confidence 0.2)
SHAKY = model([0.6, 0.3, 0.45, 0.9])  # right on 0 and 2, all unsure except item 3 (wrong, confidence 0.8)


def test_accuracy_settled_share_and_the_cascade():
    judge = FakeJudge({"next": "respond"})
    result = compare(ITEMS, {"sure": SURE, "shaky": SHAKY}, judge=judge, thresholds=(0.5, 0.9))
    sure, shaky = result.models["sure"], result.models["shaky"]
    assert (sure.accuracy, shaky.accuracy) == (1.0, 0.5)
    at_half = {m: r.thresholds[0] for m, r in result.models.items()}
    assert (at_half["sure"].settled, at_half["sure"].settled_accuracy) == (0.75, 1.0)
    assert (at_half["shaky"].settled, at_half["shaky"].settled_accuracy) == (0.25, 0.0)
    # Judge always says respond: right on items 2 and 3 only.
    assert at_half["sure"].cascade_accuracy == 1.0 and at_half["sure"].judge_share == 0.25
    assert at_half["shaky"].cascade_accuracy == 0.25 and at_half["shaky"].judge_share == 0.75
    assert result.judge_accuracy == 0.5
    assert sure.input_tokens_mean == 10


def test_the_judge_is_asked_once_per_item_and_only_when_needed():
    judge = FakeJudge({"next": "respond"})
    compare(ITEMS, {"sure": SURE, "shaky": SHAKY}, judge=judge, thresholds=(0.5, 0.9))
    asked = sorted(call[0].state for call in judge.calls)
    assert asked == ["item 0", "item 1", "item 2", "item 3"]  # every item is below 0.9 for some model, once each
    judge = FakeJudge({"next": "respond"})
    compare(ITEMS, {"sure": SURE}, judge=judge, thresholds=(0.5,))
    assert [call[0].state for call in judge.calls] == ["item 3"]


def test_recommended_threshold():
    mixed = model([0.99, 0.95, 0.3, 0.7])  # the last is wrong at confidence 0.4
    result = compare(ITEMS, {"mixed": mixed, "shaky": SHAKY}, thresholds=(0.3, 0.5, 0.9), target_accuracy=0.9)
    assert result.models["mixed"].thresholds[0].settled_accuracy == 0.75  # at 0.3 the wrong one is kept
    assert result.models["mixed"].recommended_threshold == 0.5
    assert result.models["shaky"].recommended_threshold is None
    assert "act alone at >= 0.5" in result.table()


def test_a_failed_call_counts_as_wrong_and_is_reported():
    result = compare(ITEMS, {"flaky": model([0.99, 0.99, 0.01, 0.01], fail_on={1})}, thresholds=(0.5,))
    flaky = result.models["flaky"]
    assert (flaky.failed, flaky.accuracy) == (1, 0.75)
    assert "(1 failed)" in result.table()


def test_records_from_different_places_merge(tmp_path):
    save_records(collect(ITEMS, {"sure": SURE}), tmp_path / "gpu.jsonl")
    save_records(collect(ITEMS, {"shaky": SHAKY}), tmp_path / "laptop.jsonl")
    merged = load_records(tmp_path / "gpu.jsonl") + load_records(tmp_path / "laptop.jsonl")
    assert set(report(ITEMS, merged).models) == {"sure", "shaky"}


def test_cli_reports_saved_records(tmp_path):
    items = tmp_path / "items.jsonl"
    items.write_text("".join(json.dumps(i) + "\n" for i in ITEMS))
    save_records(collect(ITEMS, {"sure": SURE}), tmp_path / "r.jsonl")
    out, err = io.StringIO(), io.StringIO()
    code = main(["compare", str(items), "--records", str(tmp_path / "r.jsonl"), "--json", str(tmp_path / "rep.json"),
                 "--thresholds", "0.5"], stdout=out, stderr=err)
    assert code == 0, err.getvalue()
    assert out.getvalue().splitlines()[1].startswith("sure")
    assert json.loads((tmp_path / "rep.json").read_text())["models"]["sure"]["accuracy"] == 1.0


def test_cli_needs_something_to_compare(tmp_path):
    items = tmp_path / "items.jsonl"
    items.write_text(json.dumps(ITEMS[0]) + "\n")
    err = io.StringIO()
    assert main(["compare", str(items)], stdout=io.StringIO(), stderr=err) == 2
    assert "nothing to compare" in err.getvalue()


@pytest.mark.parametrize("bad", ["systemone/x"])
def test_model_strings_name_the_server(bad):
    """systemone/ needs an address; compare takes it after '@'."""
    with pytest.raises(ValueError, match="base_url"):
        collect(ITEMS[:1], [bad])
