"""Builds the pull-request comment for the Decider live test: test output and notebook outputs."""

import json
import os
import pathlib

FENCE = "`" * 3


def read(path: str) -> str:
    file = pathlib.Path(path)
    return file.read_text() if file.exists() else ""


def notebook_outputs(path: str) -> dict:
    """{code cell index: what the cell printed} for the executed notebook."""
    raw = read(path)
    if not raw:
        return {}
    outputs = {}
    for index, cell in enumerate(json.loads(raw)["cells"]):
        text = "".join(
            "".join(o.get("text", [])) or "".join(o.get("data", {}).get("text/plain", []))
            for o in cell.get("outputs", [])
        )
        if cell["cell_type"] == "code" and text:
            outputs[str(index)] = text
    return outputs


tests = read("live-tests.txt")
outputs = notebook_outputs("executed/strands_decider.executed.ipynb")
sha = os.getenv("GITHUB_SHA", "")[:7]

print(f"### Decider live test ({os.getenv('DECIDER_CHECKPOINT')}, {os.getenv('DECIDER_DEVICE')}) at {sha}\n")
print(f"<details open><summary>pytest tests/integration</summary>\n\n{FENCE}text\n{tests[-30000:] or 'no output'}\n{FENCE}\n</details>\n")
print(f"<details><summary>Notebook outputs</summary>\n\n{FENCE}json\n{json.dumps(outputs, indent=1)}\n{FENCE}\n</details>")
