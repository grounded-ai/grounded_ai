# grounded-ai

Universal evaluation interface for LLM application outputs. Single `Evaluator` factory routes to backends: Grounded AI SLM (local fine-tuned Phi-4), OpenAI, Anthropic, AWS Bedrock, HuggingFace, Strands Decider (local decision model over HTTP). Includes an OTel trace converter for evaluating agent traces.

## Project Layout

```
grounded_ai/
  __init__.py          # Evaluator factory + public API
  base.py              # BaseEvaluator ABC
  schemas.py           # EvaluationInput, EvaluationOutput, EvaluationError
  backends/
    openai.py
    anthropic.py
    bedrock.py
    huggingface.py
    decider.py         # /v1/systemone client: one typed question per output-schema field
    grounded_ai_slm/
      backend.py       # PEFT adapter loading, prompt formatting, XML parsing
      prompts.py       # Jinja2 templates for TOXICITY / RAG_RELEVANCE / HALLUCINATION
  otel/
    schemas.py         # GenAISpan, GenAIConversation, MessagePart, TokenUsage
    converter.py       # TraceConverter: OTLP + LangSmith → GenAIConversation
tests/
examples/backends/     # Jupyter notebooks per backend
examples/otel/
```

## Key Design Decisions

- **Lazy backend imports** in `_load_backend` — heavy dependencies (torch, boto3) only load when needed.
- **`EvaluationInput.formatted_prompt`** is a Pydantic `computed_field` that renders a Jinja2 `base_template`. The template can be overridden at instantiation for per-call customization.
- **`BaseEvaluator.evaluate()`** normalizes dict → schema then delegates to `_call_backend()`. Runtime `output_schema` overrides the instance default.
- **OTel module** is standalone — `TraceConverter.from_otlp()` and `from_langsmith()` normalize spans into `GenAIConversation`, which serializes to an evaluation string.

---

## Known Issues & Prioritized Fix List

### P0 — Correctness bugs

**1. Silent kwarg drop in `Evaluator.evaluate()`** (`grounded_ai/__init__.py:70-98`)
Runtime kwargs like `temperature=0.7` are passed to `EvaluationInput(**kwargs)` when `input_data is None`. Pydantic v2 silently ignores unknown fields, so they never reach the backend. When `input_data` is provided (BaseModel/str/GenAIConversation), kwargs are consumed but not forwarded to `self.backend.evaluate()` at all.
- Fix: separate input-construction kwargs from backend-runtime kwargs; forward the latter to `backend.evaluate()`.

**2. SLM hardcodes `confidence=1.0`** (`grounded_ai/backends/grounded_ai_slm/backend.py:241`)
The model outputs `<rating>` and `<reasoning>` XML tags only. Confidence is always returned as `1.0` regardless of actual model certainty. This makes the field misleading for every SLM evaluation.
- Fix: either drop the field from SLM output, set it to `0.0` explicitly, or have the model output a confidence token.

**3. Inconsistent error handling across backends**
- OpenAI + Anthropic: catch-all `except` → return `EvaluationError`
- SLM backend: no try/except in `_call_backend` — exceptions propagate raw
- HuggingFace `_evaluate_classification`: no error handling at all
- Fix: wrap `_call_backend` in each backend (or in `BaseEvaluator.evaluate()`) with consistent error normalization.

### P1 — Dependency / packaging

**4. `openai`, `anthropic`, `boto3` are hard dependencies** (`pyproject.toml`)
Every install pulls all three SDKs unconditionally. These should be optional extras (e.g., `pip install grounded-ai[openai]`), the same way `[slm]` is handled. Most users only need one provider.

**5. License mismatch**
README badge says MIT. `pyproject.toml` classifier says `Apache Software License`. Pick one and make it consistent everywhere.

**6. `requires-python = ">=3.8"` is unvalidated**
CI matrix only tests 3.10, 3.11, 3.12. Either test 3.8/3.9 or narrow the declared minimum.

### P2 — Code quality

**7. `_enforce_strict_schema` is copy-pasted** (`backends/anthropic.py` and `backends/bedrock.py`)
The bedrock version is more complete — it also handles `$defs`, `definitions`, and arrays. The anthropic version misses those cases. Extract to a shared utility in `grounded_ai/utils.py` or similar.

**8. Hardcoded Anthropic beta string** (`backends/anthropic.py:104`)
`betas=["structured-outputs-2025-11-13"]` — date-stamped beta identifiers get removed when a feature graduates to stable. When that happens, every Anthropic evaluation call will break with a confusing API error.

**9. HuggingFace text-generation path is broken for eval purposes** (`backends/huggingface.py:164-170`)
The default `EvaluationOutput` path always returns `score=0.0`, `confidence=0.0`, `label="generated_text"`. It stuffs the raw generation into `reasoning`. This is only meaningful if the caller uses a custom output schema, but the README doesn't make that caveat clear. `text-classification` (Prompt Guard) works correctly.

### P3 — Missing features

**10. No async support** — evaluating at CI scale requires async. All backends are synchronous.

**11. No batch evaluation** — no `evaluate_batch(items)` convenience method.

**12. No retry/backoff logic** — rate limits silently become `EvaluationError` with no recovery.

**13. No caching** — repeated evaluations on the same inputs re-call the API every time.

### Nitpicks

- `base_template` is a field on `EvaluationInput`, so it appears in `model_dump()` and is passed into Jinja's render context (accessible as `{{ base_template }}` inside templates). Harmless but leaky.
- `GenAIConversation.total_tokens` calls `usage.compute_total()` directly — that's a `@model_validator`, not a regular method. Works today but fragile if the model implementation changes.
- README API reference for `evaluate()` incorrectly shows `eval_mode` as a call-site parameter. It's an `__init__` arg on `GroundedAISLMBackend` only.
- `GenAISpan` uses both Pydantic aliases (`alias="gen_ai.system"`) and Python attribute names (`gen_ai_system`). Correct usage, but adds cognitive overhead for attribute access.
