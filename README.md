# GroundedAI

![CI](https://github.com/grounded-ai/grounded_ai/actions/workflows/ci.yml/badge.svg)
[![PyPI](https://img.shields.io/pypi/v/grounded-ai)](https://pypi.org/project/grounded-ai/)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

**The Universal Evaluation Interface for LLM Applications.**

`grounded-ai` provides a unified, type-safe Python API to evaluate your LLM application's outputs. It supports a wide range of backends, from specialized local models to frontier LLMs (OpenAI, Anthropic).

We standardize the evaluation interface while keeping everything modular. Define your own Inputs, Outputs, System Prompts, and prompt formatting logic—or use our defaults.

## Why Grounded AI?

Most evaluation libraries are black boxes. **Grounded AI** is different:

1.  **Standardization**: A single, type-safe function (`evaluate()`) for *any* backend (Grounded AI SLM, HuggingFace, OpenAI, Anthropic).
2.  **Modularity**: Don't like our prompts? **Change them.** Don't like our schemas? **Bring your own.** Every part of the pipeline is customizable.
3.  **Evaluations Made Easy**: JSON-mode and schema validation are handled for you. Just focus on your data.
4.  **Privacy First**: First-class support for running evaluations 100% locally on your own GPU.

## Decoupled Architecture

Grounded AI is built on a philosophy of separation of concerns:

1.  **No Metric Lock-in**: Unlike other eval libraries that lock you into their pre-defined, black-box metrics, Grounded AI puts you in control. Evaluations are just Pydantic schemas. Need a specific "Brand Voice Compliance" metric? Define it yourself in seconds. You are never limited to what the vendor provides.
2.  **Model / Provider Agnostic Backends**: The evaluation *definition* is decoupled from the *execution engine*. You can run the exact same metric on **GPT-4o** for high-precision audits, or switch to a local **Llama Guard** model for high-volume CI/CD checks—without changing a single line of your validation logic.


## Implementation Status

| Backend | Status | Description |
| :--- | :--- | :--- |
| **Grounded AI SLM** | ✅ | specialized local models (Phi-4 based) for Hallucination, Toxicity, and RAG Relevance. |
| **OpenAI** | ✅ | Uses `gpt-4o`/`mini` with strict Structured Outputs. |
| **Anthropic** | ✅ | Uses `claude-4-5` series with Beta Structured Outputs. |
| **HuggingFace** | ✅ | Run any generic HF model locally. |
| **Integrations** | 🏗️ **Planned** | LiteLLM |

## Backend Capabilities

| Feature | Grounded AI SLM | OpenAI | Anthropic | HuggingFace |
| :--- | :--- | :--- | :--- | :--- |
| **System Prompt Fallback** | ✅ `SYSTEM_PROMPT_BASE` | ✅ `default` if None | ✅ `default` if None | ✅ `default` if None |
| **Input Formatting** | 🛠️ Specialized Jinja | ✅ `formatted_prompt` | ✅ `formatted_prompt` | ✅ `formatted_prompt` |
| **Schema Validation** | ⚡ Regex Parsing | 🔒 Native `response_format` | 🔒 Native `json_schema` | ⚡ Generic Injection |


## Installation

**Basic (LLM Providers only):**
```bash
pip install grounded-ai
```

**Local Inference Support (GPU Recommended):**
```bash
pip install grounded-ai[slm]
```

## Quick Start

### 1. Evaluation with SLM's
Run specialized models locally on your GPU. No API keys needed.

```python
from grounded_ai import Evaluator

# Auto-downloads the localized judge model
evaluator = Evaluator("grounded-ai/phi4-mini-judge", device="cuda")

# Check for Hallucinations
result = evaluator.evaluate(
    response="London is the capital of France.",
    context="Paris is the capital of France.",
    eval_mode="HALLUCINATION"
)
print(result.label) # 'hallucinated'
```

### 2. Evaluation with Proprietary Models
Use GPT-4o or Claude for high-precision auditing. We handle the structured output complexity.

```python
import os
os.environ["OPENAI_API_KEY"] = "sk-..."

evaluator = Evaluator("openai/gpt-4o")

result = evaluator.evaluate(
    response="The user is asking for illegal streaming sites.",
    system_prompt="Is this content safe?"
)
print(result)
# EvaluationOutput(score=1.0, label='unsafe', ...)
```

### 3. Custom Metrics
Define your OWN metrics using Pydantic. Use this for "Brand Compliance", "Code Quality", or anything specific to your business.

```python
from pydantic import BaseModel

class BrandCheck(BaseModel):
    tone_compliant: bool
    forbidden_words: list[str]

evaluator = Evaluator("openai/gpt-4o")

result = evaluator.evaluate(
    response="Our product is kinda cheap.",
    output_schema=BrandCheck
)
# Returns a typed object directly!
print(result.forbidden_words) # ['kinda', 'cheap']
```

### 4. Agent Trace Evaluation
Flatten complex agent traces (OpenTelemetry, LangSmith) into a linear story for evaluation.

```python
from grounded_ai.otel import TraceConverter

# 1. Convert scattered OTel spans into a logical conversation
conversation = TraceConverter.from_otlp(raw_spans)

# 2. Extract the reasoning chain (Thought -> Tool -> Observation -> Answer)
# This unifies the agent's logic flow.
reasoning = conversation.get_reasoning_chain()

# 3. Evaluate the full flow
evaluator = Evaluator("openai/gpt-4o")
result = evaluator.evaluate(
    response="\n".join(reasoning),
    system_prompt="Did the agent complete the task correctly?"
)
```

## API Reference

### `Evaluator` Factory

```python
Evaluator(
    model: str,      # e.g., "grounded-ai/...", "openai/...", "anthropic/..."
    eval_mode: str,  # Required for Grounded AI SLMs only ("TOXICITY", "HALLUCINATION", "RAG_RELEVANCE")
    **kwargs         # Backend-specific args (e.g. quantization=True, temperature=0.1)
)
```

### `evaluate()`

```python
evaluate(
    response: str,              # The primary content to evaluate from the model or user
    query: Optional[str],       # User question
    context: Optional[str]      # Retrieved context or ground truth
) -> EvaluationOutput | EvaluationError
```

### Output Schema

```python
class EvaluationOutput(BaseModel):
    score: float       # 0.0 to 1.0 (0.0 = Good/Faithful, 1.0 = Bad/Hallucinated/Toxic)
    label: str         # e.g. "faithful", "toxic", "relevant"
    confidence: float  # 0.0 to 1.0
    reasoning: str     # Explanation
```

## Contributing

We welcome contributions! Please feel free to submit a Pull Request or open an Issue on GitHub.

## License

MIT 
