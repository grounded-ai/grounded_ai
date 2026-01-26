# GroundedAI

![CI](https://github.com/grounded-ai/grounded_ai/actions/workflows/ci.yml/badge.svg)
[![PyPI](https://img.shields.io/pypi/v/grounded-ai)](https://pypi.org/project/grounded-ai/)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

**The Universal Evaluation Interface for LLM Applications.**

`grounded-ai` provides a unified, type-safe Python API to evaluate your LLM application's outputs using either local Small Language Models (SLMs) or frontier Cloud Models (OpenAI, Anthropic).

We standardize the **Inputs** (Response, Query, Context) and **Outputs** (Score, Label, Confidence, Reasoning) regardless of the underlying model backend.

## Features

- **Unified API**: Switch between local SLMs and cloud providers just by changing the model string.
- **Strict Typing**: All inputs (`EvaluationInput`) and outputs (`EvaluationOutput`) are Pydantic models.
- **Local Privacy**: Run Grounded AI's fine-tuned evaluation models locally on your GPU (zero data egress).
- **Structured Outputs**: Native support for OpenAI (`parse`) and Anthropic (`tool_use`) JSON constraints.
- **Customizable**: Use Jinja2 templates to format your prompts exactly how you want.

## Implementation Status

| Backend | Status | Description |
| :--- | :--- | :--- |
| **Grounded AI SLM** | ✅ **Production** | specialized local models (Phi-4 based) for Hallucination, Toxicity, and RAG Relevance. |
| **OpenAI** | ✅ **Production** | Uses `gpt-4o`/`mini` with strict Structured Outputs. |
| **Anthropic** | ✅ **Production** | Uses `claude-4-5` series with Beta Structured Outputs. |
| **HuggingFace** | 🚧 **Beta** | Run any generic HF model locally. |
| **Integrations** | 🏗️ **Planned** | LangSmith Tracing, OpenTelemetry, AWS Bedrock. |

## Backend Capabilities

| Feature | OpenAI | Anthropic | HuggingFace | Grounded AI SLM |
| :--- | :--- | :--- | :--- | :--- |
| **System Prompt Fallback** | ✅ `default` if None | ✅ `default` if None | ✅ `default` if None | ✅ `SYSTEM_PROMPT_BASE` |
| **Input Formatting** | ✅ `formatted_prompt` | ✅ `formatted_prompt` | ✅ `formatted_prompt` | 🛠️ Specialized Jinja |
| **Kwargs Passthrough** | ✅ Merged | ✅ Merged | ✅ Merged | ✅ Merged |
| **Schema Validation** | 🔒 Native `response_format` | 🔒 Native `json_schema` | ⚡ Generic Injection | ⚡ Regex Parsing |

## Installation

**Basic (Cloud Backends only):**
```bash
pip install grounded-ai
```

**Local SLM Support (GPU Recommended):**
```bash
pip install grounded-ai[slm]
```

## Quick Start

### 1. Using Grounded AI Local SLMs (Recommended)

Run specialized evaluation models locally.

```python
from grounded_ai import Evaluator, EvalMode

# Initialize for Hallucination detection
evaluator = Evaluator(
    "grounded-ai/hallucination-v1",
    eval_mode=EvalMode.HALLUCINATION,
    device="cuda" # or "cpu"
)

# Detect if the response contradicts the context
result = evaluator.evaluate(
    response="London is the capital of France.",
    query="What is the capital?",
    context="Paris is the capital of France."
)

print(result)
# score=1.0 label='hallucinated' confidence=0.99 reasoning='Contradicts context Paris.'
```

### 2. Using OpenAI (GPT-4o)

```python
import os
os.environ["OPENAI_API_KEY"] = "sk-..."

evaluator = Evaluator("openai/gpt-4o")

result = evaluator.evaluate(
    response="The moon is made of cheese.",
    context="The moon is made of rock."
)
```

### 3. Using Anthropic (Claude)

```python
import os
os.environ["ANTHROPIC_API_KEY"] = "sk-ant-..."

evaluator = Evaluator("anthropic/claude-haiku-4-5-20251001")

result = evaluator.evaluate(response="This content is safe.")
```

### 4. Custom Templating

You can completely customize how your inputs are presented to the model using Jinja2 templates. Variables `{{ response }}`, `{{ context }}`, `{{ query }}` are available.

```python
from grounded_ai import Evaluator, EvaluationInput

evaluator = Evaluator("openai/gpt-4o")

# Define a custom template
custom_template = """
SYSTEM: You are a strict code reviewer.
CODE: {{ response }}

{% if context %}
CONTEXT: {{ context }}
{% endif %}

Evaluate the code quality.
"""

input_data = EvaluationInput(
    response="def foo(): pass",
    context="Python coding standards",
    base_template=custom_template
)

# Prompts the model with your custom format!
result = evaluator.evaluate(input_data)
```

## API Reference

### `Evaluator` Factory

```python
Evaluator(
    model: str,          # e.g., "grounded-ai/...", "openai/...", "anthropic/..."
    eval_mode: str,      # Optional: "TOXICITY", "HALLUCINATION", "RAG_RELEVANCE"
    **kwargs             # Backend-specific args (e.g. quantization=True, temperature=0.1)
)
```

### `evaluate()`

```python
evaluate(
    response: str,              # The primary content to evaluate (output/document)
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

We welcome contributions! Please see `TODOs.md` for the current roadmap and `design.md` for architecture details.

## License

Apache 2.0
