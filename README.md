# GroundedAI

**The Universal Evaluation Interface for LLM Applications.**

`grounded-ai` provides a unified, type-safe Python API to evaluate your LLM application's outputs using either local Small Language Models (SLMs) or frontier Cloud Models (OpenAI, Anthropic).

We standardize the **Inputs** (Text, Query, Context) and **Outputs** (Score, Label, Confidence, Reasoning) regardless of the underlying model backend.

## Features

- **Unified API**: Switch between local SLMs and cloud providers just by changing the model string.
- **Strict Typing**: All inputs and outputs are Pydantic models.
- **Local Privacy**: Run Grounded AI's fine-tuned evaluation models locally on your GPU.
- **Structured Outputs**: Native support for OpenAI and Anthropic JSON constraints.

## Implementation Status

| Backend | Status | Description |
| :--- | :--- | :--- |
| **Grounded AI SLM** | ✅ **Production** | Our specialized local models for Hallucination, Toxicity, and RAG Relevance. |
| **OpenAI** | ✅ **Production** | Uses `gpt-4o`/`mini` with strict Structured Outputs (`parse`). |
| **Anthropic** | ✅ **Production** | Uses `claude-haiku-4-5` with Beta Structured Outputs (JSON Schema). |
| **HuggingFace** | 🚧 **Beta** | Run any generic HF model locally (naive implementation). |
| **Integrations** | 🏗️ **Planned** | LangSmith Tracing, OpenTelemetry, AWS Bedrock. |

## Installation

```bash
pip install grounded-ai
# OR for local GPU support
pip install grounded-ai[gpu]
```

## Quick Start

### 1. Using Grounded AI Local SLMs (Recommended)

Run specialized evaluation models locally (requires GPU/CUDA recommended).

```python
from grounded_ai import Evaluator, EvalMode

# Initialize for Hallucination detection
evaluator = Evaluator(
    "grounded-ai/hallucination-v1",
    eval_mode=EvalMode.HALLUCINATION
)

result = evaluator.evaluate(
    text="London is the capital of France.",
    query="What is the capital?",
    context="Paris is the capital of France."
)

print(result)
# score=1.0 label='hallucinated' confidence=0.99 reasoning='Contradicts context.'
```

### 2. Using OpenAI (GPT-4o)

```python
import os
os.environ["OPENAI_API_KEY"] = "sk-..."

evaluator = Evaluator("openai/gpt-4o")

result = evaluator.evaluate(
    text="The moon is made of cheese.",
    context="The moon is made of rock."
)
```

### 3. Using Anthropic (Claude)

```python
import os
os.environ["ANTHROPIC_API_KEY"] = "sk-ant-..."

evaluator = Evaluator("anthropic/claude-haiku-4-5-20251001")

result = evaluator.evaluate(text="This content is safe.")
```

### 4. Custom Templating (New!)

You can completely customize how your inputs are presented to the model using Jinja2 templates directly in the input schema.

```python
from grounded_ai import Evaluator, EvaluationInput

evaluator = Evaluator("openai/gpt-4o")

# Define a custom template (Jinja2 supported)
custom_template = """
SYSTEM: You are a strict code reviewer.
CODE: {{ text }}

{% if context %}
CONTEXT: {{ context }}
{% endif %}

Evaluate the code quality.
"""

input_data = EvaluationInput(
    text="def foo(): pass",
    context="Python coding standards",
    base_template=custom_template
)

# prompt sent: 
# SYSTEM: You are a strict code reviewer.
# CODE: def foo(): pass
# CONTEXT: Python coding standards
# Evaluate the code quality.

result = evaluator.evaluate(input_data)
```

## API Reference

### `Evaluator` Factory

```python
Evaluator(
    model: str,          # e.g., "grounded-ai/...", "openai/...", "anthropic/..."
    eval_mode: str,      # Optional: "TOXICITY", "HALLUCINATION", "RAG_RELEVANCE"
    **kwargs             # Backend-specific args (e.g. quantization=True, temperature=0.0)
)
```

### `evaluate()`

```python
evaluate(
    text: str,                  # The content to evaluate (response/document)
    query: Optional[str],       # User question
    context: Optional[str],     # Retrieved context
    reference: Optional[str]    # Ground truth answer
) -> EvaluationOutput | EvaluationError
```

### Output Schema

```python
class EvaluationOutput(BaseModel):
    score: float       # 0.0 to 1.0
    label: str         # e.g. "faithful", "toxic"
    confidence: float  # 0.0 to 1.0
    reasoning: str     # Explanation
```

## Contributing

We welcome contributions! Please see `TODOs.md` for the current roadmap.

## License

Apache 2.0
