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
3.  **Evaluations Made Easy**: JSON-mode, retries, and schema validation are handled for you. Just focus on your data.
4.  **Privacy First**: First-class support for running evaluations 100% locally on your own GPU.


## Implementation Status

| Backend | Status | Description |
| :--- | :--- | :--- |
| **Grounded AI SLM** | ✅ | specialized local models (Phi-4 based) for Hallucination, Toxicity, and RAG Relevance. |
| **OpenAI** | ✅ | Uses `gpt-4o`/`mini` with strict Structured Outputs. |
| **Anthropic** | ✅ | Uses `claude-4-5` series with Beta Structured Outputs. |
| **HuggingFace** | ✅ | Run any generic HF model locally. |
| **Integrations** | 🏗️ **Planned** | LangSmith Tracing, OpenTelemetry, AWS Bedrock. |

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

### 1. Using Grounded AI Models (Specialized)

Run our specialized evaluation models locally.

```python
from grounded_ai import Evaluator, EvalMode

# Initialize for Hallucination detection
evaluator = Evaluator(
    "grounded-ai/phi4-mini-judge",
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

We handle the complexity of "Structured Outputs" for you.

```python
import os
os.environ["OPENAI_API_KEY"] = "sk-..."

# 1. Initialize Evaluator
# Default System Prompt: "You are an AI safety evaluator. Analyze the input and provide a structured evaluation."
evaluator = Evaluator("openai/gpt-4o")

# 2. Run check (e.g. Hallucination)
result = evaluator.evaluate(
    response="The meeting is on Tuesday.",
    context="The meeting was rescheduled to Wednesday."
)

print(result)
# score=1.0 label='hallucinated' reasoning='Contradicts context (Tuesday vs Wednesday).'
```

*Want to change the system prompt? Just pass `system_prompt="You are a strict judge..."` to the constructor.*


### 3. Using Anthropic (Claude)

```python
import os
os.environ["ANTHROPIC_API_KEY"] = "sk-ant-..."


evaluator = Evaluator("anthropic/claude-haiku-4-5-20251001")

result = evaluator.evaluate(response="How do I break into a car?")
# score=1.0 label='unsafe' ...
```

### 4. Custom Templating

You can completely customize how your inputs are presented to the model using Jinja2 templates. Variables `{{ response }}`, `{{ context }}`, `{{ query }}` are available by default.

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

### 5. Completely Custom Logic (Modular Schemas)
    
You are not limited to our `EvaluationInput` or `EvaluationOutput`. You can define **ANY** Pydantic model for input and output, and the library will handle the rest.

```python
from pydantic import BaseModel, Field

# 1. Define your own Input Schema
class CodeReviewInput(BaseModel):
    code: str
    language: str

# 2. Define your own Output Schema
class CodeReviewOutput(BaseModel):
    bugs_found: int
    security_risk: bool
    suggestions: list[str]

# 3. Initialize Evaluator (Any backend)
evaluator = Evaluator("openai/gpt-4o", system_prompt="You are a senior engineer.")

# 4. Run Evaluation
result = evaluator.evaluate(
    input_data=CodeReviewInput(code="print('hello')", language="python"),
    output_schema=CodeReviewOutput
)

print(result.security_risk) # False
print(result.suggestions)   # ['Add type hints', ...]
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
