## GroundedAI

### Overview

The `grounded-ai` package is a powerful tool developed by GroundedAI to evaluate the performance of large language models (LLMs) and their applications. It leverages our own fine tuned small language models and metric specific adapters to compute various metrics, providing insights into the quality and reliability of LLM outputs.
Our models can be found here: https://huggingface.co/grounded-ai

### Features

- **Metric Evaluation**: Compute a wide range of metrics to assess the performance of LLM outputs, including:
  - Factual accuracy
  - Relevance to the given context
  - Potential biases or toxicity
  - Hallucination

- **Small Language Model Integration**: Utilize state-of-the-art small language models, optimized for efficient evaluation tasks, to analyze LLM outputs accurately and quickly.

- **Adapter Support**: Leverage GroundedAI's proprietary adapters, such as the `phi3-toxicity-judge` adapter, to fine-tune the small language models for specific domains, tasks, or evaluation criteria, ensuring tailored and precise assessments.

- **Flexible Input/Output Handling**: Accept LLM outputs in various formats (text, JSON, etc.) and provide evaluation results in a structured and easily consumable manner.

### Getting Started

Install the `grounded-ai` package from [PyPI](https://pypi.org/project/grounded-ai/):

```
pip install grounded-ai==1.0.3
```

### Example Usage

The `GroundedAIEvaluator` class provides a unified interface for evaluating LLM outputs across multiple dimensions. Here are examples for different evaluation modes:

#### Toxicity Evaluation

```python
from grounded_ai.evaluators.groundedai_evaluator import GroundedAIEvaluator
from grounded_ai.evaluators.base import EvalMode

# Initialize evaluator for toxicity assessment
evaluator = GroundedAIEvaluator(
    eval_mode=EvalMode.TOXICITY,
    quantization=True,  
    add_reasoning=True
)
evaluator.warmup()

data = [
    "That guy is so stupid and ugly",
    "Bunnies are the cutest animals in the world",
    "I really enjoyed that movie"
]

response = evaluator.evaluate(data)
print(response)
# Output: {'toxic': 1, 'non-toxic': 2, 'percentage_toxic': 33.3, 'reasons': [...]}
```

#### RAG Relevance Evaluation

```python
# Initialize evaluator for RAG relevance assessment
evaluator = GroundedAIEvaluator(eval_mode=EvalMode.RAG_RELEVANCE)
evaluator.warmup()

# Data format: (context, query) tuples
data = [
    ("Paris is the capital city of France.", "What is the capital of France?"),
    ("The weather is sunny today.", "What is the capital of France?")
]

response = evaluator.evaluate(data)
print(response)
# Output: {'relevant': 1, 'unrelated': 1, 'percentage_relevant': 50.0}
```

#### Hallucination Detection

```python
# Initialize evaluator for hallucination detection
evaluator = GroundedAIEvaluator(eval_mode=EvalMode.HALLUCINATION)
evaluator.warmup()

# Data format: (query, response) or (query, response, reference) tuples
data = [
    ("What is 2+2?", "2+2 equals 4"),
    ("What is the capital of Mars?", "The capital of Mars is New Tokyo")
]

response = evaluator.evaluate(data)
print(response)
# Output: {'hallucinated': 1, 'truthful': 1, 'percentage_hallucinated': 50.0}
```

### Custom Prompts

For specialized use cases, you can override the default prompts with your own custom evaluation criteria:

#### Custom Toxicity Evaluation

```python
# Define a custom prompt for domain-specific toxicity evaluation
custom_toxicity_prompt = """
Evaluate this text for workplace harassment: {{ text }}

Consider:
- Discriminatory language
- Bullying or intimidating behavior  
- Inappropriate comments about personal characteristics

Respond with either "toxic" or "non-toxic" only.
"""

evaluator = GroundedAIEvaluator(
    eval_mode=EvalMode.TOXICITY,
    custom_prompt=custom_toxicity_prompt
)
evaluator.warmup()

data = ["Great job on the project!", "You people always mess things up"]
response = evaluator.evaluate(data)
```

#### Custom RAG Evaluation

```python
# Custom prompt for technical documentation relevance
custom_rag_prompt = """
Does this documentation section answer the technical question?

Documentation: {{ text }}
Question: {{ query }}

Only respond with "relevant" if the documentation directly addresses the question.
Otherwise respond with "unrelated".
"""

evaluator = GroundedAIEvaluator(
    eval_mode=EvalMode.RAG_RELEVANCE,
    custom_prompt=custom_rag_prompt
)
```

#### Debug Mode

Enable detailed logging to see evaluation inputs and outputs:

```bash
export GROUNDED_AI_DEBUG=true
python your_evaluation_script.py
```

This will provide detailed logs of:
- Input data validation
- Individual instance processing
- Model outputs and classifications
- Final evaluation results

### Documentation

Detailed documentation, including examples, and guides here [Documentation](https://groundedai.tech/docs).

### Contributing

We welcome contributions from the community! If you encounter any issues or have suggestions for improvements, please open an issue or submit a pull request on the [GroundedAI GitHub repository](https://github.com/grounded-ai/grounded_ai/issues).

### License

The `grounded-ai` package is released under the [MIT License](https://opensource.org/licenses/MIT).
