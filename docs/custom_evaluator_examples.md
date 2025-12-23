# Custom Evaluator Examples

This document shows how to use the CustomEvaluator framework for maximum flexibility with your own models and evaluation logic.

## Quick Start Examples

### 1. OpenAI GPT-4 Evaluator

```python
from grounded_ai.evaluators.custom_evaluator import CustomEvaluator
import openai

# Create OpenAI model runner
def openai_model_runner(prompt: str) -> str:
    client = openai.OpenAI()  # Uses OPENAI_API_KEY env var
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=100
    )
    return response.choices[0].message.content

# Create evaluation function
def sentiment_evaluate_func(data, responses):
    """Count positive, negative, and neutral sentiments"""
    positive = sum(1 for r in responses if "positive" in r.lower())
    negative = sum(1 for r in responses if "negative" in r.lower())
    neutral = len(responses) - positive - negative
    
    return {
        "positive": positive,
        "negative": negative, 
        "neutral": neutral,
        "total": len(responses),
        "positive_rate": positive / len(responses) if responses else 0
    }

# Create sentiment evaluator
evaluator = CustomEvaluator(
    model_runner=openai_model_runner,
    prompt_template="Classify sentiment as POSITIVE, NEGATIVE, or NEUTRAL:\n\n{{ text }}\n\nSentiment:",
    evaluate_func=sentiment_evaluate_func
)

# Run evaluation
data = [
    {"text": "I love this product!"},
    {"text": "This is terrible"},
    {"text": "It's okay, nothing special"}
]

results = evaluator.evaluate(data)
print(results)
# Output: {'positive': 1, 'negative': 1, 'neutral': 1, 'total': 3, 'positive_rate': 0.33, 'metadata': {...}}
```

### 2. Custom Local Model

```python
import requests

def my_local_model_runner(prompt: str) -> str:
    """Example using a local API endpoint"""
    response = requests.post(
        "http://localhost:8080/generate",
        json={"prompt": prompt, "max_tokens": 50}
    )
    return response.json()["text"]

def accuracy_evaluate_func(data, responses):
    """Evaluate accuracy by checking for keywords"""
    correct = 0
    for i, response in enumerate(responses):
        # Custom logic to determine if response is correct
        if "correct" in response.lower() or "yes" in response.lower():
            correct += 1
    
    return {
        "correct": correct,
        "total": len(responses),
        "accuracy": correct / len(responses) if responses else 0
    }

evaluator = CustomEvaluator(
    model_runner=my_local_model_runner,
    prompt_template="""
    Task: Evaluate if this response answers the question correctly.
    
    Question: {{ question }}
    Response: {{ response }}
    
    Answer with only: CORRECT or INCORRECT
    """,
    evaluate_func=accuracy_evaluate_func
)
```

### 3. HuggingFace Model with Custom Input/Output Formatting

```python
def my_hf_runner(prompt: str) -> str:
    from transformers import pipeline
    
    # Use any HuggingFace model
    pipe = pipeline("text-generation", model="microsoft/DialoGPT-medium")
    response = pipe(prompt, max_length=100, num_return_sequences=1)
    return response[0]["generated_text"][len(prompt):].strip()

def custom_format_input(instance):
    """Custom input formatting instead of template"""
    return f"System: You are an expert evaluator.\nTask: Rate this text from 1-5 for quality: {instance['text']}\nRating:"

def custom_format_output(response):
    """Extract just the numeric rating"""
    import re
    score_match = re.search(r'(\d+)', response)
    return score_match.group(1) if score_match else "0"

def rating_evaluate_func(data, responses):
    """Calculate rating statistics"""
    scores = [int(r) for r in responses if r.isdigit()]
    return {
        "average_score": sum(scores) / len(scores) if scores else 0,
        "high_scores": sum(1 for s in scores if s >= 4),
        "low_scores": sum(1 for s in scores if s <= 2),
        "scores": scores
    }

evaluator = CustomEvaluator(
    model_runner=my_hf_runner,
    format_input_func=custom_format_input,
    format_output_func=custom_format_output,
    evaluate_func=rating_evaluate_func,
    debug=True
)

data = [{"text": "This is a well-written article about AI."}]
results = evaluator.evaluate(data)
```

### 4. Anthropic Claude Evaluator

```python
import anthropic

def claude_model_runner(prompt: str) -> str:
    client = anthropic.Anthropic()  # Uses ANTHROPIC_API_KEY env var
    
    response = client.messages.create(
        model="claude-3-haiku-20240307",
        max_tokens=100,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.content[0].text

def toxicity_evaluate_func(data, responses):
    """Evaluate toxicity levels"""
    toxic_keywords = ["hate", "violence", "harassment", "abuse"]
    
    toxic_count = 0
    for response in responses:
        if any(keyword in response.lower() for keyword in toxic_keywords):
            toxic_count += 1
    
    return {
        "toxic_count": toxic_count,
        "clean_count": len(responses) - toxic_count,
        "toxicity_rate": toxic_count / len(responses) if responses else 0,
        "is_safe": toxic_count == 0
    }

evaluator = CustomEvaluator(
    model_runner=claude_model_runner,
    prompt_template="""
    Analyze this content for toxicity and harmful language:
    
    Content: {{ text }}
    
    Respond with TOXIC or SAFE and a brief explanation.
    """,
    evaluate_func=toxicity_evaluate_func
)
```

### 5. Complex RAG Evaluation

```python
def rag_evaluation_func(data, responses):
    """Multi-criteria RAG evaluation"""
    results = {
        "relevance_scores": [],
        "faithfulness_scores": [],
        "completeness_scores": []
    }
    
    for i, response in enumerate(responses):
        # Extract scores from LLM response (assumes structured output)
        lines = response.strip().split('\n')
        relevance = next((line for line in lines if "Relevance:" in line), "Relevance: 0")
        faithfulness = next((line for line in lines if "Faithfulness:" in line), "Faithfulness: 0")
        completeness = next((line for line in lines if "Completeness:" in line), "Completeness: 0")
        
        # Extract numeric scores
        import re
        relevance_score = int(re.search(r'(\d+)', relevance).group(1)) if re.search(r'(\d+)', relevance) else 0
        faithfulness_score = int(re.search(r'(\d+)', faithfulness).group(1)) if re.search(r'(\d+)', faithfulness) else 0
        completeness_score = int(re.search(r'(\d+)', completeness).group(1)) if re.search(r'(\d+)', completeness) else 0
        
        results["relevance_scores"].append(relevance_score)
        results["faithfulness_scores"].append(faithfulness_score)
        results["completeness_scores"].append(completeness_score)
    
    # Calculate averages
    results["avg_relevance"] = sum(results["relevance_scores"]) / len(results["relevance_scores"]) if results["relevance_scores"] else 0
    results["avg_faithfulness"] = sum(results["faithfulness_scores"]) / len(results["faithfulness_scores"]) if results["faithfulness_scores"] else 0
    results["avg_completeness"] = sum(results["completeness_scores"]) / len(results["completeness_scores"]) if results["completeness_scores"] else 0
    
    return results

evaluator = CustomEvaluator(
    model_runner=openai_model_runner,  # Reuse from example 1
    prompt_template="""
    Evaluate this RAG response on three criteria (1-5 scale):
    
    Question: {{ question }}
    Context: {{ context }}
    Response: {{ response }}
    
    Rate each aspect:
    
    Relevance: How well does the response address the question?
    Faithfulness: How well does the response stay true to the provided context?
    Completeness: How complete is the response?
    
    Format:
    Relevance: [1-5]
    Faithfulness: [1-5] 
    Completeness: [1-5]
    """,
    evaluate_func=rag_evaluation_func
)

# Example RAG data
rag_data = [
    {
        "question": "What is machine learning?",
        "context": "Machine learning is a subset of AI that enables computers to learn without explicit programming...",
        "response": "Machine learning allows computers to learn from data automatically."
    }
]

results = evaluator.evaluate(rag_data)
print(f"Average relevance: {results['avg_relevance']}")
```

## Advanced Patterns

### Using Custom Validators

```python
from grounded_ai.validators.rag_data import RAGData

def validate_and_format(instance):
    """Validate input data and format for evaluation"""
    # Use existing validators
    rag_item = RAGData(
        question=instance["question"],
        context=instance["context"], 
        response=instance["response"]
    )
    
    # Custom formatting
    return f"Q: {rag_item.question}\nC: {rag_item.context}\nR: {rag_item.response}"

evaluator = CustomEvaluator(
    model_runner=my_model_runner,
    format_input_func=validate_and_format,
    evaluate_func=my_evaluate_func
)
```

### Batch Processing with Metadata

```python
def batch_evaluate_func(data, responses):
    """Process results in batches with metadata tracking"""
    results = {"batch_results": []}
    
    for i, (instance, response) in enumerate(zip(data, responses)):
        item_result = {
            "instance_id": instance.get("id", i),
            "input_length": len(instance.get("text", "")),
            "output_length": len(response),
            "processed_at": instance.get("timestamp"),
            "score": len(response.split())  # Simple word count score
        }
        results["batch_results"].append(item_result)
    
    # Aggregate stats
    results["total_processed"] = len(responses)
    results["avg_input_length"] = sum(r["input_length"] for r in results["batch_results"]) / len(results["batch_results"])
    results["avg_output_length"] = sum(r["output_length"] for r in results["batch_results"]) / len(results["batch_results"])
    
    return results

evaluator = CustomEvaluator(
    model_runner=my_model_runner,
    prompt_template="{{ text }}",
    evaluate_func=batch_evaluate_func,
    metadata={"evaluator_version": "1.0", "batch_size": 100}
)
```

## Key Benefits

### 1. Bring Your Own Model
- OpenAI, Anthropic, Cohere APIs
- Local models via Transformers  
- Custom API endpoints
- Local inference servers
- Any callable that takes a string and returns a string

### 2. Flexible Evaluation Logic
- Custom scoring functions
- Multi-criteria evaluation 
- Complex aggregations
- Domain-specific metrics
- Integration with existing evaluation frameworks

### 3. Minimal Dependencies
- Only requires `jinja2` for templates
- All model-specific dependencies are optional
- Users import only what they need
- No vendor lock-in

## Best Practices

### Error Handling

```python
def robust_model_runner(prompt: str) -> str:
    try:
        # Your model call here
        return model.generate(prompt)
    except Exception as e:
        print(f"Model error: {e}")
        return "ERROR: Model failed"

def robust_evaluate_func(data, responses):
    # Handle errors in responses
    valid_responses = [r for r in responses if not r.startswith("ERROR:")]
    error_count = len(responses) - len(valid_responses)
    
    # Your evaluation logic here
    return {
        "valid_responses": len(valid_responses),
        "errors": error_count,
        "error_rate": error_count / len(responses) if responses else 0
    }
```

### Performance Optimization

```python
# For batch processing, consider concurrent execution
import concurrent.futures

def batch_model_runner(prompts):
    """Process multiple prompts concurrently"""
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        responses = list(executor.map(single_model_runner, prompts))
    return responses

# Or implement batch processing in evaluate_func
def batch_evaluate_func(data, responses):
    # Process in chunks for memory efficiency
    chunk_size = 100
    results = []
    
    for i in range(0, len(responses), chunk_size):
        chunk = responses[i:i+chunk_size]
        chunk_result = process_chunk(chunk)
        results.append(chunk_result)
    
    return {"chunk_results": results}
```

### Debugging Tips

```python
evaluator = CustomEvaluator(
    model_runner=my_model_runner,
    prompt_template="{{ text }}",
    evaluate_func=my_evaluate_func,
    debug=True  # Enable debug output
)

# Debug output shows:
# - Evaluator initialization 
# - Number of instances being processed
# - Template rendering
# - Model calls
# - Evaluation function calls
```