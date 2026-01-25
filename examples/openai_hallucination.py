import os
import sys
from grounded_ai import Evaluator

def main():
    # Verify API Key
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY environment variable not set.")
        sys.exit(1)

    print("Initializing Grounded AI Evaluator (OpenAI Backend)...")
    
    # Initialize with a specific OpenAI model
    # Note: 'output_schema' defaults to standard EvaluationOutput if not provided
    evaluator = Evaluator(model="openai/gpt-4o")

    context = """
    Grounded AI is a new Python library for reliable LLM evaluation.
    It supports multiple backends including OpenAI, Anthropic, and local SLMs.
    It uses Pydantic for type-safe structured outputs.
    """

    # Test Case 1: Faithful
    query = "What libraries does Grounded AI support?"
    response_faithful = "It supports OpenAI, Anthropic, and local SLMs."
    
    print(f"\n--- Evaluation 1 (Faithful) ---")
    result = evaluator.evaluate(
        query=query,
        context=context,
        text=response_faithful
    )
    print(f"Label: {result.label} | Score: {result.score}")
    print(f"Reasoning: {result.reasoning}")

    # Test Case 2: Hallucination
    response_hallucinated = "It supports Google Gemini and Llama 3 natively."
    
    print(f"\n--- Evaluation 2 (Hallucinated) ---")
    result = evaluator.evaluate(
        query=query,
        context=context,
        text=response_hallucinated
    )
    print(f"Label: {result.label} | Score: {result.score}")
    print(f"Reasoning: {result.reasoning}")

if __name__ == "__main__":
    main()
