import os
import sys
from grounded_ai import Evaluator

def main():
    # Verify API Key
    if not os.getenv("ANTHROPIC_API_KEY"):
        print("Error: ANTHROPIC_API_KEY environment variable not set.")
        sys.exit(1)

    print("Initializing Grounded AI Evaluator (Anthropic Backend)...")
    
    # Initialize with a specific Anthropic model
    # Uses the 'anthropic/' prefix to trigger auto-detection
    evaluator = Evaluator(model="anthropic/claude-3-5-sonnet-20241022")

    context = """
    The Apollo 11 mission landed the first humans on the Moon.
    Neil Armstrong and Buzz Aldrin walked on the lunar surface.
    Michael Collins remained in orbit in the Command Module.
    """

    # Test Case 1: Accurate
    query = "Who stayed in orbit?"
    response_accurate = "Michael Collins remained in orbit."
    
    print(f"\n--- Evaluation 1 (Accurate) ---")
    result = evaluator.evaluate(
        query=query,
        context=context,
        text=response_accurate
    )
    # Note: Anthropic backend uses 'structured outputs' beta for this guaranteed schema
    print(f"Label: {result.label} | Score: {result.score}")
    print(f"Reasoning: {result.reasoning}")

    # Test Case 2: Inaccurate
    response_inaccurate = "Buzz Aldrin stayed in the orbiter while Neil went down alone."
    
    print(f"\n--- Evaluation 2 (Inaccurate) ---")
    result = evaluator.evaluate(
        query=query,
        context=context,
        text=response_inaccurate
    )
    print(f"Label: {result.label} | Score: {result.score}")
    print(f"Reasoning: {result.reasoning}")

if __name__ == "__main__":
    main()
