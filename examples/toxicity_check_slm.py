from grounded_ai import Evaluator

def main():
    print("Initializing Grounded AI Toxicity Evaluator (SLM)...")
    evaluator = Evaluator(
        model="grounded-ai/phi-4-mini-toxicity", 
        eval_mode="TOXICITY",
        device="cpu"
    )

    # 1. Clean Text
    text_clean = "I really enjoyed the workshop today. The team was super helpful!"
    print(f"\n--- Test Case 1: Clean Text ---")
    print(f"Input: '{text_clean}'")
    
    result = evaluator.evaluate(text=text_clean)
    
    print(f"Label: {result.label}")
    print(f"Score: {result.score} (0=safe, 1=toxic)")

    # 2. Toxic Text
    text_toxic = "You are completely useless and nobody likes you."
    print(f"\n--- Test Case 2: Toxic Text ---")
    print(f"Input: '{text_toxic}'")
    
    result = evaluator.evaluate(text=text_toxic)
    
    print(f"Label: {result.label}")
    print(f"Score: {result.score}")
    if result.reasoning:
        print(f"Reasoning: {result.reasoning}")

if __name__ == "__main__":
    main()
