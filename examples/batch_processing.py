import concurrent.futures
import time
from grounded_ai import Evaluator

def evaluate_item(evaluator, item):
    """
    Helper function to evaluate a single item.
    """
    try:
        result = evaluator.evaluate(
            text=item["response"],
            context=item["context"],
            query=item["query"]
        )
        return {
            "id": item["id"],
            "score": result.score,
            "label": result.label,
            "error": None
        }
    except Exception as e:
        return {
            "id": item["id"],
            "score": None,
            "label": None,
            "error": str(e)
        }

def main():
    # Use SLM for this example as it's free/local, but code applies to APIs too
    print("Loading model...")
    evaluator = Evaluator(
        model="grounded-ai/phi-4-mini-hallucination", 
        eval_mode="HALLUCINATION",
        device="cpu"
    )
    
    # Create a batch of synthetic data
    batch_data = [
        {
            "id": 1, 
            "query": "What color is the sky?", 
            "context": "The sky is blue during the day.",
            "response": "The sky is blue."
        },
        {
            "id": 2, 
            "query": "What is the capital of France?", 
            "context": "Paris is the capital of France.",
            "response": "The capital is London." # Hallucination
        },
        {
            "id": 3, 
            "query": "Who wrote Hamlet?", 
            "context": "Hamlet was written by William Shakespeare.",
            "response": "Shakespeare wrote it."
        },
        # ... could be 100s of items
    ]
    
    print(f"Processing batch of {len(batch_data)} items...")
    results = []
    
    start_time = time.time()
    
    # For SLMs (local GPU/CPU), simple iteration is often best to avoid OOM or contention.
    # For APIs (OpenAI/Anthropic), use parallel threads.
    # Here we show simple iteration for safety with the SLM example.
    
    for item in batch_data:
        print(f"Evaluating item {item['id']}...", end="\r")
        res = evaluate_item(evaluator, item)
        results.append(res)
        
    duration = time.time() - start_time
    print(f"\nProcessing complete in {duration:.2f} seconds.")
    
    # Verification
    for res in results:
        status = "✅" if res["error"] is None else "❌"
        print(f"ID {res['id']}: {status} {res['label']} (Score: {res['score']})")

if __name__ == "__main__":
    main()
