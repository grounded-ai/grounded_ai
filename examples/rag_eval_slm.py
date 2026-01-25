from grounded_ai import Evaluator

def main():
    print("Initializing Grounded AI RAG Evaluator (SLM)...")
    # We explicitly set eval_mode for the SLM backend
    evaluator = Evaluator(
        model="grounded-ai/phi-4-mini-rag", 
        eval_mode="RAG_RELEVANCE",
        device="cpu" # Using CPU for broad compatibility in this example
    )

    query = "What are the benefits of vitamin D?"
    document_text = "Vitamin D is essential for strong bones because it helps the body use calcium from the diet."
    
    # 1. Relevant Case
    print(f"\n--- Test Case 1: Relevant Document ---")
    print(f"Query: {query}")
    print(f"Document: {document_text}")
    
    # Note: 'text' field carries the document content for RAG, 'query' carries the question
    result = evaluator.evaluate(text=document_text, query=query)
    
    print(f"Result: {result.label.upper()}")
    print(f"Score: {result.score}")
    print(f"Confidence: {result.confidence}")
    if result.reasoning:
        print(f"Reasoning: {result.reasoning}")

    # 2. Irrelevant Case
    irrelevant_text = "The Eiffel Tower is located in Paris, France and was constructed in 1889."
    
    print(f"\n--- Test Case 2: Irrelevant Document ---")
    print(f"Query: {query}")
    print(f"Document: {irrelevant_text}")
    
    result = evaluator.evaluate(text=irrelevant_text, query=query)
    
    print(f"Result: {result.label.upper()}")
    print(f"Score: {result.score}")

if __name__ == "__main__":
    main()
