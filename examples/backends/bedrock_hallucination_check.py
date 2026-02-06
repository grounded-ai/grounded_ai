from typing import List, Optional

from pydantic import BaseModel, Field

from grounded_ai import EvaluationError, EvaluationInput, Evaluator

# --- 1. Define Custom Evaluation Schema ---
# The power of the library is defining exactly HOW you want to judge the output.
# Here we define a strict schema for checking Hallucinations/Groundedness.


class Citation(BaseModel):
    statement: str = Field(description="The specific statement made in the response")
    is_supported: bool = Field(
        description="True if the statement is supported by the context"
    )
    evidence: Optional[str] = Field(
        description="Quote from context supporting the statement, if applicable"
    )


class GroundednessEvaluation(BaseModel):
    score: float = Field(
        description="0.0 to 1.0 score where 1.0 is fully grounded in context"
    )
    reasoning: str = Field(description="High level explanation of the score")
    citations: List[Citation] = Field(
        description="Line-by-line analysis of the response"
    )


def main():
    print("--- Bedrock Backend: Hallucination/Groundedness Check ---\n")

    # 2. Initialize Evaluator with an AWS Bedrock model
    # Requires AWS credentials (AWS_PROFILE or env vars)
    try:
        evaluator = Evaluator(
            model="bedrock/us.anthropic.claude-sonnet-4-5-20250929-v1:0",
            region_name="us-east-1",
        )
    except ImportError as e:
        print(f"Skipping: {e}")
        return
    except Exception as e:
        print(f"Initialization Failed (Check AWS Creds): {e}")
        return

    # 3. Setup the RAG Scenario
    # We want to check if the 'response' is actually supported by the 'context'.

    rag_context = """
    Product: SuperBattery 3000
    Specs:
    - Battery Life: 24 hours on standard load.
    - Charging: Fast charge 0-80% in 15 minutes.
    - Weight: 150 grams.
    - Warranty: 2 years limited warranty.
    - Colors: Black, Silver, and Midnight Blue.
    """

    user_query = "Tell me about the battery life and warranty of the SuperBattery."

    # A response that contains a hallucination ("Red" color is not in context, "5 year" warranty is wrong)
    ai_response = "The SuperBattery 3000 lasts for 24 hours. It comes with a 5-year warranty and is available in Red."

    print(f"Context:\n{rag_context.strip()}")
    print(f"\nResponse to Evaluate:\n{ai_response}\n")

    # 4. Run Evaluation
    # We pass the input as an EvaluationInput object which formats the prompt automatically using Jinja2
    input_payload = EvaluationInput(
        query=user_query, context=rag_context, response=ai_response
    )

    print("Running evaluation (LLM-as-a-Judge)...")

    try:
        # The backend will force the LLM to output exactly matching our GroundednessEvaluation schema
        result = evaluator.evaluate(
            input_data=input_payload,
            output_schema=GroundednessEvaluation,
            system_prompt="You are a strict groundedness evaluator. Verify every claim against the context.",
        )

        if isinstance(result, EvaluationError):
            print(f"\nEvaluation failed with error: {result.error_code}")
            print(f"Message: {result.message}")
            if result.details:
                print(f"Details: {result.details}")
            return

        # 5. Review Results
        print("\n--- Evaluation Result ---")
        print(f"Score: {result.score}/1.0")
        print(f"Reasoning: {result.reasoning}\n")

        print("Citations Analysis:")
        for citation in result.citations:
            status = "✅" if citation.is_supported else "❌"
            print(f' {status} Statement: "{citation.statement}"')
            if citation.is_supported:
                print(f'    Evidence: "{citation.evidence}"')
            else:
                print("    Issue: Not supported by context.")

    except Exception as e:
        print(f"\nEvaluation failed (Is AWS Configured?): {e}")


if __name__ == "__main__":
    main()
