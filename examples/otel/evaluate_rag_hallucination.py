
import os
from pydantic import BaseModel, Field
from grounded_ai import Evaluator
from grounded_ai.otel import TraceConverter

# 1. Define Standard OTel GenAI Span
# Modified to demonstrate a clear RAG failure (Hallucination)
# Using 'gen_ai.system' and message lists as per semantic conventions.
raw_span = {
    "name": "llm",
    "trace_id": "0x6c80880dbeb609e2ed41e06a6397a0dd",
    "span_id": "0xd9bdedf0df0b7208",
    "start_time": "2024-05-08T21:46:11.480777Z",
    "end_time": "2024-05-08T21:46:35.368042Z",
    "status": {
        "code": 1  # OK
    },
    "attributes": {
        "gen_ai.system": "openai",
        "gen_ai.request.model": "gpt-4-turbo-preview",
        "gen_ai.request.temperature": 0.1,
        
        "gen_ai.input.messages": [
            {
                "role": "system",
                "content": r"""
The following is a friendly conversation between a user and an AI assistant.
The assistant is talkative and provides lots of specific details from its context.
If the assistant does not know the answer to a question, it truthfully says it
does not know.

Here are the relevant documents for the context:

page_label: 7
file_path: /Users/mikeldking/work/openinference/python/examples/llama-index-new/backend/data/101.pdf

Domestic Mail Manual • Updated 7-9-23101
101.6.4Retail Mail: Physical Standards for Letters, Cards, Flats, and Parcels
a. No piece may weigh more than 70 pounds.
b. The combined length and girth of a piece (the length of its longest side plus 
the distance around its thickest part) may not exceed 108 inches.
c. Lower size or weight standards apply to mail addressed to certain APOs and 
FPOs, subject to 703.2.0  and 703.4.0  and for Department of State mail, 
subject to 703.3.0 .

page_label: 4
file_path: /Users/mikeldking/work/openinference/python/examples/llama-index-new/backend/data/101.pdf

Domestic Mail Manual • Updated 7-9-23101
6.0 Additional Physical Standards for First-Class Mail and 
USPS Ground Advantage — Retail
[7-9-23]
6.1 Maximum Weight
6.1.1   First-Class Mail
First-Class Mail (letters and flats) must not exceed 13 ounces. 
6.1.2   USPS Ground Advantage — Retail
USPS Ground Advantage — Retail mail must not exceed 70 pounds.

Instruction: Based on the above documents, provide a detailed answer for the user question below.
Answer "don't know" if not present in the document.
"""
            },
            {
                "role": "user",
                "content": "What is the maximum weight for a First-Class Mail letter?"
            }
        ],
        
        "gen_ai.output.messages": [
            {
                "role": "assistant",
                "content": "Based on the documents provided, a First-Class Mail letter must not exceed 70 pounds."
            }
        ]
    }
}

# 2. Convert to Agent Trace (GenAIConversation)
# We wrap the single span in a list for OTLP conversion
print("Converting Trace...")
conversation = TraceConverter.from_otlp([raw_span])
span = conversation.spans[0]

print(f"Trace ID: {conversation.conversation_id}")
print(f"Span ID: {span.span_id}")
print(f"System: {span.gen_ai_system}")

# 3. Define Metric Schema
class CorrectnessEval(BaseModel):
    is_faithful: bool = Field(description="True if the answer is supported by the provided context.")
    is_correct: bool = Field(description="True if the answer is factually correct based on the context.")
    reasoning: str = Field(description="Explanation of discrepancies.")

# Set API Key (Placeholder - User must provide)
if "OPENAI_API_KEY" not in os.environ:
    os.environ["OPENAI_API_KEY"] = "YOUR_OPENAI_API_KEY"

# 4. Initialize Evaluator
evaluator = Evaluator(
    model="openai/gpt-4o-mini",
    system_prompt="You are a RAG evaluator. Verify if the Assistant's response is supported by the System Prompt context."
)

# 5. Evaluate
# Use the helper to get simple dicts of the full conversation history
full_history = conversation.get_full_conversation()

# Extract context and query helper
system_context = next((m["content"] for m in full_history if m["role"] == "system"), "No context found")
user_query = next((m["content"] for m in full_history if m["role"] == "user"), "No query found")
assistant_answer = next((m["content"] for m in full_history if m["role"] == "assistant"), "No answer found")

eval_payload = (
    f"Retrieved Context:\n{system_context[:2000]}...\n\n" # Truncate large context for display/eval prompt safety
    f"User Question: {user_query}\n\n"
    f"Model Answer: {assistant_answer}"
)

print("\nEvaluating RAG Correctness...")
try:
    result = evaluator.evaluate(
        response=eval_payload,
        output_schema=CorrectnessEval
    )
    print(f"Faithful: {result.is_faithful}")
    print(f"Reasoning: {result.reasoning}")
except Exception as e:
    print(f"Evaluation failed (missing API key?): {e}")
