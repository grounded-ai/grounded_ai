import os

from pydantic import BaseModel, Field

from grounded_ai import Evaluator
from grounded_ai.otel import TraceConverter

# Set API Key for demo purposes (User provided)
if "OPENAI_API_KEY" not in os.environ:
    os.environ["OPENAI_API_KEY"] = "YOUR_OPENAI_API_KEY"

# 1. Define Standard OTel GenAI Trace Data
# Utilizing nested 'parts' list as per latest Semantic Conventions
raw_span = {
    "trace_id": "a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4",
    "span_id": "1a2b3c4d1a2b3c4d",
    "name": "chat gpt-4o-mini",
    "start_time": "2026-01-15T10:30:00Z",
    "end_time": "2026-01-15T10:30:02Z",
    "attributes": {
        "gen_ai.system": "openai",
        "gen_ai.request.model": "gpt-4o-mini",
        "gen_ai.usage.input_tokens": 12,
        "gen_ai.usage.output_tokens": 16,
        "gen_ai.input.messages": [
            {
                "role": "system",
                "parts": [{"type": "text", "content": "You are a helpful assistant."}],
            },
            {
                "role": "user",
                "parts": [
                    {"type": "text", "content": "What is the capital of France?"}
                ],
            },
        ],
        "gen_ai.output.messages": [
            {
                "role": "assistant",
                "parts": [
                    {"type": "text", "content": "The capital of France is Paris."}
                ],
            }
        ],
    },
}

# 2. Convert
print(f"Parsing Trace ID: {raw_span['trace_id']}")
# Note: from_otlp handles standard list-of-dicts for messages automatically now
conversation = TraceConverter.from_otlp([raw_span])
span = conversation.spans[0]

# 3. Validation
print(f"Parsed Model: {span.gen_ai_request_model}")
print(f"Input Messages: {len(span.gen_ai_input_messages)}")

# Verify system prompt content
system_msg = span.gen_ai_input_messages[0]
content = system_msg.parts[0].content
print(f"System Prompt: {content}")

if content != "You are a helpful assistant.":
    print("FAILED: Incorrect parsing of message parts")
    exit(1)

# Verify Output
output_msg = span.gen_ai_output_messages[0]
response = output_msg.parts[0].content
print(f"Response: {response}")

if response != "The capital of France is Paris.":
    print("FAILED: Incorrect parsing of output parts")
    exit(1)

print("SUCCESS: Standard OTel GenAI trace parsed correctly.")

# 4. Real Evaluation
# We evaluate if the model's response is factually correct given the user query.


class ResponseCorrectness(BaseModel):
    is_correct: bool = Field(
        description="True if the response accurately answers the user's question."
    )
    explanation: str = Field(description="Reasoning for the score.")


print("\n--- Running Evaluator ---")
evaluator = Evaluator(
    model="openai/gpt-4o",
    system_prompt="You are an expert judge. Evaluate the accuracy of the assistant's response.",
)

# Construct evaluation input from the parsed span
eval_input = (
    f"User Question: {span.gen_ai_input_messages[1].parts[0].content}\n"
    f"Assistant Response: {span.gen_ai_output_messages[0].parts[0].content}"
)

try:
    result = evaluator.evaluate(response=eval_input, output_schema=ResponseCorrectness)
    # Check if result is the schema or an error
    if isinstance(result, ResponseCorrectness):
        print(f"Is Correct: {result.is_correct}")
        print(f"Explanation: {result.explanation}")
    else:
        print(f"Evaluation returned error object: {result}")

except Exception as e:
    print(f"Script execution failed: {e}")
    # print("Ensure OPENAI_API_KEY is set in your environment.")
