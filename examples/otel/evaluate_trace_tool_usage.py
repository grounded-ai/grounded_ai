
import os
from pydantic import BaseModel, Field
from grounded_ai import Evaluator
from grounded_ai.otel import TraceConverter

# 1. Simulate a Trace with Tool Usage
# Scenario: User asks for weather, Assistant calls tool.
trace_data = [
    {
        "trace_id": "trace-tool-demo",
        "span_id": "span-1",
        "name": "chat",
        "start_time": "2024-01-01T12:00:00Z",
        "end_time": "2024-01-01T12:00:01Z",
        "attributes": {
            "gen_ai.system": "openai",
            "gen_ai.request.model": "gpt-4-turbo",
            "gen_ai.input.messages": [
                {"role": "user", "content": "What's the weather like in New York?"}
            ],
            "gen_ai.output.messages": [
                {
                    "role": "assistant",
                    # Standard OTel 'parts' for Tool Calls
                    "parts": [
                        {
                            "type": "tool_call",
                            "id": "call_123",
                            "name": "get_weather",
                            "arguments": '{"location": "New York, NY", "unit": "celsius"}'
                        }
                    ]
                }
            ]
        }
    }
]

# 2. Convert Trace
print("Parsing Trace...")
conversation = TraceConverter.from_otlp(trace_data)
span = conversation.spans[0]

# 3. Define Evaluation Schema
class ToolUsageCheck(BaseModel):
    is_valid_tool: bool = Field(description="True if the tool call matches the user request.")
    missing_params: list[str] = Field(description="List of required parameters that are missing.")
    argument_quality: str = Field(description="Assessment of the argument values.")

# 4. Initialize Evaluator
if "OPENAI_API_KEY" not in os.environ:
    os.environ["OPENAI_API_KEY"] = "YOUR_OPENAI_API_KEY"

print("\nRunning Evaluation...")
try:
    evaluator = Evaluator("openai/gpt-4o")
    
    # Extract the tool call part
    tool_call = span.gen_ai_output_messages[0].parts[0]
    user_query = span.gen_ai_input_messages[0].parts[0].content
    
    eval_input = (
        f"User Query: {user_query}\n"
        f"Tool Called: {tool_call.name}\n"
        f"Arguments: {tool_call.arguments}"
    )
    
    result = evaluator.evaluate(
        response=eval_input,
        output_schema=ToolUsageCheck,
        system_prompt="You are a QA engineer. Verify the tool usage against the user query."
    )
    
    print(f"Valid Tool: {result.is_valid_tool}")
    print(f"Quality: {result.argument_quality}")

except Exception as e:
    print(f"Skipped execution: {e}")
