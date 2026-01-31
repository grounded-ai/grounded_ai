import os

from pydantic import BaseModel

from grounded_ai import Evaluator
from grounded_ai.otel import TraceConverter

# 1. Simulate a Multi-Turn Agent Trace (2 Spans)
# This represents a "Chain of Thought" or "ReAct" loop.
# Span 1: The agent decides to call a tool.
# Span 2: The agent receives the tool result and comments on it.

trace_data = [
    # --- Turn 1: Thoughts & Action ---
    {
        "trace_id": "trace-agent-multi-turn",
        "span_id": "span-1",
        "name": "agent_step_1",
        "start_time": "2024-01-01T10:00:00Z",
        "end_time": "2024-01-01T10:00:02Z",
        "attributes": {
            "gen_ai.system": "openai",
            "gen_ai.request.model": "gpt-4",
            # Input: User query
            "gen_ai.input.messages": [
                {"role": "user", "content": "What is the weather in Tokyo?"}
            ],
            # Output: Tool Call
            "gen_ai.output.messages": [
                {
                    "role": "assistant",
                    "parts": [
                        {"type": "text", "content": "I should checks the weather api."},
                        {
                            "type": "tool_call",
                            "name": "get_weather",
                            "arguments": '{"city": "Tokyo"}',
                        },
                    ],
                }
            ],
        },
    },
    # --- Turn 2: Observation & Response ---
    {
        "trace_id": "trace-agent-multi-turn",
        "span_id": "span-2",
        "name": "agent_step_2",
        "start_time": "2024-01-01T10:00:05Z",
        "end_time": "2024-01-01T10:00:07Z",
        "attributes": {
            "gen_ai.system": "openai",
            "gen_ai.request.model": "gpt-4",
            # Input: Previous history + Tool Result (Observation)
            # Note: In some systems, full history is logged. Ideally we just look at the new IO.
            # Here we simulate the LLM call input which includes history.
            "gen_ai.input.messages": [
                {"role": "user", "content": "What is the weather in Tokyo?"},
                {"role": "assistant", "content": "I should checks the weather api."},
                {"role": "tool", "content": "15 degrees Celsius, Sunny"},
            ],
            # Output: Final Answer
            "gen_ai.output.messages": [
                {
                    "role": "assistant",
                    "content": "The weather in Tokyo is 15 degrees Celsius and Sunny.",
                }
            ],
        },
    },
]

# 2. Convert Trace
print("Parsing Multi-Turn Trace...")
conversation = TraceConverter.from_otlp(trace_data)
print(f"Trace ID: {conversation.conversation_id}")
print(f"Spans Found: {len(conversation.spans)}")

# 3. Extract Reasoning Chain
# We want to see the Agent's thought process across spans.
print("\n--- Agent Reasoning Chain ---")
reasoning = conversation.get_reasoning_chain()
for i, step in enumerate(reasoning):
    print(f"Step {i + 1}: {step}")

# 4. Evaluate Final Response faithfulness to Tools
# We will use the output of the *last* span and the *tool parameters* from the first span.
if "OPENAI_API_KEY" not in os.environ:
    if "OPENAI_API_KEY" not in os.environ:
        os.environ["OPENAI_API_KEY"] = "YOUR_OPENAI_API_KEY"


class AgentEvaluation(BaseModel):
    plan_followed: bool
    final_answer_correct: bool


print("\n--- Evaluating Agent Performance ---")
try:
    evaluator = Evaluator("openai/gpt-4o")

    # Construct a view of the trace for the judge
    trace_summary = (
        f"User: What is the weather in Tokyo?\n"
        f"Agent Step 1: {reasoning[0]} (Call get_weather)\n"
        f"Observation: 15 degrees Celsius, Sunny\n"
        f"Agent Final: {reasoning[1]}"
    )

    result = evaluator.evaluate(
        response=trace_summary,
        output_schema=AgentEvaluation,
        system_prompt="Evaluate if the agent followed a logical plan and used the observation correctly.",
    )

    print(f"Plan Followed: {result.plan_followed}")
    print(f"Correct: {result.final_answer_correct}")

except Exception as e:
    print(f"Evaluation skipped: {e}")
