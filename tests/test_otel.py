
"""
Unit tests for OTel GenAI Schema and Converter.
"""
import pytest
from datetime import datetime, timezone
import json

from grounded_ai.otel import (
    TraceConverter,
    GenAISpan,
    GenAIConversation,
    GenAIMessage,
    MessagePart,
    TokenUsage,
)

# === Mock Data ===

MOCK_OTLP_SPAN = {
    "trace_id": "trace-123",
    "span_id": "span-123",
    "name": "chat gpt-4",
    "start_time": "2024-01-01T10:00:00Z",
    "end_time": "2024-01-01T10:00:01Z",
    "attributes": {
        "gen_ai.system": "openai",
        "gen_ai.request.model": "gpt-4",
        "gen_ai.usage.input_tokens": 10,
        "gen_ai.usage.output_tokens": 20,
        "gen_ai.input.messages": json.dumps([{"role": "user", "content": "Hello"}]),
        "gen_ai.output.messages": json.dumps([{"role": "assistant", "content": "Hi there"}]),
    }
}

MOCK_PHOENIX_SPAN = {
    "trace_id": "trace-456",
    "span_id": "span-456",
    "name": "LLM",
    "attributes": {
        "openinference.span.kind": "LLM",
        "llm.model_name": "gpt-4",
        "llm.token_count.prompt": 50,
        "llm.token_count.completion": 60,
        "llm.input_messages.0.message.role": "user",
        "llm.input_messages.0.message.content": "Question?",
        "llm.output_messages.0.message.content": "Answer.",
    },
    "start_time": "2024-01-01T10:00:00Z",
    "end_time": "2024-01-01T10:00:02Z",
}

MOCK_LANGSMITH_RUN = {
    "id": "run-789",
    "run_type": "chain",
    "start_time": datetime.now(timezone.utc),
    "end_time": datetime.now(timezone.utc),
    "child_runs": [
        {
            "id": "run-llm-1",
            "run_type": "llm",
            "start_time": datetime.now(timezone.utc),
            "end_time": datetime.now(timezone.utc),
            "inputs": {"messages": [{"type": "user", "content": "Hello"}]},
            "outputs": {"generations": [[{"text": "Hi"}]]},
            "extra": {"token_usage": {"prompt_tokens": 5, "completion_tokens": 5}}
        }
    ]
}

# === Tests ===

def test_gen_ai_schema_validation():
    """Test standard valid span creation."""
    span = GenAISpan(
        trace_id="t1",
        span_id="s1",
        name="test",
        start_time=datetime.now(timezone.utc),
        end_time=datetime.now(timezone.utc),
        gen_ai_system="openai",
        gen_ai_request_model="gpt-4",
    )
    assert span.gen_ai_system == "openai"
    assert span.kind == "CLIENT"

def test_trace_converter_otlp():
    """Test OTLP conversion with JSON messages."""
    conversation = TraceConverter.from_otlp([MOCK_OTLP_SPAN])
    assert len(conversation.spans) == 1
    span = conversation.spans[0]
    
    assert span.trace_id == "trace-123"
    assert span.gen_ai_system == "openai"
    assert len(span.gen_ai_input_messages) == 1
    assert span.gen_ai_input_messages[0].parts[0].content == "Hello"
    assert span.usage.total_tokens == 30 
    assert span.latency_ms == 1000.0

def test_trace_converter_phoenix():
    """Test Phoenix conversion with flattened attributes."""
    conversation = TraceConverter.from_phoenix({"spans": [MOCK_PHOENIX_SPAN], "trace_id": "trace-456"})
    assert len(conversation.spans) == 1
    span = conversation.spans[0]
    
    assert span.trace_id == "trace-456"
    assert span.gen_ai_request_model == "gpt-4"
    # Check robust flattened message parsing
    assert span.gen_ai_input_messages[0].role == "user"
    assert span.gen_ai_input_messages[0].parts[0].content == "Question?"
    # Check default role for output
    assert span.gen_ai_output_messages[0].role == "assistant"
    assert span.gen_ai_output_messages[0].parts[0].content == "Answer."

def test_trace_converter_langsmith():
    """Test LangSmith conversion."""
    conversation = TraceConverter.from_langsmith(MOCK_LANGSMITH_RUN)
    # Recursively finds the LLM run
    assert len(conversation.spans) == 1
    span = conversation.spans[0]
    
    assert span.span_id == "run-llm-1"
    assert span.gen_ai_input_messages[0].role == "user"
    assert span.gen_ai_output_messages[0].parts[0].content == "Hi"

def test_conversation_helpers():
    """Test helper methods like get_full_conversation."""
    conversation = TraceConverter.from_otlp([MOCK_OTLP_SPAN])
    
    # get_full_conversation -> List[Dict]
    full_history = conversation.get_full_conversation()
    assert len(full_history) == 2
    assert full_history[0] == {"role": "user", "content": "Hello"}
    assert full_history[1] == {"role": "assistant", "content": "Hi there"}
    
    # get_reasoning_chain
    reasoning = conversation.get_reasoning_chain()
    assert reasoning == ["Hi there"]

def test_trace_converter_otlp_list():
    """Test OTLP conversion with Python list messages (non-serialized)."""
    mock_span = MOCK_OTLP_SPAN.copy()
    mock_span["attributes"] = MOCK_OTLP_SPAN["attributes"].copy()
    mock_span["attributes"]["gen_ai.input.messages"] = [{"role": "user", "content": "List format"}]
    
    conversation = TraceConverter.from_otlp([mock_span])
    span = conversation.spans[0]
    assert span.gen_ai_input_messages[0].parts[0].content == "List format"

def test_empty_input():
    """Test error handling."""
    with pytest.raises(ValueError):
        TraceConverter.from_otlp([])
