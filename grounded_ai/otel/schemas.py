"""
OpenTelemetry GenAI Semantic Convention Models.

Pydantic models for LLM traces following OpenTelemetry GenAI semantic conventions
as used by LogFire and other OTel-compatible platforms.
"""

from datetime import datetime
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, computed_field, model_validator

# === Message Format (GenAI Semantic Convention) ===


class MessagePart(BaseModel):
    """A part within a message (text, tool_call, etc.)."""

    type: Literal["text", "tool_call", "tool_call_response"] = Field(
        ..., description="Type of message part"
    )
    content: Optional[str] = Field(None, description="Text content")

    # For tool calls
    id: Optional[str] = Field(None, description="Tool call ID")
    name: Optional[str] = Field(None, description="Tool/function name")
    arguments: Optional[Dict[str, Any]] = Field(None, description="Tool arguments")
    response: Optional[str] = Field(None, description="Tool response")


class GenAIMessage(BaseModel):
    """
    Message format following OpenTelemetry GenAI semantic conventions.
    Used in gen_ai.input.messages and gen_ai.output.messages.
    """

    role: Literal["system", "user", "assistant", "tool"] = Field(
        ..., description="Message role"
    )
    parts: List[MessagePart] = Field(
        default_factory=list, description="Message parts (text, tool calls, etc.)"
    )


# === Token Usage (gen_ai.usage.*) ===


class TokenUsage(BaseModel):
    """Token usage following gen_ai.usage.* attributes."""

    input_tokens: int = Field(0, description="gen_ai.usage.input_tokens")
    output_tokens: int = Field(0, description="gen_ai.usage.output_tokens")
    total_tokens: Optional[int] = Field(None, description="Total tokens")

    @model_validator(mode="after")
    def compute_total(self):
        """Auto-compute total if not provided."""
        if self.total_tokens is None:
            self.total_tokens = self.input_tokens + self.output_tokens
        return self


# === GenAI Span (The core LLM trace) ===


class GenAISpan(BaseModel):
    """
    A single LLM invocation following OpenTelemetry GenAI semantic conventions.
    This is what Blue Guardrails and other OTel platforms expect.
    """

    # Identity (OpenTelemetry core)
    trace_id: str = Field(..., description="trace_id")
    span_id: str = Field(..., description="span_id")
    parent_span_id: Optional[str] = Field(None, description="Parent span ID")

    # Span metadata
    name: str = Field(..., description="Span name (e.g., 'chat gpt-4')")
    kind: Literal["CLIENT", "INTERNAL", "SERVER"] = Field(
        "CLIENT", description="Span kind (use CLIENT for LLM calls)"
    )

    # Timing
    start_time: datetime = Field(..., description="Span start time")
    end_time: datetime = Field(..., description="Span end time")

    # Status
    status: Literal["UNSET", "OK", "ERROR"] = Field("UNSET", description="Span status")

    # GenAI Semantic Convention Attributes (gen_ai.*)
    gen_ai_system: str = Field(
        ..., alias="gen_ai.system", description="Provider (openai, anthropic, etc.)"
    )
    gen_ai_request_model: str = Field(
        ..., alias="gen_ai.request.model", description="Model requested"
    )
    gen_ai_response_model: Optional[str] = Field(
        None, alias="gen_ai.response.model", description="Model used"
    )
    gen_ai_response_id: Optional[str] = Field(
        None, alias="gen_ai.response.id", description="Provider response ID"
    )

    # Messages (THE KEY ATTRIBUTES)
    gen_ai_input_messages: List[GenAIMessage] = Field(
        default_factory=list,
        alias="gen_ai.input.messages",
        description="Input messages",
    )
    gen_ai_output_messages: List[GenAIMessage] = Field(
        default_factory=list,
        alias="gen_ai.output.messages",
        description="Output messages",
    )

    # Usage
    usage: TokenUsage = Field(default_factory=TokenUsage, description="Token usage")

    # Optional: Request parameters
    gen_ai_request_temperature: Optional[float] = Field(
        None, alias="gen_ai.request.temperature"
    )
    gen_ai_request_max_tokens: Optional[int] = Field(
        None, alias="gen_ai.request.max_tokens"
    )
    gen_ai_response_finish_reasons: Optional[List[str]] = Field(
        None, alias="gen_ai.response.finish_reasons"
    )

    # Optional: Conversation grouping
    gen_ai_conversation_id: Optional[str] = Field(None, alias="gen_ai.conversation.id")

    # Additional attributes (catch-all for platform-specific data)
    attributes: Dict[str, Any] = Field(
        default_factory=dict, description="Additional span attributes"
    )

    class Config:
        populate_by_name = True

    @computed_field
    @property
    def latency_ms(self) -> float:
        """Span duration in milliseconds."""
        return (self.end_time - self.start_time).total_seconds() * 1000


# === Conversation (Multiple spans grouped) ===


class GenAIConversation(BaseModel):
    """
    A conversation is a collection of GenAI spans grouped together.
    Blue Guardrails groups spans by gen_ai.conversation.id or trace_id.
    """

    # Identity
    conversation_id: str = Field(..., description="Conversation/trace ID")

    # Spans in chronological order
    spans: List[GenAISpan] = Field(default_factory=list, description="LLM spans")

    # Metadata
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Conversation metadata"
    )

    @model_validator(mode="after")
    def ensure_chronological_order(self):
        """Ensure spans are sorted chronologically."""
        self.spans = sorted(self.spans, key=lambda s: s.start_time)
        return self

    @computed_field
    @property
    def total_tokens(self) -> TokenUsage:
        """Total tokens across all spans."""
        usage = TokenUsage()
        for span in self.spans:
            usage.input_tokens += span.usage.input_tokens
            usage.output_tokens += span.usage.output_tokens
        usage.compute_total()
        return usage

    @computed_field
    @property
    def total_latency_ms(self) -> float:
        """Total conversation duration."""
        if not self.spans:
            return 0.0
        start = min(s.start_time for s in self.spans)
        end = max(s.end_time for s in self.spans)
        return (end - start).total_seconds() * 1000

    def get_all_messages(self) -> List[GenAIMessage]:
        """Get all messages in chronological order."""
        messages = []
        for span in self.spans:
            messages.extend(span.gen_ai_input_messages)
            messages.extend(span.gen_ai_output_messages)
        return messages

    def get_reasoning_chain(self) -> List[str]:
        """
        Extract the chronological chain of LLM reasoning.

        Returns a list of assistant messages showing the agent's thought process.
        Useful for analyzing decision-making logic and debugging unexpected behavior.

        Example:
            ["I need to search for weather data",
             "Based on the results, Paris is 18°C",
             "The capital of France is Paris and it's currently 18°C"]
        """
        reasoning = []
        for span in self.spans:  # Already sorted chronologically
            # Extract assistant messages from output
            for message in span.gen_ai_output_messages:
                if message.role == "assistant":
                    # Get text content from parts
                    for part in message.parts:
                        if part.type == "text" and part.content:
                            reasoning.append(part.content)
        return reasoning

    def get_full_conversation(self) -> List[Dict[str, str]]:
        """
        Get the complete conversation in chronological order.

        Returns all messages (system, user, assistant, tool) as simple dicts.
        Useful for replaying the entire conversation or feeding to another LLM.

        Example:
            [{"role": "system", "content": "You are helpful"},
             {"role": "user", "content": "What's the weather?"},
             {"role": "assistant", "content": "Let me check..."},
             {"role": "tool", "content": "{temp: 18}"},
             {"role": "assistant", "content": "It's 18°C"}]
        """
        conversation = []
        for span in self.spans:
            # Input messages
            for message in span.gen_ai_input_messages:
                for part in message.parts:
                    if part.type == "text" and part.content:
                        conversation.append(
                            {"role": message.role, "content": part.content}
                        )
            # Output messages
            for message in span.gen_ai_output_messages:
                for part in message.parts:
                    if part.type == "text" and part.content:
                        conversation.append(
                            {"role": message.role, "content": part.content}
                        )
        return conversation

    def get_tool_usage_summary(self) -> List[Dict[str, Any]]:
        """
        Extract all tool calls and their results chronologically.

        Useful for evaluating whether the agent used tools correctly.

        Example:
            [{"tool": "get_weather",
              "arguments": {"location": "Paris"},
              "result": "18°C sunny",
              "span_id": "abc123"}]
        """
        tool_calls = []
        for span in self.spans:
            # Find tool calls in output messages
            for message in span.gen_ai_output_messages:
                if message.role == "assistant":
                    for part in message.parts:
                        if part.type == "tool_call":
                            tool_calls.append(
                                {
                                    "tool": part.name,
                                    "arguments": part.arguments,
                                    "call_id": part.id,
                                    "span_id": span.span_id,
                                }
                            )

            # Find tool responses in input messages
            for message in span.gen_ai_input_messages:
                if message.role == "tool":
                    for part in message.parts:
                        if part.type == "tool_call_response":
                            # Match with the call
                            for tc in tool_calls:
                                if tc.get("call_id") == part.id:
                                    tc["result"] = part.response

        return tool_calls
