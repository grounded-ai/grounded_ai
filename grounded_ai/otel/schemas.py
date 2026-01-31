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

    def get_full_conversation(self) -> List[Dict[str, Any]]:
        """
        Get the complete conversation history including tool calls and outputs.
        Deduplicates messages based on content to handle overlapping span histories.

        Returns:
            List of message dicts (OpenAI/Anthropic compatible format).
        """
        messages = []
        seen = set()

        for msg in self.get_all_messages():
            content_parts = []
            tool_calls = []
            tool_call_id = None

            for p in msg.parts:
                if p.type == "text" and p.content:
                    content_parts.append(p.content)
                elif p.type == "tool_call_response":
                    content_parts.append(p.response)
                    # Link to the call if ID present
                    if p.id:
                        tool_call_id = p.id
                elif p.type == "tool_call":
                    tool_calls.append(
                        {
                            "id": p.id,
                            "type": "function",
                            "function": {"name": p.name, "arguments": p.arguments},
                        }
                    )

            # Build the dict
            msg_dict = {
                "role": msg.role,
                "content": "\n".join(filter(None, content_parts))
                if content_parts
                else None,
            }

            if tool_calls:
                msg_dict["tool_calls"] = tool_calls
            if tool_call_id:
                msg_dict["tool_call_id"] = tool_call_id

            # Deduplication Signature
            # We use a tuple of stable representation of key fields
            # (role, content, num_tool_calls, first_tool_id)
            # This avoids adding the exact same message twice if spans overlap context.
            sig = (
                msg.role,
                msg_dict["content"],
                len(tool_calls),
                tool_calls[0]["id"] if tool_calls else None,
                tool_call_id,
            )

            if sig not in seen:
                seen.add(sig)
                messages.append(msg_dict)

        return messages
