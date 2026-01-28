"""
OpenTelemetry-compliant Agent Trace module.

Provides schemas and converters for working with agent traces from
various observability platforms (LangSmith, Phoenix, OpenLLMetry, etc.).
"""

from .schemas import (
    # Core types
    SpanContext,
    TokenUsage,
    Message,
    ToolCallRequest,
    # Span types
    LLMSpan,
    ToolSpan,
    RetrievalSpan,
    AgentSpan,
    # Top-level trace
    AgentTrace,
)

from .converter import (
    TraceConverter,
    convert_traces,
)

__all__ = [
    # Core types
    "SpanContext",
    "TokenUsage",
    "Message",
    "ToolCallRequest",
    # Span types
    "LLMSpan",
    "ToolSpan",
    "RetrievalSpan",
    "AgentSpan",
    # Top-level trace
    "AgentTrace",
    # Converter
    "TraceConverter",
    "convert_traces",
]
