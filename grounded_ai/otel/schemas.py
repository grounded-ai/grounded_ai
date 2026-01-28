"""
OpenTelemetry-compliant Agent Trace Schemas.

Provides Pydantic models for representing agent execution traces in a format
compatible with OpenLLMetry semantic conventions. These schemas serve as a
universal adapter for agent traces exported from any observability platform.
"""

from datetime import datetime
from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, Field, computed_field, model_validator


# === Core OTel-Compatible Types ===


class SpanContext(BaseModel):
    """OpenTelemetry span context identifying a span within a trace."""

    trace_id: str = Field(..., description="Unique identifier for the entire trace")
    span_id: str = Field(..., description="Unique identifier for this span")
    parent_span_id: Optional[str] = Field(
        None, description="Span ID of the parent span, if any"
    )


class TokenUsage(BaseModel):
    """Token usage metrics for LLM calls (OpenLLMetry: gen_ai.usage.*)."""

    input_tokens: int = Field(0, alias="gen_ai.usage.input_tokens")
    output_tokens: int = Field(0, alias="gen_ai.usage.output_tokens")
    total_tokens: int = Field(0)

    @model_validator(mode="after")
    def compute_total(self):
        """Auto-compute total if not provided."""
        if self.total_tokens == 0:
            self.total_tokens = self.input_tokens + self.output_tokens
        return self


# === Message Types ===


class Message(BaseModel):
    """A message in an LLM conversation."""

    role: Literal["system", "user", "assistant", "tool"] = Field(
        ..., description="Role of the message author"
    )
    content: str = Field(..., description="Content of the message")
    tool_call_id: Optional[str] = Field(
        None, description="ID of the tool call this message responds to"
    )
    name: Optional[str] = Field(None, description="Name for tool messages")


class ToolCallRequest(BaseModel):
    """A tool/function call requested by the LLM."""

    id: str = Field(..., description="Unique identifier for this tool call")
    name: str = Field(..., description="Name of the tool/function to call")
    arguments: Dict[str, Any] = Field(
        default_factory=dict, description="Arguments passed to the tool"
    )


# === Span Types ===


class LLMSpan(BaseModel):
    """
    Represents a single LLM invocation within an agent trace.
    Aligned with OpenLLMetry gen_ai.* semantic conventions.
    """

    span_type: Literal["llm"] = "llm"
    context: SpanContext

    # Provider info (OpenLLMetry: gen_ai.system, gen_ai.request.model)
    provider: str = Field(..., description="LLM provider (openai, anthropic, etc.)")
    model: str = Field(..., description="Model identifier")

    # Request
    messages: List[Message] = Field(
        default_factory=list, description="Input messages to the LLM"
    )
    temperature: Optional[float] = Field(None, alias="gen_ai.request.temperature")
    max_tokens: Optional[int] = Field(None, alias="gen_ai.request.max_tokens")

    # Response
    completion: Optional[str] = Field(None, description="LLM completion text")
    tool_calls: List[ToolCallRequest] = Field(
        default_factory=list, description="Tool calls requested by the LLM"
    )

    # Metrics
    usage: TokenUsage = Field(default_factory=TokenUsage)
    latency_ms: float = Field(..., description="Latency in milliseconds")

    # Status
    status: Literal["ok", "error"] = Field(..., description="Span execution status")
    error: Optional[str] = Field(None, description="Error message if status is error")

    # Timing
    start_time: datetime = Field(..., description="Span start timestamp")
    end_time: datetime = Field(..., description="Span end timestamp")


class ToolSpan(BaseModel):
    """Represents a tool/function execution within an agent trace."""

    span_type: Literal["tool"] = "tool"
    context: SpanContext

    # Tool info
    tool_name: str = Field(..., description="Name of the executed tool")
    tool_call_id: str = Field(
        ..., description="ID linking to the LLM's tool call request"
    )

    # Execution
    arguments: Dict[str, Any] = Field(
        default_factory=dict, description="Arguments passed to the tool"
    )
    result: Any = Field(None, description="Result returned by the tool")

    # Status
    status: Literal["ok", "error"] = Field(..., description="Tool execution status")
    error: Optional[str] = Field(None, description="Error message if status is error")
    latency_ms: float = Field(..., description="Execution latency in milliseconds")

    # Timing
    start_time: datetime = Field(..., description="Execution start timestamp")
    end_time: datetime = Field(..., description="Execution end timestamp")


class RetrievalSpan(BaseModel):
    """Represents a retrieval/RAG operation within an agent trace."""

    span_type: Literal["retrieval"] = "retrieval"
    context: SpanContext

    # Query
    query: str = Field(..., description="The retrieval query")

    # Results
    documents: List[Dict[str, Any]] = Field(
        default_factory=list, description="Retrieved documents"
    )
    scores: List[float] = Field(
        default_factory=list, description="Relevance scores for retrieved documents"
    )

    # Metadata
    vector_store: Optional[str] = Field(None, description="Vector store identifier")
    top_k: Optional[int] = Field(None, description="Number of documents requested")

    # Status
    status: Literal["ok", "error"] = Field(..., description="Retrieval status")
    error: Optional[str] = Field(None, description="Error message if status is error")
    latency_ms: float = Field(..., description="Retrieval latency in milliseconds")

    # Timing
    start_time: datetime = Field(..., description="Retrieval start timestamp")
    end_time: datetime = Field(..., description="Retrieval end timestamp")


# Union type for all span variants
AgentSpan = Union[LLMSpan, ToolSpan, RetrievalSpan]


# === Top-Level Agent Trace ===


class AgentTrace(BaseModel):
    """
    Complete agent execution trace.

    This is the unified model that traces from any source (LangSmith, Phoenix,
    OpenLLMetry, etc.) get converted into for evaluation. Spans are guaranteed
    to be in chronological order.
    """

    # Identity
    trace_id: str = Field(..., description="Unique identifier for this trace")

    # Agent metadata
    agent_name: Optional[str] = Field(None, description="Name of the agent")
    agent_version: Optional[str] = Field(None, description="Version of the agent")

    # Input/Output
    input: str = Field(..., description="Original user request/task")
    output: Optional[str] = Field(None, description="Final agent output")
    status: Literal["success", "error", "timeout"] = Field(
        ..., description="Overall trace status"
    )

    # The trace - ordered list of spans
    spans: List[AgentSpan] = Field(
        default_factory=list, description="Chronologically ordered spans"
    )

    # Timing
    start_time: datetime = Field(..., description="Trace start timestamp")
    end_time: datetime = Field(..., description="Trace end timestamp")

    # Arbitrary metadata for evaluation context
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional trace metadata"
    )

    @model_validator(mode="after")
    def ensure_chronological_order(self):
        """Guarantee spans are always in chronological order."""
        self.spans = sorted(self.spans, key=lambda s: s.start_time)
        return self

    # === Computed Properties for Evaluation ===

    @computed_field
    @property
    def total_latency_ms(self) -> float:
        """Total trace duration in milliseconds."""
        return (self.end_time - self.start_time).total_seconds() * 1000

    @computed_field
    @property
    def total_tokens(self) -> TokenUsage:
        """Aggregate token usage across all LLM spans."""
        usage = TokenUsage()
        for span in self.spans:
            if isinstance(span, LLMSpan):
                usage.input_tokens += span.usage.input_tokens
                usage.output_tokens += span.usage.output_tokens
                usage.total_tokens += span.usage.total_tokens
        return usage

    @computed_field
    @property
    def llm_call_count(self) -> int:
        """Number of LLM calls in the trace."""
        return sum(1 for s in self.spans if s.span_type == "llm")

    @computed_field
    @property
    def tool_call_count(self) -> int:
        """Number of tool calls in the trace."""
        return sum(1 for s in self.spans if s.span_type == "tool")

    # === Helper Methods for Evaluation ===

    def get_llm_spans(self) -> List[LLMSpan]:
        """Get all LLM spans in chronological order."""
        return [s for s in self.spans if s.span_type == "llm"]

    def get_tool_spans(self) -> List[ToolSpan]:
        """Get all tool spans in chronological order."""
        return [s for s in self.spans if s.span_type == "tool"]

    def get_retrieval_spans(self) -> List[RetrievalSpan]:
        """Get all retrieval spans in chronological order."""
        return [s for s in self.spans if s.span_type == "retrieval"]

    def get_reasoning_chain(self) -> List[str]:
        """Extract the chain of LLM completions for evaluation."""
        return [s.completion for s in self.get_llm_spans() if s.completion]

    def get_tool_call_pairs(self) -> List[Dict[str, Any]]:
        """
        Get tool calls with their arguments and results.
        Useful for evaluating tool parameterization correctness.
        """
        return [
            {
                "name": s.tool_name,
                "arguments": s.arguments,
                "result": s.result,
                "status": s.status,
                "error": s.error,
            }
            for s in self.get_tool_spans()
        ]

    def get_error_spans(self) -> List[AgentSpan]:
        """Get all spans that resulted in errors."""
        return [s for s in self.spans if s.status == "error"]
