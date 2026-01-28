import pytest
from datetime import datetime, timezone, timedelta

from grounded_ai.otel import (
    AgentTrace,
    AgentSpan,
    LLMSpan,
    ToolSpan,
    RetrievalSpan,
    TraceConverter,
    convert_traces,
)
from grounded_ai.otel.schemas import (
    SpanContext,
    TokenUsage,
    Message,
    ToolCallRequest,
)


# === Fixtures ===


@pytest.fixture
def base_time():
    """Base timestamp for test spans."""
    return datetime(2026, 1, 15, 10, 0, 0, tzinfo=timezone.utc)


@pytest.fixture
def span_context():
    """Default span context for tests."""
    return SpanContext(trace_id="trace-1", span_id="span-1")


@pytest.fixture
def sample_llm_span(base_time, span_context):
    """Sample LLM span for tests."""
    return LLMSpan(
        context=span_context,
        provider="openai",
        model="gpt-4",
        messages=[Message(role="user", content="Hello")],
        completion="Hi there!",
        usage=TokenUsage(input_tokens=10, output_tokens=20),
        latency_ms=150.0,
        status="ok",
        start_time=base_time,
        end_time=base_time + timedelta(milliseconds=150),
    )


@pytest.fixture
def sample_tool_span(base_time):
    """Sample tool span for tests."""
    return ToolSpan(
        context=SpanContext(trace_id="trace-1", span_id="span-2"),
        tool_name="calculator",
        tool_call_id="tc-1",
        arguments={"x": 5, "y": 3},
        result=8,
        status="ok",
        latency_ms=10.0,
        start_time=base_time + timedelta(milliseconds=150),
        end_time=base_time + timedelta(milliseconds=160),
    )


# === SpanContext Tests ===


class TestSpanContext:
    def test_creation(self):
        """Test SpanContext basic creation."""
        ctx = SpanContext(
            trace_id="trace-123",
            span_id="span-456",
            parent_span_id="span-000",
        )
        assert ctx.trace_id == "trace-123"
        assert ctx.span_id == "span-456"
        assert ctx.parent_span_id == "span-000"

    def test_optional_parent(self):
        """Test SpanContext with no parent."""
        ctx = SpanContext(trace_id="trace-123", span_id="span-456")
        assert ctx.parent_span_id is None


# === TokenUsage Tests ===


class TestTokenUsage:
    def test_auto_total(self):
        """Test TokenUsage auto-computes total."""
        usage = TokenUsage(input_tokens=100, output_tokens=50)
        assert usage.total_tokens == 150

    def test_explicit_total(self):
        """Test TokenUsage respects explicit total."""
        usage = TokenUsage(input_tokens=100, output_tokens=50, total_tokens=200)
        assert usage.total_tokens == 200

    def test_defaults(self):
        """Test TokenUsage defaults to zero."""
        usage = TokenUsage()
        assert usage.input_tokens == 0
        assert usage.output_tokens == 0
        assert usage.total_tokens == 0


# === Message Tests ===


class TestMessage:
    def test_creation(self):
        """Test Message basic creation."""
        msg = Message(role="user", content="Hello, world!")
        assert msg.role == "user"
        assert msg.content == "Hello, world!"

    def test_tool_fields(self):
        """Test Message with tool-related fields."""
        msg = Message(
            role="tool",
            content="Result: 42",
            tool_call_id="call-123",
            name="calculator",
        )
        assert msg.role == "tool"
        assert msg.tool_call_id == "call-123"
        assert msg.name == "calculator"


# === LLMSpan Tests ===


class TestLLMSpan:
    def test_creation(self, sample_llm_span):
        """Test LLMSpan basic creation."""
        assert sample_llm_span.span_type == "llm"
        assert sample_llm_span.provider == "openai"
        assert sample_llm_span.model == "gpt-4"
        assert sample_llm_span.completion == "Hi there!"

    def test_with_tool_calls(self, base_time, span_context):
        """Test LLMSpan with tool calls."""
        span = LLMSpan(
            context=span_context,
            provider="openai",
            model="gpt-4",
            messages=[],
            tool_calls=[
                ToolCallRequest(id="tc-1", name="search", arguments={"q": "test"})
            ],
            latency_ms=100.0,
            status="ok",
            start_time=base_time,
            end_time=base_time + timedelta(milliseconds=100),
        )
        assert len(span.tool_calls) == 1
        assert span.tool_calls[0].name == "search"


# === ToolSpan Tests ===


class TestToolSpan:
    def test_creation(self, sample_tool_span):
        """Test ToolSpan basic creation."""
        assert sample_tool_span.span_type == "tool"
        assert sample_tool_span.tool_name == "calculator"
        assert sample_tool_span.result == 8

    def test_error_status(self, base_time):
        """Test ToolSpan with error status."""
        span = ToolSpan(
            context=SpanContext(trace_id="trace-1", span_id="span-2"),
            tool_name="api_call",
            tool_call_id="tc-2",
            arguments={"url": "http://example.com"},
            status="error",
            error="Connection timeout",
            latency_ms=5000.0,
            start_time=base_time,
            end_time=base_time + timedelta(milliseconds=5000),
        )
        assert span.status == "error"
        assert span.error == "Connection timeout"


# === RetrievalSpan Tests ===


class TestRetrievalSpan:
    def test_creation(self, base_time):
        """Test RetrievalSpan basic creation."""
        span = RetrievalSpan(
            context=SpanContext(trace_id="trace-1", span_id="span-3"),
            query="What is Python?",
            documents=[
                {"content": "Python is a programming language", "metadata": {}}
            ],
            scores=[0.95],
            vector_store="pinecone",
            top_k=5,
            status="ok",
            latency_ms=50.0,
            start_time=base_time,
            end_time=base_time + timedelta(milliseconds=50),
        )
        assert span.span_type == "retrieval"
        assert span.query == "What is Python?"
        assert len(span.documents) == 1
        assert span.scores[0] == 0.95


# === AgentTrace Tests ===


class TestAgentTrace:
    @pytest.fixture
    def create_llm_span(self, base_time):
        """Factory to create LLM spans with offset."""
        def _create(span_id: str, offset_ms: int, completion: str) -> LLMSpan:
            start = base_time + timedelta(milliseconds=offset_ms)
            return LLMSpan(
                context=SpanContext(trace_id="trace-1", span_id=span_id),
                provider="openai",
                model="gpt-4",
                messages=[Message(role="user", content="test")],
                completion=completion,
                usage=TokenUsage(input_tokens=10, output_tokens=20),
                latency_ms=100.0,
                status="ok",
                start_time=start,
                end_time=start + timedelta(milliseconds=100),
            )
        return _create

    @pytest.fixture
    def create_tool_span(self, base_time):
        """Factory to create tool spans with offset."""
        def _create(span_id: str, offset_ms: int, name: str) -> ToolSpan:
            start = base_time + timedelta(milliseconds=offset_ms)
            return ToolSpan(
                context=SpanContext(trace_id="trace-1", span_id=span_id),
                tool_name=name,
                tool_call_id=f"tc-{span_id}",
                arguments={"arg": "value"},
                result="result",
                status="ok",
                latency_ms=50.0,
                start_time=start,
                end_time=start + timedelta(milliseconds=50),
            )
        return _create

    def test_creation(self, base_time, create_llm_span, create_tool_span):
        """Test AgentTrace basic creation."""
        spans = [
            create_llm_span("span-1", 0, "First response"),
            create_tool_span("span-2", 100, "search"),
            create_llm_span("span-3", 150, "Final response"),
        ]

        trace = AgentTrace(
            trace_id="trace-1",
            agent_name="test-agent",
            input="User query",
            output="Final response",
            status="success",
            spans=spans,
            start_time=base_time,
            end_time=base_time + timedelta(milliseconds=250),
        )

        assert trace.trace_id == "trace-1"
        assert trace.agent_name == "test-agent"
        assert len(trace.spans) == 3

    def test_chronological_ordering(self, base_time, create_llm_span, create_tool_span):
        """Test that spans are automatically sorted chronologically."""
        # Create spans out of order
        span3 = create_llm_span("span-3", 200, "Third")
        span1 = create_llm_span("span-1", 0, "First")
        span2 = create_tool_span("span-2", 100, "search")

        trace = AgentTrace(
            trace_id="trace-1",
            input="test",
            status="success",
            spans=[span3, span1, span2],  # Deliberately out of order
            start_time=base_time,
            end_time=base_time + timedelta(milliseconds=300),
        )

        # Verify chronological order
        assert trace.spans[0].context.span_id == "span-1"
        assert trace.spans[1].context.span_id == "span-2"
        assert trace.spans[2].context.span_id == "span-3"

    def test_computed_total_latency(self, base_time):
        """Test total_latency_ms computed field."""
        trace = AgentTrace(
            trace_id="trace-1",
            input="test",
            status="success",
            spans=[],
            start_time=base_time,
            end_time=base_time + timedelta(milliseconds=500),
        )
        assert trace.total_latency_ms == 500.0

    def test_computed_total_tokens(self, base_time, create_llm_span):
        """Test total_tokens computed field."""
        spans = [
            create_llm_span("span-1", 0, "First"),
            create_llm_span("span-2", 100, "Second"),
        ]

        trace = AgentTrace(
            trace_id="trace-1",
            input="test",
            status="success",
            spans=spans,
            start_time=base_time,
            end_time=base_time + timedelta(milliseconds=200),
        )

        # Each span has 10 input + 20 output = 30 total, x2 spans = 60
        assert trace.total_tokens.input_tokens == 20
        assert trace.total_tokens.output_tokens == 40
        assert trace.total_tokens.total_tokens == 60

    def test_llm_call_count(self, base_time, create_llm_span, create_tool_span):
        """Test llm_call_count computed field."""
        spans = [
            create_llm_span("span-1", 0, "First"),
            create_tool_span("span-2", 100, "search"),
            create_llm_span("span-3", 150, "Second"),
        ]

        trace = AgentTrace(
            trace_id="trace-1",
            input="test",
            status="success",
            spans=spans,
            start_time=base_time,
            end_time=base_time + timedelta(milliseconds=250),
        )

        assert trace.llm_call_count == 2

    def test_tool_call_count(self, base_time, create_llm_span, create_tool_span):
        """Test tool_call_count computed field."""
        spans = [
            create_llm_span("span-1", 0, "First"),
            create_tool_span("span-2", 100, "search"),
            create_tool_span("span-3", 150, "calculate"),
        ]

        trace = AgentTrace(
            trace_id="trace-1",
            input="test",
            status="success",
            spans=spans,
            start_time=base_time,
            end_time=base_time + timedelta(milliseconds=200),
        )

        assert trace.tool_call_count == 2

    def test_get_llm_spans(self, base_time, create_llm_span, create_tool_span):
        """Test get_llm_spans helper method."""
        spans = [
            create_llm_span("span-1", 0, "First"),
            create_tool_span("span-2", 100, "search"),
            create_llm_span("span-3", 150, "Second"),
        ]

        trace = AgentTrace(
            trace_id="trace-1",
            input="test",
            status="success",
            spans=spans,
            start_time=base_time,
            end_time=base_time + timedelta(milliseconds=250),
        )

        llm_spans = trace.get_llm_spans()
        assert len(llm_spans) == 2
        assert all(s.span_type == "llm" for s in llm_spans)

    def test_get_tool_spans(self, base_time, create_llm_span, create_tool_span):
        """Test get_tool_spans helper method."""
        spans = [
            create_llm_span("span-1", 0, "First"),
            create_tool_span("span-2", 100, "search"),
            create_tool_span("span-3", 150, "calculate"),
        ]

        trace = AgentTrace(
            trace_id="trace-1",
            input="test",
            status="success",
            spans=spans,
            start_time=base_time,
            end_time=base_time + timedelta(milliseconds=200),
        )

        tool_spans = trace.get_tool_spans()
        assert len(tool_spans) == 2
        assert all(s.span_type == "tool" for s in tool_spans)

    def test_get_reasoning_chain(self, base_time, create_llm_span, create_tool_span):
        """Test get_reasoning_chain helper method."""
        spans = [
            create_llm_span("span-1", 0, "I need to search"),
            create_tool_span("span-2", 100, "search"),
            create_llm_span("span-3", 150, "Based on results, the answer is X"),
        ]

        trace = AgentTrace(
            trace_id="trace-1",
            input="test",
            status="success",
            spans=spans,
            start_time=base_time,
            end_time=base_time + timedelta(milliseconds=250),
        )

        chain = trace.get_reasoning_chain()
        assert len(chain) == 2
        assert chain[0] == "I need to search"
        assert chain[1] == "Based on results, the answer is X"

    def test_get_tool_call_pairs(self, base_time, create_tool_span):
        """Test get_tool_call_pairs helper method."""
        spans = [
            create_tool_span("span-1", 0, "search"),
            create_tool_span("span-2", 50, "calculate"),
        ]

        trace = AgentTrace(
            trace_id="trace-1",
            input="test",
            status="success",
            spans=spans,
            start_time=base_time,
            end_time=base_time + timedelta(milliseconds=100),
        )

        pairs = trace.get_tool_call_pairs()
        assert len(pairs) == 2
        assert pairs[0]["name"] == "search"
        assert pairs[1]["name"] == "calculate"
        assert "arguments" in pairs[0]
        assert "result" in pairs[0]
        assert "status" in pairs[0]

    def test_get_error_spans(self, base_time, create_llm_span):
        """Test get_error_spans helper method."""
        error_span = ToolSpan(
            context=SpanContext(trace_id="trace-1", span_id="span-2"),
            tool_name="api_call",
            tool_call_id="tc-2",
            arguments={},
            status="error",
            error="Failed",
            latency_ms=100.0,
            start_time=base_time + timedelta(milliseconds=100),
            end_time=base_time + timedelta(milliseconds=200),
        )

        spans = [
            create_llm_span("span-1", 0, "First"),
            error_span,
            create_llm_span("span-3", 200, "Second"),
        ]

        trace = AgentTrace(
            trace_id="trace-1",
            input="test",
            status="error",
            spans=spans,
            start_time=base_time,
            end_time=base_time + timedelta(milliseconds=300),
        )

        errors = trace.get_error_spans()
        assert len(errors) == 1
        assert errors[0].status == "error"


# === TraceConverter OTLP Tests ===


class TestTraceConverterOTLP:
    def test_from_otlp_llm_span(self):
        """Test converting OTLP LLM spans."""
        otlp_spans = [
            {
                "trace_id": "trace-123",
                "span_id": "span-1",
                "start_time": "2026-01-15T10:00:00Z",
                "end_time": "2026-01-15T10:00:01Z",
                "attributes": {
                    "openinference.span.kind": "llm",
                    "gen_ai.system": "openai",
                    "gen_ai.request.model": "gpt-4",
                    "gen_ai.completion": "Hello!",
                    "gen_ai.usage.input_tokens": 10,
                    "gen_ai.usage.output_tokens": 5,
                },
                "status": {"code": 0},
            }
        ]

        trace = TraceConverter.from_otlp(otlp_spans)

        assert trace.trace_id == "trace-123"
        assert len(trace.spans) == 1
        assert trace.spans[0].span_type == "llm"
        assert trace.spans[0].provider == "openai"
        assert trace.spans[0].model == "gpt-4"

    def test_from_otlp_tool_span(self):
        """Test converting OTLP tool spans."""
        otlp_spans = [
            {
                "trace_id": "trace-123",
                "span_id": "span-1",
                "start_time": "2026-01-15T10:00:00Z",
                "end_time": "2026-01-15T10:00:01Z",
                "attributes": {
                    "openinference.span.kind": "tool",
                    "tool.name": "calculator",
                    "tool.parameters": {"x": 5},
                    "tool.output": 10,
                },
                "status": {"code": 0},
            }
        ]

        trace = TraceConverter.from_otlp(otlp_spans)

        assert len(trace.spans) == 1
        assert trace.spans[0].span_type == "tool"
        assert trace.spans[0].tool_name == "calculator"

    def test_from_otlp_empty_raises(self):
        """Test that empty span list raises ValueError."""
        with pytest.raises(ValueError):
            TraceConverter.from_otlp([])


# === TraceConverter LangSmith Tests ===


class TestTraceConverterLangSmith:
    def test_from_langsmith_basic(self):
        """Test converting basic LangSmith run."""
        langsmith_run = {
            "id": "run-123",
            "trace_id": "trace-123",
            "name": "test-agent",
            "run_type": "chain",
            "inputs": {"input": "What is 2+2?"},
            "outputs": {"output": "4"},
            "start_time": "2026-01-15T10:00:00Z",
            "end_time": "2026-01-15T10:00:05Z",
            "child_runs": [
                {
                    "id": "run-456",
                    "trace_id": "trace-123",
                    "run_type": "llm",
                    "inputs": {"messages": []},
                    "outputs": {"generations": [[{"text": "The answer is 4"}]]},
                    "start_time": "2026-01-15T10:00:01Z",
                    "end_time": "2026-01-15T10:00:02Z",
                    "extra": {"invocation_params": {"model_name": "gpt-4"}},
                    "child_runs": [],
                }
            ],
        }

        trace = TraceConverter.from_langsmith(langsmith_run)

        assert trace.trace_id == "run-123"
        assert trace.agent_name == "test-agent"
        assert trace.input == "What is 2+2?"
        assert trace.output == "4"
        assert len(trace.spans) == 1  # Only LLM child run becomes a span

    def test_from_langsmith_with_tools(self):
        """Test converting LangSmith run with tool calls."""
        langsmith_run = {
            "id": "run-123",
            "run_type": "chain",
            "inputs": {"input": "Search for Python"},
            "outputs": {"output": "Results"},
            "start_time": "2026-01-15T10:00:00Z",
            "end_time": "2026-01-15T10:00:05Z",
            "child_runs": [
                {
                    "id": "run-tool",
                    "run_type": "tool",
                    "name": "search",
                    "inputs": {"query": "Python"},
                    "outputs": {"output": "Python is..."},
                    "start_time": "2026-01-15T10:00:01Z",
                    "end_time": "2026-01-15T10:00:02Z",
                    "child_runs": [],
                }
            ],
        }

        trace = TraceConverter.from_langsmith(langsmith_run)

        assert len(trace.spans) == 1
        assert trace.spans[0].span_type == "tool"
        assert trace.spans[0].tool_name == "search"


# === TraceConverter Phoenix Tests ===


class TestTraceConverterPhoenix:
    def test_from_phoenix_basic(self):
        """Test converting basic Phoenix trace."""
        phoenix_trace = {
            "trace_id": "trace-123",
            "name": "test-agent",
            "input": "Hello",
            "output": "Hi there!",
            "status": "success",
            "start_time": "2026-01-15T10:00:00Z",
            "end_time": "2026-01-15T10:00:05Z",
            "spans": [
                {
                    "trace_id": "trace-123",
                    "span_id": "span-1",
                    "span_kind": "llm",
                    "start_time": "2026-01-15T10:00:01Z",
                    "end_time": "2026-01-15T10:00:02Z",
                    "status_code": "OK",
                    "attributes": {
                        "llm.provider": "openai",
                        "llm.model_name": "gpt-4",
                        "llm.output_messages": [{"content": "Hi there!"}],
                    },
                }
            ],
        }

        trace = TraceConverter.from_phoenix(phoenix_trace)

        assert trace.trace_id == "trace-123"
        assert trace.input == "Hello"
        assert len(trace.spans) == 1
        assert trace.spans[0].span_type == "llm"


# === TraceConverter Auto-Detect Tests ===


class TestTraceConverterAutoDetect:
    def test_auto_detect_langsmith(self):
        """Test auto-detection of LangSmith format."""
        data = {
            "id": "run-123",
            "run_type": "chain",
            "inputs": {"input": "test"},
            "outputs": {},
            "start_time": "2026-01-15T10:00:00Z",
            "end_time": "2026-01-15T10:00:01Z",
            "child_runs": [],
        }

        trace = TraceConverter.auto_detect(data)
        assert isinstance(trace, AgentTrace)

    def test_auto_detect_otlp_list(self):
        """Test auto-detection of OTLP span list."""
        data = [
            {
                "trace_id": "trace-123",
                "span_id": "span-1",
                "start_time": "2026-01-15T10:00:00Z",
                "end_time": "2026-01-15T10:00:01Z",
                "attributes": {
                    "openinference.span.kind": "llm",
                    "gen_ai.system": "openai",
                    "gen_ai.request.model": "gpt-4",
                },
                "status": {"code": 0},
            }
        ]

        trace = TraceConverter.auto_detect(data)
        assert isinstance(trace, AgentTrace)

    def test_auto_detect_unknown_raises(self):
        """Test that unknown format raises ValueError."""
        with pytest.raises(ValueError):
            TraceConverter.auto_detect({"unknown": "format"})


# === convert_traces Utility Tests ===


class TestConvertTracesUtility:
    def test_returns_list(self):
        """Test convert_traces always returns a list."""
        data = {
            "id": "run-123",
            "run_type": "chain",
            "inputs": {"input": "test"},
            "outputs": {},
            "start_time": "2026-01-15T10:00:00Z",
            "end_time": "2026-01-15T10:00:01Z",
            "child_runs": [],
        }

        traces = convert_traces(data, source="langsmith")

        assert isinstance(traces, list)
        assert len(traces) == 1
        assert isinstance(traces[0], AgentTrace)

    def test_batch_langsmith(self):
        """Test convert_traces handles batch of LangSmith runs."""
        data = [
            {
                "id": "run-1",
                "run_type": "chain",
                "inputs": {"input": "test1"},
                "outputs": {},
                "start_time": "2026-01-15T10:00:00Z",
                "end_time": "2026-01-15T10:00:01Z",
                "child_runs": [],
            },
            {
                "id": "run-2",
                "run_type": "chain",
                "inputs": {"input": "test2"},
                "outputs": {},
                "start_time": "2026-01-15T10:00:02Z",
                "end_time": "2026-01-15T10:00:03Z",
                "child_runs": [],
            },
        ]

        traces = convert_traces(data, source="langsmith")

        assert len(traces) == 2
        assert traces[0].trace_id == "run-1"
        assert traces[1].trace_id == "run-2"

    def test_invalid_source_raises(self):
        """Test convert_traces raises for invalid source."""
        with pytest.raises(ValueError):
            convert_traces({}, source="invalid")
