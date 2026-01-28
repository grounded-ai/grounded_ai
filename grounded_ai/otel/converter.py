"""
Trace Converter utilities.

Provides converters to transform agent traces from various observability
platforms into the unified AgentTrace model for evaluation.
"""

from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Optional, Union

from .schemas import (
    AgentSpan,
    AgentTrace,
    LLMSpan,
    Message,
    RetrievalSpan,
    SpanContext,
    TokenUsage,
    ToolCallRequest,
    ToolSpan,
)


class TraceConverter:
    """
    Converts traces from various observability sources to AgentTrace.

    Supported sources:
    - OTLP/OpenLLMetry: Raw OpenTelemetry spans with gen_ai.* attributes
    - LangSmith: LangChain's observability platform exports
    - Phoenix: Arize Phoenix trace exports

    All converters guarantee chronological ordering of spans.
    """

    @classmethod
    def from_otlp(
        cls,
        spans: List[Dict[str, Any]],
        trace_input: Optional[str] = None,
    ) -> AgentTrace:
        """
        Convert raw OTLP spans with OpenLLMetry semantic conventions.

        Args:
            spans: List of OTLP span dictionaries with gen_ai.* attributes
            trace_input: Optional override for the trace input (user request)

        Returns:
            AgentTrace with spans in chronological order
        """
        if not spans:
            raise ValueError("Cannot convert empty span list")

        parsed_spans: List[AgentSpan] = []
        trace_id = None

        for span in spans:
            trace_id = trace_id or span.get("trace_id")
            parsed = cls._parse_otlp_span(span)
            if parsed:
                parsed_spans.append(parsed)

        # Sort chronologically
        parsed_spans.sort(key=lambda s: s.start_time)

        # Extract input/output from first/last LLM spans if not provided
        llm_spans = [s for s in parsed_spans if s.span_type == "llm"]
        inferred_input = trace_input
        inferred_output = None

        if llm_spans and not inferred_input:
            first_llm = llm_spans[0]
            user_msgs = [m for m in first_llm.messages if m.role == "user"]
            if user_msgs:
                inferred_input = user_msgs[-1].content

        if llm_spans:
            inferred_output = llm_spans[-1].completion

        # Determine overall status
        has_errors = any(s.status == "error" for s in parsed_spans)
        status = "error" if has_errors else "success"

        return AgentTrace(
            trace_id=trace_id or "unknown",
            input=inferred_input or "",
            output=inferred_output,
            status=status,
            spans=parsed_spans,
            start_time=parsed_spans[0].start_time,
            end_time=parsed_spans[-1].end_time,
        )

    @classmethod
    def from_langsmith(
        cls,
        run: Dict[str, Any],
    ) -> AgentTrace:
        """
        Convert a LangSmith run export to AgentTrace.

        Args:
            run: LangSmith run dictionary (from export or API)

        Returns:
            AgentTrace with spans in chronological order
        """
        spans: List[AgentSpan] = []
        cls._parse_langsmith_run(run, spans)

        # Sort chronologically
        spans.sort(key=lambda s: s.start_time)

        # Extract input/output
        trace_input = ""
        if "inputs" in run:
            inputs = run["inputs"]
            if isinstance(inputs, dict):
                trace_input = inputs.get("input", inputs.get("question", str(inputs)))
            else:
                trace_input = str(inputs)

        trace_output = None
        if "outputs" in run and run["outputs"]:
            outputs = run["outputs"]
            if isinstance(outputs, dict):
                trace_output = outputs.get("output", outputs.get("answer", str(outputs)))
            else:
                trace_output = str(outputs)

        # Determine status
        status = "error" if run.get("error") else "success"

        return AgentTrace(
            trace_id=run.get("id", run.get("trace_id", "unknown")),
            agent_name=run.get("name"),
            input=trace_input,
            output=trace_output,
            status=status,
            spans=spans,
            start_time=cls._parse_timestamp(run.get("start_time")),
            end_time=cls._parse_timestamp(run.get("end_time")),
            metadata=run.get("extra", {}),
        )

    @classmethod
    def from_phoenix(
        cls,
        trace: Dict[str, Any],
    ) -> AgentTrace:
        """
        Convert an Arize Phoenix trace export to AgentTrace.

        Args:
            trace: Phoenix trace dictionary

        Returns:
            AgentTrace with spans in chronological order
        """
        spans: List[AgentSpan] = []

        for span_data in trace.get("spans", []):
            parsed = cls._parse_phoenix_span(span_data)
            if parsed:
                spans.append(parsed)

        # Sort chronologically
        spans.sort(key=lambda s: s.start_time)

        # Extract input/output
        trace_input = trace.get("input", "")
        trace_output = trace.get("output")

        if not trace_input and spans:
            # Try to infer from first span
            first_span = spans[0]
            if isinstance(first_span, LLMSpan):
                user_msgs = [m for m in first_span.messages if m.role == "user"]
                if user_msgs:
                    trace_input = user_msgs[-1].content

        return AgentTrace(
            trace_id=trace.get("trace_id", trace.get("id", "unknown")),
            agent_name=trace.get("name"),
            input=trace_input,
            output=trace_output,
            status=trace.get("status", "success"),
            spans=spans,
            start_time=cls._parse_timestamp(trace.get("start_time")),
            end_time=cls._parse_timestamp(trace.get("end_time")),
            metadata=trace.get("metadata", {}),
        )

    @classmethod
    def auto_detect(
        cls,
        data: Union[Dict[str, Any], List[Dict[str, Any]]],
    ) -> AgentTrace:
        """
        Auto-detect the trace format and convert accordingly.

        Args:
            data: Trace data in unknown format

        Returns:
            AgentTrace with spans in chronological order

        Raises:
            ValueError: If format cannot be detected
        """
        # List of spans - likely OTLP
        if isinstance(data, list):
            return cls.from_otlp(data)

        # Dict with specific keys
        if isinstance(data, dict):
            # LangSmith format
            if "run_type" in data or "child_runs" in data:
                return cls.from_langsmith(data)

            # Phoenix format
            if "spans" in data and isinstance(data["spans"], list):
                # Check if spans have Phoenix-specific attributes
                if data["spans"] and "span_kind" in data["spans"][0]:
                    return cls.from_phoenix(data)
                # Could also be OTLP wrapped
                return cls.from_otlp(data["spans"])

            # Single OTLP span
            if "trace_id" in data and "span_id" in data:
                return cls.from_otlp([data])

        raise ValueError(
            "Could not auto-detect trace format. "
            "Please use a specific converter method (from_otlp, from_langsmith, from_phoenix)."
        )

    # === Private Helper Methods ===

    @classmethod
    def _parse_otlp_span(cls, span: Dict[str, Any]) -> Optional[AgentSpan]:
        """Parse a single OTLP span into the appropriate AgentSpan type."""
        attributes = span.get("attributes", {})

        # Determine span type from attributes
        span_kind = attributes.get("openinference.span.kind", "").lower()
        gen_ai_system = attributes.get("gen_ai.system")

        context = SpanContext(
            trace_id=span.get("trace_id", ""),
            span_id=span.get("span_id", ""),
            parent_span_id=span.get("parent_span_id"),
        )

        start_time = cls._parse_timestamp(span.get("start_time"))
        end_time = cls._parse_timestamp(span.get("end_time"))
        latency_ms = (end_time - start_time).total_seconds() * 1000

        status = "ok" if span.get("status", {}).get("code", 0) != 2 else "error"
        error = span.get("status", {}).get("message") if status == "error" else None

        # LLM span
        if span_kind == "llm" or gen_ai_system:
            return LLMSpan(
                context=context,
                provider=gen_ai_system or attributes.get("llm.provider", "unknown"),
                model=attributes.get("gen_ai.request.model", attributes.get("llm.model_name", "unknown")),
                messages=cls._parse_messages(attributes.get("gen_ai.input.messages", [])),
                temperature=attributes.get("gen_ai.request.temperature"),
                max_tokens=attributes.get("gen_ai.request.max_tokens"),
                completion=attributes.get("gen_ai.completion", attributes.get("llm.output", "")),
                tool_calls=cls._parse_tool_calls(attributes.get("gen_ai.tool_calls", [])),
                usage=TokenUsage(
                    input_tokens=attributes.get("gen_ai.usage.input_tokens", 0),
                    output_tokens=attributes.get("gen_ai.usage.output_tokens", 0),
                ),
                latency_ms=latency_ms,
                status=status,
                error=error,
                start_time=start_time,
                end_time=end_time,
            )

        # Tool span
        if span_kind == "tool":
            return ToolSpan(
                context=context,
                tool_name=attributes.get("tool.name", span.get("name", "unknown")),
                tool_call_id=attributes.get("tool.call_id", ""),
                arguments=attributes.get("tool.parameters", {}),
                result=attributes.get("tool.output"),
                status=status,
                error=error,
                latency_ms=latency_ms,
                start_time=start_time,
                end_time=end_time,
            )

        # Retrieval span
        if span_kind == "retriever":
            return RetrievalSpan(
                context=context,
                query=attributes.get("retrieval.query", ""),
                documents=attributes.get("retrieval.documents", []),
                scores=attributes.get("retrieval.scores", []),
                vector_store=attributes.get("retrieval.vector_store"),
                top_k=attributes.get("retrieval.top_k"),
                status=status,
                error=error,
                latency_ms=latency_ms,
                start_time=start_time,
                end_time=end_time,
            )

        # Unknown span type - skip
        return None

    @classmethod
    def _parse_langsmith_run(
        cls,
        run: Dict[str, Any],
        spans: List[AgentSpan],
        parent_span_id: Optional[str] = None,
    ) -> None:
        """Recursively parse LangSmith run and child runs into spans."""
        run_type = run.get("run_type", "")
        run_id = run.get("id", "")
        trace_id = run.get("trace_id", run_id)

        context = SpanContext(
            trace_id=trace_id,
            span_id=run_id,
            parent_span_id=parent_span_id,
        )

        start_time = cls._parse_timestamp(run.get("start_time"))
        end_time = cls._parse_timestamp(run.get("end_time"))
        latency_ms = (end_time - start_time).total_seconds() * 1000

        status = "error" if run.get("error") else "ok"
        error = run.get("error")

        # LLM run
        if run_type == "llm":
            inputs = run.get("inputs", {})
            outputs = run.get("outputs", {})

            messages = []
            if "messages" in inputs:
                messages = cls._parse_langsmith_messages(inputs["messages"])
            elif "prompt" in inputs:
                messages = [Message(role="user", content=inputs["prompt"])]

            completion = ""
            if "generations" in outputs:
                gens = outputs["generations"]
                if gens and len(gens) > 0 and len(gens[0]) > 0:
                    completion = gens[0][0].get("text", "")
            elif "output" in outputs:
                completion = outputs["output"]

            extra = run.get("extra", {})
            invocation = extra.get("invocation_params", {})

            spans.append(LLMSpan(
                context=context,
                provider=invocation.get("_type", "unknown"),
                model=invocation.get("model_name", invocation.get("model", "unknown")),
                messages=messages,
                temperature=invocation.get("temperature"),
                max_tokens=invocation.get("max_tokens"),
                completion=completion,
                tool_calls=[],
                usage=TokenUsage(
                    input_tokens=run.get("prompt_tokens", 0),
                    output_tokens=run.get("completion_tokens", 0),
                ),
                latency_ms=latency_ms,
                status=status,
                error=error,
                start_time=start_time,
                end_time=end_time,
            ))

        # Tool run
        elif run_type == "tool":
            inputs = run.get("inputs", {})
            outputs = run.get("outputs", {})

            spans.append(ToolSpan(
                context=context,
                tool_name=run.get("name", "unknown"),
                tool_call_id=run_id,
                arguments=inputs,
                result=outputs.get("output") if outputs else None,
                status=status,
                error=error,
                latency_ms=latency_ms,
                start_time=start_time,
                end_time=end_time,
            ))

        # Retriever run
        elif run_type == "retriever":
            inputs = run.get("inputs", {})
            outputs = run.get("outputs", {})

            documents = []
            if "documents" in outputs:
                documents = [
                    {"content": doc.get("page_content", ""), "metadata": doc.get("metadata", {})}
                    for doc in outputs["documents"]
                ]

            spans.append(RetrievalSpan(
                context=context,
                query=inputs.get("query", ""),
                documents=documents,
                scores=[],
                status=status,
                error=error,
                latency_ms=latency_ms,
                start_time=start_time,
                end_time=end_time,
            ))

        # Recursively process child runs
        for child in run.get("child_runs", []):
            cls._parse_langsmith_run(child, spans, parent_span_id=run_id)

    @classmethod
    def _parse_phoenix_span(cls, span: Dict[str, Any]) -> Optional[AgentSpan]:
        """Parse a Phoenix span into the appropriate AgentSpan type."""
        span_kind = span.get("span_kind", "").lower()

        context = SpanContext(
            trace_id=span.get("trace_id", ""),
            span_id=span.get("span_id", span.get("id", "")),
            parent_span_id=span.get("parent_span_id"),
        )

        start_time = cls._parse_timestamp(span.get("start_time"))
        end_time = cls._parse_timestamp(span.get("end_time"))
        latency_ms = (end_time - start_time).total_seconds() * 1000

        status = "ok" if span.get("status_code", "OK") == "OK" else "error"
        error = span.get("status_message") if status == "error" else None

        attributes = span.get("attributes", {})

        if span_kind == "llm":
            return LLMSpan(
                context=context,
                provider=attributes.get("llm.provider", "unknown"),
                model=attributes.get("llm.model_name", "unknown"),
                messages=cls._parse_messages(attributes.get("llm.input_messages", [])),
                completion=attributes.get("llm.output_messages", [{}])[0].get("content", "") if attributes.get("llm.output_messages") else "",
                tool_calls=[],
                usage=TokenUsage(
                    input_tokens=attributes.get("llm.token_count.prompt", 0),
                    output_tokens=attributes.get("llm.token_count.completion", 0),
                ),
                latency_ms=latency_ms,
                status=status,
                error=error,
                start_time=start_time,
                end_time=end_time,
            )

        if span_kind == "tool":
            return ToolSpan(
                context=context,
                tool_name=attributes.get("tool.name", span.get("name", "unknown")),
                tool_call_id=span.get("span_id", ""),
                arguments=attributes.get("tool.parameters", {}),
                result=attributes.get("tool.output"),
                status=status,
                error=error,
                latency_ms=latency_ms,
                start_time=start_time,
                end_time=end_time,
            )

        if span_kind == "retriever":
            return RetrievalSpan(
                context=context,
                query=attributes.get("retriever.query", ""),
                documents=attributes.get("retriever.documents", []),
                scores=attributes.get("retriever.scores", []),
                status=status,
                error=error,
                latency_ms=latency_ms,
                start_time=start_time,
                end_time=end_time,
            )

        return None

    @classmethod
    def _parse_timestamp(cls, ts: Any) -> datetime:
        """Parse various timestamp formats into datetime."""
        if ts is None:
            return datetime.now(timezone.utc)

        if isinstance(ts, datetime):
            return ts

        if isinstance(ts, (int, float)):
            # Assume nanoseconds if very large, otherwise seconds
            if ts > 1e12:
                return datetime.fromtimestamp(ts / 1e9, tz=timezone.utc)
            return datetime.fromtimestamp(ts, tz=timezone.utc)

        if isinstance(ts, str):
            # Try ISO format
            try:
                return datetime.fromisoformat(ts.replace("Z", "+00:00"))
            except ValueError:
                pass

        return datetime.now(timezone.utc)

    @classmethod
    def _parse_messages(cls, messages: List[Any]) -> List[Message]:
        """Parse message list into Message objects."""
        result = []
        for msg in messages:
            if isinstance(msg, dict):
                result.append(Message(
                    role=msg.get("role", "user"),
                    content=msg.get("content", ""),
                    tool_call_id=msg.get("tool_call_id"),
                    name=msg.get("name"),
                ))
            elif isinstance(msg, str):
                result.append(Message(role="user", content=msg))
        return result

    @classmethod
    def _parse_langsmith_messages(cls, messages: List[Any]) -> List[Message]:
        """Parse LangSmith message format."""
        result = []
        for msg in messages:
            if isinstance(msg, list) and len(msg) > 0:
                msg = msg[0]  # LangSmith wraps in extra list

            if isinstance(msg, dict):
                # Handle different message class formats
                msg_type = msg.get("type", msg.get("_type", ""))
                content = msg.get("content", "")

                role = "user"
                if "human" in msg_type.lower():
                    role = "user"
                elif "ai" in msg_type.lower() or "assistant" in msg_type.lower():
                    role = "assistant"
                elif "system" in msg_type.lower():
                    role = "system"
                elif "tool" in msg_type.lower() or "function" in msg_type.lower():
                    role = "tool"

                result.append(Message(role=role, content=content))

        return result

    @classmethod
    def _parse_tool_calls(cls, tool_calls: List[Any]) -> List[ToolCallRequest]:
        """Parse tool call list into ToolCallRequest objects."""
        result = []
        for tc in tool_calls:
            if isinstance(tc, dict):
                result.append(ToolCallRequest(
                    id=tc.get("id", ""),
                    name=tc.get("name", tc.get("function", {}).get("name", "")),
                    arguments=tc.get("arguments", tc.get("function", {}).get("arguments", {})),
                ))
        return result


def convert_traces(
    data: Union[Dict[str, Any], List[Dict[str, Any]]],
    source: Literal["otlp", "langsmith", "phoenix", "auto"] = "auto",
) -> List[AgentTrace]:
    """
    Convert trace export to list of AgentTrace models.

    All traces returned with spans in chronological order.

    Args:
        data: Raw trace export (single trace or batch)
        source: Trace source format, or "auto" to detect

    Returns:
        List of AgentTrace models, each with chronologically ordered spans

    Example:
        >>> traces = convert_traces(langsmith_export, source="langsmith")
        >>> for trace in traces:
        ...     print(f"Trace {trace.trace_id}: {trace.llm_call_count} LLM calls")
    """
    converter_map = {
        "otlp": TraceConverter.from_otlp,
        "langsmith": TraceConverter.from_langsmith,
        "phoenix": TraceConverter.from_phoenix,
        "auto": TraceConverter.auto_detect,
    }

    converter = converter_map.get(source)
    if not converter:
        raise ValueError(f"Unknown source: {source}. Supported: {list(converter_map.keys())}")

    # Handle batch of traces
    if isinstance(data, list) and source == "langsmith":
        return [TraceConverter.from_langsmith(run) for run in data]

    # Single trace
    result = converter(data)

    # Normalize to list
    if isinstance(result, AgentTrace):
        return [result]
    return result
