"""
Trace Converter utilities.

Provides converters to transform agent traces from various observability
platforms into the unified GenAI OTel-Semantic Convention models.
"""

import json
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Union

from .schemas import (
    GenAIConversation,
    GenAIMessage,
    GenAISpan,
    MessagePart,
    TokenUsage,
)


class TraceConverter:
    """
    Converts traces from various observability sources to GenAIConversation.

    Supported sources:
    - OTLP/OpenLLMetry: Raw OpenTelemetry spans with gen_ai.* attributes
    - LangSmith: LangChain's observability platform exports
    """

    @classmethod
    def from_otlp(
        cls,
        spans: List[Dict[str, Any]],
        conversation_id: Optional[str] = None,
    ) -> GenAIConversation:
        """
        Convert raw OTLP spans with OpenLLMetry semantic conventions.

        Args:
            spans: List of OTLP span dictionaries
            conversation_id: Optional ID override

        Returns:
            GenAIConversation with chronological GenAI spans
        """
        if not spans:
            raise ValueError("Cannot convert empty span list")

        gen_ai_spans: List[GenAISpan] = []
        trace_id = conversation_id

        for span in spans:
            parsed = cls._parse_otlp_span(span)
            if parsed:
                gen_ai_spans.append(parsed)
                if not trace_id:
                    trace_id = parsed.trace_id

        return GenAIConversation(
            conversation_id=trace_id or "unknown",
            spans=gen_ai_spans,
        )

    @classmethod
    def from_langsmith(cls, run: Dict[str, Any]) -> GenAIConversation:
        """
        Convert a LangSmith run export to GenAIConversation.
        Recursively extracts LLM runs as GenAIs spans.
        """
        gen_ai_spans: List[GenAISpan] = []
        cls._parse_langsmith_run_recursive(run, gen_ai_spans)

        return GenAIConversation(
            conversation_id=run.get("id", run.get("trace_id", "unknown")),
            spans=gen_ai_spans,
            metadata=run.get("extra", {}),
        )

    # === Private Parsers ===

    @classmethod
    def _parse_otlp_span(cls, span: Dict[str, Any]) -> Optional[GenAISpan]:
        """Parse OTLP span to GenAISpan."""
        attrs = span.get("attributes", {})

        # We generally filter for spans that have gen_ai.system
        # Removed OpenInference/Arize support
        if not attrs.get("gen_ai.system"):
            return None

        start_time = cls._parse_timestamp(span.get("start_time"))
        end_time = cls._parse_timestamp(span.get("end_time"))

        status_code = span.get("status", {}).get("code", 1)
        status = "OK"
        if status_code == 2:
            status = "ERROR"

        # Usage
        usage = TokenUsage(
            input_tokens=attrs.get("gen_ai.usage.input_tokens", 0),
            output_tokens=attrs.get("gen_ai.usage.output_tokens", 0),
        )

        return GenAISpan(
            trace_id=span.get("trace_id")
            or span.get("context", {}).get("trace_id", ""),
            span_id=span.get("span_id") or span.get("context", {}).get("span_id", ""),
            parent_span_id=span.get("parent_span_id")
            or span.get("context", {}).get("parent_id"),
            name=span.get("name", "llm"),
            kind="CLIENT",
            start_time=start_time,
            end_time=end_time,
            status=status,
            gen_ai_system=attrs.get("gen_ai.system") or "unknown",
            gen_ai_request_model=attrs.get("gen_ai.request.model") or "unknown",
            gen_ai_response_model=attrs.get("gen_ai.response.model"),
            gen_ai_input_messages=cls._parse_otlp_messages(attrs, "input"),
            gen_ai_output_messages=cls._parse_otlp_messages(attrs, "output"),
            usage=usage,
            gen_ai_request_temperature=attrs.get("gen_ai.request.temperature"),
            gen_ai_request_max_tokens=attrs.get("gen_ai.request.max_tokens"),
            attributes=attrs,
        )

    @classmethod
    def _parse_langsmith_run_recursive(
        cls, run: Dict[str, Any], spans: List[GenAISpan]
    ):
        """Recursively find LLM runs."""
        run_type = run.get("run_type")

        if run_type == "llm":
            spans.append(cls._parse_langsmith_llm_run(run))

        for child in run.get("child_runs", []):
            cls._parse_langsmith_run_recursive(child, spans)

    @classmethod
    def _parse_langsmith_llm_run(cls, run: Dict[str, Any]) -> GenAISpan:
        start_time = cls._parse_timestamp(run.get("start_time"))
        end_time = cls._parse_timestamp(run.get("end_time"))

        extra = run.get("extra", {})
        invocation = extra.get("invocation_params", {})

        # Parse Messages
        inputs = run.get("inputs", {})
        input_msgs = []
        if "messages" in inputs:
            input_msgs = cls._parse_langchain_messages(inputs["messages"])
        elif "prompt" in inputs:
            input_msgs = [
                GenAIMessage(
                    role="user",
                    parts=[MessagePart(type="text", content=inputs["prompt"])],
                )
            ]

        # Parse Output
        outputs = run.get("outputs", {})
        output_msgs = []
        if "generations" in outputs:
            # LangChain generations format
            for gen_list in outputs["generations"]:
                for gen in gen_list:
                    msg = gens_to_message(gen)
                    if msg:
                        output_msgs.append(msg)
        elif "output" in outputs:
            # Simple string output
            output_msgs.append(
                GenAIMessage(
                    role="assistant",
                    parts=[MessagePart(type="text", content=outputs["output"])],
                )
            )

        return GenAISpan(
            trace_id=run.get("trace_id", str(run.get("id"))),
            span_id=str(run.get("id")),
            parent_span_id=str(run.get("parent_run_id"))
            if run.get("parent_run_id")
            else None,
            name=run.get("name", "llm"),
            kind="CLIENT",
            start_time=start_time,
            end_time=end_time,
            status="ERROR" if run.get("error") else "OK",
            gen_ai_system=invocation.get("_type", "unknown"),
            gen_ai_request_model=invocation.get("model_name")
            or invocation.get("model")
            or "unknown",
            gen_ai_input_messages=input_msgs,
            gen_ai_output_messages=output_msgs,
            usage=TokenUsage(
                input_tokens=run.get("prompt_tokens", 0),
                output_tokens=run.get("completion_tokens", 0)
                or extra.get("token_usage", {}).get("completion_tokens", 0),
            ),
            gen_ai_request_temperature=invocation.get("temperature"),
            attributes=extra,
        )

    # === Helpers ===

    @classmethod
    def _parse_timestamp(cls, ts: Any) -> datetime:
        if ts is None:
            return datetime.now(timezone.utc)
        if isinstance(ts, datetime):
            return ts
        if isinstance(ts, (int, float)):
            return datetime.fromtimestamp(
                ts if ts < 1e11 else ts / 1e9, tz=timezone.utc
            )
        if isinstance(ts, str):
            try:
                return datetime.fromisoformat(ts.replace("Z", "+00:00"))
            except ValueError:
                pass
        return datetime.now(timezone.utc)

    @classmethod
    def _parse_otlp_messages(cls, attrs: Dict, direction: str) -> List[GenAIMessage]:
        # 1. Try standard gen_ai.{direction}.messages (list or JSON string)
        key = f"gen_ai.{direction}.messages"
        val = attrs.get(key)
        if val:
            if isinstance(val, str):
                try:
                    val = json.loads(val)
                except Exception:
                    val = []
            if isinstance(val, list):
                # Parse list of dicts to GenAIMessages
                return cls._parse_list_of_message_dicts(val)

        return []

    @staticmethod
    def _parse_list_of_message_dicts(msgs: List[Dict]) -> List[GenAIMessage]:
        results = []
        for m in msgs:
            role = m.get("role", "user")
            parts = []

            # 1. Standard GenAI 'parts' list
            if "parts" in m and isinstance(m["parts"], list):
                for p in m["parts"]:
                    # Convert dict to MessagePart if needed
                    if isinstance(p, dict):
                        args = p.get("arguments")
                        if isinstance(args, str):
                            try:
                                args = json.loads(args)
                            except ValueError:
                                # Coerce non-JSON string to dict wrap if necessary
                                if args:
                                    args = {"raw_arguments": args}
                                else:
                                    args = None

                        parts.append(
                            MessagePart(
                                type=p.get("type", "text"),
                                content=p.get("content"),
                                id=p.get("id"),
                                name=p.get("name"),
                                arguments=args,
                                response=p.get("response"),
                            )
                        )

            # 2. Simplified 'content' string
            elif "content" in m:
                content = m.get("content")
                if content:
                    parts.append(MessagePart(type="text", content=str(content)))

            results.append(GenAIMessage(role=role, parts=parts))
        return results

    @staticmethod
    def _parse_langchain_messages(
        messages: List[Union[Dict, Any]],
    ) -> List[GenAIMessage]:
        result = []
        for m in messages:
            # LangSmith message dict
            if hasattr(m, "dict"):
                m = m.dict()
            if not isinstance(m, dict):
                continue

            role = "user"
            type_ = m.get("type", m.get("_type", "")).lower()
            if "system" in type_:
                role = "system"
            elif "ai" in type_ or "assistant" in type_:
                role = "assistant"
            elif "tool" in type_ or "function" in type_:
                role = "tool"

            parts = []
            content = m.get("content")
            if content:
                parts.append(MessagePart(type="text", content=content))

            # Tool calls in message?
            tool_calls = m.get("tool_calls", [])
            for tc in tool_calls:
                parts.append(
                    MessagePart(
                        type="tool_call",
                        id=tc.get("id"),
                        name=tc.get("name") or tc.get("function", {}).get("name"),
                        arguments=tc.get("args")
                        or tc.get("function", {}).get("arguments"),
                    )
                )

            result.append(GenAIMessage(role=role, parts=parts))
        return result


def gens_to_message(gen: Dict) -> Optional[GenAIMessage]:
    """Helper to convert LangChain Generation dict to GenAIMessage."""
    text = gen.get("text", "")
    msg_dict = gen.get("message", {})

    parts = []
    if text:
        parts.append(MessagePart(type="text", content=text))

    tool_calls = msg_dict.get("tool_calls", [])
    for tc in tool_calls:
        # Check if already added (sometimes text includes tool call repr)
        parts.append(
            MessagePart(
                type="tool_call",
                id=tc.get("id"),
                name=tc.get("function", {}).get("name"),
                arguments=tc.get("function", {}).get("arguments"),
            )
        )

    if not parts:
        return None

    return GenAIMessage(role="assistant", parts=parts)


def convert_traces(trace_data: List[Dict[str, Any]]) -> List[GenAIConversation]:
    """
    Convenience wrapper to convert a list of raw traces into GenAIConversations.
    In OTLP, input is often a list of spans, potentially mixed from different traces.
    This helper groups them by trace_id and converts.
    """
    # Group by trace_id
    traces = {}
    for span in trace_data:
        tid = span.get("trace_id") or span.get("context", {}).get("trace_id")
        if tid:
            if tid not in traces:
                traces[tid] = []
            traces[tid].append(span)

    # Convert each group
    conversations = []
    for tid, spans in traces.items():
        try:
            conversations.append(TraceConverter.from_otlp(spans, conversation_id=tid))
        except ValueError:
            continue

    return conversations
