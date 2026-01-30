
"""
Trace Converter utilities.

Provides converters to transform agent traces from various observability
platforms into the unified GenAI OTel-Semantic Convention models.
"""

from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Optional, Union
import json

from .schemas import (
    GenAISpan,
    GenAIConversation,
    GenAIMessage,
    MessagePart,
    TokenUsage,
)


class TraceConverter:
    """
    Converts traces from various observability sources to GenAIConversation.

    Supported sources:
    - OTLP/OpenLLMetry: Raw OpenTelemetry spans with gen_ai.* attributes
    - LangSmith: LangChain's observability platform exports
    - Phoenix: Arize Phoenix trace exports
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
            metadata=run.get("extra", {})
        )

    @classmethod
    def from_phoenix(cls, trace: Dict[str, Any]) -> GenAIConversation:
        """
        Convert an Arize Phoenix trace export.
        """
        gen_ai_spans: List[GenAISpan] = []
        spans_list = trace.get("spans", [])
        
        for span in spans_list:
            # Only interested in LLM spans for GenAI Schema
            # (Vector retrieval can be simulated as system context or looked at later)
            if span.get("span_kind") == "LLM" or span.get("attributes", {}).get("openinference.span.kind") == "LLM":
                 parsed = cls._parse_phoenix_span(span)
                 if parsed:
                     gen_ai_spans.append(parsed)

        return GenAIConversation(
            conversation_id=trace.get("trace_id", "unknown"),
            spans=gen_ai_spans,
        )

    # === Private Parsers ===

    @classmethod
    def _parse_otlp_span(cls, span: Dict[str, Any]) -> Optional[GenAISpan]:
        """Parse OTLP span to GenAISpan."""
        attrs = span.get("attributes", {})
        
        # We generally filter for spans that have gen_ai.system or openinference.span.kind=LLM
        if not (attrs.get("gen_ai.system") or attrs.get("openinference.span.kind") == "LLM"):
             # In strict GenAI convention, we might only care about LLM calls
             return None

        start_time = cls._parse_timestamp(span.get("start_time"))
        end_time = cls._parse_timestamp(span.get("end_time"))
        
        status_code = span.get("status", {}).get("code", 1)  # 1=OK normally in OTLP JSON? Check standard. 
        # Actually OTLP JSON: status: { code: 1 (OK), 2 (ERROR) }
        status = "OK"
        if status_code == 2:
            status = "ERROR"

        # Usage
        usage = TokenUsage(
            input_tokens=attrs.get("gen_ai.usage.input_tokens", 0),
            output_tokens=attrs.get("gen_ai.usage.output_tokens", 0)
        )

        return GenAISpan(
            trace_id=span.get("trace_id") or span.get("context", {}).get("trace_id", ""),
            span_id=span.get("span_id") or span.get("context", {}).get("span_id", ""),
            parent_span_id=span.get("parent_span_id") or span.get("context", {}).get("parent_id"),
            name=span.get("name", "llm"),
            kind="CLIENT",
            start_time=start_time,
            end_time=end_time,
            status=status,
            gen_ai_system=attrs.get("gen_ai.system") or attrs.get("llm.provider") or "unknown",
            gen_ai_request_model=attrs.get("gen_ai.request.model") or attrs.get("llm.model_name") or "unknown",
            gen_ai_response_model=attrs.get("gen_ai.response.model"),
            gen_ai_input_messages=cls._parse_otlp_messages(attrs, "input"),
            gen_ai_output_messages=cls._parse_otlp_messages(attrs, "output"),
            usage=usage,
            gen_ai_request_temperature=attrs.get("gen_ai.request.temperature"),
            gen_ai_request_max_tokens=attrs.get("gen_ai.request.max_tokens"),
            attributes=attrs
        )

    @classmethod
    def _parse_langsmith_run_recursive(cls, run: Dict[str, Any], spans: List[GenAISpan]):
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
             input_msgs = [GenAIMessage(role="user", parts=[MessagePart(type="text", content=inputs["prompt"])])]

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
             output_msgs.append(GenAIMessage(role="assistant", parts=[MessagePart(type="text", content=outputs["output"])]))

        return GenAISpan(
            trace_id=run.get("trace_id", str(run.get("id"))),
            span_id=str(run.get("id")),
            parent_span_id=str(run.get("parent_run_id")) if run.get("parent_run_id") else None,
            name=run.get("name", "llm"),
            kind="CLIENT",
            start_time=start_time,
            end_time=end_time,
            status="ERROR" if run.get("error") else "OK",
            gen_ai_system=invocation.get("_type", "unknown"),
            gen_ai_request_model=invocation.get("model_name") or invocation.get("model") or "unknown",
            gen_ai_input_messages=input_msgs,
            gen_ai_output_messages=output_msgs,
            usage=TokenUsage(
                input_tokens=run.get("prompt_tokens", 0),
                output_tokens=run.get("completion_tokens", 0) or extra.get("token_usage", {}).get("completion_tokens", 0)
            ),
            gen_ai_request_temperature=invocation.get("temperature"),
            attributes=extra
        )

    @classmethod
    def _parse_phoenix_span(cls, span: Dict[str, Any]) -> GenAISpan:
        attrs = span.get("attributes", {})
        start_time = cls._parse_timestamp(span.get("start_time"))
        end_time = cls._parse_timestamp(span.get("end_time"))
        
        # Phoenix attributes mapping
        # llm.input_messages is typically a list of dicts or JSON string
        input_msgs = cls._parse_phoenix_messages(attrs, "input")
        output_msgs = cls._parse_phoenix_messages(attrs, "output")
        
        return GenAISpan(
            trace_id=span.get("trace_id", ""),
            span_id=span.get("span_id", ""),
            parent_span_id=span.get("parent_id"),
            name=span.get("name", "llm"),
            kind="CLIENT",
            start_time=start_time,
            end_time=end_time,
            status="OK" if span.get("status", {}).get("status_code") == "OK" else "ERROR",
            gen_ai_system="unknown", # Phoenix often doesn't explicitly state provider separately from model
            gen_ai_request_model=attrs.get("llm.model_name", "unknown"),
            gen_ai_input_messages=input_msgs,
            gen_ai_output_messages=output_msgs,
            usage=TokenUsage(
                input_tokens=attrs.get("llm.token_count.prompt", 0),
                output_tokens=attrs.get("llm.token_count.completion", 0)
            ),
            gen_ai_request_temperature=attrs.get("llm.invocation_parameters", {}).get("temperature") if isinstance(attrs.get("llm.invocation_parameters"), dict) else None,
            attributes=attrs
        )

    # === Helpers ===

    @classmethod
    def _parse_timestamp(cls, ts: Any) -> datetime:
        if ts is None: return datetime.now(timezone.utc)
        if isinstance(ts, datetime): return ts
        if isinstance(ts, (int, float)):
            return datetime.fromtimestamp(ts if ts < 1e11 else ts/1e9, tz=timezone.utc)
        if isinstance(ts, str):
            try: return datetime.fromisoformat(ts.replace("Z", "+00:00"))
            except: pass
        return datetime.now(timezone.utc)

    @classmethod
    def _parse_otlp_messages(cls, attrs: Dict, direction: str) -> List[GenAIMessage]:
        # 1. Try standard gen_ai.{direction}.messages (list or JSON string)
        key = f"gen_ai.{direction}.messages"
        val = attrs.get(key)
        if val:
            if isinstance(val, str):
                try: val = json.loads(val)
                except: val = []
            if isinstance(val, list):
                # Parse list of dicts to GenAIMessages
                return cls._parse_list_of_message_dicts(val)
                
        # 2. Fallback to Phoenix/OpenInference flattened attributes
        # (reuse phoenix parser logic logic, but accessible here)
        return cls._parse_phoenix_messages(attrs, direction)

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
                    # Handle strict validation or loose dict? Pydantic handles loose dicts often
                    # We map fields manually to be safe against extra fields
                    if isinstance(p, dict):
                        parts.append(MessagePart(
                            type=p.get("type", "text"),
                            content=p.get("content"),
                            id=p.get("id"),
                            name=p.get("name"),
                            arguments=p.get("arguments"),
                            response=p.get("response")
                        ))

            # 2. Simplified 'content' string (fallback/legacy)
            elif "content" in m:
                content = m.get("content")
                if content:
                    parts.append(MessagePart(type="text", content=str(content)))

            results.append(GenAIMessage(role=role, parts=parts))
        return results

    @staticmethod
    def _parse_langchain_messages(messages: List[Union[Dict, Any]]) -> List[GenAIMessage]:
        result = []
        for m in messages:
            # LangSmith message dict
            if hasattr(m, "dict"): m = m.dict()
            if not isinstance(m, dict): continue
            
            role = "user"
            type_ = m.get("type", m.get("_type", "")).lower()
            if "system" in type_: role = "system"
            elif "ai" in type_ or "assistant" in type_: role = "assistant"
            elif "tool" in type_ or "function" in type_: role = "tool"
            
            parts = []
            content = m.get("content")
            if content:
                parts.append(MessagePart(type="text", content=content))
                
            # Tool calls in message?
            tool_calls = m.get("tool_calls", [])
            for tc in tool_calls:
                 parts.append(MessagePart(
                     type="tool_call",
                     id=tc.get("id"),
                     name=tc.get("name") or tc.get("function", {}).get("name"),
                     arguments=tc.get("args") or tc.get("function", {}).get("arguments")
                 ))
                 
            result.append(GenAIMessage(role=role, parts=parts))
        return result

    @staticmethod
    def _parse_phoenix_messages(attrs: Dict, direction: str) -> List[GenAIMessage]:
        # Phoenix uses attributes like 'llm.input_messages.0.message.content'
        # Need to iterate keys to reconstruct list
        messages = []
        prefix = "llm.input_messages" if direction == "input" else "llm.output_messages"
        
        # Simple heuristic: look for index 0..N
        i = 0
        while True:
            role_key = f"{prefix}.{i}.message.role"
            content_key = f"{prefix}.{i}.message.content"
            
            if role_key not in attrs and content_key not in attrs:
                 break
                 
            default_role = "user" if direction == "input" else "assistant"
            role = attrs.get(role_key, default_role)
            content = attrs.get(content_key, "")
            
            messages.append(GenAIMessage(
                role=role,
                parts=[MessagePart(type="text", content=content)]
            ))
            i += 1
            
        return messages

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
        parts.append(MessagePart(
             type="tool_call",
             id=tc.get("id"),
             name=tc.get("function", {}).get("name"),
             arguments=tc.get("function", {}).get("arguments")
        ))
        
    if not parts: return None
        
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
            if tid not in traces: traces[tid] = []
            traces[tid].append(span)
            
    # Convert each group
    conversations = []
    for tid, spans in traces.items():
        try:
            conversations.append(TraceConverter.from_otlp(spans, conversation_id=tid))
        except ValueError:
            continue
        
    return conversations
