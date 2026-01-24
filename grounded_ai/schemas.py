from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

# --- Standard Evaluation Schemas ---

class EvaluationInput(BaseModel):
    """Standard input for Grounded AI evaluators."""
    text: str
    context: Optional[str] = None
    reference: Optional[str] = None
    query: Optional[str] = None

class EvaluationOutput(BaseModel):
    """Standard output for Grounded AI evaluators."""
    score: float = Field(ge=0.0, le=1.0, description="Numerical score between 0 and 1")
    label: str = Field(description="Classification label (e.g., 'hallucination', 'faithful')")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence score of the evaluation")
    reasoning: Optional[str] = Field(None, description="Explanation for the evaluation result")

class EvaluationError(BaseModel):
    """Standard error response when evaluation fails."""
    error_code: str = Field(description="Short code identifying the error type (e.g., 'PROVIDER_ERROR', 'RATE_LIMIT')")
    message: str = Field(description="Human-readable error message")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional context about the error")

# --- Trace / OTEL Schemas (formerly trace_data.py) ---

class LLMMessage(BaseModel):
    role: str = Field(..., description="The role of the message author (e.g., 'system', 'user', 'assistant')")
    content: str = Field(..., description="The content of the message")

class LLMUsage(BaseModel):
    prompt_tokens: int = Field(0, alias="gen_ai.usage.input_tokens")
    completion_tokens: int = Field(0, alias="gen_ai.usage.output_tokens")
    total_tokens: int = Field(0)

class LLMOTelLog(BaseModel):
    """GenAI Semantic Conventions (Version 2) compliant log."""
    model_name: str = Field(..., alias="gen_ai.request.model")
    system_instructions: Optional[str] = Field(None, alias="gen_ai.system_instructions")
    input_messages: List[LLMMessage] = Field(..., alias="gen_ai.input.messages")
    output_messages: List[LLMMessage] = Field(..., alias="gen_ai.output.messages")
    
    # Metadata and Performance
    latency_ms: float = Field(..., description="Round-trip time in milliseconds")
    temperature: Optional[float] = Field(None, alias="gen_ai.request.temperature")
    usage: LLMUsage
    
    metadata: Dict[str, Any] = Field(default_factory=dict)
