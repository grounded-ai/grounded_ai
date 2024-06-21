from typing import List, Tuple
from pydantic import BaseModel, Field, field_validator, ValidationError

class EvaluationInstance(BaseModel):
    query: str
    text: str

class BaseEvaluationData(BaseModel):
    instances: List[Tuple[str, str]] = Field(..., min_items=1)

    @field_validator('instances')
    @classmethod
    def validate_instances(cls, instances):
        validated_instances = []
        for instance in instances:
            if len(instance) != 2:
                raise ValueError("Each instance must be a tuple of length 2 (query, text)")
            query, text = instance
            validated_instances.append(EvaluationInstance(query=query, text=text))
        return validated_instances