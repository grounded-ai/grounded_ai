from typing import List

from pydantic import BaseModel, Field, model_validator


class EvaluationInstance(BaseModel):
    query: str
    context: str


class RagData(BaseModel):
    instances: List[EvaluationInstance] = Field(..., min_items=1)

    @model_validator(mode="before")
    @classmethod
    def validate_instances(cls, values: dict) -> dict:
        raw_instances = values.get("instances", [])
        validated_instances = []
        for i, instance in enumerate(raw_instances):
            if isinstance(instance, (tuple, list)) and len(instance) == 2:
                query, context = instance
                if isinstance(query, str) and isinstance(context, str):
                    validated_instances.append(
                        EvaluationInstance(query=query, context=context)
                    )
                else:
                    raise ValueError(
                        f"Both query and context at index {i} must be strings"
                    )
            else:
                raise ValueError(
                    f"Instance at index {i} must be a tuple or list of length 2 (query, context)"
                )
        values["instances"] = validated_instances
        return values
