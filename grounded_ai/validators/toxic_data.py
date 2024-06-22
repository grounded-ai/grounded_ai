from typing import List

from pydantic import BaseModel, Field, model_validator


class EvaluationInstance(BaseModel):
    text: str


class ToxicityData(BaseModel):
    instances: List[EvaluationInstance] = Field(..., min_items=1)

    @model_validator(mode="before")
    @classmethod
    def validate_instances(cls, values: dict) -> dict:
        raw_instances = values.get("instances", [])
        validated_instances = []
        for i, instance in enumerate(raw_instances):
            if isinstance(instance, str):
                validated_instances.append(EvaluationInstance(text=instance))
            else:
                raise ValueError(f"Instance at index {i} must be a string")
        values["instances"] = validated_instances
        return values
