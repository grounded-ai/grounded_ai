from typing import List, Optional

from pydantic import BaseModel, Field, model_validator


class EvaluationInstance(BaseModel):
    query: str
    response: str
    reference: Optional[str] = None


class HallucinationData(BaseModel):
    instances: List[EvaluationInstance] = Field(..., min_items=1)

    @model_validator(mode="before")
    @classmethod
    def validate_instances(cls, values: dict) -> dict:
        raw_instances = values.get("instances", [])
        if not raw_instances:
            raise ValueError("At least one instance is required")

        # Check that all instances have the same number of elements
        instance_length = len(raw_instances[0])
        if not all(len(instance) == instance_length for instance in raw_instances):
            raise ValueError(
                "All instances must have the same number of elements (2 or 3)"
            )

        validated_instances = []
        for i, instance in enumerate(raw_instances):
            if not isinstance(instance, (tuple, list)):
                raise ValueError(f"Instance at index {i} must be a tuple or list")

            if len(instance) == 2:
                query, response = instance
                if not isinstance(query, str) or not isinstance(response, str):
                    raise ValueError(f"Query and response at index {i} must be strings")
                validated_instances.append(
                    EvaluationInstance(query=query, response=response)
                )
            elif len(instance) == 3:
                query, response, reference = instance
                if (
                    not isinstance(query, str)
                    or not isinstance(response, str)
                    or (reference is not None and not isinstance(reference, str))
                ):
                    raise ValueError(
                        f"Query, response, and reference (if provided) at index {i} must be strings"
                    )
                validated_instances.append(
                    EvaluationInstance(
                        query=query, response=response, reference=reference
                    )
                )
            else:
                raise ValueError(
                    f"Instance at index {i} must be a tuple or list of length 2 or 3"
                )

        values["instances"] = validated_instances
        return values
