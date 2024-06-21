from typing import List, Optional, Tuple

from pydantic import BaseModel, Field, field_validator


class EvaluationInstance(BaseModel):
    query: str
    response: str
    reference: Optional[str] = None


class EvaluationData(BaseModel):
    instances: List[Tuple[str, str, Optional[str]]] = Field(..., min_items=1)

    @field_validator("instances")
    @classmethod
    def validate_instances(cls, instances):
        if not all(len(instance) == len(instances[0]) for instance in instances):
            raise ValueError(
                "All instances must have the same number of elements (2 or 3)"
            )

        validated_instances = []
        for instance in instances:
            if len(instance) == 2:
                query, response = instance
                validated_instances.append(
                    EvaluationInstance(query=query, response=response)
                )
            elif len(instance) == 3:
                query, response, reference = instance
                if reference:  # Only include reference if it's not None or empty string
                    validated_instances.append(
                        EvaluationInstance(
                            query=query, response=response, reference=reference
                        )
                    )
                else:
                    validated_instances.append(
                        EvaluationInstance(query=query, response=response)
                    )
            else:
                raise ValueError("Each instance must be a tuple of length 2 or 3")
        return validated_instances
