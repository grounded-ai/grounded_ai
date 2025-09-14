from typing import List, Optional

from pydantic import BaseModel, Field, model_validator, computed_field
import re

def extract_rating(response: str) -> Optional[int]:
    """
    Extracts a rating from the model's response.

    Args:
        response (str): The model's response containing the rating.

    Returns:
        Optional[int]: The extracted rating as an integer, or None if not found.
    """
    match = re.search(r"<rating>(.*?)</rating>", response)
    if match:
        return match.group(1)
    return None


def extract_reasoning(response: str) -> Optional[str]:
    """
    Extracts reasoning from the model's response.

    Args:
        response (str): The model's response containing the reasoning.

    Returns:
        Optional[str]: The extracted reasoning as a string, or None if not found.
    """
    match = re.search(r"<reasoning>(.*?)</reasoning>", response, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None

class OutputInstance(BaseModel):
    raw_response: str
    
    @computed_field
    @property
    def rating(self) -> str:
        return extract_rating(self.raw_response)
    
    @computed_field
    @property
    def reasoning(self) -> Optional[str]:
        return extract_reasoning(self.raw_response)
