from abc import ABC, abstractmethod
from typing import Type, Any, Union, Dict
from pydantic import BaseModel

from .schemas import EvaluationInput, EvaluationOutput, EvaluationError

class BaseEvaluator(ABC):
    """
    Abstract base class for all backends (SLM, OpenAI, Anthropic, Custom).
    """
    
    # ... (properties unchanged) ...

    @abstractmethod
    def evaluate(self, input_data: Union[BaseModel, Dict[str, Any]]) -> Union[EvaluationOutput, EvaluationError]:
        """
        Run the evaluation.
        
        Args:
            input_data: Either an instance of input_schema or a dictionary matching it.
            
        Returns:
            An instance of EvaluationOutput or EvaluationError on failure.
        """
        pass
