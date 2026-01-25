from abc import ABC, abstractmethod
from typing import Type, Any, Union, Dict
from pydantic import BaseModel

from .schemas import EvaluationInput, EvaluationOutput, EvaluationError


class BaseEvaluator(ABC):
    """
    Abstract base class for all backends (SLM, OpenAI, Anthropic, Custom).
    """

    def __init__(
        self,
        input_schema: Type[BaseModel] = None,
        output_schema: Type[BaseModel] = None,
    ):
        self._input_schema = input_schema
        self._output_schema = output_schema

    @property
    def input_schema(self) -> Type[BaseModel]:
        return self._input_schema or EvaluationInput

    @property
    def output_schema(self) -> Type[BaseModel]:
        return self._output_schema or EvaluationOutput

    def evaluate(
        self,
        input_data: Union[BaseModel, Dict[str, Any]],
        output_schema: Type[BaseModel] = None,
    ) -> Union[BaseModel, EvaluationError]:
        """
        Run the evaluation.

        Args:
            input_data: Either an instance of input_schema or a dictionary matching it.
            output_schema: Optional Pydantic model to override the default output_schema for this call.

        Returns:
            An instance of the effective output_schema or EvaluationError on failure.
        """
        # 1. Standardize Input
        if isinstance(input_data, dict):
            input_data = self.input_schema(**input_data)

        # 2. Resolve Output Schema
        target_schema = output_schema or self.output_schema

        # 3. Delegate to specific backend implementation
        return self._call_backend(input_data, target_schema)

    @abstractmethod
    def _call_backend(
        self, input_data: BaseModel, output_schema: Type[BaseModel]
    ) -> Union[BaseModel, EvaluationError]:
        """
        Implementation specific logic to call the model.

        Args:
            input_data: The validated input model instance.
            output_schema: The target output schema class to populate.
        """
        pass
