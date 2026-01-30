import sys

# Pre-mock dependencies before imports
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import pytest

if TYPE_CHECKING:
    from grounded_ai.backends.grounded_ai_slm.backend import (
        EvalMode,
        GroundedAISLMBackend,
    )
else:
    # We will import these dynamically in the fixture to allow mocking imports
    GroundedAISLMBackend = None
    EvalMode = None

from grounded_ai import Evaluator
from grounded_ai.schemas import EvaluationOutput


class TestSLMBackend:
    @pytest.fixture(autouse=True)
    def mock_imports(self):
        """
        Safely mock heavy dependencies (torch, transformers, peft) for this test module only.
        This prevents polluting sys.modules for other integration tests.
        """
        mock_modules = {
            "torch": MagicMock(),
            "transformers": MagicMock(),
            "peft": MagicMock(),
        }

        # If the backend was already imported (e.g. by another test), we force a reload
        if "grounded_ai.backends.grounded_ai_slm.backend" in sys.modules:
            del sys.modules["grounded_ai.backends.grounded_ai_slm.backend"]

        with patch.dict(sys.modules, mock_modules):
            # Now safe to import
            from grounded_ai.backends.grounded_ai_slm.backend import (
                EvalMode as EM,
            )
            from grounded_ai.backends.grounded_ai_slm.backend import (
                GroundedAISLMBackend as GB,
            )

            # Inject into global namespace for tests to use (or attach to self/fixture)
            global GroundedAISLMBackend, EvalMode
            GroundedAISLMBackend = GB
            EvalMode = EM

            yield

        # Cleanup: we rely on patch.dict to restore sys.modules, but we might need to purge our tainted backend module
        if "grounded_ai.backends.grounded_ai_slm.backend" in sys.modules:
            del sys.modules["grounded_ai.backends.grounded_ai_slm.backend"]

    @pytest.fixture
    def mock_deps(self):
        """Fixture to patch the internal backend imports/calls if needed again."""
        with patch(
            "grounded_ai.backends.grounded_ai_slm.backend.pipeline"
        ) as mock_pipeline, patch(
            "grounded_ai.backends.grounded_ai_slm.backend.AutoModelForCausalLM"
        ) as mock_automodel, patch(
            "grounded_ai.backends.grounded_ai_slm.backend.AutoTokenizer"
        ) as mock_tokenizer, patch(
            "grounded_ai.backends.grounded_ai_slm.backend.PeftModel"
        ) as mock_peft, patch(
            "grounded_ai.backends.grounded_ai_slm.backend.PeftConfig"
        ):
            # Setup default behavior for pipeline
            mock_generator = MagicMock()
            # Simulation of model output for "Accurate" -> Faithful
            mock_output = [
                {
                    "generated_text": [
                        {"role": "user", "content": "..."},
                        {
                            "role": "assistant",
                            "content": "<rating>Accurate</rating><reasoning>Matches context.</reasoning>",
                        },
                    ]
                }
            ]
            mock_generator.return_value = mock_output
            mock_pipeline.return_value = mock_generator

            yield {
                "pipeline": mock_pipeline,
                "automodel": mock_automodel,
                "tokenizer": mock_tokenizer,
                "peft": mock_peft,
            }

    def test_system_prompt_override(self, mock_deps):
        """Test that system prompt can be overridden."""
        # Setup mock
        mock_deps["pipeline"].return_value.return_value = [
            {
                "generated_text": [
                    {
                        "role": "assistant",
                        "content": "<rating>Accurate</rating><reasoning>OK</reasoning>",
                    }
                ]
            }
        ]

        evaluator = Evaluator(
            "grounded-ai/hallucination-v1",
            eval_mode="HALLUCINATION",
            system_prompt="You are a strict judge.",
        )

        evaluator.evaluate(response="test")

        # Verify call args on the pipeline instance, not the factory
        # mock_deps["pipeline"] creates the instance (return_value)
        pipeline_instance = mock_deps["pipeline"].return_value
        call_args = pipeline_instance.call_args
        messages = call_args[0][0]  # first arg to passed to pipeline() call
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == "You are a strict judge."

    def test_factory_routing(self, mock_deps):
        """Test that the factory routes 'grounded-ai/*' to SLM backend."""
        evaluator = Evaluator("grounded-ai/hallucination-v1", eval_mode="hallucination")
        assert isinstance(evaluator.backend, GroundedAISLMBackend)
        assert evaluator.backend.task == EvalMode.HALLUCINATION

    def test_hallucination_flow(self, mock_deps):
        """Test a full hallucination evaluation flow."""
        evaluator = Evaluator(
            "grounded-ai/hallucination-v1", eval_mode=EvalMode.HALLUCINATION
        )

        result = evaluator.evaluate(
            response="London is the capital.",  # This maps to RESPONSE in hallucination
            query="What is the capital?",
            context="London is the capital of UK.",
        )

        assert isinstance(result, EvaluationOutput)
        assert result.score == 0.0  # Accurate -> 0.0 hallucination score
        assert result.label == "faithful"
        assert result.reasoning == "Matches context."

    def test_toxicity_flow(self, mock_deps):
        """Test toxicity flow logic."""
        # Setup mock to return 'Toxic'
        mock_deps["pipeline"].return_value.return_value = [
            {
                "generated_text": [
                    {
                        "role": "assistant",
                        "content": "<rating>Toxic</rating><reasoning>Rude language.</reasoning>",
                    }
                ]
            }
        ]

        evaluator = Evaluator("grounded-ai/toxic-judge-v1", eval_mode="TOXICITY")
        assert evaluator.backend.task == EvalMode.TOXICITY

        result = evaluator.evaluate(response="You are stupid.")

        assert result.score == 1.0  # Toxic -> 1.0
        assert result.label == "toxic"

    def test_rag_flow(self, mock_deps):
        """Test RAG flow logic."""
        # Setup mock to return 'Relevant'
        mock_deps["pipeline"].return_value.return_value = [
            {
                "generated_text": [
                    {
                        "role": "assistant",
                        "content": "<rating>Relevant</rating><reasoning>Direct answer.</reasoning>",
                    }
                ]
            }
        ]

        evaluator = Evaluator(
            "grounded-ai/rag-relevance-v1", eval_mode=EvalMode.RAG_RELEVANCE
        )
        assert evaluator.backend.task == EvalMode.RAG_RELEVANCE

        result = evaluator.evaluate(
            query="Question?",
            response="Document content.",  # 'text' is the reference text in RAG relevance inputs
        )

        assert result.score == 1.0  # Relevant -> 1.0
        assert result.label == "relevant"
