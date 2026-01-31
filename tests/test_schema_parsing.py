import unittest

from pydantic import computed_field

from grounded_ai.schemas import EvaluationInput, EvaluationOutput


class TestSchemaParsing(unittest.TestCase):
    def test_default_input_templating(self):
        """Test that the default EvaluationInput templates correctly."""
        input_data = EvaluationInput(
            response="This is a test submission.",
            query="Is this a test?",
            context="Testing context.",
        )

        # Strip whitespace for cleaner testing
        prompt = input_data.formatted_prompt.strip()

        # Check that all components are present in the default template
        self.assertIn("Query: Is this a test?", prompt)
        self.assertIn("Context: Testing context.", prompt)
        # Fix: New template logic uses strict fields
        self.assertIn("Response:", prompt)
        self.assertIn("This is a test submission.", prompt)

    def test_partial_fields_templating(self):
        """Test templating when optional fields are None."""
        input_data = EvaluationInput(response="Just text here.")

        prompt = input_data.formatted_prompt

        # In the new implementation, None fields are excluded entirely
        self.assertIn("Response:", prompt)
        self.assertIn("Just text here.", prompt)
        self.assertNotIn("Query:", prompt)
        self.assertNotIn("Context:", prompt)

    def test_runtime_template_override(self):
        """Test overriding the base_template at runtime instantiation."""
        # Update to use JINJA syntax {{ }}
        custom_tmpl = "STRICT FORMAT :: Q: {{ query }} -> A: {{ response }}"

        input_data = EvaluationInput(
            response="My Answer", query="My Question", base_template=custom_tmpl
        )

        expected = "STRICT FORMAT :: Q: My Question -> A: My Answer"
        self.assertEqual(input_data.formatted_prompt, expected)

    def test_custom_schema_inheritance(self):
        """Test defining a subclass with completely custom logic."""

        class TraceInput(EvaluationInput):
            trace_id: str
            span_content: str

            # Override the computed field logic entirely
            @computed_field
            @property
            def formatted_prompt(self) -> str:
                return f"Analyze Trace {self.trace_id}: {self.span_content}"

        trace_data = TraceInput(
            response="Ignored base text",  # Required by base but ignored in override
            trace_id="0x1234",
            span_content="DB Query failed",
        )

        self.assertEqual(
            trace_data.formatted_prompt, "Analyze Trace 0x1234: DB Query failed"
        )
        self.assertIsInstance(trace_data, EvaluationInput)

    def test_model_dump_includes_computed(self):
        """Ensure computed fields are included when dumping."""
        input_data = EvaluationInput(response="Dump test")

        dump = input_data.model_dump()

        # Check standard fields
        self.assertEqual(dump["response"], "Dump test")

        # computed_field should appear in the dump
        self.assertIn("formatted_prompt", dump)
        self.assertIn("Response:", dump["formatted_prompt"])
        self.assertIn("Dump test", dump["formatted_prompt"])

    def test_output_schema_instantiation(self):
        """Simple check that Output schema works as expected."""
        output = EvaluationOutput(
            score=0.9, label="pass", confidence=1.0, reasoning="Looks good"
        )
        self.assertEqual(output.score, 0.9)
        self.assertEqual(output.label, "pass")


if __name__ == "__main__":
    unittest.main()
