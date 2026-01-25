import unittest
from pydantic import BaseModel, computed_field
from grounded_ai.schemas import EvaluationInput, EvaluationOutput

class TestSchemaParsing(unittest.TestCase):
    
    def test_default_input_templating(self):
        """Test that the default EvaluationInput templates correctly."""
        input_data = EvaluationInput(
            response="This is a test submission.",
            query="Is this a test?",
            conresponse="Testing context.",
            reference="Test reference."
        )
        
        # Strip whitespace for cleaner testing
        prompt = input_data.formatted_prompt.strip()
        
        # Check that all components are present in the default template
        self.assertIn("Query: Is this a test?", prompt)
        self.assertIn("Context: Testing context.", prompt)
        self.assertIn("Reference: Test reference.", prompt)
        # Fix: assertIn is loose enough for substrings, but let's be safe regarding newlines
        self.assertIn("Content to Evaluate:", prompt)
        self.assertIn("This is a test submission.", prompt)

    def test_partial_fields_templating(self):
        """Test templating when optional fields are None."""
        input_data = EvaluationInput(response="Just text here.")
        
        prompt = input_data.formatted_prompt
        
        # In the new implementation, None fields are excluded entirely
        self.assertIn("Content to Evaluate:", prompt)
        self.assertIn("Just text here.", prompt)
        self.assertNotIn("Query:", prompt) 
        self.assertNotIn("Context:", prompt)

    def test_runtime_template_override(self):
        """Test overriding the base_template at runtime instantiation."""
        # Update to use JINJA syntax {{ }}
        custom_tmpl = "STRICT FORMAT :: Q: {{ query }} -> A: {{ text }}"
        
        input_data = EvaluationInput(
            response="My Answer",
            query="My Question",
            base_template=custom_tmpl
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
            response="Ignored base text", # Required by base but ignored in override
            trace_id="0x1234",
            span_content="DB Query failed"
        )
        
        self.assertEqual(trace_data.formatted_prompt, "Analyze Trace 0x1234: DB Query failed")
        
        # Ensure it still looks like an EvaluationInput for duck typing if needed
        self.assertIsInstance(trace_data, EvaluationInput)

    def test_model_dump_includes_computed(self):
        """Ensure computed fields are included when dumping."""
        input_data = EvaluationInput(response="Dump test")
        
        # model_dump() usually doesn't include computed fields by default in older v2
        # but usage in code might expect access via property. 
        # Let's check if we strictly need it in the dict.
        # Actually, standard pydantic v2 'model_dump' does NOT include computed fields unless requested?
        # Wait, @computed_field decorator DOES include it in serialization (dump types 'json' or 'python').
        
        dump = input_data.model_dump()
        
        # Check standard fields
        self.assertEqual(dump['response'], "Dump test")
        
        # computed_field should appear in the dump
        self.assertIn('formatted_prompt', dump)
        self.assertIn("Content to Evaluate:", dump['formatted_prompt'])
        self.assertIn("Dump test", dump['formatted_prompt'])

    def test_output_schema_instantiation(self):
        """Simple check that Output schema works as expected."""
        output = EvaluationOutput(
            score=0.9,
            label="pass",
            confidence=1.0,
            reasoning="Looks good"
        )
        self.assertEqual(output.score, 0.9)
        self.assertEqual(output.label, "pass")

if __name__ == '__main__':
    unittest.main()
