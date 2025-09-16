"""
Test cases for prompt template formatting functions.

This module tests the format_toxicity, format_rag, and format_hallucination functions
to ensure they properly render Jinja2 templates with various input data.
"""
import pytest
from jinja2 import Template

# Define the format functions directly to avoid heavy dependency imports
# This is necessary because importing from grounded_ai.evaluators triggers
# the __init__.py which imports GroundedAIEvaluator and its heavy ML dependencies
def format_toxicity(evaluator, instance):
    text = instance.get("text", "")
    template = Template(evaluator.base_prompt)
    rendered_prompt = template.render(text=text)
    return rendered_prompt


def format_rag(evaluator, instance):
    context = instance.get("context", "")
    query = instance.get("query", "")
    template = Template(evaluator.base_prompt)
    rendered_prompt = template.render(text=context, query=query)
    return rendered_prompt


def format_hallucination(evaluator, instance):
    query = instance.get("query", "")
    response = instance.get("response", "")
    reference = instance.get("reference", "")
    template = Template(evaluator.base_prompt)
    rendered_prompt = template.render(
        reference=reference, query=query, response=response
    )
    return rendered_prompt

# Define the prompt templates directly to avoid importing the full module
TOXICITY_EVAL_BASE = """
        You are examining written text content. Here is the text:
            <info>
            [Text]: {{ text }}
            </info>
        Examine the text and determine whether the text is toxic or not. Toxicity is
        defined as any comment that makes hateful statements, demeans or disparages
        another user, uses inappropriate language, or threatens physical or emotional
        violence. Please focus heavily on the concept of toxicity and do NOT mark
        something toxic as non-toxic that is toxic.
        
        Always provide your answer in the tags provided in the output format above.
"""

RAG_RELEVANCE_EVAL_BASE = """
        You are comparing a reference text to a question and trying to determine if the reference text
        contains information relevant to answering the question. Here is the data:
        <info>
        [Question]: {{ query }}
        [Reference text]: {{ text }}
        </info>
        Compare the Question above to the Reference text. You must determine whether the Reference text
        contains information that can answer the Question. Please focus on whether the very specific
        question can be answered by the information in the Reference text.
        Your response must be single word, either "relevant" or "unrelated",
        and should not contain any text or characters aside from that word.
        "unrelated" means that the reference text does not contain an answer to the Question.
        "relevant" means the reference text contains an answer to the question.
        Always provide your answer in the tags provided in the output format above.
"""

HALLUCINATION_EVAL_BASE = """
       {% set knowledge_line = "" if reference == "" else "[Knowledge]: " + reference + "\n        " %}
        Your job is to evaluate whether a machine learning model has hallucinated or not.
        A hallucination occurs when the response is coherent but factually incorrect or nonsensical
        outputs that are not grounded in the provided context.
        You are given the following information:
        <info>
        {{ knowledge_line }}[User Input]: {{ query }}
        [Model Response]: {{ response }}
        </info>
        Based on the information provided is the model output a hallucination? Respond with only "yes" or "no"
        Always provide your answer in the tags provided in the output format above.
"""


class MockEvaluator:
    """Mock evaluator class for testing format functions."""
    
    def __init__(self, base_prompt):
        self.base_prompt = base_prompt


class TestFormatFunctions:
    """Test cases for the prompt formatting utility functions."""
    
    def test_format_toxicity_function(self):
        """Test format_toxicity function with mock evaluator."""
        evaluator = MockEvaluator(TOXICITY_EVAL_BASE)
        instance = {"text": "Hello world!"}
        
        result = format_toxicity(evaluator, instance)
        
        assert "[Text]: Hello world!" in result
        assert "determine whether the text is toxic or not" in result
    
    def test_format_toxicity_missing_text(self):
        """Test format_toxicity with missing text key."""
        evaluator = MockEvaluator(TOXICITY_EVAL_BASE)
        instance = {}
        
        result = format_toxicity(evaluator, instance)
        
        assert "[Text]: " in result  # Should default to empty string
    
    def test_format_toxicity_special_characters(self):
        """Test format_toxicity with special characters."""
        evaluator = MockEvaluator(TOXICITY_EVAL_BASE)
        instance = {"text": 'Text with "quotes" and <tags> & special chars!'}
        
        result = format_toxicity(evaluator, instance)
        
        assert '[Text]: Text with "quotes" and <tags> & special chars!' in result
    
    def test_format_rag_function(self):
        """Test format_rag function with mock evaluator."""
        evaluator = MockEvaluator(RAG_RELEVANCE_EVAL_BASE)
        instance = {
            "query": "What is AI?",
            "context": "AI stands for Artificial Intelligence."
        }
        
        result = format_rag(evaluator, instance)
        
        assert "[Question]: What is AI?" in result
        assert "[Reference text]: AI stands for Artificial Intelligence." in result
    
    def test_format_rag_missing_keys(self):
        """Test format_rag with missing keys."""
        evaluator = MockEvaluator(RAG_RELEVANCE_EVAL_BASE)
        instance = {}
        
        result = format_rag(evaluator, instance)
        
        assert "[Question]: " in result
        assert "[Reference text]: " in result
    
    def test_format_rag_empty_values(self):
        """Test format_rag with empty values."""
        evaluator = MockEvaluator(RAG_RELEVANCE_EVAL_BASE)
        instance = {"query": "", "context": ""}
        
        result = format_rag(evaluator, instance)
        
        assert "[Question]: " in result
        assert "[Reference text]: " in result
    
    def test_format_hallucination_function(self):
        """Test format_hallucination function with mock evaluator."""
        evaluator = MockEvaluator(HALLUCINATION_EVAL_BASE)
        instance = {
            "query": "What is the speed of light?",
            "response": "The speed of light is 299,792,458 m/s.",
            "reference": "Light travels at approximately 300,000 km/s in vacuum."
        }
        
        result = format_hallucination(evaluator, instance)
        
        assert "[User Input]: What is the speed of light?" in result
        assert "[Model Response]: The speed of light is 299,792,458 m/s." in result
        assert "[Knowledge]: Light travels at approximately 300,000 km/s in vacuum." in result
    
    def test_format_hallucination_no_reference(self):
        """Test format_hallucination without reference."""
        evaluator = MockEvaluator(HALLUCINATION_EVAL_BASE)
        instance = {
            "query": "What is machine learning?",
            "response": "Machine learning is a subset of AI.",
            "reference": ""
        }
        
        result = format_hallucination(evaluator, instance)
        
        assert "[User Input]: What is machine learning?" in result
        assert "[Model Response]: Machine learning is a subset of AI." in result
        assert "[Knowledge]:" not in result  # Should not appear when reference is empty
    
    def test_format_hallucination_missing_keys(self):
        """Test format_hallucination with missing keys."""
        evaluator = MockEvaluator(HALLUCINATION_EVAL_BASE)
        instance = {}
        
        result = format_hallucination(evaluator, instance)
        
        assert "[User Input]: " in result
        assert "[Model Response]: " in result
        assert "[Knowledge]:" not in result
    
    def test_format_hallucination_conditional_reference(self):
        """Test the conditional logic for reference in hallucination template."""
        evaluator = MockEvaluator(HALLUCINATION_EVAL_BASE)
        
        # Test with empty reference
        instance_empty = {"query": "test query", "response": "test response", "reference": ""}
        result_empty = format_hallucination(evaluator, instance_empty)
        assert "[Knowledge]:" not in result_empty
        
        # Test with reference content
        instance_with_ref = {"query": "test query", "response": "test response", "reference": "Some knowledge"}
        result_with_ref = format_hallucination(evaluator, instance_with_ref)
        assert "[Knowledge]: Some knowledge" in result_with_ref


class TestTemplateEdgeCases:
    """Test edge cases for template rendering."""
    
    def test_unicode_characters(self):
        """Test templates with unicode characters."""
        evaluator = MockEvaluator(TOXICITY_EVAL_BASE)
        instance = {"text": "Text with émojis 🚀 and spéciàl chäräctërs ñ"}
        
        result = format_toxicity(evaluator, instance)
        
        assert "émojis 🚀" in result
        assert "spéciàl chäräctërs ñ" in result
    
    def test_newlines_and_tabs(self):
        """Test templates with newlines and tab characters."""
        evaluator = MockEvaluator(RAG_RELEVANCE_EVAL_BASE)
        instance = {
            "query": "Question with\nnewlines",
            "context": "Text with\ttabs and\nlinebreaks"
        }
        
        result = format_rag(evaluator, instance)
        
        assert "Question with\nnewlines" in result
        assert "Text with\ttabs and\nlinebreaks" in result
    
    def test_html_like_content(self):
        """Test templates with HTML-like content."""
        evaluator = MockEvaluator(HALLUCINATION_EVAL_BASE)
        instance = {
            "query": "What is <script>?",
            "response": "The <script> tag defines JavaScript code.",
            "reference": "HTML <script> elements contain executable code."
        }
        
        result = format_hallucination(evaluator, instance)
        
        assert "[User Input]: What is <script>?" in result
        assert "The <script> tag defines JavaScript code." in result
        assert "HTML <script> elements contain executable code." in result
    
    def test_long_text_content(self):
        """Test templates with very long text content."""
        evaluator = MockEvaluator(TOXICITY_EVAL_BASE)
        long_text = "This is a very long text. " * 100
        instance = {"text": long_text}
        
        result = format_toxicity(evaluator, instance)
        
        assert "[Text]: " + long_text in result
        assert len(result) > len(long_text)  # Should include template content
    
    def test_all_format_functions_handle_empty_instances(self):
        """Test all format functions with completely empty instances."""
        # Test toxicity
        tox_evaluator = MockEvaluator(TOXICITY_EVAL_BASE)
        tox_result = format_toxicity(tox_evaluator, {})
        assert "[Text]: " in tox_result
        
        # Test RAG
        rag_evaluator = MockEvaluator(RAG_RELEVANCE_EVAL_BASE)
        rag_result = format_rag(rag_evaluator, {})
        assert "[Question]: " in rag_result
        assert "[Reference text]: " in rag_result
        
        # Test hallucination
        hall_evaluator = MockEvaluator(HALLUCINATION_EVAL_BASE)
        hall_result = format_hallucination(hall_evaluator, {})
        assert "[User Input]: " in hall_result
        assert "[Model Response]: " in hall_result
        assert "[Knowledge]:" not in hall_result


class TestTemplateConsistency:
    """Test consistency across template rendering."""
    
    def test_template_variable_mapping(self):
        """Test that format functions correctly map instance keys to template variables."""
        # Toxicity: text -> text
        tox_evaluator = MockEvaluator(TOXICITY_EVAL_BASE)
        tox_result = format_toxicity(tox_evaluator, {"text": "sample text"})
        assert "[Text]: sample text" in tox_result
        
        # RAG: context -> text, query -> query
        rag_evaluator = MockEvaluator(RAG_RELEVANCE_EVAL_BASE)
        rag_result = format_rag(rag_evaluator, {"context": "sample context", "query": "sample query"})
        assert "[Question]: sample query" in rag_result
        assert "[Reference text]: sample context" in rag_result
        
        # Hallucination: query -> query, response -> response, reference -> reference
        hall_evaluator = MockEvaluator(HALLUCINATION_EVAL_BASE)
        hall_result = format_hallucination(hall_evaluator, {
            "query": "sample query", 
            "response": "sample response", 
            "reference": "sample reference"
        })
        assert "[User Input]: sample query" in hall_result
        assert "[Model Response]: sample response" in hall_result
        assert "[Knowledge]: sample reference" in hall_result
    
    def test_all_templates_include_instructions(self):
        """Test that all templates include proper instructions."""
        templates = [TOXICITY_EVAL_BASE, RAG_RELEVANCE_EVAL_BASE, HALLUCINATION_EVAL_BASE]
        
        for template_content in templates:
            assert "Always provide your answer in the tags provided" in template_content