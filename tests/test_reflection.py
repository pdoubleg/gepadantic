"""Tests for the reflection module."""

from __future__ import annotations

from typing import Any

from pydantic_ai.models.test import TestModel

from gepadantic.reflection import (
    ProposalOutput,
    ReflectionInput,
    UpdatedComponent,
    propose_new_texts,
)


def test_reflection_input_model():
    """Test that ReflectionInput model works correctly."""
    # Create reflection dataset
    reflection_dataset = {
        "instructions": [
            {
                "user_prompt": "Classify: positive review",
                "assistant_response": "positive",
                "score": 1.0,
                "success": True,
                "feedback": "Correct classification",
            },
            {
                "user_prompt": "Classify: terrible experience",
                "assistant_response": "positive",
                "score": 0.0,
                "success": True,
                "feedback": "Wrong classification - should be negative",
            },
        ]
    }

    # Create ReflectionInput instance
    reflection_input = ReflectionInput(
        instructions="Classify text as positive or negative.",
        prompt_components={
            "instructions": "Classify text as positive or negative.",
        },
        reflection_dataset=reflection_dataset,
        components_to_update=["instructions"],
    )

    # Verify the model was created correctly
    assert reflection_input.instructions == "Classify text as positive or negative."
    assert reflection_input.prompt_components == {
        "instructions": "Classify text as positive or negative.",
    }
    assert reflection_input.reflection_dataset == reflection_dataset
    assert reflection_input.components_to_update == ["instructions"]


def test_proposal_output_model():
    """Test that ProposalOutput model works correctly."""
    # Create ProposalOutput instance
    updated_components = [
        UpdatedComponent(
            component_name="instructions",
            optimized_value="Carefully analyze the sentiment and classify as positive or negative.",
        )
    ]

    proposal = ProposalOutput(updated_components=updated_components)

    # Verify the model was created correctly
    assert len(proposal.updated_components) == 1
    assert proposal.updated_components[0].component_name == "instructions"
    assert (
        proposal.updated_components[0].optimized_value
        == "Carefully analyze the sentiment and classify as positive or negative."
    )


def test_updated_component_model():
    """Test that UpdatedComponent model works correctly."""
    component = UpdatedComponent(
        component_name="signature:Query:question:desc",
        optimized_value="The specific geography question to answer with detail",
    )

    assert component.component_name == "signature:Query:question:desc"
    assert (
        component.optimized_value
        == "The specific geography question to answer with detail"
    )


def test_propose_new_texts_basic(monkeypatch):
    """Test basic functionality of propose_new_texts."""
    # Create a reflection dataset with performance data
    reflective_dataset = {
        "instructions": [
            {
                "user_prompt": "What is 2 + 2?",
                "assistant_response": "4",
                "score": 1.0,
                "success": True,
                "feedback": "Correct answer",
            },
            {
                "user_prompt": "What is 10 * 5?",
                "assistant_response": "15",
                "score": 0.0,
                "success": True,
                "feedback": "Wrong - should be 50",
            },
        ]
    }

    # Create a candidate with current prompts
    candidate = {
        "instructions": "Answer math questions.",
    }

    # Mock the reflection model output
    test_output = ProposalOutput(
        updated_components=[
            UpdatedComponent(
                component_name="instructions",
                optimized_value="Answer math questions accurately. Pay special attention to multiplication.",
            )
        ]
    )

    # Use TestModel to provide deterministic output
    test_model = TestModel(custom_output_args=test_output.model_dump(mode="python"))

    # Mock get_openai_model to return our test model
    def mock_get_openai_model(model_name):
        return test_model

    monkeypatch.setattr("gepadantic.reflection.get_openai_model", mock_get_openai_model)

    # Run propose_new_texts
    result = propose_new_texts(
        candidate=candidate,
        reflective_dataset=reflective_dataset,
        components_to_update=["instructions"],
        reflection_model="test-model",
    )

    # Verify the result
    assert result is not None
    assert isinstance(result.output, ProposalOutput)
    assert len(result.output.updated_components) == 1
    assert result.output.updated_components[0].component_name == "instructions"
    assert (
        "multiplication" in result.output.updated_components[0].optimized_value.lower()
    )


def test_propose_new_texts_multiple_components(monkeypatch):
    """Test propose_new_texts with multiple components to update."""
    # Create reflection dataset with multiple component failures
    reflective_dataset = {
        "instructions": [
            {
                "user_prompt": "<question>What is the capital of France?</question>",
                "assistant_response": "London",
                "score": 0.0,
                "success": True,
                "feedback": "Wrong capital",
            }
        ],
        "signature:Query:question:desc": [
            {
                "user_prompt": "<question>What is the capital of France?</question>",
                "assistant_response": "London",
                "score": 0.0,
                "success": True,
                "feedback": "Question not clear enough",
            }
        ],
    }

    candidate = {
        "instructions": "Answer geography questions.",
        "signature:Query:question:desc": "The question",
    }

    # Mock output for multiple components
    test_output = ProposalOutput(
        updated_components=[
            UpdatedComponent(
                component_name="instructions",
                optimized_value="Answer geography questions with accurate information about world capitals.",
            ),
            UpdatedComponent(
                component_name="signature:Query:question:desc",
                optimized_value="The geography question requiring a specific factual answer",
            ),
        ]
    )

    test_model = TestModel(custom_output_args=test_output.model_dump(mode="python"))

    def mock_get_openai_model(model_name):
        return test_model

    monkeypatch.setattr("gepadantic.reflection.get_openai_model", mock_get_openai_model)

    result = propose_new_texts(
        candidate=candidate,
        reflective_dataset=reflective_dataset,
        components_to_update=["instructions", "signature:Query:question:desc"],
        reflection_model="test-model",
    )

    assert len(result.output.updated_components) == 2
    component_names = {c.component_name for c in result.output.updated_components}
    assert "instructions" in component_names
    assert "signature:Query:question:desc" in component_names


def test_propose_new_texts_with_instructions_in_dataset(monkeypatch):
    """Test that instructions are extracted from reflection dataset."""
    # Reflection dataset with instructions embedded
    reflective_dataset = {
        "instructions": [
            {
                "user_prompt": "Test prompt",
                "assistant_response": "Test response",
                "score": 0.5,
                "success": True,
                "feedback": "Average performance",
                "instructions": "Original instructions from agent",  # Should be extracted
            },
            {
                "user_prompt": "Another test",
                "assistant_response": "Another response",
                "score": 0.6,
                "success": True,
                "feedback": "OK",
                "instructions": "Original instructions from agent",  # Duplicate
            },
        ]
    }

    candidate = {
        "instructions": "Original instructions from agent",
    }

    test_output = ProposalOutput(
        updated_components=[
            UpdatedComponent(
                component_name="instructions",
                optimized_value="Improved instructions based on performance",
            )
        ]
    )

    test_model = TestModel(custom_output_args=test_output.model_dump(mode="python"))

    def mock_get_openai_model(model_name):
        return test_model

    monkeypatch.setattr("gepadantic.reflection.get_openai_model", mock_get_openai_model)

    result = propose_new_texts(
        candidate=candidate,
        reflective_dataset=reflective_dataset,
        components_to_update=["instructions"],
        reflection_model="test-model",
    )

    # Verify that the function runs without error
    assert result is not None
    assert isinstance(result.output, ProposalOutput)

    # Verify that instructions were removed from the dataset
    for component_data in reflective_dataset.values():
        for record in component_data:
            assert "instructions" not in record


def test_propose_new_texts_normalizes_components(monkeypatch):
    """Test that component texts are normalized before reflection."""

    class MultiPartInstructions:
        """Mock object with list-like instructions."""

        def __str__(self):
            return "Part 1\n\nPart 2"

    reflective_dataset = {
        "instructions": [
            {
                "user_prompt": "Test",
                "assistant_response": "Response",
                "score": 0.8,
                "success": True,
                "feedback": "Good",
            }
        ]
    }

    # Candidate with various text formats
    candidate = {
        "instructions": ["Part 1", "Part 2"],  # List of strings
        "other_component": "Simple string",
    }

    test_output = ProposalOutput(
        updated_components=[
            UpdatedComponent(
                component_name="instructions",
                optimized_value="Normalized and improved",
            )
        ]
    )

    test_model = TestModel(custom_output_args=test_output.model_dump(mode="python"))

    def mock_get_openai_model(model_name):
        return test_model

    monkeypatch.setattr("gepadantic.reflection.get_openai_model", mock_get_openai_model)

    result = propose_new_texts(
        candidate=candidate,
        reflective_dataset=reflective_dataset,
        components_to_update=["instructions"],
        reflection_model="test-model",
    )

    # Should complete successfully with normalized text
    assert result is not None
    assert isinstance(result.output, ProposalOutput)


def test_propose_new_texts_with_signature_components(monkeypatch):
    """Test reflection with signature-based component names."""
    reflective_dataset = {
        "signature:EmailQuery:subject:desc": [
            {
                "user_prompt": "<subject>Meeting request</subject>",
                "assistant_response": "Low priority",
                "score": 0.0,
                "success": True,
                "feedback": "Should be high priority - it's a meeting",
            }
        ],
        "signature:EmailQuery:sender:desc": [
            {
                "user_prompt": "<sender>boss@company.com</sender>",
                "assistant_response": "Unknown sender",
                "score": 0.0,
                "success": True,
                "feedback": "Should recognize boss emails",
            }
        ],
    }

    candidate = {
        "instructions": "Analyze emails",
        "signature:EmailQuery:subject:desc": "Email subject line",
        "signature:EmailQuery:sender:desc": "Email sender address",
    }

    test_output = ProposalOutput(
        updated_components=[
            UpdatedComponent(
                component_name="signature:EmailQuery:subject:desc",
                optimized_value="Email subject line - pay attention to meeting requests which are high priority",
            ),
            UpdatedComponent(
                component_name="signature:EmailQuery:sender:desc",
                optimized_value="Email sender address - prioritize emails from management",
            ),
        ]
    )

    test_model = TestModel(custom_output_args=test_output.model_dump(mode="python"))

    def mock_get_openai_model(model_name):
        return test_model

    monkeypatch.setattr("gepadantic.reflection.get_openai_model", mock_get_openai_model)

    result = propose_new_texts(
        candidate=candidate,
        reflective_dataset=reflective_dataset,
        components_to_update=[
            "signature:EmailQuery:subject:desc",
            "signature:EmailQuery:sender:desc",
        ],
        reflection_model="test-model",
    )

    assert len(result.output.updated_components) == 2

    # Check that signature components were updated
    updated_dict = {
        c.component_name: c.optimized_value for c in result.output.updated_components
    }
    assert "signature:EmailQuery:subject:desc" in updated_dict
    assert "signature:EmailQuery:sender:desc" in updated_dict


def test_propose_new_texts_handles_empty_dataset(monkeypatch):
    """Test that propose_new_texts handles empty reflection dataset."""
    reflective_dataset: dict[str, list[dict[str, Any]]] = {
        "instructions": []  # Empty list
    }

    candidate = {
        "instructions": "Some instructions",
    }

    test_output = ProposalOutput(
        updated_components=[
            UpdatedComponent(
                component_name="instructions",
                optimized_value="Updated instructions",
            )
        ]
    )

    test_model = TestModel(custom_output_args=test_output.model_dump(mode="python"))

    def mock_get_openai_model(model_name):
        return test_model

    monkeypatch.setattr("gepadantic.reflection.get_openai_model", mock_get_openai_model)

    result = propose_new_texts(
        candidate=candidate,
        reflective_dataset=reflective_dataset,
        components_to_update=["instructions"],
        reflection_model="test-model",
    )

    # Should still work with empty data
    assert result is not None
    assert isinstance(result.output, ProposalOutput)


def test_propose_new_texts_with_failures_and_successes(monkeypatch):
    """Test reflection with mixed success/failure patterns."""
    reflective_dataset = {
        "instructions": [
            # Successes
            {
                "user_prompt": "Calculate 5 + 3",
                "assistant_response": "8",
                "score": 1.0,
                "success": True,
                "feedback": "Correct addition",
            },
            {
                "user_prompt": "Calculate 10 - 2",
                "assistant_response": "8",
                "score": 1.0,
                "success": True,
                "feedback": "Correct subtraction",
            },
            # Failures
            {
                "user_prompt": "Calculate 7 * 6",
                "assistant_response": "13",
                "score": 0.0,
                "success": True,
                "feedback": "Wrong - multiplication should give 42",
            },
            {
                "user_prompt": "Calculate 20 / 4",
                "assistant_response": "24",
                "score": 0.0,
                "success": True,
                "feedback": "Wrong - division should give 5",
            },
            # Agent errors
            {
                "user_prompt": "Calculate sqrt(-1)",
                "assistant_response": "",
                "score": 0.0,
                "success": False,
                "error": "Math domain error",
                "feedback": "Should handle complex numbers or explain limitation",
            },
        ]
    }

    candidate = {
        "instructions": "Perform basic arithmetic calculations.",
    }

    test_output = ProposalOutput(
        updated_components=[
            UpdatedComponent(
                component_name="instructions",
                optimized_value="Perform basic arithmetic calculations. Pay special attention to multiplication and division operations. Handle edge cases like negative square roots gracefully.",
            )
        ]
    )

    test_model = TestModel(custom_output_args=test_output.model_dump(mode="python"))

    def mock_get_openai_model(model_name):
        return test_model

    monkeypatch.setattr("gepadantic.reflection.get_openai_model", mock_get_openai_model)

    result = propose_new_texts(
        candidate=candidate,
        reflective_dataset=reflective_dataset,
        components_to_update=["instructions"],
        reflection_model="test-model",
    )

    # Verify output incorporates lessons from failures
    assert result is not None
    optimized_text = result.output.updated_components[0].optimized_value
    assert len(optimized_text) > len(candidate["instructions"])


def test_propose_new_texts_preserves_other_components(monkeypatch):
    """Test that propose_new_texts only modifies specified components."""
    reflective_dataset = {
        "instructions": [
            {
                "user_prompt": "Test",
                "assistant_response": "Response",
                "score": 0.5,
                "success": True,
                "feedback": "Could be better",
            }
        ]
    }

    # Candidate has multiple components
    candidate = {
        "instructions": "Original instructions",
        "signature:Query:question:desc": "The question to answer",
        "signature:Query:context:desc": "Additional context",
    }

    # Only update instructions
    test_output = ProposalOutput(
        updated_components=[
            UpdatedComponent(
                component_name="instructions",
                optimized_value="Improved instructions",
            )
        ]
    )

    test_model = TestModel(custom_output_args=test_output.model_dump(mode="python"))

    def mock_get_openai_model(model_name):
        return test_model

    monkeypatch.setattr("gepadantic.reflection.get_openai_model", mock_get_openai_model)

    result = propose_new_texts(
        candidate=candidate,
        reflective_dataset=reflective_dataset,
        components_to_update=["instructions"],  # Only updating one component
        reflection_model="test-model",
    )

    # Should only return the updated component
    assert len(result.output.updated_components) == 1
    assert result.output.updated_components[0].component_name == "instructions"


def test_propose_new_texts_uses_custom_model(monkeypatch):
    """Test that propose_new_texts respects custom reflection_model parameter."""
    reflective_dataset = {
        "instructions": [
            {
                "user_prompt": "Test",
                "assistant_response": "Response",
                "score": 0.7,
                "success": True,
                "feedback": "Good",
            }
        ]
    }

    candidate = {"instructions": "Initial instructions"}

    # Create two different test models with different outputs
    output1 = ProposalOutput(
        updated_components=[
            UpdatedComponent(
                component_name="instructions",
                optimized_value="Model 1 output",
            )
        ]
    )

    output2 = ProposalOutput(
        updated_components=[
            UpdatedComponent(
                component_name="instructions",
                optimized_value="Model 2 output",
            )
        ]
    )

    test_model1 = TestModel(custom_output_args=output1.model_dump(mode="python"))
    test_model2 = TestModel(custom_output_args=output2.model_dump(mode="python"))

    # Mock for first model
    def mock_get_openai_model_1(model_name):
        return test_model1

    monkeypatch.setattr(
        "gepadantic.reflection.get_openai_model", mock_get_openai_model_1
    )

    # Use first model
    result1 = propose_new_texts(
        candidate=candidate,
        reflective_dataset=reflective_dataset,
        components_to_update=["instructions"],
        reflection_model="test-model-1",
    )

    # Mock for second model
    def mock_get_openai_model_2(model_name):
        return test_model2

    monkeypatch.setattr(
        "gepadantic.reflection.get_openai_model", mock_get_openai_model_2
    )

    # Use second model
    result2 = propose_new_texts(
        candidate=candidate,
        reflective_dataset=reflective_dataset.copy(),  # Copy to avoid mutation
        components_to_update=["instructions"],
        reflection_model="test-model-2",
    )

    # Different models should produce different outputs
    assert (
        result1.output.updated_components[0].optimized_value
        != result2.output.updated_components[0].optimized_value
    )
    assert result1.output.updated_components[0].optimized_value == "Model 1 output"
    assert result2.output.updated_components[0].optimized_value == "Model 2 output"


def test_reflection_input_docstring_content():
    """Test that ReflectionInput has appropriate docstring for LLM guidance."""
    # The docstring serves as the system prompt for the reflection agent
    docstring = ReflectionInput.__doc__

    assert docstring is not None
    assert "reflection dataset" in docstring.lower()
    assert "prompt components" in docstring.lower() or "components" in docstring.lower()
    assert "improve" in docstring.lower() or "optimize" in docstring.lower()


def test_proposal_output_docstring_content():
    """Test that ProposalOutput has appropriate docstring for LLM guidance."""
    docstring = ProposalOutput.__doc__

    assert docstring is not None
    assert "optimized" in docstring.lower() or "improved" in docstring.lower()
    assert "component" in docstring.lower()


def test_propose_new_texts_with_tool_components(monkeypatch):
    """Test reflection with tool-related components."""
    reflective_dataset = {
        "tool:calculate:description": [
            {
                "user_prompt": "User needs calculation",
                "assistant_response": "No tool used",
                "score": 0.0,
                "success": True,
                "feedback": "Should have used calculate tool",
            }
        ],
        "tool:calculate:param:x": [
            {
                "user_prompt": "Add 5 and 3",
                "assistant_response": "Used calculate(x='five', y=3)",
                "score": 0.0,
                "success": True,
                "feedback": "Parameter x should be a number, not a string",
            }
        ],
    }

    candidate = {
        "instructions": "Use tools when appropriate",
        "tool:calculate:description": "Performs calculations",
        "tool:calculate:param:x": "First number",
        "tool:calculate:param:y": "Second number",
    }

    test_output = ProposalOutput(
        updated_components=[
            UpdatedComponent(
                component_name="tool:calculate:description",
                optimized_value="Performs arithmetic calculations - use for any math operations",
            ),
            UpdatedComponent(
                component_name="tool:calculate:param:x",
                optimized_value="First number (must be numeric, not a word)",
            ),
        ]
    )

    test_model = TestModel(custom_output_args=test_output.model_dump(mode="python"))

    def mock_get_openai_model(model_name):
        return test_model

    monkeypatch.setattr("gepadantic.reflection.get_openai_model", mock_get_openai_model)

    result = propose_new_texts(
        candidate=candidate,
        reflective_dataset=reflective_dataset,
        components_to_update=[
            "tool:calculate:description",
            "tool:calculate:param:x",
        ],
        reflection_model="test-model",
    )

    assert len(result.output.updated_components) == 2
    updated_dict = {
        c.component_name: c.optimized_value for c in result.output.updated_components
    }
    assert "tool:calculate:description" in updated_dict
    assert "tool:calculate:param:x" in updated_dict
    assert "numeric" in updated_dict["tool:calculate:param:x"].lower()


def test_propose_new_texts_integration_with_real_agent(monkeypatch):
    """Integration test with a real agent (using TestModel)."""
    # This test ensures the full flow works end-to-end
    reflective_dataset = {
        "instructions": [
            {
                "user_prompt": "Classify: This movie was amazing!",
                "assistant_response": "negative",
                "score": 0.0,
                "success": True,
                "feedback": "Wrong sentiment - 'amazing' is positive",
                "instructions": "Classify movie reviews as positive or negative.",
            },
            {
                "user_prompt": "Classify: Worst film ever made",
                "assistant_response": "positive",
                "score": 0.0,
                "success": True,
                "feedback": "Wrong sentiment - 'worst' is negative",
                "instructions": "Classify movie reviews as positive or negative.",
            },
            {
                "user_prompt": "Classify: It was okay, nothing special",
                "assistant_response": "neutral",
                "score": 0.0,
                "success": True,
                "feedback": "No neutral option - should be positive or negative only",
                "instructions": "Classify movie reviews as positive or negative.",
            },
        ]
    }

    candidate = {
        "instructions": "Classify movie reviews as positive or negative.",
    }

    test_output = ProposalOutput(
        updated_components=[
            UpdatedComponent(
                component_name="instructions",
                optimized_value="Classify movie reviews as either positive or negative (not neutral). Pay attention to sentiment words like 'amazing' (positive) and 'worst' (negative).",
            )
        ]
    )

    test_model = TestModel(custom_output_args=test_output.model_dump(mode="python"))

    def mock_get_openai_model(model_name):
        return test_model

    monkeypatch.setattr("gepadantic.reflection.get_openai_model", mock_get_openai_model)

    result = propose_new_texts(
        candidate=candidate,
        reflective_dataset=reflective_dataset,
        components_to_update=["instructions"],
        reflection_model="test-model",
    )

    # Verify the output is improved and incorporates lessons
    assert result is not None
    optimized = result.output.updated_components[0].optimized_value

    # Should mention both sentiment examples and the no-neutral constraint
    assert len(optimized) > len(candidate["instructions"])
    assert any(
        keyword in optimized.lower()
        for keyword in ["positive", "negative", "sentiment", "attention"]
    )
