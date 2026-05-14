"""Integration tests for pydantic-ai GEPA adapter."""

from typing import Any

import time_machine
from inline_snapshot import snapshot
from pydantic_ai import Agent
from pydantic_ai.messages import (
    ModelRequest,
    ModelResponse,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models.test import TestModel

from gepadantic.adapter import PydanticAIGEPAAdapter
from gepadantic.components import (
    extract_seed_candidate,
    get_component_names,
)
from gepadantic.schema import DataInst, DataInstWithPrompt, RolloutOutput, Trajectory


def test_extract_seed_candidate():
    """Test extracting prompts from an agent."""
    agent = Agent(
        TestModel(),
        instructions="Be helpful",
    )

    candidate = extract_seed_candidate(agent)

    assert candidate["instructions"] == "Be helpful"
    assert len(candidate) == 1


def test_get_component_names():
    """Test getting optimizable component names."""
    agent = Agent(
        TestModel(),
        instructions="Instructions",
    )

    components = get_component_names(agent)

    assert "instructions" in components
    assert len(components) == 1


def test_process_data_instance():
    """Test processing a single data instance."""
    agent = Agent(
        TestModel(custom_output_text="Test response"), instructions="Be helpful"
    )

    def metric(
        data_inst: DataInst, output: RolloutOutput[Any]
    ) -> tuple[float, str | None]:
        if output.success:
            return (0.8, "Good")
        return (0.0, "Failed")

    adapter = PydanticAIGEPAAdapter(agent, metric)

    # Test without traces
    data_inst = DataInstWithPrompt(
        user_prompt=UserPromptPart(content="Hello"),
        message_history=None,
        metadata={},
        case_id="test-4",
    )
    result = adapter.process_data_instance(data_inst, capture_traces=False)

    assert "output" in result
    assert "score" in result
    assert result["output"].success is True
    assert result["score"] == 0.8
    assert "trajectory" not in result

    # Test with traces
    result_with_trace = adapter.process_data_instance(data_inst, capture_traces=True)

    assert "output" in result_with_trace
    assert "score" in result_with_trace
    assert "trajectory" in result_with_trace
    assert result_with_trace["output"].success is True
    assert result_with_trace["score"] == 0.8
    assert result_with_trace["trajectory"].final_output == "Test response"


@time_machine.travel("2023-01-01", tick=False)
def test_make_reflective_dataset():
    """Test making a reflective dataset."""
    agent = Agent(
        TestModel(custom_output_text="Test response"), instructions="Be helpful"
    )

    def metric(
        data_inst: DataInst, output: RolloutOutput[Any]
    ) -> tuple[float, str | None]:
        if output.success:
            return (0.8, "Good")
        return (0.0, "Failed")

    adapter = PydanticAIGEPAAdapter(agent, metric)
    candidate = extract_seed_candidate(agent)

    data_inst = DataInstWithPrompt(
        user_prompt=UserPromptPart(content="Hello"),
        message_history=None,
        metadata={},
        case_id="test-4",
    )
    result = adapter.evaluate([data_inst], candidate, capture_traces=True)

    reflective_dataset = adapter.make_reflective_dataset(
        candidate, result, ["instructions"]
    )
    assert reflective_dataset == snapshot(
        {
            "traces": [
                {
                    "system_prompt": "Be helpful",
                    "user_prompt": "Hello",
                    "assistant_response": "Test response",
                    "score": 0.8,
                    "success": True,
                    "feedback": "Good",
                }
            ]
        }
    )


def test_make_reflective_dataset_uses_shared_trace_bucket():
    """Reflection records are not repeated once per component."""
    agent = Agent(
        TestModel(custom_output_text="Test response"), instructions="Be helpful"
    )

    def metric(
        data_inst: DataInst, output: RolloutOutput[Any]
    ) -> tuple[float, str | None]:
        return (0.8, "Good") if output.success else (0.0, "Failed")

    adapter = PydanticAIGEPAAdapter(agent, metric)
    candidate = extract_seed_candidate(agent)

    data_inst = DataInstWithPrompt(
        user_prompt=UserPromptPart(content="Hello"),
        message_history=None,
        metadata={},
        case_id="test-shared",
    )
    result = adapter.evaluate([data_inst], candidate, capture_traces=True)

    reflective_dataset = adapter.make_reflective_dataset(
        candidate,
        result,
        ["instructions", "signature:Input:field:desc"],
    )

    assert list(reflective_dataset) == ["traces"]
    assert len(reflective_dataset["traces"]) == 1


def test_reflective_record_truncates_tool_returns():
    """Tool traces keep args and bounded return output without full message bloat."""
    trajectory = Trajectory(
        messages=[
            ModelRequest(
                [UserPromptPart(content="Look this up")],
                instructions="Use tools carefully",
            ),
            ModelResponse(
                [
                    ToolCallPart(
                        tool_name="lookup",
                        args={"query": "alpha"},
                        tool_call_id="call-1",
                    )
                ],
                provider_name="test-provider",
            ),
            ModelRequest(
                [
                    ToolReturnPart(
                        tool_name="lookup",
                        content="abcdefghijklmnopqrstuvwxyz",
                        tool_call_id="call-1",
                    )
                ]
            ),
            ModelResponse([TextPart(content="Done")]),
        ],
        final_output="Done",
        instructions="Use tools carefully",
    )

    record = trajectory.to_reflective_record(max_tool_return_chars=5)

    assert record["system_prompt"] == "Use tools carefully"
    assert record["user_prompt"] == "Look this up"
    assert record["assistant_response"] == "Done"
    assert record["tool_trace"] == [
        {
            "type": "tool_call",
            "tool_name": "lookup",
            "arguments": '{"query":"alpha"}',
            "tool_call_id": "call-1",
        },
        {
            "type": "tool_return",
            "tool_name": "lookup",
            "content": "abcde... [truncated 21 chars]",
            "tool_call_id": "call-1",
        },
    ]
    assert "messages" not in record
    assert "run_usage" not in record
