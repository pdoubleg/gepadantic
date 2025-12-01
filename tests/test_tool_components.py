from __future__ import annotations

from pydantic_ai import Agent
from pydantic_ai.models.test import TestModel
from pydantic_ai.tools import ToolDefinition

from gepadantic.tool_components import (
    ToolComponentInfo,
    ToolOptimizationManager,
    _description_key,
    _extract_components,
    _format_path,
    _iter_schema_descriptions,
    _parameter_key,
    _set_schema_description,
    _unwrap_agent,
    get_or_create_tool_optimizer,
    get_tool_optimizer,
)


def test_description_key():
    """Test that _description_key generates correct component keys."""
    assert _description_key("calculate") == "tool:calculate:description"
    assert _description_key("fetch_data") == "tool:fetch_data:description"
    assert _description_key("my_tool") == "tool:my_tool:description"


def test_parameter_key_simple():
    """Test _parameter_key with simple paths."""
    assert _parameter_key("calculate", ("x",)) == "tool:calculate:param:x"
    assert _parameter_key("calculate", ("y",)) == "tool:calculate:param:y"
    assert _parameter_key("fetch_data", ("url",)) == "tool:fetch_data:param:url"


def test_parameter_key_nested():
    """Test _parameter_key with nested paths."""
    assert (
        _parameter_key("process", ("config", "timeout"))
        == "tool:process:param:config.timeout"
    )
    assert (
        _parameter_key("api_call", ("headers", "auth", "token"))
        == "tool:api_call:param:headers.auth.token"
    )


def test_parameter_key_with_arrays():
    """Test _parameter_key with array segments."""
    assert _parameter_key("batch", ("items[]",)) == "tool:batch:param:items[]"
    assert (
        _parameter_key("transform", ("data", "values[]"))
        == "tool:transform:param:data.values[]"
    )


def test_format_path_simple():
    """Test _format_path with simple paths."""
    assert _format_path(("x",)) == "x"
    assert _format_path(("config", "timeout")) == "config.timeout"
    assert _format_path(("a", "b", "c")) == "a.b.c"


def test_format_path_with_arrays():
    """Test _format_path with array notation."""
    assert _format_path(("items", "[]")) == "items[]"
    assert _format_path(("data", "values", "[]")) == "data.values[]"
    assert _format_path(("[]",)) == "[]"


def test_iter_schema_descriptions_simple():
    """Test extracting descriptions from simple parameter schemas."""
    schema = {
        "type": "object",
        "properties": {
            "x": {"type": "number", "description": "First number"},
            "y": {"type": "number", "description": "Second number"},
        },
    }

    descriptions = list(_iter_schema_descriptions(schema))
    assert len(descriptions) == 2
    assert (("x",), "First number") in descriptions
    assert (("y",), "Second number") in descriptions


def test_iter_schema_descriptions_nested():
    """Test extracting descriptions from nested parameter schemas."""
    schema = {
        "type": "object",
        "properties": {
            "config": {
                "type": "object",
                "description": "Configuration settings",
                "properties": {
                    "timeout": {"type": "number", "description": "Request timeout"},
                    "retries": {"type": "number", "description": "Number of retries"},
                },
            }
        },
    }

    descriptions = list(_iter_schema_descriptions(schema))
    # Root-level description for "config" should be included
    assert (("config",), "Configuration settings") in descriptions
    # Nested descriptions
    assert (("config", "timeout"), "Request timeout") in descriptions
    assert (("config", "retries"), "Number of retries") in descriptions


def test_iter_schema_descriptions_array():
    """Test extracting descriptions from array schemas."""
    schema = {
        "type": "object",
        "properties": {
            "items": {
                "type": "array",
                "description": "List of items",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string", "description": "Item name"}
                    },
                },
            }
        },
    }

    descriptions = list(_iter_schema_descriptions(schema))
    assert (("items",), "List of items") in descriptions
    assert (("items", "[]", "name"), "Item name") in descriptions


def test_iter_schema_descriptions_empty():
    """Test _iter_schema_descriptions with empty or invalid schemas."""
    # Empty dict
    assert list(_iter_schema_descriptions({})) == []

    # Non-dict
    assert list(_iter_schema_descriptions("not a schema")) == []  # type: ignore

    # Schema without descriptions
    schema = {
        "type": "object",
        "properties": {"x": {"type": "number"}},
    }
    assert list(_iter_schema_descriptions(schema)) == []


def test_set_schema_description_simple():
    """Test setting description on simple schema paths."""
    schema = {
        "type": "object",
        "properties": {"x": {"type": "number", "description": "Old description"}},
    }

    result = _set_schema_description(schema, ("x",), "New description")
    assert result is True
    assert schema["properties"]["x"]["description"] == "New description"


def test_set_schema_description_nested():
    """Test setting description on nested schema paths."""
    schema = {
        "type": "object",
        "properties": {
            "config": {
                "type": "object",
                "properties": {
                    "timeout": {"type": "number", "description": "Old timeout desc"}
                },
            }
        },
    }

    result = _set_schema_description(
        schema, ("config", "timeout"), "New timeout description"
    )
    assert result is True
    assert (
        schema["properties"]["config"]["properties"]["timeout"]["description"]
        == "New timeout description"
    )


def test_set_schema_description_array():
    """Test setting description on array item schemas."""
    schema = {
        "type": "object",
        "properties": {
            "items": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string", "description": "Old name"}
                    },
                },
            }
        },
    }

    result = _set_schema_description(schema, ("items", "[]", "name"), "New name")
    assert result is True
    assert (
        schema["properties"]["items"]["items"]["properties"]["name"]["description"]
        == "New name"
    )


def test_set_schema_description_no_change():
    """Test that _set_schema_description returns False when value is unchanged."""
    schema = {
        "type": "object",
        "properties": {"x": {"type": "number", "description": "Same description"}},
    }

    result = _set_schema_description(schema, ("x",), "Same description")
    assert result is False


def test_set_schema_description_invalid_path():
    """Test that _set_schema_description handles invalid paths gracefully."""
    schema = {
        "type": "object",
        "properties": {"x": {"type": "number"}},
    }

    # Non-existent path
    result = _set_schema_description(schema, ("nonexistent",), "Description")
    assert result is False

    # Invalid nested path
    result = _set_schema_description(schema, ("x", "invalid"), "Description")
    assert result is False


def test_tool_component_info():
    """Test ToolComponentInfo dataclass."""
    info = ToolComponentInfo(
        description_key="tool:calculate:description",
        parameter_paths={
            "tool:calculate:param:x": ("x",),
            "tool:calculate:param:y": ("y",),
        },
    )

    assert info.description_key == "tool:calculate:description"
    assert len(info.parameter_paths) == 2
    assert info.parameter_paths["tool:calculate:param:x"] == ("x",)
    assert info.parameter_paths["tool:calculate:param:y"] == ("y",)


def test_extract_components_basic():
    """Test extracting components from basic tool definitions."""
    tool_def = ToolDefinition(
        name="calculate",
        description="Perform arithmetic calculations",
        parameters_json_schema={
            "type": "object",
            "properties": {
                "x": {"type": "number", "description": "First number"},
                "y": {"type": "number", "description": "Second number"},
            },
            "required": ["x", "y"],
        },
    )

    seed_components, component_info = _extract_components([tool_def])

    # Check seed components
    assert "tool:calculate:description" in seed_components
    assert (
        seed_components["tool:calculate:description"]
        == "Perform arithmetic calculations"
    )
    assert "tool:calculate:param:x" in seed_components
    assert seed_components["tool:calculate:param:x"] == "First number"
    assert "tool:calculate:param:y" in seed_components
    assert seed_components["tool:calculate:param:y"] == "Second number"

    # Check component info
    assert "calculate" in component_info
    info = component_info["calculate"]
    assert info.description_key == "tool:calculate:description"
    assert len(info.parameter_paths) == 2


def test_extract_components_no_description():
    """Test extracting components from tools without descriptions."""
    tool_def = ToolDefinition(
        name="simple_tool",
        description=None,
        parameters_json_schema={
            "type": "object",
            "properties": {
                "arg": {"type": "string"}  # No description
            },
        },
    )

    seed_components, component_info = _extract_components([tool_def])

    # Should not extract components without descriptions
    assert "tool:simple_tool:description" not in seed_components
    assert "tool:simple_tool:param:arg" not in seed_components

    # But should still have component info
    assert "simple_tool" in component_info
    assert component_info["simple_tool"].description_key is None


def test_extract_components_multiple_tools():
    """Test extracting components from multiple tool definitions."""
    tool1 = ToolDefinition(
        name="add",
        description="Add two numbers",
        parameters_json_schema={
            "type": "object",
            "properties": {"a": {"type": "number", "description": "First value"}},
        },
    )

    tool2 = ToolDefinition(
        name="multiply",
        description="Multiply two numbers",
        parameters_json_schema={
            "type": "object",
            "properties": {"a": {"type": "number", "description": "First factor"}},
        },
    )

    seed_components, component_info = _extract_components([tool1, tool2])

    # Both tools should have their components
    assert "tool:add:description" in seed_components
    assert "tool:multiply:description" in seed_components
    assert "tool:add:param:a" in seed_components
    assert "tool:multiply:param:a" in seed_components

    # Component info for both
    assert "add" in component_info
    assert "multiply" in component_info


def test_unwrap_agent_basic():
    """Test unwrapping a basic agent."""
    agent = Agent(TestModel(), instructions="Test instructions")
    unwrapped = _unwrap_agent(agent)
    assert unwrapped is agent


def test_unwrap_agent_with_wrapper():
    """Test unwrapping nested WrapperAgent layers."""
    from pydantic_ai.agent.wrapper import WrapperAgent

    base_agent = Agent(TestModel(), instructions="Base agent")
    wrapped_once = WrapperAgent(base_agent)
    wrapped_twice = WrapperAgent(wrapped_once)

    unwrapped = _unwrap_agent(wrapped_twice)
    assert unwrapped is base_agent


def test_tool_optimization_manager_initialization():
    """Test ToolOptimizationManager initialization."""
    agent = Agent(TestModel(), instructions="Test agent")

    manager = ToolOptimizationManager(agent)

    assert manager is not None
    assert manager._base_agent is agent
    assert manager._seed_components is None
    assert manager._tool_components == {}


def test_tool_optimization_manager_with_tools():
    """Test ToolOptimizationManager with an agent that has tools."""
    agent = Agent(TestModel(), instructions="Calculator agent")

    @agent.tool_plain
    def calculate(x: int, y: int) -> int:
        """Add two numbers together.

        Args:
            x: The first number
            y: The second number
        """
        return x + y

    manager = ToolOptimizationManager(agent)
    seed_components = manager.get_seed_components()

    # Should extract tool components
    assert "tool:calculate:description" in seed_components
    assert "tool:calculate:param:x" in seed_components
    assert "tool:calculate:param:y" in seed_components


def test_tool_optimization_manager_get_component_keys():
    """Test getting component keys from manager."""
    agent = Agent(TestModel(), instructions="Test agent")

    @agent.tool_plain
    def greet(name: str) -> str:
        """Greet someone.

        Args:
            name: The person's name
        """
        return f"Hello, {name}!"

    manager = ToolOptimizationManager(agent)
    keys = manager.get_component_keys()

    assert "tool:greet:description" in keys
    assert "tool:greet:param:name" in keys


def test_tool_optimization_manager_filter_candidate():
    """Test that manager filters candidates to tool-related keys."""
    agent = Agent(TestModel(), instructions="Test agent")
    manager = ToolOptimizationManager(agent)

    candidate = {
        "instructions": "Some instructions",
        "tool:calculate:description": "Tool description",
        "tool:calculate:param:x": "Parameter x",
        "signature:Input:field:desc": "Signature field",
    }

    filtered = manager._filter_candidate(candidate)

    # Should only include tool: keys
    assert filtered is not None
    assert "tool:calculate:description" in filtered
    assert "tool:calculate:param:x" in filtered
    assert "instructions" not in filtered
    assert "signature:Input:field:desc" not in filtered


def test_tool_optimization_manager_filter_candidate_empty():
    """Test that filtering returns None for candidates with no tool keys."""
    agent = Agent(TestModel(), instructions="Test agent")
    manager = ToolOptimizationManager(agent)

    candidate = {
        "instructions": "Some instructions",
        "signature:Input:field:desc": "Signature field",
    }

    filtered = manager._filter_candidate(candidate)
    assert filtered is None


def test_tool_optimization_manager_candidate_context():
    """Test candidate context manager."""
    agent = Agent(TestModel(), instructions="Test agent")
    manager = ToolOptimizationManager(agent)

    candidate = {
        "tool:calculate:description": "Updated description",
    }

    # Initially no candidate
    assert manager._candidate_var.get() is None

    # Inside context, candidate should be set
    with manager.candidate_context(candidate):
        active_candidate = manager._candidate_var.get()
        assert active_candidate is not None
        assert "tool:calculate:description" in active_candidate

    # After context, should be cleared
    assert manager._candidate_var.get() is None


def test_get_tool_optimizer_none():
    """Test get_tool_optimizer when no optimizer is attached."""
    agent = Agent(TestModel(), instructions="Test agent")
    optimizer = get_tool_optimizer(agent)
    assert optimizer is None


def test_get_or_create_tool_optimizer():
    """Test get_or_create_tool_optimizer creates and attaches optimizer."""
    agent = Agent(TestModel(), instructions="Test agent")

    # First call creates optimizer
    optimizer1 = get_or_create_tool_optimizer(agent)
    assert optimizer1 is not None
    assert isinstance(optimizer1, ToolOptimizationManager)

    # Second call returns same instance
    optimizer2 = get_or_create_tool_optimizer(agent)
    assert optimizer2 is optimizer1


def test_get_tool_optimizer_after_creation():
    """Test get_tool_optimizer retrieves created optimizer."""
    agent = Agent(TestModel(), instructions="Test agent")

    # Create optimizer
    created = get_or_create_tool_optimizer(agent)

    # get_tool_optimizer should now find it
    retrieved = get_tool_optimizer(agent)
    assert retrieved is created


def test_tool_optimization_manager_caching():
    """Test that seed components are cached after first extraction."""
    agent = Agent(TestModel(), instructions="Test agent")

    @agent.tool_plain
    def test_tool(arg: str) -> str:
        """Test tool.

        Args:
            arg: Test argument
        """
        return arg

    manager = ToolOptimizationManager(agent)

    # First call extracts components
    components1 = manager.get_seed_components()
    assert "tool:test_tool:description" in components1

    # Second call returns cached components
    components2 = manager.get_seed_components()
    assert components2 == components1
    assert components2 is not components1  # Returns a copy


def test_tool_optimization_manager_with_nested_params():
    """Test manager with tools that have nested parameter schemas."""
    # Note: We can't easily test with real nested params via decorators,
    # so we'll test the internal logic with a mock ToolDefinition

    tool_def = ToolDefinition(
        name="complex_tool",
        description="A tool with nested parameters",
        parameters_json_schema={
            "type": "object",
            "properties": {
                "config": {
                    "type": "object",
                    "description": "Configuration object",
                    "properties": {
                        "timeout": {
                            "type": "number",
                            "description": "Timeout in seconds",
                        },
                        "retries": {
                            "type": "number",
                            "description": "Number of retries",
                        },
                    },
                }
            },
        },
    )

    seed_components, component_info = _extract_components([tool_def])

    assert "tool:complex_tool:param:config" in seed_components
    assert "tool:complex_tool:param:config.timeout" in seed_components
    assert "tool:complex_tool:param:config.retries" in seed_components


def test_tool_optimization_manager_apply_candidate():
    """Test applying a candidate to modify tool definitions."""
    original_tool = ToolDefinition(
        name="calculate",
        description="Original description",
        parameters_json_schema={
            "type": "object",
            "properties": {
                "x": {"type": "number", "description": "Original x description"}
            },
        },
    )

    agent = Agent(TestModel(), instructions="Test agent")
    manager = ToolOptimizationManager(agent)

    # Set up component info manually for testing
    manager._tool_components["calculate"] = ToolComponentInfo(
        description_key="tool:calculate:description",
        parameter_paths={"tool:calculate:param:x": ("x",)},
    )

    candidate = {
        "tool:calculate:description": "Updated description",
        "tool:calculate:param:x": "Updated x description",
    }

    modified_tool = manager._apply_candidate_to_tool(original_tool, candidate)

    # Tool should be modified
    assert modified_tool is not original_tool
    assert modified_tool.description == "Updated description"
    assert (
        modified_tool.parameters_json_schema["properties"]["x"]["description"]
        == "Updated x description"
    )


def test_tool_optimization_manager_apply_candidate_no_changes():
    """Test that apply_candidate returns same tool when no changes needed."""
    original_tool = ToolDefinition(
        name="calculate",
        description="Same description",
        parameters_json_schema={
            "type": "object",
            "properties": {
                "x": {"type": "number", "description": "Same x description"}
            },
        },
    )

    agent = Agent(TestModel(), instructions="Test agent")
    manager = ToolOptimizationManager(agent)

    manager._tool_components["calculate"] = ToolComponentInfo(
        description_key="tool:calculate:description",
        parameter_paths={"tool:calculate:param:x": ("x",)},
    )

    # Candidate with same values
    candidate = {
        "tool:calculate:description": "Same description",
        "tool:calculate:param:x": "Same x description",
    }

    modified_tool = manager._apply_candidate_to_tool(original_tool, candidate)

    # Should return same tool instance when nothing changes
    assert modified_tool is original_tool


def test_tool_optimization_manager_partial_candidate():
    """Test applying a candidate that only modifies some components."""
    original_tool = ToolDefinition(
        name="calculate",
        description="Original description",
        parameters_json_schema={
            "type": "object",
            "properties": {
                "x": {"type": "number", "description": "Original x"},
                "y": {"type": "number", "description": "Original y"},
            },
        },
    )

    agent = Agent(TestModel(), instructions="Test agent")
    manager = ToolOptimizationManager(agent)

    manager._tool_components["calculate"] = ToolComponentInfo(
        description_key="tool:calculate:description",
        parameter_paths={
            "tool:calculate:param:x": ("x",),
            "tool:calculate:param:y": ("y",),
        },
    )

    # Candidate that only updates x parameter
    candidate = {
        "tool:calculate:param:x": "Updated x description",
    }

    modified_tool = manager._apply_candidate_to_tool(original_tool, candidate)

    # Only x should be updated, y should remain the same
    assert (
        modified_tool.parameters_json_schema["properties"]["x"]["description"]
        == "Updated x description"
    )
    assert (
        modified_tool.parameters_json_schema["properties"]["y"]["description"]
        == "Original y"
    )
    # Description should remain unchanged
    assert modified_tool.description == "Original description"


def test_tool_optimization_manager_prepare_wrapper():
    """Test that the prepare wrapper correctly modifies tool definitions."""
    agent = Agent(TestModel(), instructions="Test agent")

    @agent.tool_plain
    def calculate(x: int, y: int) -> int:
        """Add two numbers.

        Args:
            x: First number
            y: Second number
        """
        return x + y

    manager = ToolOptimizationManager(agent)

    # Get initial components
    initial_components = manager.get_seed_components()
    assert "tool:calculate:description" in initial_components

    # Apply a candidate
    candidate = {
        "tool:calculate:description": "Perform addition on two numbers",
        "tool:calculate:param:x": "The first operand",
    }

    with manager.candidate_context(candidate):
        # Run the agent to trigger tool preparation
        # We're just verifying the context manager works, not inspecting the result
        _ = agent.run_sync("Calculate 2 + 3")

    # After context, original descriptions should be used again
    _ = agent.run_sync("Calculate 5 + 7")
