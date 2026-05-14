"""Utilities for exposing and applying tool components to GEPA."""

from __future__ import annotations

import asyncio
import contextvars
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass, replace
from threading import Thread
from typing import Any, Iterable

from pydantic_ai._run_context import RunContext
from pydantic_ai.agent import AbstractAgent
from pydantic_ai.agent.wrapper import WrapperAgent
from pydantic_ai.models import Model
from pydantic_ai.models.test import TestModel
from pydantic_ai.tools import ToolDefinition
from pydantic_ai.usage import RunUsage

ToolCandidate = dict[str, str]


def _run_coro_sync(
    coro: asyncio.Future[Any] | asyncio.coroutines.Coroutine[Any, Any, Any],
) -> Any:
    """Run a coroutine in sync context, even if an event loop is already running."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)

    result: list[Any] = []
    error: list[BaseException] = []

    def _runner() -> None:
        try:
            result.append(asyncio.run(coro))
        except BaseException as exc:  # noqa: BLE001
            error.append(exc)

    thread = Thread(target=_runner)
    thread.start()
    thread.join()

    if error:
        raise error[0]
    return result[0] if result else None


def _unwrap_agent(agent: AbstractAgent[Any, Any]) -> AbstractAgent[Any, Any]:
    """Return the innermost agent (unwrap WrapperAgent layers)."""
    current = agent
    while isinstance(current, WrapperAgent):
        current = current.wrapped
    return current


def _build_run_context(agent: AbstractAgent[Any, Any]) -> RunContext[Any]:
    """Create a minimal RunContext for tool preparation."""
    model = getattr(agent, "model", None)
    if not isinstance(model, Model):
        model_instance: Model = TestModel()
    else:
        model_instance = model

    return RunContext(deps=None, model=model_instance, usage=RunUsage())


def _description_key(tool_name: str) -> str:
    return f"tool:{tool_name}:description"


def _output_description_key(tool_name: str) -> str:
    return f"output:{tool_name}:description"


def _format_path(path: tuple[str, ...]) -> str:
    formatted: list[str] = []
    for segment in path:
        if segment == "[]":
            if not formatted:
                formatted.append("[]")
            else:
                formatted[-1] = f"{formatted[-1]}[]"
        else:
            formatted.append(segment)
    return ".".join(formatted)


def _parameter_key(tool_name: str, path: tuple[str, ...]) -> str:
    return f"tool:{tool_name}:param:{_format_path(path)}"


def _output_parameter_key(tool_name: str, path: tuple[str, ...]) -> str:
    return f"output:{tool_name}:param:{_format_path(path)}"


def _iter_schema_descriptions(
    schema: Any, path: tuple[str, ...] = ()
) -> Iterable[tuple[tuple[str, ...], str]]:
    """Yield (path, description) pairs for a tool parameter schema."""
    if not isinstance(schema, dict):
        return

    description = schema.get("description")
    if isinstance(description, str) and description.strip() and path:
        yield path, description

    schema_type = schema.get("type")
    if schema_type == "object":
        properties = schema.get("properties") or {}
        if isinstance(properties, dict):
            for name, subschema in properties.items():
                if isinstance(subschema, dict):
                    yield from _iter_schema_descriptions(subschema, path + (name,))
    if schema_type == "array":
        items = schema.get("items")
        if isinstance(items, dict):
            yield from _iter_schema_descriptions(items, path + ("[]",))


def _set_schema_description(
    schema: dict[str, Any], path: tuple[str, ...], value: str
) -> bool:
    """Set a description on a copied schema, returning True if modified."""
    target: dict[str, Any] | None = schema
    for segment in path:
        if target is None:
            return False
        if segment == "[]":
            next_target = target.get("items")
            if not isinstance(next_target, dict):
                return False
            target = next_target
        else:
            properties = target.get("properties")
            if not isinstance(properties, dict):
                return False
            next_target = properties.get(segment)
            if not isinstance(next_target, dict):
                return False
            target = next_target

    if target is None:
        return False

    current = target.get("description")
    if current == value:
        return False
    target["description"] = value
    return True


@dataclass
class ToolComponentInfo:
    """Component metadata for a single tool."""

    description_key: str | None
    parameter_paths: dict[str, tuple[str, ...]]


def _extract_components(
    tool_defs: Iterable[ToolDefinition],
) -> tuple[dict[str, str], dict[str, ToolComponentInfo]]:
    """Extract component seeds and metadata from tool definitions."""
    return _extract_components_with_keys(
        tool_defs,
        description_key_func=_description_key,
        parameter_key_func=_parameter_key,
    )


def _extract_output_components(
    tool_defs: Iterable[ToolDefinition],
) -> tuple[dict[str, str], dict[str, ToolComponentInfo]]:
    """Extract component seeds and metadata from output tool definitions."""
    return _extract_components_with_keys(
        tool_defs,
        description_key_func=_output_description_key,
        parameter_key_func=_output_parameter_key,
    )


def _extract_components_with_keys(
    tool_defs: Iterable[ToolDefinition],
    *,
    description_key_func: Any,
    parameter_key_func: Any,
) -> tuple[dict[str, str], dict[str, ToolComponentInfo]]:
    """Extract component seeds and metadata using caller-provided key builders."""
    seed_components: dict[str, str] = {}
    component_info: dict[str, ToolComponentInfo] = {}

    for tool_def in tool_defs:
        description_key: str | None = None
        parameter_paths: dict[str, tuple[str, ...]] = {}

        if isinstance(tool_def.description, str) and tool_def.description.strip():
            description_key = description_key_func(tool_def.name)
            seed_components[description_key] = tool_def.description

        for path, desc in _iter_schema_descriptions(tool_def.parameters_json_schema):
            key = parameter_key_func(tool_def.name, path)
            seed_components[key] = desc
            parameter_paths[key] = path

        component_info[tool_def.name] = ToolComponentInfo(
            description_key=description_key,
            parameter_paths=parameter_paths,
        )

    return seed_components, component_info


class ToolOptimizationManager:
    """Manage tool component extraction and candidate application."""

    def __init__(self, agent: AbstractAgent[Any, Any]) -> None:
        self._base_agent = _unwrap_agent(agent)
        self._base_prepare = getattr(self._base_agent, "_prepare_tools", None)
        self._candidate_var: contextvars.ContextVar[ToolCandidate | None] = (
            contextvars.ContextVar("gepa_tool_candidate", default=None)
        )
        self._tool_components: dict[str, ToolComponentInfo] = {}
        self._seed_components: dict[str, str] | None = None

        # Install wrapper only once.
        if getattr(self._base_agent, "_gepa_tool_prepare_wrapper", None) is None:
            setattr(
                self._base_agent, "_gepa_tool_prepare_wrapper", self._prepare_wrapper
            )
            self._base_agent._prepare_tools = self._prepare_wrapper  # type: ignore[assignment]

    def get_seed_components(self) -> dict[str, str]:
        """Return cached seed components, collecting them if necessary."""
        if self._seed_components is not None:
            return dict(self._seed_components)

        try:
            tool_defs = _run_coro_sync(self._fetch_tool_definitions()) or []
        except Exception:
            return {}

        if not tool_defs:
            return {}

        seed_components, component_info = _extract_components(tool_defs)
        self._seed_components = dict(seed_components)
        self._tool_components = component_info
        return dict(seed_components)

    def get_component_keys(self) -> list[str]:
        """Return all known component keys."""
        components = self.get_seed_components()
        return list(components.keys())

    @contextmanager
    def candidate_context(self, candidate: dict[str, str] | None) -> Iterable[None]:
        """Context manager to apply a candidate during tool preparation."""
        filtered = self._filter_candidate(candidate)
        token = self._candidate_var.set(filtered)
        try:
            yield
        finally:
            self._candidate_var.reset(token)

    async def _fetch_tool_definitions(self) -> list[ToolDefinition]:
        toolset = self._base_agent._get_toolset()  # type: ignore[attr-defined]
        ctx = _build_run_context(self._base_agent)

        async with toolset:
            tools_dict = await toolset.get_tools(ctx)
        return [tool.tool_def for tool in tools_dict.values()]

    def _filter_candidate(
        self, candidate: dict[str, str] | None
    ) -> ToolCandidate | None:
        if not candidate:
            return None
        filtered = {
            key: value for key, value in candidate.items() if key.startswith("tool:")
        }
        return filtered or None

    async def _prepare_wrapper(
        self, ctx: RunContext[Any], tool_defs: list[ToolDefinition]
    ) -> list[ToolDefinition] | None:
        prepared = tool_defs
        if self._base_prepare:
            prepared_result = await self._base_prepare(ctx, tool_defs)
            if prepared_result is None:
                return None
            prepared = prepared_result

        seed_components, component_info = _extract_components(prepared)
        if self._seed_components is None:
            self._seed_components = dict(seed_components)
        else:
            for key, value in seed_components.items():
                self._seed_components.setdefault(key, value)
        self._tool_components = component_info

        candidate = self._candidate_var.get()
        if not candidate:
            return prepared

        modified: list[ToolDefinition] = []
        changed = False
        for tool_def in prepared:
            new_def = self._apply_candidate_to_tool(tool_def, candidate)
            if new_def is not tool_def:
                changed = True
            modified.append(new_def)

        return modified if changed else prepared

    def _apply_candidate_to_tool(
        self, tool_def: ToolDefinition, candidate: ToolCandidate
    ) -> ToolDefinition:
        info = self._tool_components.get(tool_def.name)
        if not info:
            return tool_def

        updates: dict[str, Any] = {}

        if info.description_key:
            raw_value = candidate.get(info.description_key)
            if raw_value is not None:
                new_description = str(raw_value)
                if tool_def.description != new_description:
                    updates["description"] = new_description

        schema_copy: dict[str, Any] | None = None
        schema_changed = False

        for key, path in info.parameter_paths.items():
            if key not in candidate:
                continue
            raw_value = candidate[key]
            if raw_value is None:
                continue
            new_description = str(raw_value)
            if schema_copy is None:
                schema_copy = deepcopy(tool_def.parameters_json_schema)
            if _set_schema_description(schema_copy, path, new_description):
                schema_changed = True

        if schema_changed and schema_copy is not None:
            updates["parameters_json_schema"] = schema_copy

        if updates:
            return replace(tool_def, **updates)
        return tool_def


class OutputToolOptimizationManager:
    """Manage output model component extraction and candidate application."""

    def __init__(
        self,
        agent: AbstractAgent[Any, Any],
        *,
        output_type: Any | None = None,
    ) -> None:
        self._base_agent = _unwrap_agent(agent)
        self._base_prepare = getattr(self._base_agent, "_prepare_output_tools", None)
        self._candidate_var: contextvars.ContextVar[ToolCandidate | None] = (
            contextvars.ContextVar("gepa_output_tool_candidate", default=None)
        )
        self._tool_components: dict[str, ToolComponentInfo] = {}
        self._seed_components: dict[str, str] | None = None
        self._output_types: list[Any] = []

        if output_type is not None:
            self.add_output_type(output_type)

        # Install wrapper only once.
        if getattr(self._base_agent, "_gepa_output_tool_prepare_wrapper", None) is None:
            setattr(
                self._base_agent,
                "_gepa_output_tool_prepare_wrapper",
                self._prepare_wrapper,
            )
            self._base_agent._prepare_output_tools = self._prepare_wrapper  # type: ignore[assignment]

    def add_output_type(self, output_type: Any) -> None:
        """Record an output type whose schema should seed output components."""
        if all(existing is not output_type for existing in self._output_types):
            self._output_types.append(output_type)
        self._ingest_output_type(output_type)

    def get_seed_components(self) -> dict[str, str]:
        """Return cached seed components, collecting them if necessary."""
        if self._seed_components is not None:
            return dict(self._seed_components)

        self._seed_components = {}
        for output_type in self._output_types:
            self._ingest_output_type(output_type)

        if not self._seed_components:
            tool_defs = self._collect_default_output_tool_defs()
            self._ingest_tool_defs(tool_defs)

        return dict(self._seed_components)

    def get_component_keys(self) -> list[str]:
        """Return all known output component keys."""
        components = self.get_seed_components()
        return list(components.keys())

    @contextmanager
    def candidate_context(self, candidate: dict[str, str] | None) -> Iterable[None]:
        """Context manager to apply a candidate during output tool preparation."""
        filtered = self._filter_candidate(candidate)
        token = self._candidate_var.set(filtered)
        try:
            yield
        finally:
            self._candidate_var.reset(token)

    def _filter_candidate(
        self, candidate: dict[str, str] | None
    ) -> ToolCandidate | None:
        if not candidate:
            return None
        filtered = {
            key: value for key, value in candidate.items() if key.startswith("output:")
        }
        return filtered or None

    def _collect_default_output_tool_defs(self) -> list[ToolDefinition]:
        toolset = getattr(self._base_agent, "_output_toolset", None)
        tool_defs = (
            getattr(toolset, "_tool_defs", None) if toolset is not None else None
        )
        if isinstance(tool_defs, list):
            return list(tool_defs)
        return []

    def _collect_output_type_tool_defs(self, output_type: Any) -> list[ToolDefinition]:
        prepare_output_schema = getattr(
            self._base_agent, "_prepare_output_schema", None
        )
        if prepare_output_schema is None:
            return []
        try:
            output_schema = prepare_output_schema(output_type)
        except Exception:
            return []
        toolset = getattr(output_schema, "toolset", None)
        tool_defs = (
            getattr(toolset, "_tool_defs", None) if toolset is not None else None
        )
        if isinstance(tool_defs, list):
            return list(tool_defs)
        return []

    def _ingest_output_type(self, output_type: Any) -> None:
        self._ingest_tool_defs(self._collect_output_type_tool_defs(output_type))

    def _ingest_tool_defs(self, tool_defs: Iterable[ToolDefinition]) -> None:
        seed_components, component_info = _extract_output_components(tool_defs)
        if self._seed_components is None:
            self._seed_components = {}
        for key, value in seed_components.items():
            self._seed_components.setdefault(key, value)
        self._tool_components.update(component_info)

    async def _prepare_wrapper(
        self, ctx: RunContext[Any], tool_defs: list[ToolDefinition]
    ) -> list[ToolDefinition] | None:
        prepared = tool_defs
        if self._base_prepare:
            prepared_result = await self._base_prepare(ctx, tool_defs)
            if prepared_result is None:
                return None
            prepared = prepared_result

        self._ingest_tool_defs(prepared)

        candidate = self._candidate_var.get()
        if not candidate:
            return prepared

        modified: list[ToolDefinition] = []
        changed = False
        for tool_def in prepared:
            new_def = self._apply_candidate_to_tool(tool_def, candidate)
            if new_def is not tool_def:
                changed = True
            modified.append(new_def)

        return modified if changed else prepared

    def _apply_candidate_to_tool(
        self, tool_def: ToolDefinition, candidate: ToolCandidate
    ) -> ToolDefinition:
        info = self._tool_components.get(tool_def.name)
        if not info:
            return tool_def

        updates: dict[str, Any] = {}

        if info.description_key:
            raw_value = candidate.get(info.description_key)
            if raw_value is not None:
                new_description = str(raw_value)
                if tool_def.description != new_description:
                    updates["description"] = new_description

        schema_copy: dict[str, Any] | None = None
        schema_changed = False

        for key, path in info.parameter_paths.items():
            if key not in candidate:
                continue
            raw_value = candidate[key]
            if raw_value is None:
                continue
            new_description = str(raw_value)
            if schema_copy is None:
                schema_copy = deepcopy(tool_def.parameters_json_schema)
            if _set_schema_description(schema_copy, path, new_description):
                schema_changed = True

        if schema_changed and schema_copy is not None:
            updates["parameters_json_schema"] = schema_copy

        if updates:
            return replace(tool_def, **updates)
        return tool_def


def get_tool_optimizer(
    agent: AbstractAgent[Any, Any],
) -> ToolOptimizationManager | None:
    """Return the installed tool optimization manager for an agent, if any."""
    base_agent = _unwrap_agent(agent)
    manager = getattr(base_agent, "_gepa_tool_optimizer", None)
    if isinstance(manager, ToolOptimizationManager):
        return manager
    return None


def get_or_create_tool_optimizer(
    agent: AbstractAgent[Any, Any],
) -> ToolOptimizationManager:
    """Retrieve or attach a tool optimization manager to an agent."""
    base_agent = _unwrap_agent(agent)
    manager = getattr(base_agent, "_gepa_tool_optimizer", None)
    if isinstance(manager, ToolOptimizationManager):
        return manager

    manager = ToolOptimizationManager(base_agent)
    setattr(base_agent, "_gepa_tool_optimizer", manager)
    return manager


def get_output_tool_optimizer(
    agent: AbstractAgent[Any, Any],
) -> OutputToolOptimizationManager | None:
    """Return the installed output tool optimization manager for an agent, if any."""
    base_agent = _unwrap_agent(agent)
    manager = getattr(base_agent, "_gepa_output_tool_optimizer", None)
    if isinstance(manager, OutputToolOptimizationManager):
        return manager
    return None


def get_or_create_output_tool_optimizer(
    agent: AbstractAgent[Any, Any],
    *,
    output_type: Any | None = None,
) -> OutputToolOptimizationManager:
    """Retrieve or attach an output tool optimization manager to an agent."""
    base_agent = _unwrap_agent(agent)
    manager = getattr(base_agent, "_gepa_output_tool_optimizer", None)
    if isinstance(manager, OutputToolOptimizationManager):
        if output_type is not None:
            manager.add_output_type(output_type)
        return manager

    manager = OutputToolOptimizationManager(base_agent, output_type=output_type)
    setattr(base_agent, "_gepa_output_tool_optimizer", manager)
    return manager
