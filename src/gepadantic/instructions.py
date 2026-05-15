"""Helpers for composing pydantic-ai instruction overrides."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from pydantic_ai.agent.abstract import Instructions
from pydantic_ai.agent.wrapper import WrapperAgent

if TYPE_CHECKING:
    from pydantic_ai.agent import AbstractAgent


def _strip_surrogate_characters(text: str) -> str:
    return "".join(char for char in text if not 0xD800 <= ord(char) <= 0xDFFF)


def unwrap_agent(agent: AbstractAgent[Any, Any]) -> AbstractAgent[Any, Any]:
    """Return the innermost agent behind any pydantic-ai wrapper layers."""
    current = agent
    while isinstance(current, WrapperAgent):
        current = current.wrapped
    return current


def instruction_literal_text(agent: AbstractAgent[Any, Any]) -> str:
    """Return static instruction text, excluding dynamic instruction callables."""
    target_agent = unwrap_agent(agent)
    instructions = getattr(target_agent, "_instructions", ())
    literal_parts = [
        instruction for instruction in instructions if isinstance(instruction, str)
    ]
    return _strip_surrogate_characters(
        "\n\n".join(part for part in literal_parts if part)
    )


def instruction_callables(agent: AbstractAgent[Any, Any]) -> tuple[Any, ...]:
    """Return dynamic instruction callables registered on the underlying agent."""
    target_agent = unwrap_agent(agent)
    instructions = getattr(target_agent, "_instructions", ())
    return tuple(
        instruction
        for instruction in instructions
        if not isinstance(instruction, str) and callable(instruction)
    )


def _instruction_parts(
    instructions: Instructions[Any] | None,
) -> tuple[str | Any, ...]:
    if instructions is None:
        return ()
    if isinstance(instructions, str) or callable(instructions):
        return (instructions,)
    if isinstance(instructions, Sequence):
        return tuple(instructions)
    return (instructions,)


def compose_instructions(
    *instructions: Instructions[Any] | None,
) -> Instructions[Any] | None:
    """Compose instruction values into a pydantic-ai-compatible override."""
    parts: list[str | Any] = []
    for instruction in instructions:
        parts.extend(_instruction_parts(instruction))

    if not parts:
        return None
    if len(parts) == 1:
        return parts[0]
    return tuple(parts)
