"""Type definitions for GEPA adapter integration."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from typing import Any, Generic, Sequence, TypeVar

from pydantic import BaseModel
from pydantic_ai.messages import (
    AudioUrl,
    BinaryContent,
    BuiltinToolCallPart,
    BuiltinToolReturnPart,
    DocumentUrl,
    FilePart,
    ImageUrl,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    RetryPromptPart,
    SystemPromptPart,
    TextPart,
    ThinkingPart,
    ToolCallPart,
    ToolReturnPart,
    UserContent,
    UserPromptPart,
    VideoUrl,
)

# Type variable for the input type
InputModelT = TypeVar("InputModelT", bound=BaseModel)

# Type variable for the output type in RolloutOutput
OutputT = TypeVar("OutputT")


@dataclass
class DataInstWithPrompt:
    """A single data instance for optimization.

    Each instance represents a single case from a pydantic-evals Dataset.
    """

    user_prompt: UserPromptPart
    message_history: list[ModelMessage] | None
    metadata: dict[str, Any]
    case_id: str  # Unique identifier for tracking


@dataclass(init=False)
class DataInstWithInput(Generic[InputModelT]):
    """A single data instance for optimization with a structured input model."""

    input: InputModelT
    message_history: list[ModelMessage] | None
    metadata: dict[str, Any]
    case_id: str  # Unique identifier for tracking

    def __init__(
        self,
        *,
        input: InputModelT | None = None,
        message_history: list[ModelMessage] | None,
        metadata: dict[str, Any],
        case_id: str,
    ) -> None:
        if input is None:
            raise TypeError("'input' must be provided.")

        self.input = input
        self.message_history = message_history
        self.metadata = metadata
        self.case_id = case_id


DataInst = DataInstWithPrompt | DataInstWithInput[Any]

DEFAULT_MAX_TOOL_RETURN_CHARS = 2000


def _truncate_text(value: str, limit: int = 2000) -> str:
    """Truncate long strings to keep reflection records readable."""
    if len(value) <= limit:
        return value
    trimmed = value[:limit]
    omitted = len(value) - limit
    return f"{trimmed}... [truncated {omitted} chars]"


def _compact_dict(data: dict[str, Any]) -> dict[str, Any]:
    """Remove keys with None values to keep payloads tidy."""
    return {key: value for key, value in data.items() if value is not None}


def _timestamp_iso(timestamp: Any) -> str | None:
    """Convert timestamp objects to ISO strings."""
    return timestamp.isoformat() if hasattr(timestamp, "isoformat") else None


def _describe_binary_content(content: BinaryContent) -> str:
    """Return a compact description of binary payloads."""
    size = len(content.data) if getattr(content, "data", None) else 0
    media_type = getattr(content, "media_type", "unknown")
    identifier = getattr(content, "identifier", None)
    identifier_str = f", id={identifier}" if identifier else ""
    return f"[binary media_type={media_type}{identifier_str}, bytes={size}]"


def _describe_user_content_item(item: UserContent) -> str:
    """Produce a concise textual description for user content parts."""
    if isinstance(item, str):
        return item
    if isinstance(item, ImageUrl | AudioUrl | DocumentUrl | VideoUrl):
        meta: list[str] = [f"url={item.url}"]
        media_type = getattr(item, "media_type", None)
        if media_type:
            meta.append(f"media_type={media_type}")
        if getattr(item, "force_download", False):
            meta.append("force_download=true")
        return f"[{item.kind} {' '.join(meta)}]"
    if isinstance(item, BinaryContent):
        return _describe_binary_content(item)
    return repr(item)


def _serialize_user_prompt_content(content: str | Sequence[UserContent]) -> str:
    """Normalize user prompt content for reflection."""
    if isinstance(content, str):
        return content
    described = [_describe_user_content_item(item) for item in content]
    return "\n".join(described)


def _serialize_tool_return(
    part: ToolReturnPart | BuiltinToolReturnPart,
    *,
    max_chars: int = DEFAULT_MAX_TOOL_RETURN_CHARS,
) -> dict[str, Any]:
    """Serialize tool return parts with compact output."""
    content_str = _truncate_text(part.model_response_str(), limit=max_chars)
    serialized = {
        "type": "tool_return",
        "tool_name": part.tool_name,
        "content": content_str,
        "tool_call_id": part.tool_call_id,
    }
    return _compact_dict(serialized)


def _serialize_tool_call(part: ToolCallPart | BuiltinToolCallPart) -> dict[str, Any]:
    """Serialize tool call parts with arguments but no provider metadata."""
    return _compact_dict(
        {
            "type": "tool_call",
            "tool_name": part.tool_name,
            "arguments": _truncate_text(part.args_as_json_str()),
            "tool_call_id": part.tool_call_id,
            "id": part.id,
        }
    )


def _serialize_retry_part(part: RetryPromptPart) -> dict[str, Any]:
    """Serialize retry prompts, keeping structured validation data readable."""
    if isinstance(part.content, str):
        reason = part.content
    else:
        reason = json.dumps(part.content, ensure_ascii=False, default=str)
    return _compact_dict(
        {
            "type": "retry_prompt",
            "tool_name": part.tool_name,
            "tool_call_id": part.tool_call_id,
            "content": _truncate_text(reason),
            "timestamp": _timestamp_iso(part.timestamp),
        }
    )


def _serialize_request_part(part: Any) -> dict[str, Any]:
    """Serialize a ModelRequest part."""
    if isinstance(part, SystemPromptPart):
        return _compact_dict(
            {
                "type": "system_prompt",
                "role": "system",
                "content": _truncate_text(part.content),
                "timestamp": _timestamp_iso(part.timestamp),
            }
        )
    if isinstance(part, UserPromptPart):
        return _compact_dict(
            {
                "type": "user_prompt",
                "role": "user",
                "content": _truncate_text(_serialize_user_prompt_content(part.content)),
                "timestamp": _timestamp_iso(part.timestamp),
            }
        )
    if isinstance(part, ToolReturnPart | BuiltinToolReturnPart):
        return _serialize_tool_return(part)
    if isinstance(part, RetryPromptPart):
        return _serialize_retry_part(part)
    return {
        "type": getattr(part, "part_kind", type(part).__name__),
        "repr": repr(part),
    }


def _serialize_response_part(part: Any) -> dict[str, Any]:
    """Serialize a ModelResponse part."""
    if isinstance(part, TextPart):
        return _compact_dict(
            {
                "type": "text",
                "role": "assistant",
                "content": _truncate_text(part.content),
                "id": part.id,
            }
        )
    if isinstance(part, ThinkingPart):
        return _compact_dict(
            {
                "type": "thinking",
                "role": "assistant",
                "content": _truncate_text(part.content),
                "id": part.id,
            }
        )
    if isinstance(part, ToolCallPart | BuiltinToolCallPart):
        serialized = _serialize_tool_call(part)
        serialized["role"] = "assistant"
        return serialized
    if isinstance(part, BuiltinToolReturnPart):
        return _serialize_tool_return(part)
    if isinstance(part, FilePart):
        return _compact_dict(
            {
                "type": "file",
                "role": "assistant",
                "description": _describe_binary_content(part.content),
                "id": part.id,
            }
        )
    if hasattr(part, "content"):
        return _compact_dict(
            {
                "type": getattr(part, "part_kind", type(part).__name__),
                "role": "assistant",
                "content": _truncate_text(str(part.content)),
            }
        )
    return _compact_dict(
        {
            "type": getattr(part, "part_kind", type(part).__name__),
            "role": "assistant",
            "repr": repr(part),
        }
    )


def _serialize_model_message(
    message: ModelMessage,
    *,
    include_instructions: bool,
) -> dict[str, Any]:
    """Serialize a model message into a reflection-friendly record."""
    if isinstance(message, ModelRequest):
        data = {
            "kind": "request",
            "parts": [_serialize_request_part(part) for part in message.parts],
        }
        if include_instructions and message.instructions is not None:
            data["instructions"] = message.instructions
        return _compact_dict(data)
    if isinstance(message, ModelResponse):
        base: dict[str, Any] = {
            "kind": "response",
            "model_name": message.model_name,
            "finish_reason": message.finish_reason,
            "timestamp": _timestamp_iso(message.timestamp),
            "parts": [_serialize_response_part(part) for part in message.parts],
        }
        return _compact_dict(base)
    return _compact_dict(
        {
            "kind": getattr(message, "kind", type(message).__name__),
            "repr": repr(message),
        }
    )


@dataclass
class Trajectory:
    """Execution trajectory capturing the agent run."""

    messages: list[ModelMessage]
    final_output: Any
    instructions: str | None = None
    error: str | None = None
    usage: dict[str, int] = field(default_factory=dict)
    data_inst: DataInst | None = None
    metric_feedback: str | None = None

    def _extract_system_prompt(self) -> str | None:
        """Extract the agent instructions/system prompt once for reflection."""
        if self.instructions:
            return self.instructions
        for msg in self.messages:
            if isinstance(msg, ModelRequest):
                if msg.instructions:
                    return msg.instructions
                for part in msg.parts:
                    if isinstance(part, SystemPromptPart):
                        return part.content
        return None

    def _extract_user_content(self, part: UserPromptPart) -> str:
        """Extract text content from a UserPromptPart."""
        if isinstance(part.content, str):
            return part.content
        elif part.content:
            # For multi-modal content, just take the first text content
            for content_item in part.content:
                if isinstance(content_item, str):
                    return content_item
            return "Multi-modal content"
        else:
            return "No content"

    def _extract_user_message(self) -> str | None:
        """Extract the first user message from the trajectory."""
        for msg in self.messages:
            if isinstance(msg, ModelRequest):
                # Look for UserPromptPart in request parts
                for part in msg.parts:
                    if isinstance(part, UserPromptPart):
                        return self._extract_user_content(part)
        return None

    def _extract_assistant_message(self) -> str | None:
        """Extract the last assistant message from the trajectory."""
        for msg in reversed(self.messages):
            if isinstance(msg, ModelResponse):
                # Look for TextPart in response parts
                for part in msg.parts:
                    if isinstance(part, TextPart):
                        return part.content
        return None

    def _extract_tool_trace(
        self,
        *,
        max_tool_return_chars: int = DEFAULT_MAX_TOOL_RETURN_CHARS,
    ) -> list[dict[str, Any]]:
        """Extract only tool calls and returns from the message transcript."""
        trace: list[dict[str, Any]] = []
        for msg in self.messages:
            parts = getattr(msg, "parts", ())
            for part in parts:
                if isinstance(part, ToolCallPart | BuiltinToolCallPart):
                    trace.append(_serialize_tool_call(part))
                elif isinstance(part, ToolReturnPart | BuiltinToolReturnPart):
                    trace.append(
                        _serialize_tool_return(
                            part,
                            max_chars=max_tool_return_chars,
                        )
                    )
                elif isinstance(part, RetryPromptPart):
                    trace.append(_serialize_retry_part(part))
        return trace

    def to_reflective_record(
        self,
        *,
        max_tool_return_chars: int = DEFAULT_MAX_TOOL_RETURN_CHARS,
    ) -> dict[str, Any]:
        """Convert to a compact record for reflection."""
        system_prompt = self._extract_system_prompt()
        user_msg = self._extract_user_message()
        assistant_msg = self._extract_assistant_message()
        tool_trace = self._extract_tool_trace(
            max_tool_return_chars=max_tool_return_chars
        )

        # Format the final output appropriately
        if assistant_msg:
            response = assistant_msg
        elif self.final_output is not None:
            # Check if it's a Pydantic model and serialize to JSON
            if isinstance(self.final_output, BaseModel):
                response = self.final_output.model_dump_json()
            else:
                response = str(self.final_output)
        else:
            response = "No output"

        record = {
            "system_prompt": system_prompt,
            "user_prompt": user_msg or "No user message",
            "assistant_response": response,
            "error": self.error,
            "tool_trace": tool_trace or None,
        }
        return _compact_dict(record)

    def _serialize_messages_with_instructions(self) -> list[dict[str, Any]]:
        """Serialize messages, attaching instructions only to the first request."""
        serialized: list[dict[str, Any]] = []
        instructions_recorded = False
        for message in self.messages:
            include_instructions = (
                isinstance(message, ModelRequest)
                and not instructions_recorded
                and getattr(message, "instructions", None) is not None
            )
            if include_instructions:
                instructions_recorded = True
            serialized.append(
                _serialize_model_message(
                    message, include_instructions=include_instructions
                )
            )
        return serialized


@dataclass
class RolloutOutput(Generic[OutputT]):
    """Output from a single agent execution.

    Generic type parameter OutputT specifies the expected type of the result
    when the execution is successful.
    """

    result: OutputT | None
    success: bool
    error_message: str | None = None

    @classmethod
    def from_success(cls, result: OutputT) -> RolloutOutput[OutputT]:
        """Create from successful execution."""
        return cls(result=result, success=True)

    @classmethod
    def from_error(cls, error: Exception) -> RolloutOutput[OutputT]:
        """Create from failed execution."""
        return cls(result=None, success=False, error_message=str(error))
