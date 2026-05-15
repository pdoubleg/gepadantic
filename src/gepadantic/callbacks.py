"""Structured callbacks for GEPA optimization runs.

This module maps GEPA's raw callback TypedDicts into a smaller event shape that
is easier to stream to terminals, APIs, and frontends.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from datetime import datetime, timezone
from pathlib import Path
from queue import Full
from typing import Any, Literal, Protocol, TextIO

from gepa.core.callbacks import (
    BudgetUpdatedEvent,
    CandidateAcceptedEvent,
    CandidateRejectedEvent,
    CandidateSelectedEvent,
    ErrorEvent,
    EvaluationEndEvent,
    EvaluationSkippedEvent,
    EvaluationStartEvent,
    IterationEndEvent,
    IterationStartEvent,
    MergeAcceptedEvent,
    MergeAttemptedEvent,
    MergeRejectedEvent,
    MinibatchSampledEvent,
    OptimizationEndEvent,
    OptimizationStartEvent,
    ParetoFrontUpdatedEvent,
    ProposalEndEvent,
    ProposalStartEvent,
    ReflectiveDatasetBuiltEvent,
    StateSavedEvent,
    ValsetEvaluatedEvent,
)
from pydantic import BaseModel, Field
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text


EventLevel = Literal["debug", "info", "warning", "error"]


class OptimizationRunEvent(BaseModel):
    """Frontend-friendly event emitted during a GEPA optimization run."""

    type: str
    """Stable event type, e.g. ``proposal_created`` or ``budget_updated``."""

    message: str
    """Human-readable summary suitable for logs or status text."""

    iteration: int | None = None
    """GEPA iteration associated with the event, when available."""

    level: EventLevel = "info"
    """Event severity for display and filtering."""

    data: dict[str, Any] = Field(default_factory=dict)
    """JSON-friendly event metadata."""

    rich: dict[str, Any] = Field(default_factory=dict)
    """Optional structured content for richer clients."""

    created_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    """UTC timestamp for streaming consumers."""


class EventEmitter(Protocol):
    """Callable that consumes normalized optimization events."""

    def __call__(self, event: OptimizationRunEvent) -> None:
        """Consume an event."""
        ...


def _truncate_text(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    omitted = len(text) - max_chars
    return f"{text[:max_chars]}... [truncated {omitted} chars]"


def _json_safe(value: Any, *, max_text_chars: int, depth: int = 0) -> Any:
    """Best-effort conversion to JSON-friendly data for events."""
    if depth > 4:
        return repr(value)
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        return _truncate_text(value, max_text_chars)
    if isinstance(value, BaseModel):
        return _json_safe(
            value.model_dump(mode="json"),
            max_text_chars=max_text_chars,
            depth=depth + 1,
        )
    if isinstance(value, dict):
        return {
            str(key): _json_safe(
                item,
                max_text_chars=max_text_chars,
                depth=depth + 1,
            )
            for key, item in value.items()
        }
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [
            _json_safe(item, max_text_chars=max_text_chars, depth=depth + 1)
            for item in value
        ]
    return _truncate_text(repr(value), max_text_chars)


def _score_summary(scores: Sequence[float] | None) -> dict[str, Any]:
    if not scores:
        return {"count": 0}
    values = list(scores)
    return {
        "count": len(values),
        "average": sum(values) / len(values),
        "min": min(values),
        "max": max(values),
    }


def _component_names(candidate: dict[str, str] | None) -> list[str]:
    return sorted(candidate.keys()) if candidate else []


def _record_count(dataset: dict[str, list[dict[str, Any]]] | None) -> int:
    if not dataset:
        return 0
    return sum(len(records) for records in dataset.values())


def _event_iteration(event: dict[str, Any]) -> int | None:
    value = event.get("iteration")
    return value if isinstance(value, int) else None


class OptimizationEventCallback:
    """Map GEPA callback events to normalized ``OptimizationRunEvent`` objects.

    Args:
        emit: Callable that receives normalized events.
        include_component_text: Include proposed/candidate component text in event
            payloads. Keep this disabled for untrusted frontend streams.
        include_raw_payload: Include a sanitized copy of the raw GEPA event.
        max_text_chars: Maximum characters retained for any single text field.
    """

    def __init__(
        self,
        emit: EventEmitter | None = None,
        *,
        include_component_text: bool = False,
        include_raw_payload: bool = False,
        max_text_chars: int = 2000,
    ) -> None:
        self.emit = emit
        self.include_component_text = include_component_text
        self.include_raw_payload = include_raw_payload
        self.max_text_chars = max_text_chars

    def _emit(
        self,
        event_type: str,
        message: str,
        *,
        iteration: int | None = None,
        level: EventLevel = "info",
        data: dict[str, Any] | None = None,
        rich: dict[str, Any] | None = None,
        raw_event: dict[str, Any] | None = None,
    ) -> None:
        payload = data.copy() if data else {}
        if self.include_raw_payload and raw_event is not None:
            payload["raw_event"] = _json_safe(
                raw_event,
                max_text_chars=self.max_text_chars,
            )
        normalized = OptimizationRunEvent(
            type=event_type,
            message=message,
            iteration=iteration,
            level=level,
            data=_json_safe(payload, max_text_chars=self.max_text_chars),
            rich=_json_safe(rich or {}, max_text_chars=self.max_text_chars),
        )
        self.handle_event(normalized)

    def handle_event(self, event: OptimizationRunEvent) -> None:
        """Handle a normalized event.

        Subclasses can override this to render, enqueue, or persist events.
        """
        if self.emit is not None:
            self.emit(event)

    def on_optimization_start(self, event: OptimizationStartEvent) -> None:
        config = event.get("config", {})
        data = {
            "trainset_size": event["trainset_size"],
            "valset_size": event["valset_size"],
            "seed_components": _component_names(event.get("seed_candidate")),
            "max_metric_calls": config.get("max_metric_calls"),
            "candidate_selection_strategy": config.get("candidate_selection_strategy"),
            "module_selector": config.get("module_selector"),
        }
        self._emit(
            "run_started",
            (
                "GEPA optimization started "
                f"({event['trainset_size']} train, {event['valset_size']} val)"
            ),
            data=data,
            raw_event=event,
        )

    def on_optimization_end(self, event: OptimizationEndEvent) -> None:
        self._emit(
            "run_completed",
            (
                "GEPA optimization completed "
                f"after {event['total_iterations']} iterations"
            ),
            data={
                "best_candidate_idx": event["best_candidate_idx"],
                "total_iterations": event["total_iterations"],
                "total_metric_calls": event["total_metric_calls"],
            },
            raw_event=event,
        )

    def on_iteration_start(self, event: IterationStartEvent) -> None:
        self._emit(
            "iteration_started",
            f"Iteration {event['iteration']} started",
            iteration=event["iteration"],
            raw_event=event,
        )

    def on_iteration_end(self, event: IterationEndEvent) -> None:
        status = "accepted" if event["proposal_accepted"] else "rejected"
        self._emit(
            "iteration_completed",
            f"Iteration {event['iteration']} completed ({status})",
            iteration=event["iteration"],
            data={"proposal_accepted": event["proposal_accepted"]},
            raw_event=event,
        )

    def on_candidate_selected(self, event: CandidateSelectedEvent) -> None:
        data: dict[str, Any] = {
            "candidate_idx": event["candidate_idx"],
            "score": event["score"],
            "components": _component_names(event.get("candidate")),
        }
        if self.include_component_text:
            data["candidate"] = event["candidate"]
        self._emit(
            "candidate_selected",
            (
                f"Selected candidate {event['candidate_idx']} "
                f"(score {event['score']:.4f})"
            ),
            iteration=event["iteration"],
            data=data,
            raw_event=event,
        )

    def on_minibatch_sampled(self, event: MinibatchSampledEvent) -> None:
        self._emit(
            "minibatch_sampled",
            f"Sampled {len(event['minibatch_ids'])} training examples",
            iteration=event["iteration"],
            data={
                "minibatch_size": len(event["minibatch_ids"]),
                "trainset_size": event["trainset_size"],
            },
            raw_event=event,
        )

    def on_evaluation_start(self, event: EvaluationStartEvent) -> None:
        self._emit(
            "evaluation_started",
            f"Evaluating {event['batch_size']} examples",
            iteration=event["iteration"],
            data={
                "candidate_idx": event["candidate_idx"],
                "batch_size": event["batch_size"],
                "capture_traces": event["capture_traces"],
                "is_seed_candidate": event["is_seed_candidate"],
                "parent_ids": list(event["parent_ids"]),
            },
            raw_event=event,
        )

    def on_evaluation_end(self, event: EvaluationEndEvent) -> None:
        summary = _score_summary(event.get("scores"))
        average = summary.get("average")
        suffix = f" (avg {average:.4f})" if isinstance(average, float) else ""
        self._emit(
            "evaluation_completed",
            f"Evaluation completed{suffix}",
            iteration=event["iteration"],
            data={
                "candidate_idx": event["candidate_idx"],
                "scores": summary,
                "has_trajectories": event["has_trajectories"],
                "is_seed_candidate": event["is_seed_candidate"],
                "parent_ids": list(event["parent_ids"]),
                "objective_scores_count": len(event["objective_scores"] or []),
            },
            raw_event=event,
        )

    def on_evaluation_skipped(self, event: EvaluationSkippedEvent) -> None:
        self._emit(
            "evaluation_skipped",
            f"Evaluation skipped: {event['reason']}",
            iteration=event["iteration"],
            level="debug",
            data={
                "candidate_idx": event["candidate_idx"],
                "reason": event["reason"],
                "scores": _score_summary(event.get("scores")),
                "is_seed_candidate": event["is_seed_candidate"],
            },
            raw_event=event,
        )

    def on_valset_evaluated(self, event: ValsetEvaluatedEvent) -> None:
        self._emit(
            "validation_evaluated",
            (
                f"Validation score {event['average_score']:.4f} "
                f"on {event['num_examples_evaluated']}/"
                f"{event['total_valset_size']} examples"
            ),
            iteration=event["iteration"],
            data={
                "candidate_idx": event["candidate_idx"],
                "average_score": event["average_score"],
                "num_examples_evaluated": event["num_examples_evaluated"],
                "total_valset_size": event["total_valset_size"],
                "parent_ids": list(event["parent_ids"]),
                "is_best_program": event["is_best_program"],
            },
            raw_event=event,
        )

    def on_reflective_dataset_built(self, event: ReflectiveDatasetBuiltEvent) -> None:
        self._emit(
            "reflective_dataset_built",
            f"Built reflective dataset for {len(event['components'])} components",
            iteration=event["iteration"],
            data={
                "candidate_idx": event["candidate_idx"],
                "components": event["components"],
                "record_count": _record_count(event.get("dataset")),
            },
            raw_event=event,
        )

    def on_proposal_start(self, event: ProposalStartEvent) -> None:
        self._emit(
            "proposal_started",
            f"Proposing updates for {len(event['components'])} components",
            iteration=event["iteration"],
            data={
                "components": event["components"],
                "record_count": _record_count(event.get("reflective_dataset")),
            },
            raw_event=event,
        )

    def on_proposal_end(self, event: ProposalEndEvent) -> None:
        new_instructions = event["new_instructions"]
        data: dict[str, Any] = {
            "components": _component_names(new_instructions),
            "component_count": len(new_instructions),
        }
        rich: dict[str, Any] = {}
        if self.include_component_text:
            data["new_instructions"] = new_instructions
            rich["component_updates"] = [
                {"component": component, "text": text}
                for component, text in new_instructions.items()
            ]
        self._emit(
            "proposal_created",
            f"Proposed updates for {len(new_instructions)} components",
            iteration=event["iteration"],
            data=data,
            rich=rich,
            raw_event=event,
        )

    def on_candidate_accepted(self, event: CandidateAcceptedEvent) -> None:
        self._emit(
            "candidate_accepted",
            (
                f"Accepted candidate {event['new_candidate_idx']} "
                f"(score {event['new_score']:.4f})"
            ),
            iteration=event["iteration"],
            data={
                "new_candidate_idx": event["new_candidate_idx"],
                "new_score": event["new_score"],
                "parent_ids": list(event["parent_ids"]),
            },
            raw_event=event,
        )

    def on_candidate_rejected(self, event: CandidateRejectedEvent) -> None:
        self._emit(
            "candidate_rejected",
            f"Rejected candidate: {event['reason']}",
            iteration=event["iteration"],
            data={
                "old_score": event["old_score"],
                "new_score": event["new_score"],
                "reason": event["reason"],
            },
            raw_event=event,
        )

    def on_merge_attempted(self, event: MergeAttemptedEvent) -> None:
        data: dict[str, Any] = {"parent_ids": list(event["parent_ids"])}
        if self.include_component_text:
            data["merged_candidate"] = event["merged_candidate"]
        self._emit(
            "merge_attempted",
            "Attempted candidate merge",
            iteration=event["iteration"],
            data=data,
            raw_event=event,
        )

    def on_merge_accepted(self, event: MergeAcceptedEvent) -> None:
        self._emit(
            "merge_accepted",
            f"Accepted merged candidate {event['new_candidate_idx']}",
            iteration=event["iteration"],
            data={
                "new_candidate_idx": event["new_candidate_idx"],
                "parent_ids": list(event["parent_ids"]),
            },
            raw_event=event,
        )

    def on_merge_rejected(self, event: MergeRejectedEvent) -> None:
        self._emit(
            "merge_rejected",
            f"Rejected merge: {event['reason']}",
            iteration=event["iteration"],
            data={"parent_ids": list(event["parent_ids"]), "reason": event["reason"]},
            raw_event=event,
        )

    def on_pareto_front_updated(self, event: ParetoFrontUpdatedEvent) -> None:
        self._emit(
            "pareto_front_updated",
            f"Pareto front now has {len(event['new_front'])} candidates",
            iteration=event["iteration"],
            data={
                "new_front": event["new_front"],
                "displaced_candidates": event["displaced_candidates"],
            },
            raw_event=event,
        )

    def on_state_saved(self, event: StateSavedEvent) -> None:
        self._emit(
            "state_saved",
            "GEPA state saved",
            iteration=event["iteration"],
            data={"run_dir": event["run_dir"]},
            raw_event=event,
        )

    def on_budget_updated(self, event: BudgetUpdatedEvent) -> None:
        remaining = event["metric_calls_remaining"]
        remaining_text = "unknown" if remaining is None else str(remaining)
        self._emit(
            "budget_updated",
            (
                f"Metric calls used: {event['metric_calls_used']} "
                f"({remaining_text} remaining)"
            ),
            iteration=event["iteration"],
            data={
                "metric_calls_used": event["metric_calls_used"],
                "metric_calls_delta": event["metric_calls_delta"],
                "metric_calls_remaining": remaining,
            },
            raw_event=event,
        )

    def on_error(self, event: ErrorEvent) -> None:
        self._emit(
            "run_error",
            str(event["exception"]),
            iteration=_event_iteration(event),
            level="error",
            data={
                "exception_type": type(event["exception"]).__name__,
                "will_continue": event["will_continue"],
            },
            raw_event=event,
        )


class EventBufferCallback(OptimizationEventCallback):
    """Collect normalized events in memory for tests or later inspection."""

    def __init__(
        self,
        *,
        max_events: int | None = None,
        include_component_text: bool = False,
        include_raw_payload: bool = False,
        max_text_chars: int = 2000,
    ) -> None:
        super().__init__(
            include_component_text=include_component_text,
            include_raw_payload=include_raw_payload,
            max_text_chars=max_text_chars,
        )
        self.max_events = max_events
        self.events: list[OptimizationRunEvent] = []

    def handle_event(self, event: OptimizationRunEvent) -> None:
        self.events.append(event)
        if self.max_events is not None and len(self.events) > self.max_events:
            del self.events[: len(self.events) - self.max_events]


class QueueOptimizationCallback(OptimizationEventCallback):
    """Push normalized events into a sync queue-like object.

    The queue should expose either ``put_nowait(event)`` or ``put(event)``. Events
    can be pushed as Pydantic models or JSON-friendly dictionaries.
    """

    def __init__(
        self,
        queue: Any,
        *,
        as_dict: bool = True,
        drop_when_full: bool = True,
        include_component_text: bool = False,
        include_raw_payload: bool = False,
        max_text_chars: int = 2000,
    ) -> None:
        super().__init__(
            include_component_text=include_component_text,
            include_raw_payload=include_raw_payload,
            max_text_chars=max_text_chars,
        )
        self.queue = queue
        self.as_dict = as_dict
        self.drop_when_full = drop_when_full
        self.dropped_events = 0

    def handle_event(self, event: OptimizationRunEvent) -> None:
        item = event.model_dump(mode="json") if self.as_dict else event
        try:
            if hasattr(self.queue, "put_nowait"):
                self.queue.put_nowait(item)
            else:
                self.queue.put(item)
        except Full:
            if not self.drop_when_full:
                raise
            self.dropped_events += 1


class JSONLRunLogger(OptimizationEventCallback):
    """Write normalized optimization events to a JSON Lines file.

    The logger uses the standard library by default. Set ``use_structlog=True``
    to render records through ``structlog.processors.JSONRenderer`` when
    structlog is installed by the host application.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        append: bool = True,
        flush: bool = True,
        run_id: str | None = None,
        extra_context: dict[str, Any] | None = None,
        use_structlog: bool = False,
        include_component_text: bool = False,
        include_raw_payload: bool = False,
        max_text_chars: int = 2000,
        encoding: str = "utf-8",
    ) -> None:
        super().__init__(
            include_component_text=include_component_text,
            include_raw_payload=include_raw_payload,
            max_text_chars=max_text_chars,
        )
        self.path = Path(path)
        self.append = append
        self.flush = flush
        self.run_id = run_id
        self.extra_context = extra_context or {}
        self.use_structlog = use_structlog
        self.encoding = encoding
        self._file: TextIO | None = None
        self._structlog_renderer: Any | None = None

        if use_structlog:
            try:
                import structlog

                self._structlog_renderer = structlog.processors.JSONRenderer()
            except ImportError as exc:
                raise ImportError(
                    "structlog is required when use_structlog=True. "
                    "Install structlog or use the default JSON renderer."
                ) from exc

    def __enter__(self) -> JSONLRunLogger:
        self.open()
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def open(self) -> None:
        """Open the JSONL file if it is not already open."""
        if self._file is not None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        mode = "a" if self.append else "w"
        self._file = self.path.open(mode, encoding=self.encoding)

    def close(self) -> None:
        """Close the JSONL file if open."""
        if self._file is None:
            return
        self._file.close()
        self._file = None

    def handle_event(self, event: OptimizationRunEvent) -> None:
        record = event.model_dump(mode="json")
        if self.run_id is not None:
            record["run_id"] = self.run_id
        if self.extra_context:
            record["context"] = _json_safe(
                self.extra_context,
                max_text_chars=self.max_text_chars,
            )

        self.open()
        assert self._file is not None
        self._file.write(self._render_record(record))
        self._file.write("\n")
        if self.flush:
            self._file.flush()

    def _render_record(self, record: dict[str, Any]) -> str:
        if self._structlog_renderer is not None:
            return self._structlog_renderer(None, "optimization_event", record)
        return json.dumps(record, ensure_ascii=False, default=str)


class RichConsoleCallback(OptimizationEventCallback):
    """Render selected GEPA callback events with Rich without parsing log text."""

    def __init__(
        self,
        *,
        console: Console | None = None,
        proposal_color: str = "cyan",
        proposal_border_style: str = "bold cyan",
        include_component_text: bool = True,
        max_text_chars: int = 2000,
        show_debug_events: bool = False,
    ) -> None:
        super().__init__(
            include_component_text=include_component_text,
            max_text_chars=max_text_chars,
        )
        self.console = console if console is not None else Console(force_terminal=True)
        self.proposal_color = proposal_color
        self.proposal_border_style = proposal_border_style
        self.show_debug_events = show_debug_events

    def handle_event(self, event: OptimizationRunEvent) -> None:
        if event.level == "debug" and not self.show_debug_events:
            return
        if event.type == "proposal_created":
            self._render_proposal(event)
        elif event.type in {
            "run_started",
            "run_completed",
            "candidate_accepted",
            "candidate_rejected",
            "validation_evaluated",
            "run_error",
        }:
            self._render_status(event)

    def _render_status(self, event: OptimizationRunEvent) -> None:
        style = {
            "error": "bold red",
            "warning": "yellow",
            "info": "white",
            "debug": "dim",
        }[event.level]
        self.console.print(event.message, style=style)

    def _render_proposal(self, event: OptimizationRunEvent) -> None:
        updates = event.rich.get("component_updates") or []
        if not updates:
            self.console.print(event.message)
            return

        for update in updates:
            component = str(update.get("component", "component"))
            content = str(update.get("text", ""))
            display_content = _truncate_text(content, self.max_text_chars)

            title_text = Text()
            if event.iteration is not None:
                title_text.append(f"Iteration {event.iteration}: ", style="bold white")
            title_text.append(component, style=f"bold {self.proposal_color}")

            panel = Panel(
                display_content,
                title=title_text,
                border_style=self.proposal_border_style,
                padding=(1, 2),
            )
            self.console.print(panel)

    def render_event_table(self, events: Sequence[OptimizationRunEvent]) -> None:
        """Render a compact event timeline table."""
        table = Table(show_header=True, header_style="bold cyan")
        table.add_column("Time")
        table.add_column("Type")
        table.add_column("Iter")
        table.add_column("Message")
        for event in events:
            table.add_row(
                event.created_at.split("T", 1)[1].split(".", 1)[0],
                event.type,
                "" if event.iteration is None else str(event.iteration),
                event.message,
            )
        self.console.print(table)


def event_stream_callback(
    emit: Callable[[dict[str, Any]], None],
    *,
    include_component_text: bool = False,
    include_raw_payload: bool = False,
    max_text_chars: int = 2000,
) -> OptimizationEventCallback:
    """Create a callback that emits JSON-friendly dictionaries.

    This is a convenient bridge for SSE/WebSocket layers.
    """

    def _emit(event: OptimizationRunEvent) -> None:
        emit(event.model_dump(mode="json"))

    return OptimizationEventCallback(
        _emit,
        include_component_text=include_component_text,
        include_raw_payload=include_raw_payload,
        max_text_chars=max_text_chars,
    )
