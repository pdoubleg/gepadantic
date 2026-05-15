"""Tests for structured GEPA callbacks."""

from __future__ import annotations

import json
from pathlib import Path
from queue import Queue
from uuid import uuid4

from rich.console import Console

from gepadantic.callbacks import (
    EventBufferCallback,
    JSONLRunLogger,
    OptimizationRunEvent,
    QueueOptimizationCallback,
    RichConsoleCallback,
    event_stream_callback,
)


def _temp_jsonl_path() -> Path:
    root = Path(".test-artifacts")
    root.mkdir(parents=True, exist_ok=True)
    return root / f"{uuid4().hex}.jsonl"


def test_event_buffer_maps_proposal_with_component_text():
    callback = EventBufferCallback(include_component_text=True)

    callback.on_proposal_end(
        {
            "iteration": 3,
            "new_instructions": {
                "instructions": "Classify the audit form carefully.",
                "tool:final_result:description": "Return normalized findings.",
            },
        }
    )

    event = callback.events[0]
    assert event.type == "proposal_created"
    assert event.iteration == 3
    assert event.data["component_count"] == 2
    assert event.data["new_instructions"]["instructions"].startswith("Classify")
    assert event.rich["component_updates"][0]["component"] == "instructions"


def test_event_buffer_omits_component_text_by_default():
    callback = EventBufferCallback()

    callback.on_candidate_selected(
        {
            "iteration": 1,
            "candidate_idx": 0,
            "candidate": {"instructions": "Sensitive prompt text"},
            "score": 0.75,
        }
    )

    event = callback.events[0]
    assert event.type == "candidate_selected"
    assert event.data["components"] == ["instructions"]
    assert "candidate" not in event.data


def test_queue_callback_emits_json_friendly_dicts():
    queue: Queue[dict[str, object]] = Queue()
    callback = QueueOptimizationCallback(queue)

    callback.on_budget_updated(
        {
            "iteration": 2,
            "metric_calls_used": 12,
            "metric_calls_delta": 3,
            "metric_calls_remaining": 18,
        }
    )

    item = queue.get_nowait()
    assert item["type"] == "budget_updated"
    assert item["iteration"] == 2
    assert item["data"]["metric_calls_remaining"] == 18


def test_event_stream_callback_emits_dicts():
    emitted: list[dict[str, object]] = []
    callback = event_stream_callback(emitted.append)

    callback.on_candidate_accepted(
        {
            "iteration": 4,
            "new_candidate_idx": 7,
            "new_score": 0.91,
            "parent_ids": [2],
        }
    )

    assert emitted[0]["type"] == "candidate_accepted"
    assert emitted[0]["data"]["new_candidate_idx"] == 7


def test_jsonl_run_logger_writes_events():
    path = _temp_jsonl_path()
    logger = JSONLRunLogger(
        path,
        run_id="run-123",
        extra_context={"project": "audit-forms"},
    )

    logger.on_optimization_start(
        {
            "seed_candidate": {"instructions": "Classify forms"},
            "trainset_size": 4,
            "valset_size": 2,
            "config": {
                "max_metric_calls": 20,
                "candidate_selection_strategy": "pareto",
                "module_selector": "all",
            },
        }
    )
    logger.on_budget_updated(
        {
            "iteration": 1,
            "metric_calls_used": 5,
            "metric_calls_delta": 5,
            "metric_calls_remaining": 15,
        }
    )
    logger.close()

    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    first = json.loads(lines[0])
    second = json.loads(lines[1])
    assert first["type"] == "run_started"
    assert first["run_id"] == "run-123"
    assert first["context"]["project"] == "audit-forms"
    assert second["type"] == "budget_updated"
    assert second["data"]["metric_calls_used"] == 5
    path.unlink(missing_ok=True)


def test_jsonl_run_logger_context_manager():
    path = _temp_jsonl_path()

    with JSONLRunLogger(path, append=False) as logger:
        logger.handle_event(
            OptimizationRunEvent(
                type="custom",
                message="Custom event",
                iteration=9,
                data={"value": 42},
            )
        )

    record = json.loads(path.read_text(encoding="utf-8"))
    assert record["type"] == "custom"
    assert record["data"]["value"] == 42
    path.unlink(missing_ok=True)


def test_rich_console_callback_renders_proposal():
    console = Console(record=True, force_terminal=False)
    callback = RichConsoleCallback(console=console)

    callback.handle_event(
        OptimizationRunEvent(
            type="proposal_created",
            message="Proposed updates",
            iteration=5,
            rich={
                "component_updates": [
                    {
                        "component": "instructions",
                        "text": "Be precise and cite the relevant audit fields.",
                    }
                ]
            },
        )
    )

    assert "instructions" in console.export_text()
