"""Import externally produced agent trajectories from JSONL.

One JSON object per line:

    {"run_id": "abc", "success": false,
     "steps": [
       {"tool": "bash", "arguments": {"cmd": "ls"}, "result": "a.txt"},
       {"tool": "bash", "arguments": {"cmd": "cat x"}, "error": "No such file"}
     ]}

``success`` is optional (null/absent = unlabeled). A step with a non-empty
``error`` becomes TOOL_CALL + TOOL_ERROR, otherwise TOOL_CALL + TOOL_RESULT.
RUN_STARTED / RUN_FINISHED are added so length-based features line up with
trajectories recorded natively by this project.

Caveat: this maps *your* agent's steps onto this project's event model. If
your tool names, error conventions or step granularity differ a lot from the
synthetic data the predictor was trained on, scores will not transfer. Check
the labeled AUROC before trusting them.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from arl.domain.events import EventType, TrajectoryEvent


class TrajectoryFormatError(ValueError):
    """A line in the input file does not match the schema."""


@dataclass(frozen=True)
class ImportedTrajectory:
    run_id: str
    success: bool | None
    events: list[TrajectoryEvent]


def _parse_line(obj: Any, line_no: int) -> ImportedTrajectory:
    if not isinstance(obj, dict):
        raise TrajectoryFormatError(f"line {line_no}: expected a JSON object")
    run_id = obj.get("run_id")
    steps = obj.get("steps")
    if not isinstance(run_id, str) or not run_id:
        raise TrajectoryFormatError(f"line {line_no}: 'run_id' must be a non-empty string")
    if not isinstance(steps, list):
        raise TrajectoryFormatError(f"line {line_no}: 'steps' must be a list")
    success = obj.get("success")
    if success is not None and not isinstance(success, bool):
        raise TrajectoryFormatError(f"line {line_no}: 'success' must be true, false or null")

    events: list[TrajectoryEvent] = []

    def add(event_type: EventType, payload: dict[str, Any]) -> None:
        events.append(
            TrajectoryEvent(
                run_id=run_id, step_index=len(events), event_type=event_type, payload=payload
            )
        )

    add(EventType.RUN_STARTED, {"kind": "run_lifecycle"})
    for i, step in enumerate(steps):
        if not isinstance(step, dict) or not isinstance(step.get("tool"), str):
            raise TrajectoryFormatError(f"line {line_no}: step {i} needs a string 'tool'")
        tool = step["tool"]
        arguments = step.get("arguments", {})
        if not isinstance(arguments, dict):
            raise TrajectoryFormatError(f"line {line_no}: step {i} 'arguments' must be an object")
        add(
            EventType.TOOL_CALL,
            {"kind": "tool_call", "tool_name": tool, "arguments": arguments},
        )
        error = step.get("error")
        if error:
            add(
                EventType.TOOL_ERROR,
                {"kind": "tool_error", "tool_name": tool, "message": str(error)},
            )
        else:
            add(
                EventType.TOOL_RESULT,
                {
                    "kind": "tool_result",
                    "tool_name": tool,
                    "success": True,
                    "result_summary": str(step.get("result", "")),
                },
            )
    add(EventType.RUN_FINISHED, {"kind": "run_lifecycle"})
    return ImportedTrajectory(run_id=run_id, success=success, events=events)


def load_jsonl(path: str | Path) -> Iterator[ImportedTrajectory]:
    with open(path, encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise TrajectoryFormatError(f"line {line_no}: invalid JSON: {exc.msg}") from exc
            yield _parse_line(obj, line_no)
