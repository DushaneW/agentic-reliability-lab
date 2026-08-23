"""Pattern detectors over a trajectory's event log.

Each detector returns evidence only when it finds a concrete, checkable
pattern (specific event ids), never a hunch. `analysis/engine.py` combines
detector outputs into a single FailureDiagnosis.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from arl.domain.events import EventType, TrajectoryEvent


@dataclass
class DetectorFinding:
    description: str
    event_ids: list[str]
    weight: float  # relative signal strength, used to rank/combine findings


def _tool_calls(events: list[TrajectoryEvent]) -> list[TrajectoryEvent]:
    return [e for e in events if e.event_type == EventType.TOOL_CALL]


def _tool_errors(events: list[TrajectoryEvent]) -> list[TrajectoryEvent]:
    return [e for e in events if e.event_type == EventType.TOOL_ERROR]


def detect_looping(events: list[TrajectoryEvent]) -> DetectorFinding | None:
    """Two or more consecutive tool calls with identical name+arguments."""
    calls = _tool_calls(events)
    repeats: list[str] = []
    for prev, cur in zip(calls, calls[1:], strict=False):
        if prev.payload.get("tool_name") == cur.payload.get("tool_name") and prev.payload.get(
            "arguments"
        ) == cur.payload.get("arguments"):
            repeats.extend([prev.id, cur.id])
    if not repeats:
        return None
    return DetectorFinding(
        description=f"{len(repeats)} events are part of an identical repeated tool call",
        event_ids=sorted(set(repeats)),
        weight=len(set(repeats)) / max(len(calls), 1),
    )


def detect_error_recovery_failure(events: list[TrajectoryEvent]) -> DetectorFinding | None:
    """A tool_error immediately followed by the identical tool call again."""
    hits: list[str] = []
    for i, event in enumerate(events):
        if event.event_type != EventType.TOOL_ERROR:
            continue
        # find the tool_call this error corresponds to (previous event)
        if i == 0:
            continue
        failed_call = events[i - 1]
        for later in events[i + 1 :]:
            if later.event_type == EventType.TOOL_CALL:
                if later.payload.get("tool_name") == failed_call.payload.get(
                    "tool_name"
                ) and later.payload.get("arguments") == failed_call.payload.get("arguments"):
                    hits.extend([event.id, later.id])
                break
    if not hits:
        return None
    return DetectorFinding(
        description="a tool error was followed by retrying the exact same failing call",
        event_ids=sorted(set(hits)),
        weight=0.9,
    )


def detect_repeated_tool_errors(events: list[TrajectoryEvent]) -> DetectorFinding | None:
    errors = _tool_errors(events)
    if len(errors) < 2:
        return None
    by_tool = Counter(e.payload.get("tool_name", "") for e in errors)
    worst_tool, count = by_tool.most_common(1)[0]
    if count < 2:
        return None
    ids = [e.id for e in errors if e.payload.get("tool_name") == worst_tool]
    return DetectorFinding(
        description=f"tool '{worst_tool}' failed {count} times in this run",
        event_ids=ids,
        weight=min(count / 5, 1.0),
    )


def detect_premature_termination(
    events: list[TrajectoryEvent], max_steps: int, success: bool
) -> DetectorFinding | None:
    if success:
        return None
    finished = [
        e
        for e in events
        if e.event_type == EventType.RUN_FINISHED and e.payload.get("reason") == "agent_finished"
    ]
    if not finished:
        return None
    steps_used = max((e.step_index for e in events), default=0) + 1
    if steps_used >= max_steps * 0.5:
        return None
    return DetectorFinding(
        description=(
            f"agent finished after only {steps_used}/{max_steps} steps without succeeding"
        ),
        event_ids=[finished[0].id],
        weight=1.0 - (steps_used / max(max_steps, 1)),
    )


def detect_tool_selection_failure(events: list[TrajectoryEvent]) -> DetectorFinding | None:
    """At least one tool_error that was NOT followed by an identical retry.

    A single isolated tool error, where the agent moved on rather than
    looping or fixating on it, looks like the agent picked the wrong tool
    or wrong arguments once rather than getting stuck. This is a weaker
    signal than the other detectors, so it carries a lower weight.
    """
    errors = _tool_errors(events)
    if not errors:
        return None
    return DetectorFinding(
        description=f"{len(errors)} tool error(s) occurred without a clear retry/loop pattern",
        event_ids=[e.id for e in errors],
        weight=0.4,
    )


def detect_timeout(events: list[TrajectoryEvent]) -> DetectorFinding | None:
    timeouts = [
        e
        for e in events
        if e.event_type == EventType.RUN_FAILED and e.payload.get("reason") == "timeout"
    ]
    if not timeouts:
        return None
    return DetectorFinding(
        description="run exceeded its timeout budget", event_ids=[timeouts[0].id], weight=1.0
    )
