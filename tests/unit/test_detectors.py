from __future__ import annotations

from arl.analysis.detectors import detect_error_recovery_failure, detect_looping
from arl.domain.events import EventType, TrajectoryEvent


def _call_event(step: int, tool: str, args: dict) -> TrajectoryEvent:
    return TrajectoryEvent(
        run_id="r1",
        step_index=step,
        event_type=EventType.TOOL_CALL,
        payload={"kind": "tool_call", "tool_name": tool, "arguments": args},
    )


def _error_event(step: int, tool: str) -> TrajectoryEvent:
    return TrajectoryEvent(
        run_id="r1",
        step_index=step,
        event_type=EventType.TOOL_ERROR,
        payload={"kind": "tool_error", "tool_name": tool, "error_type": "x", "message": "boom"},
    )


def test_detect_looping_finds_identical_consecutive_calls() -> None:
    events = [
        _call_event(0, "calculator", {"expression": "1+1"}),
        _call_event(1, "calculator", {"expression": "1+1"}),
    ]
    finding = detect_looping(events)
    assert finding is not None
    assert len(finding.event_ids) == 2


def test_detect_looping_ignores_distinct_calls() -> None:
    events = [
        _call_event(0, "calculator", {"expression": "1+1"}),
        _call_event(1, "calculator", {"expression": "2+2"}),
    ]
    assert detect_looping(events) is None


def test_detect_error_recovery_failure_finds_retry_of_same_failing_call() -> None:
    events = [
        _call_event(0, "filesystem", {"op": "read", "path": "x"}),
        _error_event(1, "filesystem"),
        _call_event(2, "filesystem", {"op": "read", "path": "x"}),
    ]
    finding = detect_error_recovery_failure(events)
    assert finding is not None


def test_detect_error_recovery_failure_ignores_adapted_retry() -> None:
    events = [
        _call_event(0, "filesystem", {"op": "read", "path": "x"}),
        _error_event(1, "filesystem"),
        _call_event(2, "filesystem", {"op": "read", "path": "y"}),
    ]
    assert detect_error_recovery_failure(events) is None
