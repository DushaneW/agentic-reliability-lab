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


def _finished(step: int, reason: str = "agent_finished") -> TrajectoryEvent:
    return TrajectoryEvent(
        run_id="r1",
        step_index=step,
        event_type=EventType.RUN_FINISHED,
        payload={"kind": "run_lifecycle", "reason": reason},
    )


def test_incorrect_output_requires_clean_agent_finish() -> None:
    from arl.analysis.detectors import detect_incorrect_output

    call = _call_event(0, "filesystem", {"op": "write"})
    assert detect_incorrect_output([call, _finished(1)]) is not None
    # any tool error -> a more specific explanation exists, so do not fire
    assert detect_incorrect_output([call, _error_event(1, "filesystem"), _finished(2)]) is None
    # run did not end by the agent finishing
    assert detect_incorrect_output([call, _finished(1, "max_steps")]) is None
    # agent did nothing at all: not "confidently wrong", just empty
    assert detect_incorrect_output([_finished(0)]) is None


def test_diagnosis_prefers_specific_detectors_over_incorrect_output() -> None:
    from arl.analysis.engine import diagnose
    from arl.domain.evaluation import EvaluationResult, FailureCategory
    from arl.domain.run import Trajectory
    from arl.domain.task import GraderSpec, Task, TaskCategory, TaskDifficulty

    task = Task(
        id="t",
        description="d",
        category=TaskCategory.CODING,
        difficulty=TaskDifficulty.EASY,
        grader=GraderSpec(kind="file_equals"),
        max_steps=4,
    )
    evaluation = EvaluationResult(
        run_id="r1", task_id="t", success=False, score=0.0, grader_kind="file_equals"
    )
    # 4 of 4 steps used, no errors -> not "premature"; falls to incorrect_output
    events = [_call_event(i, "calculator", {"expression": str(i)}) for i in range(3)]
    events.append(_finished(3))
    diagnosis = diagnose(task, Trajectory(run_id="r1", events=events), evaluation)
    assert diagnosis.category == FailureCategory.INCORRECT_OUTPUT
