from __future__ import annotations

from arl.domain.events import EventType, TrajectoryEvent
from arl.domain.run import Run, RunStatus, Trajectory


def test_trajectory_append_and_len() -> None:
    trajectory = Trajectory(run_id="r1")
    event = TrajectoryEvent(
        run_id="r1", step_index=0, event_type=EventType.RUN_STARTED, payload={"kind": "x"}
    )
    trajectory.append(event)
    assert len(trajectory) == 1
    assert trajectory.of_type(EventType.RUN_STARTED) == [event]


def test_run_defaults_to_pending() -> None:
    run = Run(task_id="t1", agent_name="reference")
    assert run.status == RunStatus.PENDING
    assert run.id  # uuid was generated


def test_event_payload_as_typed_model() -> None:
    from arl.domain.events import ToolCallPayload

    event = TrajectoryEvent(
        run_id="r1",
        step_index=0,
        event_type=EventType.TOOL_CALL,
        payload={
            "kind": "tool_call",
            "tool_name": "calculator",
            "arguments": {"expression": "1+1"},
        },
    )
    typed = event.payload_as(ToolCallPayload)
    assert isinstance(typed, ToolCallPayload)
    assert typed.tool_name == "calculator"
