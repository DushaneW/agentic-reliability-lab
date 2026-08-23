"""Builds and accumulates TrajectoryEvents for a run."""

from __future__ import annotations

from typing import Any

from arl.domain.events import EventType, TrajectoryEvent
from arl.domain.run import Trajectory
from arl.telemetry.redaction import redact


class TrajectoryRecorder:
    def __init__(self, run_id: str) -> None:
        self.run_id = run_id
        self.trajectory = Trajectory(run_id=run_id)
        self._step = 0

    def record(self, event_type: EventType, payload: dict[str, Any]) -> TrajectoryEvent:
        redacted_payload = _redact_payload(payload)
        event = TrajectoryEvent(
            run_id=self.run_id,
            step_index=self._step,
            event_type=event_type,
            payload=redacted_payload,
        )
        self._step += 1
        self.trajectory.append(event)
        return event


def _redact_payload(payload: dict[str, Any]) -> dict[str, Any]:
    return {k: redact(v) if isinstance(v, str) else v for k, v in payload.items()}
