"""Run and Trajectory: the container objects for a single task execution."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from enum import StrEnum

from pydantic import BaseModel, Field

from arl.domain.events import TrajectoryEvent


class RunStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    TIMED_OUT = "timed_out"
    ERROR = "error"


class Trajectory(BaseModel):
    """The ordered event log for one run."""

    run_id: str
    events: list[TrajectoryEvent] = Field(default_factory=list)

    def append(self, event: TrajectoryEvent) -> None:
        self.events.append(event)

    def of_type(self, event_type: str) -> list[TrajectoryEvent]:
        return [e for e in self.events if e.event_type == event_type]

    def __len__(self) -> int:
        return len(self.events)


class Run(BaseModel):
    """A single execution of an agent against a task."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    task_id: str
    agent_name: str
    experiment_id: str | None = None
    status: RunStatus = RunStatus.PENDING
    started_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    finished_at: datetime | None = None
    seed: int = 0
    config: dict[str, str] = Field(default_factory=dict)
