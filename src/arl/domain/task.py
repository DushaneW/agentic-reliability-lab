"""Benchmark task definitions and their outcomes."""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class TaskCategory(StrEnum):
    CODING = "coding"
    DEBUGGING = "debugging"
    TOOL_USE = "tool_use"
    MULTI_STEP = "multi_step"
    ERROR_RECOVERY = "error_recovery"
    LONG_HORIZON = "long_horizon"
    ADVERSARIAL = "adversarial"


class TaskDifficulty(StrEnum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


class GraderSpec(BaseModel):
    """How a task's outcome is scored.

    `kind` selects the grading strategy; `config` is grader-specific.
    Only deterministic graders are implemented in the core system —
    see docs/evaluation.md for why LLM judges are deliberately excluded
    from the critical path.
    """

    kind: str
    config: dict[str, Any] = Field(default_factory=dict)


class Task(BaseModel):
    """A single benchmark task: an objective plus a way to grade it."""

    id: str
    description: str
    category: TaskCategory
    difficulty: TaskDifficulty
    environment: str = "sandboxed_fs"
    grader: GraderSpec
    timeout_seconds: float = 30.0
    max_steps: int = 20
    metadata: dict[str, Any] = Field(default_factory=dict)


class TaskResult(BaseModel):
    """Raw outcome of running one task, before evaluation grading."""

    task_id: str
    run_id: str
    success: bool
    steps_taken: int
    wall_time_seconds: float
    terminated_reason: str
    output: dict[str, Any] = Field(default_factory=dict)
