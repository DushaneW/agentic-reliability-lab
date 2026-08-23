"""Core domain types for Agentic Reliability Lab.

These are the typed objects every other layer (runtime, evaluation, analysis,
reliability, storage) speaks in terms of. Nothing outside this package should
represent a run, task, or trajectory as a bare dict.
"""

from arl.domain.evaluation import EvaluationResult, FailureCategory, FailureDiagnosis
from arl.domain.events import (
    AgentActionPayload,
    EventType,
    InterventionPayload,
    ModelRequestPayload,
    ModelResponsePayload,
    ToolCallPayload,
    ToolErrorPayload,
    ToolResultPayload,
    TrajectoryEvent,
)
from arl.domain.experiment import Experiment, ExperimentResult
from arl.domain.metric import Metric
from arl.domain.run import Run, RunStatus, Trajectory
from arl.domain.task import Task, TaskResult
from arl.domain.tool import ToolCall, ToolResult

__all__ = [
    "AgentActionPayload",
    "EventType",
    "EvaluationResult",
    "Experiment",
    "ExperimentResult",
    "FailureCategory",
    "FailureDiagnosis",
    "InterventionPayload",
    "Metric",
    "ModelRequestPayload",
    "ModelResponsePayload",
    "Run",
    "RunStatus",
    "Task",
    "TaskResult",
    "ToolCall",
    "ToolCallPayload",
    "ToolErrorPayload",
    "ToolResult",
    "ToolResultPayload",
    "Trajectory",
    "TrajectoryEvent",
]
