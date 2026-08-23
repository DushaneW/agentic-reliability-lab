"""Typed trajectory events.

Every action an agent, tool, or the runtime takes during a run is recorded
as one of these events. The event log is the ground truth for evaluation,
failure analysis, and reliability feature extraction — nothing downstream
should need to re-derive state from anywhere else.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, Field


class EventType(StrEnum):
    RUN_STARTED = "run_started"
    MODEL_REQUEST = "model_request"
    MODEL_RESPONSE = "model_response"
    AGENT_ACTION = "agent_action"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    TOOL_ERROR = "tool_error"
    RETRY = "retry"
    INTERVENTION = "intervention"
    EVALUATION = "evaluation"
    RUN_FINISHED = "run_finished"
    RUN_FAILED = "run_failed"


class ModelRequestPayload(BaseModel):
    kind: Literal["model_request"] = "model_request"
    prompt_tokens_estimate: int
    context_size: int


class ModelResponsePayload(BaseModel):
    kind: Literal["model_response"] = "model_response"
    latency_seconds: float
    completion_tokens_estimate: int


class AgentActionPayload(BaseModel):
    kind: Literal["agent_action"] = "agent_action"
    action_type: str
    rationale: str = ""


class ToolCallPayload(BaseModel):
    kind: Literal["tool_call"] = "tool_call"
    tool_name: str
    arguments: dict[str, Any]


class ToolResultPayload(BaseModel):
    kind: Literal["tool_result"] = "tool_result"
    tool_name: str
    success: bool
    result_summary: str


class ToolErrorPayload(BaseModel):
    kind: Literal["tool_error"] = "tool_error"
    tool_name: str
    error_type: str
    message: str


class RetryPayload(BaseModel):
    kind: Literal["retry"] = "retry"
    reason: str
    attempt: int


class InterventionPayload(BaseModel):
    kind: Literal["intervention"] = "intervention"
    intervention_type: str
    reason: str
    triggered_by_probability: float | None = None


class EvaluationPayload(BaseModel):
    kind: Literal["evaluation"] = "evaluation"
    success: bool
    score: float


class RunLifecyclePayload(BaseModel):
    kind: Literal["run_lifecycle"] = "run_lifecycle"
    reason: str = ""


EventPayload = (
    ModelRequestPayload
    | ModelResponsePayload
    | AgentActionPayload
    | ToolCallPayload
    | ToolResultPayload
    | ToolErrorPayload
    | RetryPayload
    | InterventionPayload
    | EvaluationPayload
    | RunLifecyclePayload
)


class TrajectoryEvent(BaseModel):
    """One entry in a run's trajectory log."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    run_id: str
    step_index: int
    event_type: EventType
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    payload: dict[str, Any]
    metadata: dict[str, Any] = Field(default_factory=dict)

    def payload_as(self, model: type[BaseModel]) -> BaseModel:
        return model.model_validate(self.payload)
