"""Evaluation outcomes and failure diagnoses.

A FailureDiagnosis deliberately separates observed evidence from inference:
the analysis engine that produces these is rule-based and every category it
assigns must be traceable to specific trajectory events, not vibes.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field


class EvaluationResult(BaseModel):
    run_id: str
    task_id: str
    success: bool
    score: float
    grader_kind: str
    details: dict[str, str] = Field(default_factory=dict)


class FailureCategory(StrEnum):
    PLANNING_FAILURE = "planning_failure"
    TOOL_SELECTION_FAILURE = "tool_selection_failure"
    TOOL_EXECUTION_FAILURE = "tool_execution_failure"
    CONTEXT_FAILURE = "context_failure"
    REASONING_FAILURE = "reasoning_failure"
    LOOPING_FAILURE = "looping_failure"
    PREMATURE_TERMINATION = "premature_termination"
    ERROR_RECOVERY_FAILURE = "error_recovery_failure"
    TIMEOUT_FAILURE = "timeout_failure"
    INCORRECT_OUTPUT = "incorrect_output"
    SAFETY_VIOLATION = "safety_violation"
    UNKNOWN_FAILURE = "unknown_failure"
    NONE = "none"


class Evidence(BaseModel):
    """One concrete, checkable observation from the trajectory."""

    description: str
    event_ids: list[str] = Field(default_factory=list)


class FailureDiagnosis(BaseModel):
    run_id: str
    category: FailureCategory
    confidence: float
    evidence: list[Evidence] = Field(default_factory=list)
    wasted_tool_calls: int = 0
    recovered: bool = False
