"""Tool invocation domain types, distinct from the trajectory event payloads.

These are the objects the tool system passes around at call time; the
runtime converts them into TrajectoryEvents when recording history.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ToolCall(BaseModel):
    tool_name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class ToolResult(BaseModel):
    tool_name: str
    success: bool
    output: str
    error: str | None = None
    duration_seconds: float = 0.0
