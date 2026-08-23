"""Agent protocol.

An agent is anything that, given a task description, the available tools,
and the history of the run so far, decides what to do next. External agents
(wrapping a real LLM) implement this same protocol — nothing in the runtime
is coupled to the reference agent's internal strategy.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from arl.domain.events import TrajectoryEvent


class ActionKind(StrEnum):
    TOOL_CALL = "tool_call"
    FINISH = "finish"


@dataclass(frozen=True)
class AgentDecision:
    kind: ActionKind
    tool_name: str | None = None
    arguments: dict[str, Any] | None = None
    rationale: str = ""


class Agent(ABC):
    """Protocol every agent (reference or external) must implement."""

    name: str

    @abstractmethod
    def decide(
        self,
        task_description: str,
        tool_specs: list[dict[str, Any]],
        history: list[TrajectoryEvent],
    ) -> AgentDecision:
        """Return the next action given the task and trajectory so far."""
        raise NotImplementedError
