"""Reference agent.

Solves benchmark tasks using a small set of generic, category-independent
strategies (read input, transform, write output) driven by `task.metadata`.
This does not call an external LLM: it is a deterministic scripted policy,
seeded by `random.Random`, and its purpose is (a) to prove the runtime/tool
system works end-to-end and (b) to generate a labeled trajectory dataset for
the reliability model.

`FaultProfile` lets the agent's competence be dialed down in controlled,
documented ways (wrong tool choice, repeating a failed action, ignoring
errors, stopping early). This is how the benchmark suite produces a spread
of real failure trajectories without needing thousands of live LLM calls.
This is a deliberate, disclosed simplification — see
docs/research/limitations.md. It does NOT simulate the failure modes of a
real LLM agent; it simulates the *shape* of those failure modes so the
detection pipeline (features -> classifier -> early warning) can be built
and evaluated on real, reproducible data.

`scripted_steps` is a second, complementary way to control the agent,
added for `benchmarks/failures` (see docs/benchmarks.md). `FaultProfile`
rates are population-level — they describe "roughly how often" a fault
happens across a run and are what the reliability dataset is built from.
They are not precise enough to construct a single, hand-designed
trajectory that exercises one specific failure detector in isolation
(e.g. "exactly one invalid tool call, nothing else"). `scripted_steps`
forces an exact `AgentDecision` at a specific `decide()` call index,
bypassing `FaultProfile` entirely for that one call, while every
unscripted call still goes through the normal plan/fault logic
unchanged. This is strictly additive: a `ReferenceAgent` built without
`scripted_steps` (the only way any existing caller constructs one)
behaves exactly as before.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

from arl.agents.base import ActionKind, Agent, AgentDecision
from arl.domain.events import EventType, TrajectoryEvent
from arl.domain.task import Task


@dataclass(frozen=True)
class FaultProfile:
    """Probabilities (per decision) that the agent behaves incompetently."""

    wrong_tool_rate: float = 0.0
    repeat_action_rate: float = 0.0
    ignore_error_rate: float = 0.0
    premature_termination_rate: float = 0.0

    @property
    def is_clean(self) -> bool:
        return (
            self.wrong_tool_rate == 0
            and self.repeat_action_rate == 0
            and self.ignore_error_rate == 0
            and self.premature_termination_rate == 0
        )


class ReferenceAgent(Agent):
    """Generic read -> transform -> write agent for the smoke/coding benchmarks."""

    name = "reference"

    def __init__(
        self,
        fault_profile: FaultProfile | None = None,
        seed: int = 0,
        scripted_steps: dict[int, AgentDecision] | None = None,
    ) -> None:
        self.fault_profile = fault_profile or FaultProfile()
        self._rng = random.Random(seed)
        self._last_action_signature: tuple[str, ...] | None = None
        self.scripted_steps = scripted_steps or {}
        self._call_index = 0

    def decide(
        self,
        task_description: str,
        tool_specs: list[dict[str, Any]],
        history: list[TrajectoryEvent],
    ) -> AgentDecision:
        del task_description, tool_specs  # strategy is driven by op/metadata seen in history

        call_index = self._call_index
        self._call_index += 1
        if call_index in self.scripted_steps:
            # Forced decision for this exact call, bypassing FaultProfile
            # entirely — see the module docstring for why this exists
            # alongside the probabilistic fault mechanism.
            decision = self.scripted_steps[call_index]
            self._last_decision = decision
            return decision

        step = self._plan_step(history)

        if self._rng.random() < self.fault_profile.premature_termination_rate:
            return AgentDecision(kind=ActionKind.FINISH, rationale="fault: stopped early")

        if step is None:
            return AgentDecision(kind=ActionKind.FINISH, rationale="plan complete")

        decision = step
        if self._rng.random() < self.fault_profile.wrong_tool_rate:
            decision = AgentDecision(
                kind=ActionKind.TOOL_CALL,
                tool_name="calculator",
                arguments={"expression": "1/0"},
                rationale="fault: wrong tool selected",
            )

        signature = (decision.tool_name or "", str(decision.arguments))
        last_result_failed = self._last_result_failed(history)
        if last_result_failed and self._rng.random() < self.fault_profile.ignore_error_rate:
            # Repeat the exact same failing action instead of adapting.
            decision = self._last_decision or decision
            signature = (decision.tool_name or "", str(decision.arguments))
        elif self._rng.random() < self.fault_profile.repeat_action_rate and self._last_decision:
            decision = self._last_decision
            signature = (decision.tool_name or "", str(decision.arguments))

        self._last_decision = decision
        self._last_action_signature = signature
        return decision

    _last_decision: AgentDecision | None = None

    def _last_result_failed(self, history: list[TrajectoryEvent]) -> bool:
        for event in reversed(history):
            if event.event_type == EventType.TOOL_RESULT:
                return not bool(event.payload.get("success", True))
            if event.event_type == EventType.TOOL_ERROR:
                return True
        return False

    def _plan_step(self, history: list[TrajectoryEvent]) -> AgentDecision | None:
        """Generic three-phase plan: list -> read+op -> write, then finish."""
        tool_calls = [e for e in history if e.event_type == EventType.TOOL_CALL]
        n = len(tool_calls)

        if n == 0:
            return AgentDecision(
                kind=ActionKind.TOOL_CALL,
                tool_name="filesystem",
                arguments={"op": "list"},
                rationale="inspect available files",
            )
        if n == 1:
            return AgentDecision(
                kind=ActionKind.TOOL_CALL,
                tool_name="filesystem",
                arguments={"op": "read", "path": "input.txt"},
                rationale="read task input",
            )
        if n == 2:
            input_content = self._find_last_read(history)
            expression = self._build_expression(input_content)
            return AgentDecision(
                kind=ActionKind.TOOL_CALL,
                tool_name="calculator",
                arguments={"expression": expression},
                rationale="compute result",
            )
        if n == 3:
            result_value = self._find_last_calculator_result(history)
            return AgentDecision(
                kind=ActionKind.TOOL_CALL,
                tool_name="filesystem",
                arguments={"op": "write", "path": "output.txt", "content": result_value},
                rationale="write result",
            )
        return None

    @staticmethod
    def _find_last_read(history: list[TrajectoryEvent]) -> str:
        for event in reversed(history):
            if event.event_type == EventType.TOOL_RESULT and event.payload.get(
                "tool_name"
            ) == "filesystem":
                return str(event.payload.get("result_summary", ""))
        return ""

    @staticmethod
    def _build_expression(content: str) -> str:
        numbers = [tok for tok in content.replace(",", " ").split() if _is_number(tok)]
        if not numbers:
            return "0"
        return "+".join(numbers)
    @staticmethod
    def _find_last_calculator_result(history: list[TrajectoryEvent]) -> str:
        for event in reversed(history):
            if event.event_type == EventType.TOOL_RESULT and event.payload.get(
                "tool_name"
            ) == "calculator":
                return str(event.payload.get("result_summary", ""))
        return "0"


def _is_number(token: str) -> bool:
    try:
        float(token)
    except ValueError:
        return False
    return True


def agent_from_task_metadata(task: Task, seed: int) -> ReferenceAgent:
    """Builds a ReferenceAgent configured from a task's own metadata.

    Benchmark tasks may declare, under `metadata`:

    - `fault_profile`: kwargs for `FaultProfile` (e.g.
      `{"repeat_action_rate": 1.0}`), applied for the whole run.
    - `scripted_steps`: a mapping of `decide()` call index -> forced
      tool call, e.g.
      `{0: {"tool_name": "filesystem", "arguments": {"op": "bogus"}}}`.
      Each entry may set `kind: finish` instead of a tool call.

    This lets a benchmark suite (see `benchmarks/failures` and
    `docs/benchmarks.md`) declare a deterministic failure scenario
    entirely in the task YAML, the same way it already declares
    `initial_files`. Tasks that set neither key (every task in every
    benchmark suite that existed before `benchmarks/failures`) get back
    exactly the plain `ReferenceAgent(seed=seed)` they got before this
    function existed — this is an additive read of `task.metadata`, not a
    change to how any existing task is interpreted.
    """
    fault_profile_kwargs = task.metadata.get("fault_profile") or {}
    fault_profile = FaultProfile(**fault_profile_kwargs) if fault_profile_kwargs else None

    scripted_steps_raw = task.metadata.get("scripted_steps") or {}
    scripted_steps: dict[int, AgentDecision] = {}
    for index, step in scripted_steps_raw.items():
        kind = ActionKind.FINISH if step.get("kind") == "finish" else ActionKind.TOOL_CALL
        scripted_steps[int(index)] = AgentDecision(
            kind=kind,
            tool_name=step.get("tool_name"),
            arguments=step.get("arguments"),
            rationale=step.get("rationale", "scripted"),
        )

    return ReferenceAgent(
        fault_profile=fault_profile,
        seed=seed,
        scripted_steps=scripted_steps or None,
    )
