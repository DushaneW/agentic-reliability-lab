"""Minimal external agent example.

Shows how to implement `arl.agents.base.Agent` for something other than
the bundled reference agent — e.g. a real wrapper around an LLM provider.
This particular example is still not a real LLM call (no API key
assumption is made — see docs/research/limitations.md for why), but it
demonstrates the actual integration surface: `decide()` gets the task
description, the tool specs available this run, and the history so far,
and must return an AgentDecision.

Run:
    uv run python examples/minimal_agent.py
"""

from __future__ import annotations

from arl.agents.base import ActionKind, Agent, AgentDecision
from arl.domain.events import EventType, TrajectoryEvent
from arl.domain.run import Run
from arl.evaluation.benchmark import load_benchmark
from arl.evaluation.engine import evaluate
from arl.runtime.runner import run_task


class AlwaysListFirstAgent(Agent):
    """A deliberately simple agent: lists files once, then gives up.

    This is here to prove the integration surface works end to end with a
    *different* agent than the bundled reference implementation — it is
    not meant to solve any task. Expect it to fail every task in the
    benchmark suite; that's the point.
    """

    name = "always-list-first"

    def decide(
        self,
        task_description: str,
        tool_specs: list[dict],
        history: list[TrajectoryEvent],
    ) -> AgentDecision:
        del task_description, tool_specs
        already_listed = any(
            e.event_type == EventType.TOOL_CALL and e.payload.get("tool_name") == "filesystem"
            for e in history
        )
        if already_listed:
            return AgentDecision(kind=ActionKind.FINISH, rationale="done exploring, giving up")
        return AgentDecision(
            kind=ActionKind.TOOL_CALL,
            tool_name="filesystem",
            arguments={"op": "list"},
            rationale="see what's here",
        )


def main() -> None:
    tasks = load_benchmark("benchmarks/smoke")
    task = tasks[0]
    run = Run(task_id=task.id, agent_name=AlwaysListFirstAgent.name)
    outcome = run_task(task, AlwaysListFirstAgent(), run)
    evaluation = evaluate(task, outcome.task_result)
    print(f"task={task.id} success={evaluation.success} steps={outcome.task_result.steps_taken}")
    for event in outcome.trajectory.events:
        print(f"  {event.step_index}. {event.event_type.value} {event.payload}")


if __name__ == "__main__":
    main()
