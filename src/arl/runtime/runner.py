"""Executes a single agent against a single task and records the trajectory."""

from __future__ import annotations

import time
from dataclasses import dataclass

from arl.agents.base import ActionKind, Agent
from arl.domain.events import EventType
from arl.domain.run import Run, RunStatus, Trajectory
from arl.domain.task import Task, TaskResult
from arl.runtime.environment import VirtualFilesystem
from arl.telemetry.recorder import TrajectoryRecorder
from arl.tools.base import ToolRegistry
from arl.tools.calculator import CalculatorTool
from arl.tools.filesystem import FilesystemTool
from arl.tools.search import SearchTool
from arl.tools.shell import ShellTool


@dataclass
class RunOutcome:
    run: Run
    trajectory: Trajectory
    task_result: TaskResult
    final_fs: VirtualFilesystem


def build_registry(task: Task, fs: VirtualFilesystem) -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(FilesystemTool(fs))
    registry.register(CalculatorTool())
    registry.register(SearchTool(corpus=task.metadata.get("search_corpus", {})))
    registry.register(ShellTool())
    return registry


def run_task(task: Task, agent: Agent, run: Run) -> RunOutcome:
    """Run `agent` against `task`, returning the full trajectory and outcome.

    This is intentionally synchronous and single-threaded: the reference
    agent and tool set have no I/O latency worth parallelizing, and keeping
    the loop simple makes trajectories trivial to reason about and replay.
    """
    fs = VirtualFilesystem(initial_files=task.metadata.get("initial_files", {}))
    registry = build_registry(task, fs)
    recorder = TrajectoryRecorder(run.id)

    recorder.record(EventType.RUN_STARTED, {"kind": "run_lifecycle", "reason": task.id})

    start = time.monotonic()
    terminated_reason = "max_steps_exceeded"
    steps_taken = 0

    for step in range(task.max_steps):
        elapsed = time.monotonic() - start
        # Strict `>` was the bug: time.monotonic() has finite resolution
        # (coarse on some platforms, e.g. ~15ms on Windows), so two calls
        # microseconds apart can report the identical value. With
        # timeout_seconds == 0.0, `elapsed > 0.0` was then False on the
        # very first check whenever elapsed happened to read exactly 0.0,
        # letting the agent run to completion instead of timing out
        # immediately. `elapsed` is always >= 0 (time.monotonic() never
        # goes backwards), so `>=` makes a zero-second budget deterministic
        # on every platform and clock resolution, independent of timing.
        if elapsed >= task.timeout_seconds:
            terminated_reason = "timeout"
            recorder.record(EventType.RUN_FAILED, {"kind": "run_lifecycle", "reason": "timeout"})
            break

        decision = agent.decide(
            task.description, registry.specs(), recorder.trajectory.events
        )
        steps_taken = step + 1

        if decision.kind == ActionKind.FINISH:
            terminated_reason = "agent_finished"
            break

        assert decision.tool_name is not None
        arguments = decision.arguments or {}
        recorder.record(
            EventType.TOOL_CALL,
            {"kind": "tool_call", "tool_name": decision.tool_name, "arguments": arguments},
        )
        try:
            tool = registry.get(decision.tool_name)
        except KeyError:
            recorder.record(
                EventType.TOOL_ERROR,
                {
                    "kind": "tool_error",
                    "tool_name": decision.tool_name,
                    "error_type": "unknown_tool",
                    "message": f"No such tool: {decision.tool_name}",
                },
            )
            continue

        result = tool.run(arguments)
        if result.success:
            recorder.record(
                EventType.TOOL_RESULT,
                {
                    "kind": "tool_result",
                    "tool_name": result.tool_name,
                    "success": True,
                    "result_summary": result.output,
                },
            )
        else:
            recorder.record(
                EventType.TOOL_ERROR,
                {
                    "kind": "tool_error",
                    "tool_name": result.tool_name,
                    "error_type": "tool_execution_error",
                    "message": result.error or "unknown error",
                },
            )

    wall_time = time.monotonic() - start
    status = RunStatus.SUCCEEDED if terminated_reason == "agent_finished" else RunStatus.FAILED
    if terminated_reason == "timeout":
        status = RunStatus.TIMED_OUT

    recorder.record(
        EventType.RUN_FINISHED, {"kind": "run_lifecycle", "reason": terminated_reason}
    )

    run.status = status
    task_result = TaskResult(
        task_id=task.id,
        run_id=run.id,
        success=False,  # filled in by the evaluation engine, not the runner
        steps_taken=steps_taken,
        wall_time_seconds=wall_time,
        terminated_reason=terminated_reason,
        output={"files": fs.snapshot()},
    )
    return RunOutcome(run=run, trajectory=recorder.trajectory, task_result=task_result, final_fs=fs)
