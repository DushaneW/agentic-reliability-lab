from __future__ import annotations

import json

from arl.agents.llm import ChatMessage, LLMAgent
from arl.domain.events import EventType
from arl.domain.run import Run
from arl.evaluation.benchmark import load_benchmark
from arl.evaluation.engine import evaluate
from arl.runtime.runner import run_task


class PlanClient:
    """Plays a model that solves coding-001 correctly, one step per call."""

    def __init__(self, replies: list[str]) -> None:
        self.replies = replies
        self.i = 0

    def complete(self, system: str, messages: list[ChatMessage]) -> str:
        del system, messages
        reply = self.replies[min(self.i, len(self.replies) - 1)]
        self.i += 1
        return reply


def _call(tool: str, **arguments: str) -> str:
    return json.dumps({"action": "tool_call", "tool": tool, "arguments": arguments})


def test_llm_agent_solves_a_task_through_the_real_runtime_and_grader() -> None:
    task = next(t for t in load_benchmark("benchmarks/coding") if t.id == "coding-001")
    client = PlanClient(
        [
            _call("filesystem", op="list"),
            _call("filesystem", op="read", path="input.txt"),
            _call("calculator", expression="-38.31 + -40.94 + 44.25 + 30.7 + -46.51"),
            _call("filesystem", op="write", path="output.txt", content="-50.81"),
            json.dumps({"action": "finish"}),
        ]
    )
    agent = LLMAgent(client, "scripted")
    run = Run(task_id=task.id, agent_name=agent.name, seed=0)
    outcome = run_task(task, agent, run)
    assert evaluate(task, outcome.task_result).success
    assert agent.name == "llm:scripted"


def test_hallucinated_tool_is_recorded_as_a_tool_error_in_the_trajectory() -> None:
    task = next(t for t in load_benchmark("benchmarks/coding") if t.id == "coding-001")
    client = PlanClient([_call("web_browser", url="x"), json.dumps({"action": "finish"})])
    agent = LLMAgent(client, "scripted")
    outcome = run_task(task, agent, Run(task_id=task.id, agent_name=agent.name, seed=0))
    kinds = [e.event_type for e in outcome.trajectory.events]
    assert EventType.TOOL_ERROR in kinds
    assert not evaluate(task, outcome.task_result).success


def test_cli_exits_nonzero_when_every_task_is_skipped(tmp_path: object) -> None:
    from typer.testing import CliRunner

    from arl.cli.main import app

    result = CliRunner().invoke(
        app,
        [
            "benchmark", "run", "benchmarks/smoke", "--db", str(tmp_path) + "/x.db",
            "--agent", "llm", "--provider", "openai-compat",
            "--base-url", "http://127.0.0.1:9/v1", "--model", "x",
        ],
    )  # fmt: skip
    assert result.exit_code == 1
    assert "skipped" in result.output
