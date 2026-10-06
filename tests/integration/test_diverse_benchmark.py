"""Validates the hand-written `benchmarks/diverse` suite independently of its graders.

Each oracle below is the correct final filesystem state, written by hand from
the task description. If a grader's `expected` value were wrong, the oracle
would be rejected here. The reference agent is not expected to solve these
tasks; they exist for real LLM agents.
"""

from __future__ import annotations

import pytest

from arl.domain.task import Task, TaskResult
from arl.evaluation.benchmark import load_benchmark
from arl.evaluation.engine import evaluate

ORACLES: dict[str, dict[str, str]] = {
    "diverse-001": {"output.txt": "13"},
    "diverse-002": {"output.txt": "3"},
    "diverse-003": {"output.txt": "4,5,17,28,40,61,93"},
    "diverse-004": {"output.txt": "6.67"},
    "diverse-005": {"output.txt": "44.5"},
    "diverse-006": {"output.txt": "HELLO FROM THE SECOND FILE"},
    "diverse-007": {"output.txt": "4"},
    "diverse-008": {"log.txt": "start\nmiddle\nend\n"},
    "diverse-009": {"output.txt": "15"},
    "diverse-010": {"output.txt": "95 82.8"},
}

TASKS = {t.id: t for t in load_benchmark("benchmarks/diverse")}


def _result(task: Task, files: dict[str, str]) -> TaskResult:
    return TaskResult(
        task_id=task.id,
        run_id="oracle",
        success=False,
        steps_taken=1,
        wall_time_seconds=0.0,
        terminated_reason="agent_finished",
        output={"files": files},
    )


def test_oracles_cover_every_task() -> None:
    assert set(ORACLES) == set(TASKS)


@pytest.mark.parametrize("task_id", sorted(ORACLES))
def test_grader_accepts_correct_state_and_rejects_wrong_one(task_id: str) -> None:
    task = TASKS[task_id]
    assert evaluate(task, _result(task, ORACLES[task_id])).success

    wrong = {path: "HACKED" for path in ORACLES[task_id]}
    assert not evaluate(task, _result(task, wrong)).success
    assert not evaluate(task, _result(task, {})).success
