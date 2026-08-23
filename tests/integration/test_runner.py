from __future__ import annotations

from arl.agents.reference import FaultProfile, ReferenceAgent
from arl.domain.run import Run, RunStatus
from arl.evaluation.engine import evaluate
from arl.runtime.runner import run_task


def test_clean_agent_solves_sum_task(sum_task) -> None:  # noqa: ANN001
    run = Run(task_id=sum_task.id, agent_name="reference")
    agent = ReferenceAgent(seed=0)
    outcome = run_task(sum_task, agent, run)

    evaluation = evaluate(sum_task, outcome.task_result)
    assert evaluation.success is True
    assert outcome.run.status == RunStatus.SUCCEEDED
    assert outcome.task_result.terminated_reason == "agent_finished"


def test_clean_agent_is_deterministic_across_seeds(sum_task) -> None:  # noqa: ANN001
    results = []
    for seed in (0, 1, 2):
        run = Run(task_id=sum_task.id, agent_name="reference", seed=seed)
        agent = ReferenceAgent(seed=seed)  # clean fault profile: seed shouldn't matter
        outcome = run_task(sum_task, agent, run)
        results.append(evaluate(sum_task, outcome.task_result).success)
    assert all(results)


def test_high_fault_rate_agent_can_fail(sum_task) -> None:  # noqa: ANN001
    fault_profile = FaultProfile(
        wrong_tool_rate=0.9, repeat_action_rate=0.9, ignore_error_rate=0.9
    )
    failed_at_least_once = False
    for seed in range(10):
        run = Run(task_id=sum_task.id, agent_name="reference", seed=seed)
        agent = ReferenceAgent(fault_profile=fault_profile, seed=seed)
        outcome = run_task(sum_task, agent, run)
        if not evaluate(sum_task, outcome.task_result).success:
            failed_at_least_once = True
            break
    assert failed_at_least_once


def test_timeout_is_respected(sum_task) -> None:  # noqa: ANN001
    sum_task = sum_task.model_copy(update={"timeout_seconds": 0.0})
    run = Run(task_id=sum_task.id, agent_name="reference")
    agent = ReferenceAgent(seed=0)
    outcome = run_task(sum_task, agent, run)
    assert outcome.task_result.terminated_reason == "timeout"
    assert outcome.run.status == RunStatus.TIMED_OUT
