"""One test that proves the whole system works, start to finish.

benchmark -> agent -> trajectory -> evaluation -> failure analysis -> persisted result
"""

from __future__ import annotations

from pathlib import Path

from arl.agents.reference import FaultProfile, ReferenceAgent
from arl.analysis.engine import diagnose
from arl.domain.evaluation import FailureCategory
from arl.domain.run import Run
from arl.evaluation.benchmark import load_benchmark
from arl.evaluation.engine import evaluate
from arl.runtime.runner import run_task
from arl.storage.repository import Repository

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def test_full_smoke_benchmark_pipeline(tmp_path: Path) -> None:
    smoke_dir = REPO_ROOT / "benchmarks" / "smoke"
    if not smoke_dir.is_dir():
        import pytest

        pytest.skip("benchmarks/smoke not generated in this environment")

    tasks = load_benchmark(smoke_dir)
    repo = Repository(tmp_path / "e2e.db")

    for task in tasks:
        run = Run(task_id=task.id, agent_name="reference")
        agent = ReferenceAgent(seed=0)  # clean: smoke tests must always pass
        outcome = run_task(task, agent, run)
        evaluation = evaluate(task, outcome.task_result)
        diagnosis = diagnose(task, outcome.trajectory, evaluation)

        repo.save_run(run)
        repo.save_events(outcome.trajectory.events)
        repo.save_task_result(outcome.task_result)
        repo.save_evaluation_result(evaluation)
        repo.save_failure_diagnosis(diagnosis)

        assert evaluation.success, f"{task.id} failed: {evaluation.details}"
        assert diagnosis.category == FailureCategory.NONE

    persisted_runs = repo.list_runs()
    assert len(persisted_runs) == len(tasks)
    for run in persisted_runs:
        trajectory = repo.get_trajectory(run.id)
        assert len(trajectory) > 0
    repo.close()


def test_e2e_pipeline_captures_a_real_failure(tmp_path: Path) -> None:
    smoke_dir = REPO_ROOT / "benchmarks" / "smoke"
    if not smoke_dir.is_dir():
        import pytest

        pytest.skip("benchmarks/smoke not generated in this environment")

    task = load_benchmark(smoke_dir)[0]
    fault_profile = FaultProfile(repeat_action_rate=1.0)  # guaranteed to loop, never finish
    run = Run(task_id=task.id, agent_name="reference")
    agent = ReferenceAgent(fault_profile=fault_profile, seed=0)
    outcome = run_task(task, agent, run)
    evaluation = evaluate(task, outcome.task_result)
    diagnosis = diagnose(task, outcome.trajectory, evaluation)

    assert not evaluation.success
    assert diagnosis.category != FailureCategory.NONE
    assert len(diagnosis.evidence) > 0
