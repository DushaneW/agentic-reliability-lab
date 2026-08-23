from __future__ import annotations

from pathlib import Path

from arl.agents.reference import ReferenceAgent
from arl.domain.evaluation import EvaluationResult
from arl.domain.run import Run
from arl.evaluation.engine import evaluate
from arl.runtime.runner import run_task
from arl.storage.repository import Repository


def test_repository_round_trips_a_full_run(tmp_path: Path, sum_task) -> None:  # noqa: ANN001
    db_path = tmp_path / "test.db"
    repo = Repository(db_path)

    run = Run(task_id=sum_task.id, agent_name="reference")
    agent = ReferenceAgent(seed=0)
    outcome = run_task(sum_task, agent, run)
    evaluation = evaluate(sum_task, outcome.task_result)

    repo.save_run(run)
    repo.save_events(outcome.trajectory.events)
    repo.save_task_result(outcome.task_result)
    repo.save_evaluation_result(evaluation)

    loaded_run = repo.get_run(run.id)
    assert loaded_run is not None
    assert loaded_run.task_id == sum_task.id

    loaded_trajectory = repo.get_trajectory(run.id)
    assert len(loaded_trajectory) == len(outcome.trajectory)
    assert [e.event_type for e in loaded_trajectory.events] == [
        e.event_type for e in outcome.trajectory.events
    ]

    loaded_result = repo.get_task_result(run.id)
    assert loaded_result is not None
    assert loaded_result.steps_taken == outcome.task_result.steps_taken

    loaded_evals = repo.list_evaluation_results([run.id])
    assert len(loaded_evals) == 1
    assert loaded_evals[0].success == evaluation.success

    repo.close()


def test_repository_list_runs_empty_by_default(tmp_path: Path) -> None:
    repo = Repository(tmp_path / "empty.db")
    assert repo.list_runs() == []
    repo.close()


def test_save_evaluation_result_is_idempotent_on_conflict(tmp_path: Path) -> None:
    repo = Repository(tmp_path / "test.db")
    run = Run(task_id="t1", agent_name="reference")
    repo.save_run(run)
    result = EvaluationResult(
        run_id=run.id, task_id="t1", success=True, score=1.0, grader_kind="numeric_file_equals"
    )
    repo.save_evaluation_result(result)
    repo.save_evaluation_result(result)  # should not raise
    assert len(repo.list_evaluation_results([run.id])) == 1
    repo.close()
