from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from arl.agents.reference import ReferenceAgent
from arl.domain.run import Run
from arl.evaluation.engine import evaluate
from arl.runtime.runner import run_task
from arl.storage.repository import Repository


def _seed_db(db_path: Path, task) -> str:  # noqa: ANN001
    repo = Repository(db_path)
    run = Run(task_id=task.id, agent_name="reference")
    agent = ReferenceAgent(seed=0)
    outcome = run_task(task, agent, run)
    evaluation = evaluate(task, outcome.task_result)
    repo.save_run(run)
    repo.save_events(outcome.trajectory.events)
    repo.save_task_result(outcome.task_result)
    repo.save_evaluation_result(evaluation)
    repo.close()
    return run.id


def test_dashboard_endpoints_serve_real_data(tmp_path: Path, sum_task, monkeypatch) -> None:  # noqa: ANN001
    db_path = tmp_path / "dashboard_test.db"
    run_id = _seed_db(db_path, sum_task)
    monkeypatch.setenv("ARL_DB_PATH", str(db_path))

    # module-level DB_PATH is read at import time; re-import cleanly per test
    import importlib

    from arl.dashboard import app as dashboard_module

    importlib.reload(dashboard_module)

    client = TestClient(dashboard_module.app)

    overview = client.get("/api/overview")
    assert overview.status_code == 200
    assert overview.json()["total_runs"] == 1

    runs = client.get("/api/runs")
    assert runs.status_code == 200
    assert runs.json()[0]["id"] == run_id

    detail = client.get(f"/api/runs/{run_id[:8]}")
    assert detail.status_code == 200
    assert len(detail.json()["events"]) > 0

    missing = client.get("/api/runs/doesnotexist")
    assert missing.status_code == 404

    index = client.get("/")
    assert index.status_code == 200
    assert "Agentic Reliability Lab" in index.text
