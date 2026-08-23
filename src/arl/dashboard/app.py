"""Local dashboard.

A single FastAPI app: a handful of JSON endpoints over the same
`Repository` the CLI uses, plus one HTML page that fetches them client-side
with vanilla JS. No frontend build step, no framework — the data volume
(hundreds of runs on a laptop) doesn't justify one, and it keeps `arl
dashboard` a `pip install`-only feature.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from arl.storage.repository import Repository

app = FastAPI(title="Agentic Reliability Lab Dashboard")

DB_PATH = os.environ.get("ARL_DB_PATH", "arl.db")
STATIC_DIR = Path(__file__).parent / "static"
if STATIC_DIR.is_dir():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


def _repo() -> Repository:
    return Repository(DB_PATH)


@app.get("/api/overview")
def overview() -> dict[str, object]:
    repo = _repo()
    runs = repo.list_runs()
    evaluations = {e.run_id: e for e in repo.list_evaluation_results()}
    repo.close()

    total = len(runs)
    graded = [evaluations[r.id] for r in runs if r.id in evaluations]
    successes = sum(1 for e in graded if e.success)

    return {
        "total_runs": total,
        "graded_runs": len(graded),
        "success_rate": successes / len(graded) if graded else None,
        "failure_rate": 1 - successes / len(graded) if graded else None,
    }


@app.get("/api/runs")
def list_runs() -> list[dict[str, object]]:
    repo = _repo()
    runs = repo.list_runs()
    evaluations = {e.run_id: e for e in repo.list_evaluation_results()}
    diagnoses = {}
    for run in runs:
        d = repo.get_failure_diagnosis(run.id)
        if d:
            diagnoses[run.id] = d
    repo.close()

    return [
        {
            "id": r.id,
            "task_id": r.task_id,
            "agent_name": r.agent_name,
            "status": r.status.value,
            "success": evaluations[r.id].success if r.id in evaluations else None,
            "failure_category": diagnoses[r.id].category.value if r.id in diagnoses else None,
        }
        for r in runs
    ]


@app.get("/api/runs/{run_id}")
def run_detail(run_id: str) -> dict[str, object]:
    repo = _repo()
    run = repo.get_run(run_id)
    if run is None:
        for candidate in repo.list_runs():
            if candidate.id.startswith(run_id):
                run = candidate
                break
    if run is None:
        repo.close()
        raise HTTPException(status_code=404, detail="run not found")

    trajectory = repo.get_trajectory(run.id)
    task_result = repo.get_task_result(run.id)
    evaluations = repo.list_evaluation_results([run.id])
    diagnosis = repo.get_failure_diagnosis(run.id)
    repo.close()

    return {
        "id": run.id,
        "task_id": run.task_id,
        "agent_name": run.agent_name,
        "status": run.status.value,
        "task_result": task_result.model_dump() if task_result else None,
        "evaluation": evaluations[0].model_dump() if evaluations else None,
        "diagnosis": diagnosis.model_dump() if diagnosis else None,
        "events": [
            {
                "step_index": e.step_index,
                "event_type": e.event_type.value,
                "payload": e.payload,
            }
            for e in trajectory.events
        ],
    }


@app.get("/api/reliability")
def reliability_metrics() -> dict[str, object]:
    path = Path("models/reliability_metrics.json")
    if not path.exists():
        return {"trained": False}
    with open(path) as f:
        data = json.load(f)
    return {"trained": True, **data}


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return _INDEX_HTML


_INDEX_HTML = """<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>Agentic Reliability Lab</title>
<style>
  :root { color-scheme: dark; }
  body { font-family: ui-monospace, "SF Mono", Consolas, monospace; background: #0b0d10;
         color: #d8dee4; margin: 0; padding: 2rem; font-size: 14px; }
  h1 { font-size: 1.1rem; font-weight: 600; margin: 0 0 1.5rem; color: #e8ebef; }
  h2 { font-size: 0.85rem; text-transform: uppercase; letter-spacing: 0.04em;
       color: #7d8590; margin: 2rem 0 0.75rem; }
  .cards { display: flex; gap: 1rem; flex-wrap: wrap; }
  .card { background: #12151a; border: 1px solid #22262c; border-radius: 6px;
          padding: 0.9rem 1.1rem; min-width: 140px; }
  .card .value { font-size: 1.4rem; color: #e8ebef; }
  .card .label { color: #7d8590; font-size: 0.75rem; margin-top: 0.2rem; }
  table { border-collapse: collapse; width: 100%; margin-top: 0.5rem; }
  th, td { text-align: left; padding: 0.4rem 0.7rem; border-bottom: 1px solid #1c2026;
           font-size: 0.82rem; }
  th { color: #7d8590; font-weight: 500; }
  .pass { color: #57ab5a; } .fail { color: #e5534b; }
  tr { cursor: pointer; } tr:hover { background: #14171c; }
  #run-detail { display: none; margin-top: 1rem; background: #12151a;
                border: 1px solid #22262c; border-radius: 6px; padding: 1rem; }
  #run-detail pre { white-space: pre-wrap; font-size: 0.78rem; color: #b6bdc4; }
</style>
</head>
<body>
<h1>Agentic Reliability Lab</h1>

<div class="cards" id="overview-cards"></div>

<h2>Reliability model</h2>
<div class="cards" id="reliability-cards"></div>

<h2>Runs</h2>
<table id="runs-table">
  <thead><tr><th>Run</th><th>Task</th><th>Status</th><th>Result</th><th>Failure category</th></tr></thead>
  <tbody></tbody>
</table>

<div id="run-detail"></div>

<script>
async function j(url) { const r = await fetch(url); return r.json(); }

function card(value, label) {
  return `<div class="card"><div class="value">${value}</div><div class="label">${label}</div></div>`;
}

async function loadOverview() {
  const o = await j('/api/overview');
  const el = document.getElementById('overview-cards');
  el.innerHTML =
    card(o.total_runs, 'total runs') +
    card(o.success_rate !== null ? (o.success_rate * 100).toFixed(1) + '%' : '—', 'success rate') +
    card(o.graded_runs, 'graded runs');
}

async function loadReliability() {
  const r = await j('/api/reliability');
  const el = document.getElementById('reliability-cards');
  if (!r.trained) {
    el.innerHTML = card('not trained', 'run `arl reliability train`');
    return;
  }
  el.innerHTML =
    card(r.auroc.toFixed(3), 'AUROC') +
    card(r.auprc.toFixed(3), 'AUPRC') +
    card(r.precision.toFixed(2), 'precision') +
    card(r.recall.toFixed(2), 'recall') +
    card(r.n_train_examples, 'train examples');
}

async function loadRuns() {
  const runs = await j('/api/runs');
  const tbody = document.querySelector('#runs-table tbody');
  tbody.innerHTML = runs.map(r => `
    <tr onclick="loadDetail('${r.id}')">
      <td>${r.id.slice(0,8)}</td>
      <td>${r.task_id}</td>
      <td>${r.status}</td>
      <td class="${r.success ? 'pass' : 'fail'}">${r.success === null ? '—' : (r.success ? 'PASS' : 'FAIL')}</td>
      <td>${r.failure_category ?? '—'}</td>
    </tr>`).join('');
}

async function loadDetail(id) {
  const d = await j('/api/runs/' + id);
  const el = document.getElementById('run-detail');
  el.style.display = 'block';
  el.innerHTML = `<h2>Run ${d.id}</h2><pre>${JSON.stringify(d, null, 2)}</pre>`;
}

loadOverview(); loadReliability(); loadRuns();
</script>
</body>
</html>
"""
