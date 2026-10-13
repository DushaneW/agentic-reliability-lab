"""Local dashboard.

A single FastAPI app: a handful of JSON endpoints over the same
`Repository` the CLI uses, plus one HTML page that fetches them client-side
with vanilla JS. No frontend build step, no framework — the data volume
(hundreds of runs on a laptop) doesn't justify one, and it keeps `arl
dashboard` a `pip install`-only feature.

Everything the page shows (run ids, task ids, tool arguments, tool output) can
originate from an agent or from an imported JSONL file, so it is untrusted.
The page therefore never builds HTML from it: rows are created with DOM APIs
and `textContent`, and the response carries a CSP that allows only the page's
own nonce-tagged script and same-origin requests.

There is no authentication. Bind it to localhost only.
"""

from __future__ import annotations

import json
import os
import secrets
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from arl.storage.repository import Repository

app = FastAPI(title="Agentic Reliability Lab Dashboard")

DB_PATH = os.environ.get("ARL_DB_PATH", "arl.db")
METRICS_PATH = Path(os.environ.get("ARL_METRICS_PATH", "models/reliability_metrics.json"))
STATIC_DIR = Path(__file__).parent / "static"
if STATIC_DIR.is_dir():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@contextmanager
def _repo() -> Iterator[Repository]:
    repo = Repository(DB_PATH)
    try:
        yield repo
    finally:
        repo.close()


@app.get("/api/overview")
def overview() -> dict[str, object]:
    with _repo() as repo:
        runs = repo.list_runs()
        evaluations = {e.run_id: e for e in repo.list_evaluation_results()}

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
    with _repo() as repo:
        runs = repo.list_runs()
        evaluations = {e.run_id: e for e in repo.list_evaluation_results()}
        diagnoses = {}
        for run in runs:
            d = repo.get_failure_diagnosis(run.id)
            if d:
                diagnoses[run.id] = d

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
    with _repo() as repo:
        run = repo.get_run(run_id)
        if run is None:
            for candidate in repo.list_runs():
                if candidate.id.startswith(run_id):
                    run = candidate
                    break
        if run is None:
            raise HTTPException(status_code=404, detail="run not found")

        trajectory = repo.get_trajectory(run.id)
        task_result = repo.get_task_result(run.id)
        evaluations = repo.list_evaluation_results([run.id])
        diagnosis = repo.get_failure_diagnosis(run.id)

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
    if not METRICS_PATH.exists():
        return {"trained": False}
    with open(METRICS_PATH, encoding="utf-8") as f:
        data = json.load(f)
    return {"trained": True, **data}


@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    nonce = secrets.token_urlsafe(16)
    csp = (
        "default-src 'none'; "
        f"script-src 'nonce-{nonce}'; "
        f"style-src 'nonce-{nonce}'; "
        "connect-src 'self'; "
        "base-uri 'none'; "
        "form-action 'none'; "
        "frame-ancestors 'none'"
    )
    return HTMLResponse(
        _INDEX_HTML.replace("__NONCE__", nonce),
        headers={
            "Content-Security-Policy": csp,
            "X-Content-Type-Options": "nosniff",
            "Referrer-Policy": "no-referrer",
        },
    )


_INDEX_HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Agentic Reliability Lab</title>
<style nonce="__NONCE__">
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
  tbody tr { cursor: pointer; } tbody tr:hover { background: #14171c; }
  #run-detail { display: none; margin-top: 1rem; background: #12151a;
                border: 1px solid #22262c; border-radius: 6px; padding: 1rem; }
  #run-detail pre { white-space: pre-wrap; font-size: 0.78rem; color: #b6bdc4; }
  .error { color: #e5534b; }
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

<script nonce="__NONCE__">
// Run ids, task ids, tool payloads etc. are untrusted: only ever put them into
// the page with textContent, never into HTML strings.
function el(tag, text, className) {
  const node = document.createElement(tag);
  if (text !== undefined && text !== null) node.textContent = String(text);
  if (className) node.className = className;
  return node;
}

async function j(url) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(url + ' -> HTTP ' + r.status);
  return r.json();
}

function card(value, label) {
  const c = el('div', null, 'card');
  c.appendChild(el('div', value, 'value'));
  c.appendChild(el('div', label, 'label'));
  return c;
}

function fill(id, nodes) {
  document.getElementById(id).replaceChildren(...nodes);
}

function showError(id, err) {
  const box = document.getElementById(id);
  box.style.display = 'block';
  box.replaceChildren(el('div', err.message, 'error'));
}

function pct(x) { return x === null || x === undefined ? '—' : (x * 100).toFixed(1) + '%'; }

async function loadOverview() {
  const o = await j('/api/overview');
  fill('overview-cards', [
    card(o.total_runs, 'total runs'),
    card(pct(o.success_rate), 'success rate'),
    card(o.graded_runs, 'graded runs'),
  ]);
}

async function loadReliability() {
  const r = await j('/api/reliability');
  if (!r.trained) {
    fill('reliability-cards', [card('not trained', 'run arl reliability train')]);
    return;
  }
  fill('reliability-cards', [
    card(r.auroc.toFixed(3), 'AUROC (single split, all prefixes)'),
    card(r.auprc.toFixed(3), 'AUPRC'),
    card(r.precision.toFixed(2), 'precision'),
    card(r.recall.toFixed(2), 'recall'),
    card(r.n_train_examples, 'train examples'),
  ]);
}

async function loadRuns() {
  const runs = await j('/api/runs');
  const tbody = document.querySelector('#runs-table tbody');
  tbody.replaceChildren(...runs.map(r => {
    const tr = document.createElement('tr');
    tr.dataset.id = r.id;
    const result = r.success === null ? '—' : (r.success ? 'PASS' : 'FAIL');
    const resultClass = r.success === null ? '' : (r.success ? 'pass' : 'fail');
    tr.append(
      el('td', r.id.slice(0, 8)),
      el('td', r.task_id),
      el('td', r.status),
      el('td', result, resultClass),
      el('td', r.failure_category ?? '—'),
    );
    return tr;
  }));
}

async function loadDetail(id) {
  const box = document.getElementById('run-detail');
  box.style.display = 'block';
  try {
    const d = await j('/api/runs/' + encodeURIComponent(id));
    box.replaceChildren(el('h2', 'Run ' + d.id), el('pre', JSON.stringify(d, null, 2)));
  } catch (err) {
    box.replaceChildren(el('div', err.message, 'error'));
  }
}

document.querySelector('#runs-table tbody').addEventListener('click', ev => {
  const tr = ev.target.closest('tr');
  if (tr && tr.dataset.id) loadDetail(tr.dataset.id);
});

loadOverview().catch(e => showError('overview-cards', e));
loadReliability().catch(e => showError('reliability-cards', e));
loadRuns().catch(e => showError('run-detail', e));
</script>
</body>
</html>
"""
