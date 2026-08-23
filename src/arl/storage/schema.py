"""SQLite schema. One flat file database, no migrations framework —
`init_db` is idempotent (CREATE TABLE IF NOT EXISTS), which is sufficient
for a single-file local-first tool at this stage. If the schema needs real
migrations later, this is the boundary to add them at.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    id TEXT PRIMARY KEY,
    task_id TEXT NOT NULL,
    agent_name TEXT NOT NULL,
    experiment_id TEXT,
    status TEXT NOT NULL,
    started_at TEXT NOT NULL,
    finished_at TEXT,
    seed INTEGER NOT NULL,
    config_json TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS trajectory_events (
    id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL REFERENCES runs(id),
    step_index INTEGER NOT NULL,
    event_type TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    metadata_json TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_events_run_id ON trajectory_events(run_id);

CREATE TABLE IF NOT EXISTS task_results (
    run_id TEXT PRIMARY KEY REFERENCES runs(id),
    task_id TEXT NOT NULL,
    success INTEGER NOT NULL,
    steps_taken INTEGER NOT NULL,
    wall_time_seconds REAL NOT NULL,
    terminated_reason TEXT NOT NULL,
    output_json TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS evaluation_results (
    run_id TEXT PRIMARY KEY REFERENCES runs(id),
    task_id TEXT NOT NULL,
    success INTEGER NOT NULL,
    score REAL NOT NULL,
    grader_kind TEXT NOT NULL,
    details_json TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS failure_diagnoses (
    run_id TEXT PRIMARY KEY REFERENCES runs(id),
    category TEXT NOT NULL,
    confidence REAL NOT NULL,
    evidence_json TEXT NOT NULL,
    wasted_tool_calls INTEGER NOT NULL,
    recovered INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS experiments (
    id TEXT PRIMARY KEY,
    config_json TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS experiment_results (
    experiment_id TEXT PRIMARY KEY REFERENCES experiments(id),
    total_tasks INTEGER NOT NULL,
    successes INTEGER NOT NULL,
    failures INTEGER NOT NULL,
    success_rate REAL NOT NULL,
    mean_steps REAL NOT NULL,
    mean_wall_time_seconds REAL NOT NULL,
    mean_tool_calls REAL NOT NULL,
    intervention_count INTEGER NOT NULL,
    failure_category_counts_json TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS metrics (
    run_id TEXT NOT NULL,
    name TEXT NOT NULL,
    value REAL NOT NULL,
    recorded_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_metrics_run_id ON metrics(run_id);
"""


def connect(db_path: str | Path) -> sqlite3.Connection:
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA)
    conn.commit()
