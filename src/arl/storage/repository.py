"""Read/write access to the SQLite store, in terms of domain objects.

This is a thin, deliberately unabstracted repository: one class, direct SQL,
no ORM, no unit-of-work pattern. The domain is small enough that an ORM
would add indirection without buying anything.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from arl.domain.evaluation import EvaluationResult, FailureDiagnosis
from arl.domain.events import TrajectoryEvent
from arl.domain.experiment import Experiment, ExperimentResult
from arl.domain.metric import Metric
from arl.domain.run import Run, Trajectory
from arl.domain.task import TaskResult
from arl.storage.schema import connect, init_db


class Repository:
    def __init__(self, db_path: str | Path) -> None:
        self.conn = connect(db_path)
        init_db(self.conn)

    def close(self) -> None:
        self.conn.close()

    # -- runs -----------------------------------------------------------
    def save_run(self, run: Run) -> None:
        self.conn.execute(
            """INSERT INTO runs (id, task_id, agent_name, experiment_id, status,
                   started_at, finished_at, seed, config_json)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(id) DO UPDATE SET
                   status=excluded.status, finished_at=excluded.finished_at""",
            (
                run.id,
                run.task_id,
                run.agent_name,
                run.experiment_id,
                run.status.value,
                run.started_at.isoformat(),
                run.finished_at.isoformat() if run.finished_at else None,
                run.seed,
                json.dumps(run.config),
            ),
        )
        self.conn.commit()

    def list_runs(self, experiment_id: str | None = None) -> list[Run]:
        if experiment_id:
            rows = self.conn.execute(
                "SELECT * FROM runs WHERE experiment_id = ? ORDER BY started_at", (experiment_id,)
            ).fetchall()
        else:
            rows = self.conn.execute("SELECT * FROM runs ORDER BY started_at").fetchall()
        return [_row_to_run(r) for r in rows]

    def get_run(self, run_id: str) -> Run | None:
        row = self.conn.execute("SELECT * FROM runs WHERE id = ?", (run_id,)).fetchone()
        return _row_to_run(row) if row else None

    # -- trajectory events -----------------------------------------------
    def save_events(self, events: list[TrajectoryEvent]) -> None:
        self.conn.executemany(
            """INSERT INTO trajectory_events
                   (id, run_id, step_index, event_type, timestamp, payload_json, metadata_json)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            [
                (
                    e.id,
                    e.run_id,
                    e.step_index,
                    e.event_type.value,
                    e.timestamp.isoformat(),
                    json.dumps(e.payload),
                    json.dumps(e.metadata),
                )
                for e in events
            ],
        )
        self.conn.commit()

    def get_trajectory(self, run_id: str) -> Trajectory:
        rows = self.conn.execute(
            "SELECT * FROM trajectory_events WHERE run_id = ? ORDER BY step_index", (run_id,)
        ).fetchall()
        events = [_row_to_event(r) for r in rows]
        return Trajectory(run_id=run_id, events=events)

    # -- task results ------------------------------------------------------
    def save_task_result(self, result: TaskResult) -> None:
        self.conn.execute(
            """INSERT INTO task_results
                   (run_id, task_id, success, steps_taken, wall_time_seconds,
                    terminated_reason, output_json)
               VALUES (?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(run_id) DO UPDATE SET success=excluded.success""",
            (
                result.run_id,
                result.task_id,
                int(result.success),
                result.steps_taken,
                result.wall_time_seconds,
                result.terminated_reason,
                json.dumps(result.output),
            ),
        )
        self.conn.commit()

    def get_task_result(self, run_id: str) -> TaskResult | None:
        row = self.conn.execute(
            "SELECT * FROM task_results WHERE run_id = ?", (run_id,)
        ).fetchone()
        if row is None:
            return None
        return TaskResult(
            task_id=row["task_id"],
            run_id=row["run_id"],
            success=bool(row["success"]),
            steps_taken=row["steps_taken"],
            wall_time_seconds=row["wall_time_seconds"],
            terminated_reason=row["terminated_reason"],
            output=json.loads(row["output_json"]),
        )

    # -- evaluation results ------------------------------------------------
    def save_evaluation_result(self, result: EvaluationResult) -> None:
        self.conn.execute(
            """INSERT INTO evaluation_results
                   (run_id, task_id, success, score, grader_kind, details_json)
               VALUES (?, ?, ?, ?, ?, ?)
               ON CONFLICT(run_id) DO UPDATE SET success=excluded.success""",
            (
                result.run_id,
                result.task_id,
                int(result.success),
                result.score,
                result.grader_kind,
                json.dumps(result.details),
            ),
        )
        self.conn.commit()

    def list_evaluation_results(self, run_ids: list[str] | None = None) -> list[EvaluationResult]:
        if run_ids:
            placeholders = ",".join("?" * len(run_ids))
            rows = self.conn.execute(
                f"SELECT * FROM evaluation_results WHERE run_id IN ({placeholders})", run_ids
            ).fetchall()
        else:
            rows = self.conn.execute("SELECT * FROM evaluation_results").fetchall()
        return [
            EvaluationResult(
                run_id=r["run_id"],
                task_id=r["task_id"],
                success=bool(r["success"]),
                score=r["score"],
                grader_kind=r["grader_kind"],
                details=json.loads(r["details_json"]),
            )
            for r in rows
        ]

    # -- failure diagnoses ---------------------------------------------------
    def save_failure_diagnosis(self, diagnosis: FailureDiagnosis) -> None:
        self.conn.execute(
            """INSERT INTO failure_diagnoses
                   (run_id, category, confidence, evidence_json, wasted_tool_calls, recovered)
               VALUES (?, ?, ?, ?, ?, ?)
               ON CONFLICT(run_id) DO UPDATE SET category=excluded.category""",
            (
                diagnosis.run_id,
                diagnosis.category.value,
                diagnosis.confidence,
                json.dumps([e.model_dump() for e in diagnosis.evidence]),
                diagnosis.wasted_tool_calls,
                int(diagnosis.recovered),
            ),
        )
        self.conn.commit()

    def get_failure_diagnosis(self, run_id: str) -> FailureDiagnosis | None:
        row = self.conn.execute(
            "SELECT * FROM failure_diagnoses WHERE run_id = ?", (run_id,)
        ).fetchone()
        if row is None:
            return None
        from arl.domain.evaluation import Evidence, FailureCategory

        return FailureDiagnosis(
            run_id=row["run_id"],
            category=FailureCategory(row["category"]),
            confidence=row["confidence"],
            evidence=[Evidence.model_validate(e) for e in json.loads(row["evidence_json"])],
            wasted_tool_calls=row["wasted_tool_calls"],
            recovered=bool(row["recovered"]),
        )

    # -- experiments ---------------------------------------------------------
    def save_experiment(self, experiment: Experiment) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO experiments (id, config_json) VALUES (?, ?)",
            (experiment.id, experiment.config.model_dump_json()),
        )
        self.conn.commit()

    def save_experiment_result(self, result: ExperimentResult) -> None:
        self.conn.execute(
            """INSERT INTO experiment_results
                   (experiment_id, total_tasks, successes, failures, success_rate,
                    mean_steps, mean_wall_time_seconds, mean_tool_calls,
                    intervention_count, failure_category_counts_json)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(experiment_id) DO UPDATE SET success_rate=excluded.success_rate""",
            (
                result.experiment_id,
                result.total_tasks,
                result.successes,
                result.failures,
                result.success_rate,
                result.mean_steps,
                result.mean_wall_time_seconds,
                result.mean_tool_calls,
                result.intervention_count,
                json.dumps(result.failure_category_counts),
            ),
        )
        self.conn.commit()

    def get_experiment_result(self, experiment_id: str) -> ExperimentResult | None:
        row = self.conn.execute(
            "SELECT * FROM experiment_results WHERE experiment_id = ?", (experiment_id,)
        ).fetchone()
        if row is None:
            return None
        return ExperimentResult(
            experiment_id=row["experiment_id"],
            total_tasks=row["total_tasks"],
            successes=row["successes"],
            failures=row["failures"],
            success_rate=row["success_rate"],
            mean_steps=row["mean_steps"],
            mean_wall_time_seconds=row["mean_wall_time_seconds"],
            mean_tool_calls=row["mean_tool_calls"],
            intervention_count=row["intervention_count"],
            failure_category_counts=json.loads(row["failure_category_counts_json"]),
        )

    # -- metrics --------------------------------------------------------------
    def save_metric(self, metric: Metric) -> None:
        self.conn.execute(
            "INSERT INTO metrics (run_id, name, value, recorded_at) VALUES (?, ?, ?, ?)",
            (metric.run_id, metric.name, metric.value, metric.recorded_at.isoformat()),
        )
        self.conn.commit()


def _row_to_run(row: sqlite3.Row) -> Run:
    from datetime import datetime

    from arl.domain.run import RunStatus

    return Run(
        id=row["id"],
        task_id=row["task_id"],
        agent_name=row["agent_name"],
        experiment_id=row["experiment_id"],
        status=RunStatus(row["status"]),
        started_at=datetime.fromisoformat(row["started_at"]),
        finished_at=datetime.fromisoformat(row["finished_at"]) if row["finished_at"] else None,
        seed=row["seed"],
        config=json.loads(row["config_json"]),
    )


def _row_to_event(row: sqlite3.Row) -> TrajectoryEvent:
    from datetime import datetime

    from arl.domain.events import EventType

    return TrajectoryEvent(
        id=row["id"],
        run_id=row["run_id"],
        step_index=row["step_index"],
        event_type=EventType(row["event_type"]),
        timestamp=datetime.fromisoformat(row["timestamp"]),
        payload=json.loads(row["payload_json"]),
        metadata=json.loads(row["metadata_json"]),
    )
