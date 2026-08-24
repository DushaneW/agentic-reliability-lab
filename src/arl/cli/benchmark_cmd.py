"""`arl benchmark ...` commands."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from arl.agents.reference import agent_from_task_metadata
from arl.analysis.engine import diagnose
from arl.domain.run import Run
from arl.evaluation.benchmark import load_benchmark
from arl.evaluation.engine import evaluate
from arl.runtime.runner import run_task
from arl.storage.repository import Repository

app = typer.Typer(help="Run and inspect benchmark suites.")
console = Console()


@dataclass
class BenchmarkRow:
    task_id: str
    run_id: str
    success: bool
    steps: int
    category: str


@app.command("list")
def list_benchmarks(
    benchmarks_dir: str = typer.Argument("benchmarks", help="Root directory of benchmark suites"),
) -> None:
    """List available benchmark suites and task counts."""
    root = Path(benchmarks_dir)
    if not root.is_dir():
        console.print(f"[red]No such directory: {root}[/red]")
        raise typer.Exit(1)

    table = Table("Suite", "Tasks")
    for suite_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        n_tasks = len(list(suite_dir.glob("*.yaml")))
        if n_tasks:
            table.add_row(suite_dir.name, str(n_tasks))
    console.print(table)


@app.command("run")
def run_benchmark(
    path: str = typer.Argument(..., help="Path to a benchmark suite, e.g. benchmarks/smoke"),
    db: str = typer.Option("arl.db", help="SQLite database path"),
    seed: int = typer.Option(0, help="Base seed for the reference agent"),
    json_output: bool = typer.Option(False, "--json", help="Emit machine-readable JSON"),
) -> None:
    """Run every task in a benchmark suite with the reference agent and persist results."""
    tasks = load_benchmark(path)
    repo = Repository(db)
    rows: list[BenchmarkRow] = []

    for i, task in enumerate(tasks):
        run = Run(task_id=task.id, agent_name="reference", seed=seed + i)
        agent = agent_from_task_metadata(task, seed=run.seed)
        outcome = run_task(task, agent, run)
        evaluation = evaluate(task, outcome.task_result)
        diagnosis = diagnose(task, outcome.trajectory, evaluation)

        repo.save_run(run)
        repo.save_events(outcome.trajectory.events)
        repo.save_task_result(outcome.task_result)
        repo.save_evaluation_result(evaluation)
        repo.save_failure_diagnosis(diagnosis)

        rows.append(
            BenchmarkRow(
                task_id=task.id,
                run_id=run.id,
                success=evaluation.success,
                steps=outcome.task_result.steps_taken,
                category=diagnosis.category.value,
            )
        )

    repo.close()

    if json_output:
        import json as json_module

        console.print(
            json_module.dumps([row.__dict__ for row in rows], indent=2)
        )
        return

    table = Table("Task", "Run ID", "Result", "Steps", "Failure category")
    for row in rows:
        result = "[green]PASS[/green]" if row.success else "[red]FAIL[/red]"
        table.add_row(row.task_id, row.run_id[:8], result, str(row.steps), row.category)
    console.print(table)
    n_success = sum(row.success for row in rows)
    console.print(f"\n{n_success}/{len(rows)} tasks passed")
