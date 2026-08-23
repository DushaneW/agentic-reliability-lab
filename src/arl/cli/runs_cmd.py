"""`arl runs ...` commands."""

from __future__ import annotations

from typing import Any

import typer
from rich.console import Console
from rich.table import Table

from arl.domain.run import Run
from arl.storage.repository import Repository

app = typer.Typer(help="List and inspect persisted runs.")
console = Console()


@app.command("list")
def list_runs(db: str = typer.Option("arl.db", help="SQLite database path")) -> None:
    repo = Repository(db)
    runs = repo.list_runs()
    evaluations = {e.run_id: e for e in repo.list_evaluation_results()}
    repo.close()

    if not runs:
        console.print("[yellow]No runs found. Run `arl benchmark run <path>` first.[/yellow]")
        return

    table = Table("Run ID", "Task", "Agent", "Status", "Result")
    for run in runs:
        ev = evaluations.get(run.id)
        result = "-" if ev is None else ("[green]PASS[/green]" if ev.success else "[red]FAIL[/red]")
        table.add_row(run.id[:8], run.task_id, run.agent_name, run.status.value, result)
    console.print(table)


@app.command("inspect")
def inspect_run(
    run_id: str,
    db: str = typer.Option("arl.db", help="SQLite database path"),
) -> None:
    repo = Repository(db)
    run = _resolve_run(repo, run_id)
    if run is None:
        console.print(f"[red]No run matching '{run_id}'[/red]")
        raise typer.Exit(1)

    trajectory = repo.get_trajectory(run.id)
    result = repo.get_task_result(run.id)
    evaluation_results = repo.list_evaluation_results([run.id])
    repo.close()

    console.print(f"[bold]Run {run.id}[/bold]  task={run.task_id}  agent={run.agent_name}")
    if result:
        console.print(
            f"terminated: {result.terminated_reason}   steps: {result.steps_taken}   "
            f"wall_time: {result.wall_time_seconds:.3f}s"
        )
    if evaluation_results:
        ev = evaluation_results[0]
        status = "[green]SUCCESS[/green]" if ev.success else "[red]FAILURE[/red]"
        console.print(f"evaluation: {status}  ({ev.details.get('detail', '')})")

    table = Table("Step", "Event", "Detail")
    for event in trajectory.events:
        detail = _summarize_payload(event.event_type.value, event.payload)
        table.add_row(str(event.step_index), event.event_type.value, detail)
    console.print(table)


@app.command("analyze")
def analyze_run(
    run_id: str,
    db: str = typer.Option("arl.db", help="SQLite database path"),
) -> None:
    repo = Repository(db)
    run = _resolve_run(repo, run_id)
    if run is None:
        console.print(f"[red]No run matching '{run_id}'[/red]")
        raise typer.Exit(1)

    diagnosis = repo.get_failure_diagnosis(run.id)
    repo.close()

    if diagnosis is None:
        console.print("[yellow]No failure diagnosis recorded for this run.[/yellow]")
        return

    console.print(f"[bold]Run {run.id}[/bold]")
    console.print(f"Diagnosis: [bold]{diagnosis.category.value}[/bold]")
    console.print(f"Confidence: {diagnosis.confidence:.2f}")
    console.print(f"Wasted tool calls: {diagnosis.wasted_tool_calls}")
    console.print(f"Recovered: {diagnosis.recovered}")
    for evidence in diagnosis.evidence:
        event_refs = ", ".join(e[:8] for e in evidence.event_ids)
        console.print(f"  - {evidence.description} (events: {event_refs})")


def _resolve_run(repo: Repository, run_id: str) -> Run | None:
    run = repo.get_run(run_id)
    if run is not None:
        return run
    for candidate in repo.list_runs():
        if candidate.id.startswith(run_id):
            return candidate
    return None


def _summarize_payload(event_type: str, payload: dict[str, Any]) -> str:
    if event_type == "tool_call":
        return f"{payload.get('tool_name')}({payload.get('arguments')})"
    if event_type == "tool_result":
        summary = str(payload.get("result_summary", ""))
        return summary if len(summary) < 80 else summary[:77] + "..."
    if event_type == "tool_error":
        return f"{payload.get('tool_name')}: {payload.get('message')}"
    return str(payload.get("reason", ""))
