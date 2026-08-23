"""`arl report generate` — produces a Markdown report for a single run."""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console

from arl.storage.repository import Repository

app = typer.Typer(help="Generate Markdown reports.")
console = Console()


@app.command("generate")
def generate(
    run_id: str,
    db: str = typer.Option("arl.db", help="SQLite database path"),
    output: str = typer.Option("", help="Output path; defaults to reports/<run_id>.md"),
) -> None:
    repo = Repository(db)
    run = repo.get_run(run_id)
    if run is None:
        for candidate in repo.list_runs():
            if candidate.id.startswith(run_id):
                run = candidate
                break
    if run is None:
        console.print(f"[red]No run matching '{run_id}'[/red]")
        raise typer.Exit(1)

    trajectory = repo.get_trajectory(run.id)
    task_result = repo.get_task_result(run.id)
    evaluations = repo.list_evaluation_results([run.id])
    diagnosis = repo.get_failure_diagnosis(run.id)
    repo.close()

    evaluation = evaluations[0] if evaluations else None
    lines = [
        f"# Run report: {run.id}",
        "",
        f"- Task: `{run.task_id}`",
        f"- Agent: `{run.agent_name}`",
        f"- Status: `{run.status.value}`",
    ]
    if task_result:
        lines += [
            f"- Steps taken: {task_result.steps_taken}",
            f"- Wall time: {task_result.wall_time_seconds:.3f}s",
            f"- Terminated: {task_result.terminated_reason}",
        ]
    if evaluation:
        lines += [
            "",
            "## Evaluation",
            "",
            f"- Success: **{evaluation.success}**",
            f"- Score: {evaluation.score}",
        ]
    if diagnosis and diagnosis.category.value != "none":
        lines += [
            "",
            "## Failure diagnosis",
            "",
            f"- Category: **{diagnosis.category.value}**",
            f"- Confidence: {diagnosis.confidence:.2f}",
            f"- Wasted tool calls: {diagnosis.wasted_tool_calls}",
        ]
        for evidence in diagnosis.evidence:
            lines.append(f"  - {evidence.description}")

    lines += ["", "## Trajectory", ""]
    for event in trajectory.events:
        lines.append(f"{event.step_index}. `{event.event_type.value}` — {event.payload}")

    content = "\n".join(lines) + "\n"
    out_path = Path(output) if output else Path("reports") / f"{run.id}.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(content)
    console.print(f"[green]Wrote {out_path}[/green]")
