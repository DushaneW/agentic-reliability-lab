"""`arl experiment ...` commands."""

from __future__ import annotations

import uuid

import typer
import yaml
from rich.console import Console
from rich.table import Table

from arl.agents.reference import FaultProfile
from arl.domain.experiment import ExperimentConfig, ExperimentResult
from arl.evaluation.benchmark import load_benchmark
from arl.experiments.runner import run_experiment
from arl.storage.repository import Repository

app = typer.Typer(help="Run and compare reproducible experiments.")
console = Console()


@app.command("run")
def run(
    config_path: str = typer.Argument(..., help="Path to an experiment YAML config"),
    db: str = typer.Option("arl.db", help="SQLite database path"),
) -> None:
    with open(config_path) as f:
        raw = yaml.safe_load(f)
    config = ExperimentConfig.model_validate(raw)
    tasks = load_benchmark(config.benchmark_path)
    fault_profile = FaultProfile(**config.fault_profile) if config.fault_profile else None

    experiment_id = f"{config.name}-{uuid.uuid4().hex[:8]}"
    repo = Repository(db)
    result = run_experiment(experiment_id, config, tasks, repo=repo, fault_profile=fault_profile)
    repo.close()

    console.print(f"[bold]Experiment {experiment_id}[/bold]")
    _print_result(result)


@app.command("compare")
def compare(
    experiment_a: str,
    experiment_b: str,
    db: str = typer.Option("arl.db", help="SQLite database path"),
) -> None:
    repo = Repository(db)
    result_a = repo.get_experiment_result(experiment_a)
    result_b = repo.get_experiment_result(experiment_b)
    repo.close()

    if result_a is None or result_b is None:
        missing = experiment_a if result_a is None else experiment_b
        console.print(f"[red]No stored result for experiment '{missing}'[/red]")
        raise typer.Exit(1)

    table = Table("Metric", experiment_a, experiment_b, "Delta")
    table.add_row(
        "Success rate",
        f"{result_a.success_rate:.2%}",
        f"{result_b.success_rate:.2%}",
        f"{(result_b.success_rate - result_a.success_rate):+.2%}",
    )
    table.add_row(
        "Mean steps",
        f"{result_a.mean_steps:.1f}",
        f"{result_b.mean_steps:.1f}",
        f"{(result_b.mean_steps - result_a.mean_steps):+.1f}",
    )
    table.add_row(
        "Mean tool calls",
        f"{result_a.mean_tool_calls:.1f}",
        f"{result_b.mean_tool_calls:.1f}",
        f"{(result_b.mean_tool_calls - result_a.mean_tool_calls):+.1f}",
    )
    console.print(table)


def _print_result(result: ExperimentResult) -> None:
    table = Table("Metric", "Value")
    table.add_row("Total tasks", str(result.total_tasks))
    table.add_row("Successes", str(result.successes))
    table.add_row("Success rate", f"{result.success_rate:.2%}")
    table.add_row("Mean steps", f"{result.mean_steps:.1f}")
    table.add_row("Mean wall time (s)", f"{result.mean_wall_time_seconds:.3f}")
    table.add_row("Mean tool calls", f"{result.mean_tool_calls:.1f}")
    console.print(table)
    if result.failure_category_counts:
        cat_table = Table("Failure category", "Count")
        for category, count in sorted(
            result.failure_category_counts.items(), key=lambda kv: -kv[1]
        ):
            if category != "none":
                cat_table.add_row(category, str(count))
        console.print(cat_table)
