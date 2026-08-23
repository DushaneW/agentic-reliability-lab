"""`arl` CLI entrypoint."""

from __future__ import annotations

import typer

from arl.cli import benchmark_cmd, experiment_cmd, reliability_cmd, report_cmd, runs_cmd

app = typer.Typer(
    name="arl",
    help="Agentic Reliability Lab: benchmark, trace, and diagnose autonomous agent failures.",
    no_args_is_help=True,
)

app.add_typer(benchmark_cmd.app, name="benchmark")
app.add_typer(runs_cmd.app, name="runs")
app.add_typer(experiment_cmd.app, name="experiment")
app.add_typer(reliability_cmd.app, name="reliability")
app.add_typer(report_cmd.app, name="report")


if __name__ == "__main__":
    app()
