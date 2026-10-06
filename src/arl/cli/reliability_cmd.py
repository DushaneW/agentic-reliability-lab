"""`arl reliability ...` commands."""

from __future__ import annotations

import json
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from arl.evaluation.benchmark import load_benchmark
from arl.importers.jsonl import TrajectoryFormatError, load_jsonl
from arl.reliability.crossval import Protocol, evaluate_all, render_markdown
from arl.reliability.dataset import DatasetExample, generate_dataset
from arl.reliability.features import extract_features
from arl.reliability.model import (
    EvalMetrics,
    ReliabilityModel,
    evaluate_model,
    feature_importances,
    train_test_split_examples,
)

app = typer.Typer(help="Train and evaluate the reliability predictor.")
console = Console()

DEFAULT_MODEL_PATH = "models/reliability_model.pkl"
DEFAULT_METRICS_PATH = "models/reliability_metrics.json"


@app.command("train")
def train(
    benchmarks: list[str] = typer.Option(
        ["benchmarks/coding", "benchmarks/tool_use", "benchmarks/smoke"],
        help="Benchmark directories to source trajectories from",
    ),
    seeds_per_profile: int = typer.Option(8, help="Runs per (task, fault profile) combination"),
    model_path: str = typer.Option(DEFAULT_MODEL_PATH),
    metrics_path: str = typer.Option(DEFAULT_METRICS_PATH),
) -> None:
    """Generate a labeled trajectory dataset and train the baseline predictor."""
    tasks = []
    for b in benchmarks:
        tasks.extend(load_benchmark(b))

    console.print(f"Generating dataset from {len(tasks)} tasks x fault profiles x seeds...")
    examples = generate_dataset(tasks, seeds_per_profile=seeds_per_profile)
    console.print(f"{len(examples)} examples, failure rate {_failure_rate(examples):.2f}")

    train_examples, test_examples = train_test_split_examples(examples)
    model = ReliabilityModel()
    model.fit(train_examples)
    model.save(model_path)

    metrics = evaluate_model(model, test_examples)
    Path(metrics_path).parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_path, "w") as f:
        json.dump(
            {
                "auroc": metrics.auroc,
                "auprc": metrics.auprc,
                "precision": metrics.precision,
                "recall": metrics.recall,
                "f1": metrics.f1,
                "false_positive_rate": metrics.false_positive_rate,
                "n_test_examples": metrics.n_examples,
                "n_train_examples": len(train_examples),
                "per_fraction_auroc": metrics.per_fraction_auroc,
                "feature_importances": feature_importances(model),
            },
            f,
            indent=2,
        )

    console.print(f"[green]Saved model to {model_path}[/green]")
    _print_metrics(metrics)


@app.command("cv")
def cross_validate_cmd(
    benchmarks: list[str] = typer.Option(
        ["benchmarks/coding", "benchmarks/tool_use", "benchmarks/smoke"],
        help="Benchmark directories to source trajectories from",
    ),
    seeds_per_profile: int = typer.Option(10, help="Runs per (task, fault profile)"),
    protocol: list[Protocol] = typer.Option(
        [], help="Protocols to run (default: all). Repeat the flag for several."
    ),
    n_boot: int = typer.Option(300, help="Cluster-bootstrap resamples for confidence intervals"),
    output: str = typer.Option("", help="Write the markdown report to this path"),
) -> None:
    """Compare predictors and baselines under grouped cross-validation."""
    tasks = []
    for b in benchmarks:
        tasks.extend(load_benchmark(b))
    examples = generate_dataset(tasks, seeds_per_profile=seeds_per_profile)
    results = evaluate_all(examples, protocols=protocol or None, n_boot=n_boot)
    report = render_markdown(results)
    console.print(report)
    if output:
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        Path(output).write_text(report)


@app.command("score")
def score_file(
    path: str = typer.Argument(..., help="JSONL trajectories (schema: arl.importers.jsonl)"),
    model_path: str = typer.Option(DEFAULT_MODEL_PATH),
) -> None:
    """Score external trajectories with the trained predictor.

    Prints the predicted failure probability at 20/40/60/80/100% of each
    trajectory. If the file carries `success` labels, also reports AUROC per
    completion fraction. The predictor was trained on synthetic data; read
    docs/research/limitations.md before treating these scores as meaningful.
    """
    if not Path(model_path).exists():
        console.print(f"[red]No model at {model_path}. Run `arl reliability train` first.[/red]")
        raise typer.Exit(1)
    model = ReliabilityModel.load(model_path)
    fractions = (0.2, 0.4, 0.6, 0.8, 1.0)

    try:
        trajectories = list(load_jsonl(path))
    except (TrajectoryFormatError, OSError) as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from exc

    table = Table("Run", "Label", *(f"{int(f * 100)}%" for f in fractions))
    scores: dict[float, list[tuple[int, float]]] = {f: [] for f in fractions}
    for traj in trajectories:
        row: list[str] = []
        for f in fractions:
            cutoff = max(1, round(len(traj.events) * f))
            p = model.predict_proba(extract_features(traj.events[:cutoff]).as_list())
            row.append(f"{p:.2f}")
            if traj.success is not None:
                scores[f].append((0 if traj.success else 1, p))
        label = "?" if traj.success is None else ("pass" if traj.success else "fail")
        table.add_row(traj.run_id, label, *row)
    console.print(table)

    from sklearn.metrics import roc_auc_score

    auroc_table = Table("Completion", "Labeled runs", "AUROC")
    for f in fractions:
        pairs = scores[f]
        ys = [y for y, _ in pairs]
        if len(set(ys)) == 2:
            auroc_table.add_row(
                f"{int(f * 100)}%", str(len(ys)), f"{roc_auc_score(ys, [p for _, p in pairs]):.3f}"
            )
    if auroc_table.row_count:
        console.print(auroc_table)
    else:
        console.print("(need both passing and failing labeled runs to compute AUROC)")


@app.command("evaluate")
def evaluate_cmd(
    model_path: str = typer.Option(DEFAULT_MODEL_PATH),
    metrics_path: str = typer.Option(DEFAULT_METRICS_PATH),
) -> None:
    """Print the metrics recorded from the last `arl reliability train` run."""
    path = Path(metrics_path)
    if not path.exists():
        console.print(f"[red]No metrics file at {path}. Run `arl reliability train` first.[/red]")
        raise typer.Exit(1)
    with open(path) as f:
        data = json.load(f)

    table = Table("Metric", "Value")
    for key in ("auroc", "auprc", "precision", "recall", "f1", "false_positive_rate"):
        table.add_row(key, f"{data[key]:.3f}")
    console.print(table)

    fraction_table = Table("Trajectory completion", "AUROC")
    for fraction, auroc in sorted(data["per_fraction_auroc"].items()):
        fraction_table.add_row(f"{float(fraction) * 100:.0f}%", f"{auroc:.3f}")
    console.print(fraction_table)


def _failure_rate(examples: list[DatasetExample]) -> float:
    return sum(e.label_failure for e in examples) / len(examples) if examples else 0.0


def _print_metrics(metrics: EvalMetrics) -> None:
    table = Table("Metric", "Value")
    table.add_row("AUROC", f"{metrics.auroc:.3f}")
    table.add_row("AUPRC", f"{metrics.auprc:.3f}")
    table.add_row("Precision", f"{metrics.precision:.3f}")
    table.add_row("Recall", f"{metrics.recall:.3f}")
    table.add_row("F1", f"{metrics.f1:.3f}")
    table.add_row("False positive rate", f"{metrics.false_positive_rate:.3f}")
    console.print(table)
