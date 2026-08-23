"""Loads Task definitions from a benchmark directory of YAML files."""

from __future__ import annotations

from pathlib import Path

import yaml

from arl.domain.task import Task


def load_benchmark(path: str | Path) -> list[Task]:
    directory = Path(path)
    if not directory.is_dir():
        raise FileNotFoundError(f"Benchmark directory not found: {directory}")

    tasks: list[Task] = []
    for task_file in sorted(directory.glob("*.yaml")):
        with task_file.open() as f:
            raw = yaml.safe_load(f)
        tasks.append(Task.model_validate(raw))

    if not tasks:
        raise ValueError(f"No task YAML files found in {directory}")

    ids = [t.id for t in tasks]
    if len(ids) != len(set(ids)):
        raise ValueError(f"Duplicate task ids in {directory}")

    return tasks
