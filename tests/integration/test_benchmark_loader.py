from __future__ import annotations

from pathlib import Path

import pytest

from arl.evaluation.benchmark import load_benchmark

FIXTURE_DIR = Path(__file__).resolve().parent.parent / "fixtures" / "mini_benchmark"


def test_load_benchmark_reads_all_yaml_files() -> None:
    tasks = load_benchmark(FIXTURE_DIR)
    assert len(tasks) == 1
    assert tasks[0].id == "fixture-001"


def test_load_benchmark_missing_dir_raises() -> None:
    with pytest.raises(FileNotFoundError):
        load_benchmark("/nonexistent/path/xyz")


def test_real_benchmark_suites_all_load() -> None:
    root = Path(__file__).resolve().parent.parent.parent / "benchmarks"
    if not root.is_dir():
        pytest.skip("benchmarks/ not generated in this environment")
    for suite in root.iterdir():
        if suite.is_dir() and list(suite.glob("*.yaml")):
            tasks = load_benchmark(suite)
            assert len(tasks) > 0
