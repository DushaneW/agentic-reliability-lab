from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from arl.cli.main import app
from arl.evaluation.benchmark import load_benchmark
from arl.reliability.dataset import generate_dataset
from arl.reliability.model import ReliabilityModel

runner = CliRunner()


def test_score_reports_labeled_auroc(tmp_path: Path) -> None:
    tasks = load_benchmark("benchmarks/smoke")
    model = ReliabilityModel()
    model.fit(generate_dataset(tasks, seeds_per_profile=2))
    model_path = tmp_path / "m.pkl"
    model.save(model_path)

    ok = {"tool": "calculator", "arguments": {"expression": "1+1"}, "result": "2"}
    bad = {"tool": "calculator", "arguments": {"expression": "1/0"}, "error": "div by zero"}
    rows = [
        {"run_id": "good", "success": True, "steps": [ok, ok, ok]},
        {"run_id": "bad", "success": False, "steps": [bad, bad, bad, bad]},
        {"run_id": "unknown", "steps": [ok]},
    ]
    data = tmp_path / "t.jsonl"
    data.write_text("\n".join(json.dumps(r) for r in rows))

    result = runner.invoke(
        app, ["reliability", "score", str(data), "--model-path", str(model_path)]
    )
    assert result.exit_code == 0, result.output
    assert "good" in result.output and "bad" in result.output
    assert "AUROC" in result.output


def test_score_rejects_malformed_file(tmp_path: Path) -> None:
    model_path = tmp_path / "m.pkl"
    ReliabilityModel().save(model_path)
    data = tmp_path / "t.jsonl"
    data.write_text("nope")
    result = runner.invoke(
        app, ["reliability", "score", str(data), "--model-path", str(model_path)]
    )
    assert result.exit_code == 1
    assert "line 1" in result.output
