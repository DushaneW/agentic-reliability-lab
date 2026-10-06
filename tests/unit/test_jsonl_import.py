from __future__ import annotations

import json
from pathlib import Path

import pytest

from arl.domain.events import EventType
from arl.importers.jsonl import TrajectoryFormatError, load_jsonl


def _write(tmp_path: Path, *lines: object) -> Path:
    p = tmp_path / "t.jsonl"
    p.write_text("\n".join(x if isinstance(x, str) else json.dumps(x) for x in lines) + "\n")
    return p


def test_steps_become_call_plus_outcome_with_lifecycle_events(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        {
            "run_id": "a",
            "success": False,
            "steps": [
                {"tool": "bash", "arguments": {"cmd": "ls"}, "result": "x"},
                {"tool": "bash", "arguments": {"cmd": "cat y"}, "error": "no such file"},
            ],
        },
    )
    (t,) = load_jsonl(path)
    assert t.success is False
    assert [e.event_type for e in t.events] == [
        EventType.RUN_STARTED,
        EventType.TOOL_CALL,
        EventType.TOOL_RESULT,
        EventType.TOOL_CALL,
        EventType.TOOL_ERROR,
        EventType.RUN_FINISHED,
    ]
    assert [e.step_index for e in t.events] == list(range(6))


def test_unlabeled_and_blank_lines(tmp_path: Path) -> None:
    path = _write(tmp_path, {"run_id": "a", "steps": []}, "", {"run_id": "b", "steps": []})
    out = list(load_jsonl(path))
    assert [t.run_id for t in out] == ["a", "b"]
    assert all(t.success is None for t in out)


@pytest.mark.parametrize(
    "line, message",
    [
        ("not json", "invalid JSON"),
        ('{"steps": []}', "run_id"),
        ('{"run_id": "a", "steps": "x"}', "steps"),
        ('{"run_id": "a", "steps": [], "success": "yes"}', "success"),
        ('{"run_id": "a", "steps": [{"arguments": {}}]}', "tool"),
        ('{"run_id": "a", "steps": [{"tool": "t", "arguments": 3}]}', "arguments"),
        ("[1]", "JSON object"),
    ],
)
def test_malformed_lines_report_the_line_number(tmp_path: Path, line: str, message: str) -> None:
    path = _write(tmp_path, line)
    with pytest.raises(TrajectoryFormatError, match=rf"line 1.*{message}"):
        list(load_jsonl(path))
