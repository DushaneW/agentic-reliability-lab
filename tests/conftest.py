from __future__ import annotations

import pytest

from arl.domain.task import GraderSpec, Task, TaskCategory, TaskDifficulty


@pytest.fixture
def sum_task() -> Task:
    return Task(
        id="test-sum",
        description="sum the numbers in input.txt",
        category=TaskCategory.CODING,
        difficulty=TaskDifficulty.EASY,
        grader=GraderSpec(
            kind="numeric_file_equals",
            config={"path": "output.txt", "expected": 6.0, "tolerance": 1e-6},
        ),
        timeout_seconds=5.0,
        max_steps=8,
        metadata={"initial_files": {"input.txt": "1 2 3"}},
    )
