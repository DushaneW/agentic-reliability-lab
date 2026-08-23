"""Generates the benchmark/*/*.yaml task suite.

All tasks in this suite share one underlying operation — sum the numbers
found in input.txt and write the total to output.txt — because the shipped
reference agent (arl.agents.reference.ReferenceAgent) implements exactly
that one generic strategy and nothing more. See docs/research/limitations.md
for why: without a real LLM in the loop, a broader task suite would either
require task-specific scripted logic (which would make "agent failure"
analysis meaningless, since failures would be bugs in per-task code, not
agent behavior) or fake results. Difficulty and category labels here
organize the evaluation/analysis infrastructure and are exercised by the
runner and CLI; they do not yet imply the agent can plan across task types.

Run: python scripts/generate_benchmark_tasks.py
"""

from __future__ import annotations

import random
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
BENCHMARKS = ROOT / "benchmarks"


def task_yaml(
    task_id: str,
    category: str,
    difficulty: str,
    numbers: list[float],
    max_steps: int = 8,
    timeout: float = 10.0,
) -> dict:
    content = " ".join(str(n) for n in numbers)
    total = sum(numbers)
    return {
        "id": task_id,
        "description": (
            "input.txt contains a list of numbers. Compute their sum and "
            "write the result to output.txt."
        ),
        "category": category,
        "difficulty": difficulty,
        "environment": "sandboxed_fs",
        "grader": {
            "kind": "numeric_file_equals",
            "config": {"path": "output.txt", "expected": float(total), "tolerance": 1e-6},
        },
        "timeout_seconds": timeout,
        "max_steps": max_steps,
        "metadata": {"initial_files": {"input.txt": content}},
    }


def write_task(directory: Path, task: dict) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    out_path = directory / f"{task['id']}.yaml"
    with out_path.open("w") as f:
        yaml.safe_dump(task, f, sort_keys=False)


def main() -> None:
    rng = random.Random(1234)

    # smoke: fast, trivial, used for `arl benchmark run benchmarks/smoke`
    smoke_dir = BENCHMARKS / "smoke"
    write_task(smoke_dir, task_yaml("smoke-001", "coding", "easy", [1, 2, 3]))
    write_task(smoke_dir, task_yaml("smoke-002", "coding", "easy", [10, -4]))

    # coding: varying magnitude / count
    coding_dir = BENCHMARKS / "coding"
    for i in range(1, 11):
        n = rng.randint(2, 6)
        numbers = [round(rng.uniform(-50, 50), 2) for _ in range(n)]
        difficulty = "easy" if n <= 3 else "medium" if n <= 5 else "hard"
        write_task(coding_dir, task_yaml(f"coding-{i:03d}", "coding", difficulty, numbers))

    # tool_use: same op, exercises multi-call sequences with larger sets
    tool_use_dir = BENCHMARKS / "tool_use"
    for i in range(1, 6):
        n = rng.randint(6, 10)
        numbers = [round(rng.uniform(0, 100), 1) for _ in range(n)]
        write_task(tool_use_dir, task_yaml(f"tool-use-{i:03d}", "tool_use", "medium", numbers))

    # recovery: empty / degenerate inputs, still solvable by the same strategy
    recovery_dir = BENCHMARKS / "recovery"
    write_task(recovery_dir, task_yaml("recovery-001", "error_recovery", "medium", [0]))
    write_task(recovery_dir, task_yaml("recovery-002", "error_recovery", "medium", [5, -5]))
    write_task(recovery_dir, task_yaml("recovery-003", "error_recovery", "hard", [1e6, -1e6, 1]))

    # adversarial: tight step/time budgets so agent competence actually matters
    adversarial_dir = BENCHMARKS / "adversarial"
    for i in range(1, 4):
        n = rng.randint(3, 5)
        numbers = [round(rng.uniform(-10, 10), 1) for _ in range(n)]
        write_task(
            adversarial_dir,
            task_yaml(f"adversarial-{i:03d}", "adversarial", "hard", numbers, max_steps=4),
        )

    total = len(list(BENCHMARKS.rglob("*.yaml")))
    print(f"Generated {total} task files under {BENCHMARKS}")


if __name__ == "__main__":
    main()
