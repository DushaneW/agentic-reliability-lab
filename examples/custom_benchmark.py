"""Example: defining a task and a custom grader without editing the core package.

Shows the two extension points a new benchmark task type actually needs:
a Task (loaded here from an in-memory dict instead of YAML, to keep the
example self-contained) and a Grader registered under a new `kind`.

Run:
    uv run python examples/custom_benchmark.py
"""

from __future__ import annotations

from arl.agents.reference import ReferenceAgent
from arl.domain.run import Run
from arl.domain.task import GraderSpec, Task, TaskCategory, TaskDifficulty
from arl.evaluation.engine import evaluate
from arl.evaluation.graders import GRADERS
from arl.runtime.runner import run_task


class FileNotEmptyGrader:
    """Passes iff a named file exists and is non-empty.

    A genuinely different grading strategy from the bundled ones — useful
    for tasks where you care that *something* was produced, not what
    exactly it says.
    """

    def grade(self, spec, task_result):  # noqa: ANN001, ANN201
        path = spec.config["path"]
        files = task_result.output.get("files", {})
        content = files.get(path)
        success = bool(content and content.strip())
        detail = "file missing" if content is None else f"content: {content!r}"
        return success, 1.0 if success else 0.0, detail


def main() -> None:
    GRADERS["file_not_empty"] = FileNotEmptyGrader()

    task = Task(
        id="example-not-empty",
        description="Write anything to output.txt.",
        category=TaskCategory.CODING,
        difficulty=TaskDifficulty.EASY,
        grader=GraderSpec(kind="file_not_empty", config={"path": "output.txt"}),
        timeout_seconds=5.0,
        max_steps=8,
        metadata={"initial_files": {"input.txt": "7 8 9"}},
    )

    run = Run(task_id=task.id, agent_name="reference")
    outcome = run_task(task, ReferenceAgent(seed=0), run)
    evaluation = evaluate(task, outcome.task_result)
    print(f"success={evaluation.success} detail={evaluation.details}")


if __name__ == "__main__":
    main()
