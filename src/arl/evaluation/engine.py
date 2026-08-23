"""Turns a raw TaskResult into a graded EvaluationResult."""

from __future__ import annotations

from arl.domain.evaluation import EvaluationResult
from arl.domain.task import Task, TaskResult
from arl.evaluation.graders import grade


def evaluate(task: Task, task_result: TaskResult) -> EvaluationResult:
    success, score, detail = grade(task.grader, task_result)
    return EvaluationResult(
        run_id=task_result.run_id,
        task_id=task.id,
        success=success,
        score=score,
        grader_kind=task.grader.kind,
        details={"detail": detail},
    )
