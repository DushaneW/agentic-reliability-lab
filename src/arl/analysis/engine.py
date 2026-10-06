"""Combines detector findings into a single FailureDiagnosis.

Priority order below reflects specificity, not severity: timeout and
looping are unambiguous from the event log alone, while premature
termination and "unknown" require ruling other explanations out first.
"""

from __future__ import annotations

from arl.analysis.detectors import (
    DetectorFinding,
    detect_error_recovery_failure,
    detect_incorrect_output,
    detect_looping,
    detect_premature_termination,
    detect_repeated_tool_errors,
    detect_timeout,
    detect_tool_selection_failure,
)
from arl.domain.evaluation import EvaluationResult, Evidence, FailureCategory, FailureDiagnosis
from arl.domain.run import Trajectory
from arl.domain.task import Task


def diagnose(task: Task, trajectory: Trajectory, evaluation: EvaluationResult) -> FailureDiagnosis:
    events = trajectory.events

    if evaluation.success:
        return FailureDiagnosis(
            run_id=evaluation.run_id,
            category=FailureCategory.NONE,
            confidence=1.0,
            evidence=[],
            wasted_tool_calls=0,
            recovered=True,
        )

    ordered_checks: list[tuple[FailureCategory, DetectorFinding | None]] = [
        (FailureCategory.TIMEOUT_FAILURE, detect_timeout(events)),
        (FailureCategory.LOOPING_FAILURE, detect_looping(events)),
        (FailureCategory.ERROR_RECOVERY_FAILURE, detect_error_recovery_failure(events)),
        (FailureCategory.TOOL_EXECUTION_FAILURE, detect_repeated_tool_errors(events)),
        (
            FailureCategory.PREMATURE_TERMINATION,
            detect_premature_termination(events, task.max_steps, evaluation.success),
        ),
        (FailureCategory.TOOL_SELECTION_FAILURE, detect_tool_selection_failure(events)),
        (FailureCategory.INCORRECT_OUTPUT, detect_incorrect_output(events)),
    ]

    best_category = FailureCategory.UNKNOWN_FAILURE
    best_finding: DetectorFinding | None = None
    best_weight = 0.0
    for category, finding in ordered_checks:
        if finding is not None and finding.weight > best_weight:
            best_category, best_finding, best_weight = category, finding, finding.weight

    wasted = sum(1 for e in events if e.event_type.value == "tool_error")
    if best_finding is None:
        return FailureDiagnosis(
            run_id=evaluation.run_id,
            category=FailureCategory.UNKNOWN_FAILURE,
            confidence=0.3,
            evidence=[Evidence(description="task failed but no known failure pattern matched")],
            wasted_tool_calls=wasted,
            recovered=False,
        )

    return FailureDiagnosis(
        run_id=evaluation.run_id,
        category=best_category,
        confidence=round(min(0.5 + best_weight * 0.5, 0.95), 2),
        evidence=[Evidence(description=best_finding.description, event_ids=best_finding.event_ids)],
        wasted_tool_calls=wasted,
        recovered=False,
    )
