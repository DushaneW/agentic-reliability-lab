"""Deterministic graders.

Every grader here is a pure function over (final filesystem state, task
result) with no model calls involved. LLM-judge grading is out of scope for
the core system by design — see docs/evaluation.md.
"""

from __future__ import annotations

from typing import Protocol

from arl.domain.task import GraderSpec, TaskResult


class Grader(Protocol):
    def grade(self, spec: GraderSpec, task_result: TaskResult) -> tuple[bool, float, str]:
        """Return (success, score in [0,1], detail message)."""
        ...


class FileEqualsGrader:
    """Passes iff a named file's content matches an expected string exactly."""

    def grade(self, spec: GraderSpec, task_result: TaskResult) -> tuple[bool, float, str]:
        path = spec.config["path"]
        expected = str(spec.config["expected"])
        files = task_result.output.get("files", {})
        actual = files.get(path)
        if actual is None:
            return False, 0.0, f"file '{path}' was never created"
        success = actual.strip() == expected.strip()
        return success, 1.0 if success else 0.0, f"expected {expected!r}, got {actual!r}"


class NumericFileEqualsGrader:
    """Passes iff a named file's content is numerically close to an expected value.

    Exact string equality on a float is the wrong tool: two mathematically
    equivalent summations (different associativity) can legitimately produce
    '-42.84' vs '-42.839999999999996'. Deterministic does not mean bit-exact.
    """

    def grade(self, spec: GraderSpec, task_result: TaskResult) -> tuple[bool, float, str]:
        path = spec.config["path"]
        expected = float(spec.config["expected"])
        tolerance = float(spec.config.get("tolerance", 1e-6))
        files = task_result.output.get("files", {})
        actual_raw = files.get(path)
        if actual_raw is None:
            return False, 0.0, f"file '{path}' was never created"
        try:
            actual = float(actual_raw.strip())
        except ValueError:
            return False, 0.0, f"file '{path}' does not contain a number: {actual_raw!r}"
        success = abs(actual - expected) <= tolerance
        return success, 1.0 if success else 0.0, f"expected ~{expected}, got {actual}"


class FileContainsGrader:
    """Passes iff a named file's content contains an expected substring."""

    def grade(self, spec: GraderSpec, task_result: TaskResult) -> tuple[bool, float, str]:
        path = spec.config["path"]
        expected_substring = str(spec.config["contains"])
        files = task_result.output.get("files", {})
        actual = files.get(path)
        if actual is None:
            return False, 0.0, f"file '{path}' was never created"
        success = expected_substring in actual
        return success, 1.0 if success else 0.0, f"looked for {expected_substring!r} in {actual!r}"


class TerminationReasonGrader:
    """Passes iff the run terminated for an expected reason (e.g. recovery tasks)."""

    def grade(self, spec: GraderSpec, task_result: TaskResult) -> tuple[bool, float, str]:
        expected_reason = spec.config["reason"]
        success = task_result.terminated_reason == expected_reason
        reason = task_result.terminated_reason
        return success, 1.0 if success else 0.0, f"terminated_reason={reason}"


GRADERS: dict[str, Grader] = {
    "file_equals": FileEqualsGrader(),
    "numeric_file_equals": NumericFileEqualsGrader(),
    "file_contains": FileContainsGrader(),
    "termination_reason": TerminationReasonGrader(),
}


def grade(spec: GraderSpec, task_result: TaskResult) -> tuple[bool, float, str]:
    if spec.kind not in GRADERS:
        raise ValueError(f"Unknown grader kind: {spec.kind}")
    return GRADERS[spec.kind].grade(spec, task_result)
