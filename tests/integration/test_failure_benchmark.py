"""Regression tests for benchmarks/failures.

Every assertion here was captured by actually running the real
runner -> evaluate -> diagnose pipeline against these task YAML files
(see docs/benchmarks.md for the full transcript) — nothing here is a
mocked or hand-typed expected value. These tests exist to catch the
demo suite silently drifting (e.g. if a detector's weight formula
changes and a scenario's dominant category flips) without anyone
noticing.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from arl.agents.reference import agent_from_task_metadata
from arl.analysis.engine import diagnose
from arl.domain.evaluation import FailureCategory
from arl.domain.run import Run, RunStatus
from arl.evaluation.benchmark import load_benchmark
from arl.evaluation.engine import evaluate
from arl.runtime.runner import run_task

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
FAILURES_DIR = REPO_ROOT / "benchmarks" / "failures"


def _run(task_id: str, seed: int = 0):  # noqa: ANN201
    if not FAILURES_DIR.is_dir():
        pytest.skip("benchmarks/failures not present in this environment")
    tasks = {t.id: t for t in load_benchmark(FAILURES_DIR)}
    task = tasks[task_id]
    run = Run(task_id=task.id, agent_name="reference", seed=seed)
    agent = agent_from_task_metadata(task, seed=seed)
    outcome = run_task(task, agent, run)
    evaluation = evaluate(task, outcome.task_result)
    diagnosis = diagnose(task, outcome.trajectory, evaluation)
    return outcome, evaluation, diagnosis


def test_all_five_failure_tasks_are_present() -> None:
    if not FAILURES_DIR.is_dir():
        pytest.skip("benchmarks/failures not present in this environment")
    tasks = load_benchmark(FAILURES_DIR)
    assert {t.id for t in tasks} == {
        "failure-looping-001",
        "failure-error-recovery-001",
        "failure-timeout-001",
        "failure-tool-misuse-001",
        "failure-partial-recovery-001",
    }


class TestLooping:
    def test_fails_the_task(self) -> None:
        _, evaluation, _ = _run("failure-looping-001")
        assert evaluation.success is False

    def test_diagnosed_as_looping(self) -> None:
        _, _, diagnosis = _run("failure-looping-001")
        assert diagnosis.category == FailureCategory.LOOPING_FAILURE
        assert diagnosis.confidence == 0.95
        assert diagnosis.recovered is False
        assert len(diagnosis.evidence) == 1
        assert len(diagnosis.evidence[0].event_ids) > 0

    def test_deterministic_across_seeds(self) -> None:
        categories = {_run("failure-looping-001", seed=s)[2].category for s in (0, 1, 42, 9999)}
        assert categories == {FailureCategory.LOOPING_FAILURE}


class TestErrorRecovery:
    def test_fails_the_task(self) -> None:
        _, evaluation, _ = _run("failure-error-recovery-001")
        assert evaluation.success is False

    def test_diagnosed_as_error_recovery_not_looping(self) -> None:
        """The scenario also matches detect_looping (3 identical failing
        reads in a row), but detect_error_recovery_failure's fixed 0.9
        weight beats looping's ratio-based weight here — this is real
        detector-priority behavior, not something hard-coded for the
        test. See docs/benchmarks.md for the full weight calculation.
        """
        _, _, diagnosis = _run("failure-error-recovery-001")
        assert diagnosis.category == FailureCategory.ERROR_RECOVERY_FAILURE
        assert diagnosis.confidence == 0.95
        assert diagnosis.wasted_tool_calls == 3
        assert diagnosis.recovered is False

    def test_deterministic_across_seeds(self) -> None:
        categories = {
            _run("failure-error-recovery-001", seed=s)[2].category for s in (0, 1, 42, 9999)
        }
        assert categories == {FailureCategory.ERROR_RECOVERY_FAILURE}


class TestTimeout:
    def test_fails_the_task(self) -> None:
        _, evaluation, _ = _run("failure-timeout-001")
        assert evaluation.success is False

    def test_diagnosed_as_timeout(self) -> None:
        outcome, _, diagnosis = _run("failure-timeout-001")
        assert outcome.task_result.terminated_reason == "timeout"
        assert outcome.run.status == RunStatus.TIMED_OUT
        assert diagnosis.category == FailureCategory.TIMEOUT_FAILURE
        assert diagnosis.confidence == 0.95
        assert diagnosis.wasted_tool_calls == 0  # times out before any tool call

    def test_zero_steps_taken(self) -> None:
        outcome, _, _ = _run("failure-timeout-001")
        assert outcome.task_result.steps_taken == 0

    def test_deterministic_across_seeds(self) -> None:
        categories = {_run("failure-timeout-001", seed=s)[2].category for s in (0, 1, 42, 9999)}
        assert categories == {FailureCategory.TIMEOUT_FAILURE}


class TestToolMisuse:
    def test_fails_the_task(self) -> None:
        _, evaluation, _ = _run("failure-tool-misuse-001")
        assert evaluation.success is False

    def test_diagnosed_as_tool_selection_failure(self) -> None:
        _, _, diagnosis = _run("failure-tool-misuse-001")
        assert diagnosis.category == FailureCategory.TOOL_SELECTION_FAILURE
        assert diagnosis.confidence == 0.70
        assert diagnosis.wasted_tool_calls == 1
        assert diagnosis.recovered is False

    def test_not_misdiagnosed_as_looping_or_error_recovery(self) -> None:
        # A single isolated tool error with no retry must not trip the
        # detectors built for repeated/looping patterns.
        _, _, diagnosis = _run("failure-tool-misuse-001")
        assert diagnosis.category not in (
            FailureCategory.LOOPING_FAILURE,
            FailureCategory.ERROR_RECOVERY_FAILURE,
        )

    def test_deterministic_across_seeds(self) -> None:
        categories = {
            _run("failure-tool-misuse-001", seed=s)[2].category for s in (0, 1, 42, 9999)
        }
        assert categories == {FailureCategory.TOOL_SELECTION_FAILURE}


class TestPartialRecovery:
    def test_task_succeeds_despite_initial_error(self) -> None:
        outcome, evaluation, _ = _run("failure-partial-recovery-001")
        assert evaluation.success is True
        # Prove the "initial error" part actually happened, i.e. this is a
        # real recovery and not just a task with no injected fault at all.
        error_events = [
            e for e in outcome.trajectory.events if e.event_type.value == "tool_error"
        ]
        assert len(error_events) == 1

    def test_diagnosis_is_none_and_recovered(self) -> None:
        """A successful run is never assigned a failure category, by
        design (see arl/analysis/engine.py) — this asserts that design
        decision holds for a run that specifically had a mid-run error,
        not just for a fully clean run.
        """
        _, _, diagnosis = _run("failure-partial-recovery-001")
        assert diagnosis.category == FailureCategory.NONE
        assert diagnosis.confidence == 1.0
        assert diagnosis.recovered is True
        assert diagnosis.evidence == []

    def test_deterministic_across_seeds(self) -> None:
        results = {
            _run("failure-partial-recovery-001", seed=s)[1].success for s in (0, 1, 42, 9999)
        }
        assert results == {True}
