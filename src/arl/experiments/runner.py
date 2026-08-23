"""Runs a full experiment (benchmark x agent x optional intervention config)."""

from __future__ import annotations

from collections import Counter

from arl.agents.reference import FaultProfile, ReferenceAgent
from arl.analysis.engine import diagnose
from arl.domain.experiment import ExperimentConfig, ExperimentResult
from arl.domain.run import Run
from arl.domain.task import Task
from arl.evaluation.engine import evaluate
from arl.reliability.model import ReliabilityModel
from arl.runtime.runner import run_task
from arl.storage.repository import Repository


def run_experiment(
    experiment_id: str,
    config: ExperimentConfig,
    tasks: list[Task],
    repo: Repository | None = None,
    fault_profile: FaultProfile | None = None,
    reliability_model: ReliabilityModel | None = None,
) -> ExperimentResult:
    del reliability_model  # reserved: live intervention during run_task is a future extension;
    # today's intervention experiments are evaluated post-hoc (see arl.experiments.intervention
    # and docs/research/limitations.md) rather than injected mid-run.

    successes = 0
    steps_list: list[int] = []
    wall_times: list[float] = []
    tool_call_counts: list[int] = []
    category_counts: Counter[str] = Counter()

    for i, task in enumerate(tasks):
        run = Run(
            task_id=task.id,
            agent_name=config.agent_type,
            experiment_id=experiment_id,
            seed=config.seed + i,
        )
        agent = ReferenceAgent(fault_profile=fault_profile or FaultProfile(), seed=run.seed)
        outcome = run_task(task, agent, run)
        evaluation = evaluate(task, outcome.task_result)
        diagnosis = diagnose(task, outcome.trajectory, evaluation)

        successes += int(evaluation.success)
        steps_list.append(outcome.task_result.steps_taken)
        wall_times.append(outcome.task_result.wall_time_seconds)
        tool_call_counts.append(
            len([e for e in outcome.trajectory.events if e.event_type.value == "tool_call"])
        )
        category_counts[diagnosis.category.value] += 1

        if repo is not None:
            repo.save_run(run)
            repo.save_events(outcome.trajectory.events)
            repo.save_task_result(outcome.task_result)
            repo.save_evaluation_result(evaluation)
            repo.save_failure_diagnosis(diagnosis)

    n = len(tasks)
    result = ExperimentResult(
        experiment_id=experiment_id,
        total_tasks=n,
        successes=successes,
        failures=n - successes,
        success_rate=successes / n if n else 0.0,
        mean_steps=sum(steps_list) / n if n else 0.0,
        mean_wall_time_seconds=sum(wall_times) / n if n else 0.0,
        mean_tool_calls=sum(tool_call_counts) / n if n else 0.0,
        intervention_count=0,
        failure_category_counts=dict(category_counts),
    )
    if repo is not None:
        from arl.domain.experiment import Experiment

        repo.save_experiment(Experiment(id=experiment_id, config=config))
        repo.save_experiment_result(result)
    return result
