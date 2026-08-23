"""Builds a labeled dataset for the reliability predictor.

Runs every task in a benchmark directory multiple times under a spread of
FaultProfiles and random seeds, then samples each resulting trajectory at
several completion fractions (20%, 40%, ..., 100%) to produce partial-
trajectory training examples. The label is always the *actual final
outcome* of that specific run (from the real deterministic grader) — early
prefixes of a run that ultimately failed are labeled as failures, which is
what makes this an "early warning" problem rather than trivial in-sample
prediction.
"""

from __future__ import annotations

from dataclasses import dataclass

from arl.agents.reference import FaultProfile, ReferenceAgent
from arl.domain.run import Run
from arl.domain.task import Task
from arl.evaluation.engine import evaluate
from arl.reliability.features import FeatureVector, extract_features
from arl.runtime.runner import run_task

FRACTIONS = (0.2, 0.4, 0.6, 0.8, 1.0)

FAULT_PROFILES = (
    FaultProfile(),  # clean
    FaultProfile(wrong_tool_rate=0.1),
    FaultProfile(repeat_action_rate=0.2),
    FaultProfile(ignore_error_rate=0.3),
    FaultProfile(premature_termination_rate=0.15),
    FaultProfile(wrong_tool_rate=0.15, ignore_error_rate=0.3),
    FaultProfile(wrong_tool_rate=0.3, repeat_action_rate=0.2, ignore_error_rate=0.4),
    FaultProfile(
        wrong_tool_rate=0.3,
        repeat_action_rate=0.2,
        ignore_error_rate=0.5,
        premature_termination_rate=0.1,
    ),
)


@dataclass
class DatasetExample:
    run_id: str
    task_id: str
    fraction: float
    features: FeatureVector
    label_failure: int  # 1 if the completed run failed, else 0


def generate_dataset(
    tasks: list[Task], seeds_per_profile: int = 3, base_seed: int = 0
) -> list[DatasetExample]:
    examples: list[DatasetExample] = []
    seed_counter = base_seed

    for task in tasks:
        for profile in FAULT_PROFILES:
            for _seed_offset in range(seeds_per_profile):
                seed = seed_counter
                seed_counter += 1
                # A deterministic id (not uuid4()) is what makes the
                # train/test split in model.py reproducible across runs:
                # train_test_split_examples splits by run_id, and a random
                # UUID would make that split land differently every time
                # even with a fixed random_state, silently breaking the
                # "reproducible" claim in docs/research/methodology.md.
                run_id = f"{task.id}-fp{hash(profile) & 0xffff:04x}-s{seed}"
                run = Run(task_id=task.id, agent_name="reference", seed=seed, id=run_id)
                agent = ReferenceAgent(fault_profile=profile, seed=seed)
                outcome = run_task(task, agent, run)
                evaluation = evaluate(task, outcome.task_result)
                label = 0 if evaluation.success else 1

                full_events = outcome.trajectory.events
                n = len(full_events)
                if n == 0:
                    continue
                seen_lengths: set[int] = set()
                for fraction in FRACTIONS:
                    cutoff = max(1, round(n * fraction))
                    if cutoff in seen_lengths:
                        continue
                    seen_lengths.add(cutoff)
                    prefix = full_events[:cutoff]
                    examples.append(
                        DatasetExample(
                            run_id=run.id,
                            task_id=task.id,
                            fraction=fraction,
                            features=extract_features(prefix),
                            label_failure=label,
                        )
                    )
    return examples
