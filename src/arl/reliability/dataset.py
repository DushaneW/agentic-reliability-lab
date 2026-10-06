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

# Named so that the profile is an explicit group key for evaluation
# (leave-one-profile-out) and so run ids do not depend on hash().
FAULT_PROFILES: dict[str, FaultProfile] = {
    "clean": FaultProfile(),
    "wrong_tool": FaultProfile(wrong_tool_rate=0.1),
    "repeat": FaultProfile(repeat_action_rate=0.2),
    "ignore_error": FaultProfile(ignore_error_rate=0.3),
    "early_stop": FaultProfile(premature_termination_rate=0.15),
    "wrong_tool_ignore_error": FaultProfile(wrong_tool_rate=0.15, ignore_error_rate=0.3),
    "mixed_heavy": FaultProfile(wrong_tool_rate=0.3, repeat_action_rate=0.2, ignore_error_rate=0.4),
    "mixed_severe": FaultProfile(
        wrong_tool_rate=0.3,
        repeat_action_rate=0.2,
        ignore_error_rate=0.5,
        premature_termination_rate=0.1,
    ),
}


@dataclass
class DatasetExample:
    run_id: str
    task_id: str
    profile_id: str
    fraction: float
    features: FeatureVector
    label_failure: int  # 1 if the completed run failed, else 0


def generate_dataset(
    tasks: list[Task], seeds_per_profile: int = 3, base_seed: int = 0
) -> list[DatasetExample]:
    examples: list[DatasetExample] = []
    seed_counter = base_seed

    for task in tasks:
        for profile_id, profile in FAULT_PROFILES.items():
            for _seed_offset in range(seeds_per_profile):
                seed = seed_counter
                seed_counter += 1
                # Deterministic id (not uuid4()) so the run-level split is reproducible.
                run_id = f"{task.id}-{profile_id}-s{seed}"
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
                            profile_id=profile_id,
                            fraction=fraction,
                            features=extract_features(prefix),
                            label_failure=label,
                        )
                    )
    return examples
