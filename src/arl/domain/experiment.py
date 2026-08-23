"""Experiment configuration and aggregate results."""

from __future__ import annotations

from pydantic import BaseModel, Field


class InterventionConfig(BaseModel):
    enabled: bool = False
    strategy: str = "none"
    probability_threshold: float = 0.5


class ExperimentConfig(BaseModel):
    name: str
    agent_type: str
    benchmark_path: str
    seed: int = 42
    fault_profile: dict[str, float] = Field(default_factory=dict)
    intervention: InterventionConfig = Field(default_factory=InterventionConfig)


class Experiment(BaseModel):
    id: str
    config: ExperimentConfig


class ExperimentResult(BaseModel):
    experiment_id: str
    total_tasks: int
    successes: int
    failures: int
    success_rate: float
    mean_steps: float
    mean_wall_time_seconds: float
    mean_tool_calls: float
    intervention_count: int = 0
    failure_category_counts: dict[str, int] = Field(default_factory=dict)
