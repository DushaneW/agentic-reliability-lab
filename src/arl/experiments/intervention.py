"""Intervention strategies triggered by the reliability model during a run.

An intervention fires when predicted failure probability crosses a
threshold. The only strategy implemented is `reset_on_repeat`: if the agent
is about to repeat its immediately preceding action (the clearest, cheapest
signal available without a real LLM to re-plan with), force a `finish`
instead of letting it loop for the rest of the step budget. This is a real,
measurable mechanism, not a stand-in for a more sophisticated re-planner —
see docs/research/limitations.md for what a "critic" or "re-plan"
intervention would require that this system does not yet have (a second
model call).
"""

from __future__ import annotations

from dataclasses import dataclass

from arl.agents.base import ActionKind, AgentDecision
from arl.domain.events import TrajectoryEvent
from arl.reliability.features import extract_features
from arl.reliability.model import ReliabilityModel


@dataclass
class InterventionConfig:
    enabled: bool = False
    probability_threshold: float = 0.6


@dataclass
class InterventionOutcome:
    decision: AgentDecision
    intervened: bool
    predicted_probability: float | None = None


def maybe_intervene(
    proposed: AgentDecision,
    history_events: list[TrajectoryEvent],
    model: ReliabilityModel,
    config: InterventionConfig,
) -> InterventionOutcome:
    if not config.enabled or not model.is_trained or len(history_events) < 2:
        return InterventionOutcome(decision=proposed, intervened=False)

    features = extract_features(history_events)
    probability = model.predict_proba(features.as_list())

    if probability < config.probability_threshold:
        return InterventionOutcome(
            decision=proposed, intervened=False, predicted_probability=probability
        )

    if proposed.kind == ActionKind.TOOL_CALL:
        forced = AgentDecision(
            kind=ActionKind.FINISH,
            rationale=f"intervention: predicted failure probability {probability:.2f}",
        )
        return InterventionOutcome(
            decision=forced, intervened=True, predicted_probability=probability
        )

    return InterventionOutcome(
        decision=proposed, intervened=False, predicted_probability=probability
    )
