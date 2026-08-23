"""Extracts a fixed-size numeric feature vector from a trajectory prefix.

Every feature here is computed purely from the event log — nothing peeks at
the final outcome. That's what makes "early warning" (predicting from a
partial trajectory) a meaningful test rather than a leak.
"""

from __future__ import annotations

from dataclasses import dataclass

from arl.domain.events import EventType, TrajectoryEvent

FEATURE_NAMES = [
    "trajectory_length",
    "tool_call_count",
    "tool_error_count",
    "tool_error_rate",
    "repeated_action_ratio",
    "action_diversity",
    "consecutive_error_pairs",
    "steps_since_last_error",
    "distinct_tools_used",
]


@dataclass(frozen=True)
class FeatureVector:
    values: dict[str, float]

    def as_list(self) -> list[float]:
        return [self.values[name] for name in FEATURE_NAMES]


def extract_features(events: list[TrajectoryEvent]) -> FeatureVector:
    tool_calls = [e for e in events if e.event_type == EventType.TOOL_CALL]
    tool_errors = [e for e in events if e.event_type == EventType.TOOL_ERROR]
    n_calls = len(tool_calls)

    signatures = [(c.payload.get("tool_name"), str(c.payload.get("arguments"))) for c in tool_calls]
    repeated = sum(1 for a, b in zip(signatures, signatures[1:], strict=False) if a == b)
    distinct_signatures = len(set(signatures))

    consecutive_error_pairs = 0
    for a, b in zip(events, events[1:], strict=False):
        if a.event_type == EventType.TOOL_ERROR and b.event_type == EventType.TOOL_ERROR:
            consecutive_error_pairs += 1

    steps_since_last_error = float(len(events))
    for event in reversed(events):
        if event.event_type == EventType.TOOL_ERROR:
            steps_since_last_error = float(len(events) - 1 - event.step_index)
            break

    values = {
        "trajectory_length": float(len(events)),
        "tool_call_count": float(n_calls),
        "tool_error_count": float(len(tool_errors)),
        "tool_error_rate": len(tool_errors) / n_calls if n_calls else 0.0,
        "repeated_action_ratio": repeated / n_calls if n_calls else 0.0,
        "action_diversity": distinct_signatures / n_calls if n_calls else 0.0,
        "consecutive_error_pairs": float(consecutive_error_pairs),
        "steps_since_last_error": steps_since_last_error,
        "distinct_tools_used": float(len({s[0] for s in signatures})),
    }
    return FeatureVector(values=values)
